"""The turbovecdb Collection — turbovec ANN + a durable SQLite sidecar.

Storage layout (one directory per collection)::

    <dir>/store.sqlite3     durable source of truth (WAL)
    <dir>/index.tvim        rebuildable turbovec cache
    <dir>/write.lock        cross-process write lock (filelock)

SQLite::

    docs(uid INTEGER PRIMARY KEY,     -- turbovec external uint64 id
         str_id TEXT UNIQUE NOT NULL, -- caller's string id
         document TEXT, metadata TEXT,-- metadata is JSON
         vector BLOB)                 -- float32 bytes, L2-normalized
    meta(key TEXT PRIMARY KEY, value TEXT)

``meta`` holds ``dim``, ``bit_width``, ``metric``, ``next_uid`` and two
generation counters:

* ``store_gen`` — bumped on every committed write.
* ``tvim_gen``  — the ``store_gen`` the on-disk ``index.tvim`` reflects.

Concurrency. Writes take a cross-process ``filelock`` and, under it, refresh
``meta`` and reload the index if another process advanced ``store_gen`` — so
multiple writers never collide on uids or lose each other's rows. Reads are
lock-free: each query/get first reads ``store_gen`` and, if it advanced past
what this handle last saw, reloads the index (loading ``index.tvim`` when it is
current, else rebuilding from the SQLite vectors). SQLite is always the source
of truth; the ``.tvim`` is flushed on ``flush()``/``close()`` and is purely an
accelerator for the next cold start.

Distance. With ``metric="cosine"`` (the only metric today) the stored vectors
are L2-normalized and ``query`` returns ``distance = 1 - cosine ∈ [0, 2]`` via an
exact re-rank of the turbovec candidate pool — so turbovec's approximate,
unnormalized scores never reach the caller.
"""

import json
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

_log = logging.getLogger(__name__)
_SQL_CHUNK = 900  # stay under SQLITE_MAX_VARIABLE_NUMBER on all SQLite builds

import numpy as np
from filelock import FileLock, Timeout

from . import index as _idx
from .errors import DimensionMismatchError, EmbedderIdentityMismatchError, EmbedderRequiredError
from .filters import combined_sql, where_to_sql

_RERANK_FLOOR = 50
_VALID_INCLUDE = frozenset({"documents", "metadatas", "distances", "vectors"})
_LOCK_TIMEOUT = 30  # seconds; prevents indefinite blocking if a writer crashes
_SCHEMA_VERSION = 1  # increment when schema evolves; add _migrate_v<N> steps below


@dataclass
class QueryResult:
    """Flat, single-query result. Fields not in ``include`` are empty lists;
    ``vectors`` is ``None`` when not requested."""

    ids: list
    distances: list
    documents: list
    metadatas: list
    vectors: Optional[list] = None


@dataclass
class GetResult:
    ids: list
    documents: list
    metadatas: list
    vectors: Optional[list] = None


@dataclass
class HealthResult:
    """Result of a ``Collection.health()`` check.

    Attributes:
        ok: True if both SQLite and the index are healthy and coherent
        quick_check: Raw result of ``PRAGMA quick_check``
        store_gen: Current committed-write counter
        tvim_gen: Counter that the cached index reflects
        coherent: True if the cache is up-to-date with the store
        doc_count: Number of documents in the SQLite store
    """
    ok: bool
    quick_check: str
    store_gen: int
    tvim_gen: int
    coherent: bool
    doc_count: int


@dataclass
class ReembedReport:
    """Result of a ``Collection.reembed()`` operation.

    Attributes:
        total_docs: Total number of documents processed
        old_dim: Dimension before re-embedding
        new_dim: Dimension after re-embedding
        skipped: Number of documents skipped (due to empty content or drop policy)
        elapsed_seconds: Total time taken in seconds
    """
    total_docs: int
    old_dim: int
    new_dim: int
    skipped: int
    elapsed_seconds: float


def _resolve_include(include, *, default_distances):
    if include is None:
        keys = {"documents", "metadatas"}
        if default_distances:
            keys.add("distances")
    else:
        keys = {k for k in include if k in _VALID_INCLUDE}
    return {
        "documents": "documents" in keys,
        "metadatas": "metadatas" in keys,
        "distances": "distances" in keys,
        "vectors": "vectors" in keys,
    }


class Collection:
    def __init__(self, coll_dir, *, dim=None, bit_width=_idx.DEFAULT_BIT_WIDTH,
                 metric="cosine", embedder=None, lock_timeout=_LOCK_TIMEOUT):
        if metric != "cosine":
            raise ValueError(f"unsupported metric {metric!r} (only 'cosine' for now)")
        self.dir = coll_dir
        os.makedirs(coll_dir, exist_ok=True)
        self._db_path = os.path.join(coll_dir, "store.sqlite3")
        self._tvim_path = os.path.join(coll_dir, "index.tvim")
        self._metric = metric
        self._embedder = embedder
        self._tlock = threading.RLock()      # in-process structure guard
        self._flock = FileLock(os.path.join(coll_dir, "write.lock"), timeout=lock_timeout)
        self._dirty = False

        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS docs ("
            "uid INTEGER PRIMARY KEY, str_id TEXT UNIQUE NOT NULL, "
            "document TEXT, metadata TEXT, vector BLOB)"
        )
        self._conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        self._conn.commit()

        self._bit_width = int(self._meta_get("bit_width", bit_width))
        self._meta_set("bit_width", self._bit_width)
        self._meta_set("metric", metric)
        stored_dim = self._meta_get("dim", None)
        self._dim = int(stored_dim) if stored_dim is not None else None
        if self._dim is None and dim is not None:
            self._commit_dim(int(dim))
        self._next_uid = int(self._meta_get("next_uid", 0))
        self._conn.commit()

        self._migrate_schema()

        self._index = None
        self._seen_gen = -1
        self._reload_index()

    @contextmanager
    def _locked(self):
        """Acquire the in-process lock and cross-process file lock.

        Raises ``TurboVecError`` if the file lock cannot be acquired within
        the configured timeout (prevents indefinite blocking when a writer
        crashes while holding the lock).
        """
        with self._tlock:
            try:
                with self._flock:
                    yield
            except Timeout:
                from .errors import TurboVecError
                raise TurboVecError(
                    f"could not acquire write lock on {self.dir!r} within "
                    f"{self._flock.timeout:.1f}s"
                )

    # -- meta -----------------------------------------------------------------

    def _meta_get(self, key, default=None):
        row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row is not None else default

    def _meta_set(self, key, value):
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)),
        )

    def _store_gen(self):
        return int(self._meta_get("store_gen", 0))

    def _commit_dim(self, dim):
        if dim <= 0 or dim % 8 != 0:
            raise DimensionMismatchError(
                f"turbovec requires dim to be a positive multiple of 8, got {dim}"
            )
        self._dim = dim
        self._meta_set("dim", dim)

    # -- schema migration ----------------------------------------------------

    def _migrate_schema(self):
        """Forward-only schema migration. Runs at most once per collection open."""
        stored_version = int(self._meta_get("schema_version", 0))
        if stored_version >= _SCHEMA_VERSION:
            return
        # Apply pending migrations sequentially.
        for version in range(stored_version + 1, _SCHEMA_VERSION + 1):
            meth = getattr(self, f"_migrate_v{version}", None)
            if meth is not None:
                meth()
        self._meta_set("schema_version", _SCHEMA_VERSION)
        self._conn.commit()

    # -- index lifecycle ------------------------------------------------------

    def _reload_index(self):
        """(Re)build the in-memory index to reflect the current store_gen."""
        if self._dim is None:
            self._index = None
            self._seen_gen = self._store_gen()
            return
        store_gen = self._store_gen()
        tvim_gen = int(self._meta_get("tvim_gen", -1))
        if os.path.exists(self._tvim_path) and tvim_gen == store_gen:
            try:
                self._index = _idx.load_index(self._tvim_path)
                self._seen_gen = store_gen
                return
            except Exception:
                pass  # fall through to rebuild from the source of truth
        rows = self._conn.execute("SELECT uid, vector FROM docs").fetchall()
        uids = [r[0] for r in rows]
        vecs = [np.frombuffer(r[1], dtype=np.float32) for r in rows]
        self._index = _idx.build_index(self._dim, self._bit_width, uids, vecs)
        self._seen_gen = store_gen
        self._dirty = True  # in-memory index now newer than on-disk .tvim

    def _ensure_current(self):
        """Cheap staleness check for readers: reload if another writer advanced
        store_gen past what this handle last saw."""
        if self._store_gen() != self._seen_gen:
            self._reload_index()

    def _checkpoint_wal(self):
        """Truncate the WAL to prevent unbounded growth on busy writers."""
        try:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            _log.warning("WAL checkpoint failed", exc_info=True)

    def flush(self):
        """Persist the in-memory index to ``index.tvim`` (a cache for cold start).

        Takes the cross-process write lock because it writes ``tvim_gen`` to
        SQLite — that write must be serialized with other processes' writers."""
        with self._locked():
            if self._index is None or not self._dirty:
                return
            _idx.write_index_atomic(self._index, self._tvim_path)
            self._meta_set("tvim_gen", self._store_gen())
            self._conn.commit()
            self._checkpoint_wal()
            self._dirty = False

    # -- embedding ------------------------------------------------------------

    def _embed(self, texts):
        if self._embedder is None:
            raise EmbedderRequiredError(
                "text was provided but this collection has no embedder; "
                "pass vectors instead, or create the collection with embedder=..."
            )
        return np.asarray(self._embedder(list(texts)), dtype=np.float32)

    # -- writes ---------------------------------------------------------------

    def add(self, *, ids, documents=None, metadatas=None, vectors=None):
        self._write(ids=ids, documents=documents, metadatas=metadatas,
                    vectors=vectors, replace=False)

    def upsert(self, *, ids, documents=None, metadatas=None, vectors=None):
        self._write(ids=ids, documents=documents, metadatas=metadatas,
                    vectors=vectors, replace=True)

    def _resolve_vectors(self, documents, vectors, n):
        if vectors is not None:
            return _idx.l2_normalize(vectors)
        if documents is None:
            raise ValueError("add/upsert requires either vectors= or documents= (with an embedder)")
        return _idx.l2_normalize(self._embed(documents))

    def _write(self, *, ids, documents, metadatas, vectors, replace):
        n = len(ids)
        for label, seq in (("documents", documents), ("metadatas", metadatas), ("vectors", vectors)):
            if seq is not None and len(seq) != n:
                raise ValueError(f"{label} length {len(seq)} != ids length {n}")
        if n == 0:
            return
        vecs = self._resolve_vectors(documents, vectors, n)
        docs = documents if documents is not None else [""] * n

        with self._locked():
            # Refresh under the lock so concurrent writers see each other's rows.
            self._next_uid = int(self._meta_get("next_uid", 0))
            self._ensure_current()
            if self._dim is None:
                self._commit_dim(int(vecs.shape[1]))
                self._index = _idx.new_index(self._dim, self._bit_width)
            elif vecs.shape[1] != self._dim:
                raise DimensionMismatchError(
                    f"vector dim {vecs.shape[1]} != collection dim {self._dim}"
                )

            new_uids, new_vecs = [], []
            for i, str_id in enumerate(ids):
                meta_json = json.dumps(metadatas[i] if metadatas else {})
                vec_bytes = vecs[i].tobytes()
                row = self._conn.execute("SELECT uid FROM docs WHERE str_id=?", (str_id,)).fetchone()
                if row is not None:
                    if not replace:
                        raise ValueError(f"id {str_id!r} already exists (use upsert)")
                    uid = int(row[0])
                    self._index.remove(np.uint64(uid))
                    self._conn.execute(
                        "UPDATE docs SET document=?, metadata=?, vector=? WHERE uid=?",
                        (docs[i], meta_json, vec_bytes, uid),
                    )
                else:
                    uid = self._next_uid
                    self._next_uid += 1
                    self._conn.execute(
                        "INSERT INTO docs(uid, str_id, document, metadata, vector) "
                        "VALUES(?, ?, ?, ?, ?)",
                        (uid, str_id, docs[i], meta_json, vec_bytes),
                    )
                new_uids.append(uid)
                new_vecs.append(vecs[i])

            self._index.add_with_ids(
                np.ascontiguousarray(np.stack(new_vecs), dtype=np.float32),
                np.array(new_uids, dtype=np.uint64),
            )
            self._meta_set("next_uid", self._next_uid)
            self._meta_set("store_gen", self._store_gen() + 1)
            self._conn.commit()
            self._seen_gen = self._store_gen()
            self._dirty = True

    def delete(self, *, ids=None, where=None):
        with self._locked():
            self._ensure_current()
            uids = self._select_uids(ids=ids, where=where)
            if not uids:
                return
            for uid in uids:
                try:
                    self._index.remove(np.uint64(uid))
                except Exception:
                    _log.warning("ANN index remove failed for uid %d; ghost entry will clear on next reload", uid)
            for i in range(0, len(uids), _SQL_CHUNK):
                chunk = uids[i:i + _SQL_CHUNK]
                qmarks = ",".join("?" for _ in chunk)
                self._conn.execute(f"DELETE FROM docs WHERE uid IN ({qmarks})", chunk)
            self._meta_set("store_gen", self._store_gen() + 1)
            self._conn.commit()
            self._seen_gen = self._store_gen()
            self._dirty = True

    def _select_uids(self, *, ids=None, where=None):
        wsql, wparams = where_to_sql(where)
        if ids is not None:
            # Chunk str_id lookups to stay under SQLITE_MAX_VARIABLE_NUMBER.
            uid_rows = []
            for i in range(0, max(len(ids), 1), _SQL_CHUNK):
                chunk = ids[i:i + _SQL_CHUNK]
                qmarks = ",".join("?" for _ in chunk)
                clause = f"WHERE str_id IN ({qmarks})"
                if wsql:
                    clause += f" AND ({wsql})"
                uid_rows.extend(
                    self._conn.execute(
                        f"SELECT uid FROM docs {clause}", list(chunk) + wparams
                    ).fetchall()
                )
            return [int(r[0]) for r in uid_rows]
        clause = (f" WHERE {wsql}") if wsql else ""
        rows = self._conn.execute(f"SELECT uid FROM docs{clause}", wparams).fetchall()
        return [int(r[0]) for r in rows]

    # -- reads ----------------------------------------------------------------

    def query(self, *, text=None, vector=None, k=10, where=None, where_document=None,
              include=("documents", "metadatas", "distances")):
        if (text is None) == (vector is None):
            raise ValueError("exactly one of text / vector is required")
        inc = _resolve_include(include, default_distances=True)
        if text is not None:
            vector = self._embed([text])[0]
        q = _idx.l2_normalize(np.asarray(vector, dtype=np.float32))

        with self._tlock:
            self._ensure_current()
            if self._index is None or self._dim is None:
                return QueryResult([], [], [], [], [] if inc["vectors"] else None)

            allow = None
            if where or where_document:
                sql, params = combined_sql(where, where_document)
                if sql:
                    rows = self._conn.execute(
                        f"SELECT uid FROM docs WHERE {sql}", params
                    ).fetchall()
                    allow_ids = [int(r[0]) for r in rows]
                    if not allow_ids:
                        return QueryResult([], [], [], [], [] if inc["vectors"] else None)
                    allow = np.array(allow_ids, dtype=np.uint64)

            pool = max(k, _RERANK_FLOOR)
            _, cand = self._index.search(q, k=pool, allowlist=allow)
            cand_uids = [int(u) for u in cand[0].tolist()]
            hits = self._rerank(q[0], cand_uids, k, inc)
            return QueryResult(
                ids=[h["id"] for h in hits],
                distances=[h["distance"] for h in hits] if inc["distances"] else [],
                documents=[h["document"] for h in hits] if inc["documents"] else [],
                metadatas=[h["metadata"] for h in hits] if inc["metadatas"] else [],
                vectors=[h["vector"] for h in hits] if inc["vectors"] else None,
            )

    def _rerank(self, qvec, cand_uids, k, inc):
        if not cand_uids:
            return []
        qmarks = ",".join("?" for _ in cand_uids)
        rows = self._conn.execute(
            f"SELECT uid, str_id, document, metadata, vector FROM docs WHERE uid IN ({qmarks})",
            cand_uids,
        ).fetchall()
        by_uid = {int(r[0]): r for r in rows}
        scored = []
        for uid in cand_uids:  # turbovec order is the tiebreak
            r = by_uid.get(uid)
            if r is None:
                continue
            v = np.frombuffer(r[4], dtype=np.float32)
            distance = 1.0 - float(np.dot(qvec, v))  # both L2-normalized → cosine
            scored.append({
                "id": r[1],
                "document": r[2] if inc["documents"] else None,
                "metadata": json.loads(r[3]) if inc["metadatas"] and r[3] else {},
                "distance": distance,
                "vector": v.tolist() if inc["vectors"] else None,
            })
        scored.sort(key=lambda h: h["distance"])
        return scored[:k]

    def get(self, *, ids=None, where=None, where_document=None, limit=None,
            offset=None, include=("documents", "metadatas")):
        inc = _resolve_include(include, default_distances=False)
        frags, params = [], []
        if ids is not None:
            qmarks = ",".join("?" for _ in ids)
            frags.append(f"str_id IN ({qmarks})")
            params.extend(ids)
        wsql, wparams = where_to_sql(where)
        if wsql:
            frags.append(wsql)
            params.extend(wparams)
        from .filters import where_document_to_sql
        wdsql, wdparams = where_document_to_sql(where_document)
        if wdsql:
            frags.append(wdsql)
            params.extend(wdparams)
        clause = (" WHERE " + " AND ".join(frags)) if frags else ""
        sql = f"SELECT str_id, document, metadata, vector FROM docs{clause} ORDER BY uid"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
            if offset is not None:
                sql += " OFFSET ?"
                params.append(int(offset))
        with self._tlock:
            rows = self._conn.execute(sql, params).fetchall()
        out_ids, out_docs, out_metas, out_vecs = [], [], [], []
        for r in rows:
            out_ids.append(r[0])
            out_docs.append(r[1] if inc["documents"] else None)
            out_metas.append(json.loads(r[2]) if inc["metadatas"] and r[2] else {})
            if inc["vectors"]:
                out_vecs.append(np.frombuffer(r[3], dtype=np.float32).tolist())
        return GetResult(
            ids=out_ids,
            documents=out_docs,
            metadatas=out_metas,
            vectors=out_vecs if inc["vectors"] else None,
        )

    def count(self):
        with self._tlock:
            return int(self._conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0])

    def _get_embedder_identity(self, embedder):
        """Extract a stable identifier for the embedder function."""
        if hasattr(embedder, "__name__"):
            return embedder.__name__
        elif hasattr(embedder, "__class__"):
            return f"{embedder.__class__.__module__}.{embedder.__class__.__name__}"
        else:
            return "unknown_embedder"

    def _recommit_dim(self, new_dim, bit_width=None):
        """Update collection dimension and rebuild index.

        Args:
            new_dim: The new dimension (must be positive multiple of 8)
            bit_width: Optional new bit_width (2/3/4). If None, keeps current.
        """
        if new_dim <= 0 or new_dim % 8 != 0:
            raise DimensionMismatchError(
                f"turbovec requires dim to be a positive multiple of 8, got {new_dim}"
            )
        if bit_width is not None and bit_width not in (2, 3, 4):
            raise ValueError(f"bit_width must be 2, 3, or 4, got {bit_width!r}")

        with self._locked():
            self._ensure_current()
            self._dim = new_dim
            if bit_width is not None:
                self._bit_width = bit_width
            self._meta_set("dim", self._dim)
            self._meta_set("bit_width", self._bit_width)
            self._conn.commit()
            self._reload_index()
            self.flush()

    def reembed(self, embedder, *, dim=None, bit_width=None, batch_size=256, on_progress=None, skip_empty="error"):
        """Recompute every vector in place from stored documents using a new embedder.

        Args:
            embedder: Callable that takes list of texts and returns list of vectors
            dim: Optional new dimension to validate against embedder output
            bit_width: Optional new bit_width (2/3/4) for quantization
            batch_size: Number of documents to embed in each batch
            on_progress: Optional callback function with signature (done, total)
            skip_empty: Policy for empty documents ("error", "keep", or "drop")

        Returns:
            ReembedReport: Summary of re-embedding operation
        """
        import time

        # Pre-validation (before acquiring locks)
        if skip_empty not in ("error", "keep", "drop"):
            raise ValueError(f"skip_empty must be 'error', 'keep', or 'drop', got {skip_empty!r}")
        if bit_width is not None:
            if bit_width not in (2, 3, 4):
                raise ValueError(f"bit_width must be 2, 3, or 4, got {bit_width!r}")
        if not callable(embedder):
            raise ValueError("embedder must be a callable")
        if dim is not None:
            if dim <= 0 or dim % 8 != 0:
                raise DimensionMismatchError(
                    f"turbovec requires dim to be a positive multiple of 8, got {dim}"
                )

        # Get embedder identity (for GAP-1 integration)
        embedder_identity = self._get_embedder_identity(embedder)

        # Get total count (before locks)
        total_docs = self.count()
        if total_docs == 0:
            return ReembedReport(0, self._dim, self._dim, 0, 0)

        # Acquire write lock and perform re-embedding
        with self._locked():
            self._ensure_current()

            old_dim = self._dim
            old_bit_width = self._bit_width
            processed = 0
            skipped = 0
            start_time = time.time()

            cursor = self._conn.cursor()
            cursor.execute("SELECT uid, document FROM docs ORDER BY uid")

            batch = []
            batch_uids = []
            updates = []  # (vector_bytes, uid) pairs

            while True:
                row = cursor.fetchone()
                if not row:
                    break

                uid, document = row

                # Handle empty document
                if not document:
                    if skip_empty == "error":
                        raise ValueError(f"Cannot re-embed empty document with uid {uid}")
                    elif skip_empty == "drop":
                        skipped += 1
                        processed += 1
                        if on_progress:
                            on_progress(processed, total_docs)
                        continue
                    else:  # keep
                        # Keep old vector - read it from DB and re-normalize
                        cursor2 = self._conn.cursor()
                        cursor2.execute("SELECT vector FROM docs WHERE uid=?", (uid,))
                        vec_row = cursor2.fetchone()
                        if vec_row:
                            old_vec = np.frombuffer(vec_row[0], dtype=np.float32)
                            new_vec = _idx.l2_normalize(old_vec.reshape(1, -1))[0]
                            updates.append((new_vec.tobytes(), uid))
                        skipped += 1
                        processed += 1
                        if on_progress:
                            on_progress(processed, total_docs)
                        continue

                # Add to batch
                batch.append(document)
                batch_uids.append(uid)

                # Process batch when full
                if len(batch) >= batch_size:
                    # Embed batch
                    try:
                        new_vecs = np.asarray(embedder(batch), dtype=np.float32)
                    except Exception as e:
                        raise ValueError(f"Embedder failed on batch: {e}")

                    # Validate dimension consistency across batches
                    batch_dim = new_vecs.shape[1]
                    if batch_dim != old_dim and dim is not None and batch_dim != dim:
                        raise DimensionMismatchError(
                            f"Embedder returned vectors of dimension {batch_dim}, "
                            f"but dim was explicitly set to {dim}"
                        )

                    # Normalize vectors
                    new_vecs = _idx.l2_normalize(new_vecs)

                    # Collect updates
                    for i, uid in enumerate(batch_uids):
                        updates.append((new_vecs[i].tobytes(), uid))

                    # Track progress
                    processed += len(batch)
                    if on_progress:
                        on_progress(processed, total_docs)

                    # Reset batch
                    batch = []
                    batch_uids = []

            # Process remaining documents
            if batch:
                try:
                    new_vecs = np.asarray(embedder(batch), dtype=np.float32)
                except Exception as e:
                    raise ValueError(f"Embedder failed on final batch: {e}")

                # Validate dimension consistency
                batch_dim = new_vecs.shape[1]
                if batch_dim != old_dim and dim is not None and batch_dim != dim:
                    raise DimensionMismatchError(
                        f"Embedder returned vectors of dimension {batch_dim}, "
                        f"but dim was explicitly set to {dim}"
                    )

                # Normalize vectors
                new_vecs = _idx.l2_normalize(new_vecs)

                # Collect updates
                for i, uid in enumerate(batch_uids):
                    updates.append((new_vecs[i].tobytes(), uid))

                processed += len(batch)
                if on_progress:
                    on_progress(processed, total_docs)

            # Get new dimension from first update (or keep old if all were skipped)
            new_dim = old_dim
            new_bit_width = old_bit_width
            if updates:
                new_dim = len(updates[0][0]) // 4  # 4 bytes per float32

            # Update dimension and bit_width if changed
            if new_dim != old_dim or (bit_width is not None and bit_width != old_bit_width):
                if dim is not None and new_dim != dim:
                    raise DimensionMismatchError(
                        f"Embedder returned vectors of dimension {new_dim}, "
                        f"but dim was explicitly set to {dim}"
                    )
                # Update dim/bit_width in meta but don't rebuild index yet
                self._dim = new_dim
                if bit_width is not None:
                    self._bit_width = bit_width
                self._meta_set("dim", self._dim)
                self._meta_set("bit_width", self._bit_width)

            # Batch update vectors
            if updates:
                self._conn.executemany(
                    "UPDATE docs SET vector=? WHERE uid=?",
                    updates
                )

            # Update metadata
            self._meta_set("store_gen", self._store_gen() + 1)
            self._meta_set("embedder_identity", embedder_identity)
            self._conn.commit()

            # Rebuild index with new vectors and new dimension
            self._reload_index()
            self.flush()

            elapsed = time.time() - start_time
            return ReembedReport(total_docs, old_dim, new_dim, skipped, elapsed)

    @property
    def dim(self):
        return self._dim

    def health(self):
        """Run an integrity check on the collection.

        Returns:
            HealthResult with SQLite quick_check result, store-gen counters,
            cache coherence status, and document count.

        Raises:
            TurboVecError: If the SQLite quick_check reports corruption.
        """
        with self._tlock:
            qc = self._conn.execute("PRAGMA quick_check").fetchone()[0]
            store_gen = self._store_gen()
            tvim_gen = int(self._meta_get("tvim_gen", -1))
            doc_count = int(self._conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0])
            coherent = store_gen == tvim_gen
            ok = qc == "ok"
            if not ok:
                from .errors import TurboVecError
                raise TurboVecError(
                    f"SQLite integrity check failed for {self.dir!r}: {qc}"
                )
            return HealthResult(
                ok=ok,
                quick_check=qc,
                store_gen=store_gen,
                tvim_gen=tvim_gen,
                coherent=coherent,
                doc_count=doc_count,
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
        return False

    def close(self):
        with self._tlock:
            try:
                self.flush()
            finally:
                self._conn.commit()
                self._checkpoint_wal()
                self._conn.close()
