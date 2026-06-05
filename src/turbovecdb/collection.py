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
from dataclasses import dataclass
from typing import Optional

_log = logging.getLogger(__name__)
_SQL_CHUNK = 900  # stay under SQLITE_MAX_VARIABLE_NUMBER on all SQLite builds

import numpy as np
from filelock import FileLock

from . import index as _idx
from .errors import DimensionMismatchError, EmbedderRequiredError
from .filters import combined_sql, where_to_sql

_RERANK_FLOOR = 50
_VALID_INCLUDE = frozenset({"documents", "metadatas", "distances", "vectors"})


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
                 metric="cosine", embedder=None):
        if metric != "cosine":
            raise ValueError(f"unsupported metric {metric!r} (only 'cosine' for now)")
        self.dir = coll_dir
        os.makedirs(coll_dir, exist_ok=True)
        self._db_path = os.path.join(coll_dir, "store.sqlite3")
        self._tvim_path = os.path.join(coll_dir, "index.tvim")
        self._metric = metric
        self._embedder = embedder
        self._tlock = threading.RLock()      # in-process structure guard
        self._flock = FileLock(os.path.join(coll_dir, "write.lock"))  # cross-process write lock
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

        self._index = None
        self._seen_gen = -1
        self._reload_index()

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

    def flush(self):
        """Persist the in-memory index to ``index.tvim`` (a cache for cold start).

        Takes the cross-process write lock because it writes ``tvim_gen`` to
        SQLite — that write must be serialized with other processes' writers."""
        with self._tlock, self._flock:
            if self._index is None or not self._dirty:
                return
            _idx.write_index_atomic(self._index, self._tvim_path)
            self._meta_set("tvim_gen", self._store_gen())
            self._conn.commit()
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

        with self._tlock, self._flock:
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
        with self._tlock, self._flock:
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

    @property
    def dim(self):
        return self._dim

    def close(self):
        with self._tlock:
            try:
                self.flush()
            finally:
                self._conn.commit()
                self._conn.close()
