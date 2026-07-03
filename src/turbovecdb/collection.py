"""Collection — a thin Python wrapper over the Rust core (``turbovecdb._core``).

The storage engine (rusqlite store + turbovec index + query/reembed) now lives in
Rust. This wrapper owns the cross-process file lock (a Python ``filelock``, held
around every write) and the in-process lock, exposes the historical
``turbovecdb.Collection`` API, and delegates the real work to
``_core.Collection``.

The result dataclasses live here because the Rust core constructs them by name
(``turbovecdb.collection.QueryResult`` etc.).

Storage layout (one directory per collection)::

    <dir>/store.sqlite3           durable source of truth (WAL)
    <dir>/index.tvim              rebuildable turbovec cache
    <dir>/../<name>.lock          cross-process write lock (filelock)

The write lock is deliberately a *sibling* of the collection directory, not
inside it: ``delete_collection`` holds this lock while removing ``<dir>``, and
a lock file living inside the directory being deleted would let a concurrent
opener recreate a *new* lock file at the same path and believe it holds the
lock too (see issue #35 / R1).
"""

import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

from filelock import FileLock, Timeout

from ._core import Collection as _CoreCollection
from .errors import TurboVecError
from .index import DEFAULT_BIT_WIDTH

_log = logging.getLogger(__name__)
_LOCK_TIMEOUT = 30


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
    """Result of a ``Collection.health()`` check."""

    ok: bool
    quick_check: str
    store_gen: int
    tvim_gen: int
    coherent: bool
    doc_count: int


@dataclass
class ReembedReport:
    """Result of a ``Collection.reembed()`` operation."""

    total_docs: int
    old_dim: int
    new_dim: int
    skipped: int
    elapsed_seconds: float


def _inc(include):
    """Normalize an ``include`` argument for the Rust core (list or None)."""
    return list(include) if include is not None else None


def write_lock_path(coll_dir):
    """Path to a collection's cross-process write lock.

    Deliberately a *sibling* of ``coll_dir`` (``<root>/<name>.lock``), not a
    file inside it — ``Database.delete_collection`` holds this lock while
    ``rmtree``-ing ``coll_dir``, and a lock file inside the directory being
    deleted would let a concurrent opener recreate a new lock file at the
    same path and believe it holds the collection's lock too (R1)."""
    root, name = os.path.split(coll_dir)
    return os.path.join(root, name + ".lock")


class Collection:
    def __init__(self, coll_dir, *, dim=None, bit_width=DEFAULT_BIT_WIDTH,
                 metric="cosine", embedder=None, lock_timeout=_LOCK_TIMEOUT):
        self.dir = coll_dir
        self._tlock = threading.RLock()  # in-process structure guard
        # Cross-process write lock. Held around every write; the Rust core does
        # not lock, so this wrapper is the sole serialization point.
        self._flock = FileLock(write_lock_path(coll_dir), timeout=lock_timeout)
        self._has_embedder = embedder is not None
        self._core = _CoreCollection(coll_dir, dim, bit_width, metric, embedder, lock_timeout)

    def _warn_vectors_bypass(self, vectors):
        if vectors is not None and self._has_embedder:
            _log.warning(
                "add/upsert with vectors= bypasses the configured embedder; use "
                "documents= to embed via the collection's embedder"
            )

    @contextmanager
    def _locked(self):
        """Acquire the in-process lock and cross-process file lock.

        Raises ``TurboVecError`` if the file lock cannot be acquired within the
        configured timeout (prevents indefinite blocking when a writer crashes
        while holding the lock)."""
        with self._tlock:
            try:
                with self._flock:
                    yield
            except Timeout:
                raise TurboVecError(
                    f"could not acquire write lock on {self.dir!r} within "
                    f"{self._flock.timeout:.1f}s"
                )

    # -- writes (serialized by the file lock) ---------------------------------

    def add(self, *, ids, documents=None, metadatas=None, vectors=None):
        self._warn_vectors_bypass(vectors)
        with self._locked():
            self._core.add(ids, documents, metadatas, vectors)

    def upsert(self, *, ids, documents=None, metadatas=None, vectors=None):
        self._warn_vectors_bypass(vectors)
        with self._locked():
            self._core.upsert(ids, documents, metadatas, vectors)

    def delete(self, *, ids=None, where=None):
        with self._locked():
            self._core.delete(ids, where)

    def update_metadata(self, *, ids, metadatas):
        with self._locked():
            self._core.update_metadata(ids, metadatas)

    def update_documents(self, *, ids, documents):
        with self._locked():
            self._core.update_documents(ids, documents)

    def clear(self):
        with self._locked():
            self._core.clear()

    def reembed(self, embedder, *, dim=None, bit_width=None, batch_size=256,
                on_progress=None, skip_empty="error"):
        with self._locked():
            return self._core.reembed(embedder, dim, bit_width, batch_size, on_progress, skip_empty)

    # -- reads ----------------------------------------------------------------

    def query(self, *, text=None, vector=None, k=10, where=None, where_document=None,
              include=("documents", "metadatas", "distances")):
        with self._tlock:
            return self._core.query(text, vector, k, where, where_document, _inc(include))

    def get(self, *, ids=None, where=None, where_document=None, limit=None,
            offset=None, include=("documents", "metadatas")):
        with self._tlock:
            return self._core.get(ids, where, where_document, limit, offset, _inc(include))

    def count(self):
        with self._tlock:
            return self._core.count()

    def health(self):
        with self._tlock:
            return self._core.health()

    # -- lifecycle ------------------------------------------------------------

    def flush(self):
        with self._locked():
            self._core.flush()

    def close(self):
        with self._locked():
            self._core.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
        return False

    @property
    def dim(self):
        return self._core.dim

    @property
    def _bit_width(self):
        return self._core.bit_width

    @property
    def embedder_identity(self):
        """The stored embedder identity string, or ``None`` if none recorded."""
        return self._core.meta_get("embedder_identity")

    # -- introspection shims (stable across the Rust flip) --------------------

    def _meta_get(self, key, default=None):
        v = self._core.meta_get(key)
        return v if v is not None else default

    def _store_gen(self):
        return self._core.store_gen()
