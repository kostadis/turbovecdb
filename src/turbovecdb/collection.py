"""Collection — a thin, lock-free Python wrapper over the Rust core
(``turbovecdb._core``).

The storage engine (rusqlite store + turbovec index + query/reembed) and *all
locking* now live in Rust. ``_core.Collection`` owns both the in-process lock
(a ``Mutex`` around the core, acquired inside ``allow_threads``) and the
cross-process write lock (an ``flock`` on the sibling ``<root>/<name>.lock``,
held for every write and for a brand-new collection's first-creation meta
init, C7). This wrapper only shapes arguments, exposes the historical
``turbovecdb.Collection`` API, and delegates to the core — it holds no lock of
its own.

The result dataclasses live here because the Rust core constructs them by name
(``turbovecdb.collection.QueryResult`` etc.).

Storage layout (one directory per collection)::

    <dir>/store.sqlite3           durable source of truth (WAL)
    <dir>/index.tvim              rebuildable turbovec cache
    <dir>/../<name>.lock          cross-process write lock (Rust flock)

The write lock is deliberately a *sibling* of the collection directory, not
inside it: ``delete_collection`` holds this lock while removing ``<dir>``, and
a lock file living inside the directory being deleted would let a concurrent
opener recreate a *new* lock file at the same path and believe it holds the
lock too (see issue #35 / R1).

Note: the core embeds *under* its non-reentrant in-process ``Mutex``, so an
embedder — or a ``reembed`` ``on_progress`` callback — that re-enters the same
collection (e.g. calls ``count()``) would deadlock. Embedders are expected to
map text→vectors without touching the collection.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

from ._core import Collection as _CoreCollection
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
    wal_size_bytes: Optional[int] = None


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


def embedder_identity(callable_):
    """The identity string an embedder callable would be stored/checked
    under — a Python-side mirror of ``PyEmbedder::identity()``
    (``crates/turbovecdb-py/src/embedder.rs``): a plain function's
    ``__name__``, else ``f"{module}.{qualname}"`` for its class. Used by
    :meth:`Database.collection` to detect a conflicting embedder on an
    already-cached handle (C6) before it ever reaches the Rust core's
    authoritative GAP-1 guard.

    If the callable defines ``_embedder_identity`` (as a string, callable,
    or ``@property``), that takes precedence — allowing class instances
    with different configurations to produce unique identities."""
    custom = getattr(callable_, "_embedder_identity", None)
    if custom is not None:
        return custom() if callable(custom) else str(custom)
    name = getattr(callable_, "__name__", None)
    if isinstance(name, str):
        return name
    cls = type(callable_)
    return f"{getattr(cls, '__module__', '') or ''}.{getattr(cls, '__name__', '') or ''}"


class Collection:
    def __init__(self, coll_dir, *, dim=None, bit_width=DEFAULT_BIT_WIDTH,
                 metric="cosine", embedder=None, lock_timeout=_LOCK_TIMEOUT):
        self.dir = coll_dir
        self._embedder = embedder
        self._has_embedder = embedder is not None
        # All locking lives in the core: _CoreCollection's Mutex serializes
        # in-process, and its flock on the sibling <root>/<name>.lock
        # serializes across processes (including the first-creation C7 path).
        self._core = _CoreCollection(coll_dir, dim, bit_width, metric, embedder, lock_timeout)

    def _warn_vectors_bypass(self, vectors):
        if vectors is not None and self._has_embedder:
            _log.warning(
                "add/upsert with vectors= bypasses the configured embedder; use "
                "documents= to embed via the collection's embedder"
            )

    # -- writes (serialized by the Rust core's Mutex + flock) -----------------

    def _write(self, ids, documents, metadatas, vectors, *, upsert):
        # All embed-before-lock sequencing (identity pre-check → embed →
        # acquire the write lock → under-lock identity re-check → write) lives
        # in the Rust core's write path (I5/R3); the wrapper just shapes
        # arguments and delegates. The ``documents=`` case is passed straight
        # through so the core embeds it under its own lock discipline.
        self._warn_vectors_bypass(vectors)
        core_method = self._core.upsert if upsert else self._core.add
        core_method(ids, documents, metadatas, vectors)

    def add(self, *, ids, documents=None, metadatas=None, vectors=None):
        self._write(ids, documents, metadatas, vectors, upsert=False)

    def upsert(self, *, ids, documents=None, metadatas=None, vectors=None):
        self._write(ids, documents, metadatas, vectors, upsert=True)

    def delete(self, *, ids=None, where=None):
        self._core.delete(ids, where)

    def update_metadata(self, *, ids, metadatas):
        self._core.update_metadata(ids, metadatas)

    def update_documents(self, *, ids, documents):
        self._core.update_documents(ids, documents)

    def clear(self):
        self._core.clear()

    def reembed(self, embedder, *, dim=None, bit_width=None, batch_size=256,
                on_progress=None, skip_empty="error"):
        return self._core.reembed(embedder, dim, bit_width, batch_size, on_progress, skip_empty)

    # -- reads ----------------------------------------------------------------

    def query(self, *, text=None, vector=None, k=10, where=None, where_document=None,
              include=("documents", "metadatas", "distances")):
        return self._core.query(text, vector, k, where, where_document, _inc(include))

    def query_batch(self, *, vectors, k=10, where=None, where_document=None,
                    include=("documents", "metadatas", "distances")):
        return self._core.query_batch(vectors, k, where, where_document, _inc(include))

    def get(self, *, ids=None, where=None, where_document=None, limit=None,
            offset=None, include=("documents", "metadatas")):
        return self._core.get(ids, where, where_document, limit, offset, _inc(include))

    def count(self):
        return self._core.count()

    def health(self):
        return self._core.health()

    # -- lifecycle ------------------------------------------------------------

    def flush(self):
        self._core.flush()

    def close(self):
        self._core.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.flush()
        except Exception as e:
            if exc_type is None:
                raise
            _log.warning("flush failed during context manager exit: %s", e)
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
