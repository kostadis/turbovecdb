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
from .errors import EmbedderIdentityMismatchError, TurboVecError
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


def embedder_identity(callable_):
    """The identity string an embedder callable would be stored/checked
    under — a Python-side mirror of ``PyEmbedder::identity()``
    (``crates/turbovecdb-py/src/embedder.rs``): a plain function's
    ``__name__``, else ``f"{module}.{qualname}"`` for its class. Used by
    :meth:`Database.collection` to detect a conflicting embedder on an
    already-cached handle (C6) before it ever reaches the Rust core's
    authoritative GAP-1 guard."""
    name = getattr(callable_, "__name__", None)
    if isinstance(name, str):
        return name
    cls = type(callable_)
    return f"{getattr(cls, '__module__', '') or ''}.{getattr(cls, '__name__', '') or ''}"


class Collection:
    def __init__(self, coll_dir, *, dim=None, bit_width=DEFAULT_BIT_WIDTH,
                 metric="cosine", embedder=None, lock_timeout=_LOCK_TIMEOUT):
        self.dir = coll_dir
        self._tlock = threading.RLock()  # in-process structure guard
        # Cross-process write lock. Held around every write; the Rust core does
        # not lock, so this wrapper is the sole serialization point.
        lock_path = write_lock_path(coll_dir)
        # The lock file's parent (the database root) may not exist yet for a
        # brand-new database — FileLock can't create the lock file otherwise.
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        self._flock = FileLock(lock_path, timeout=lock_timeout)
        self._embedder = embedder
        self._has_embedder = embedder is not None
        # The Rust constructor does a read-then-write of bit_width/dim/
        # embedder_identity meta on first creation, with no locking of its
        # own (it runs before this wrapper exists to hold one). Without
        # this, two processes racing to create the same new collection with
        # different options could each proceed believing their own
        # in-memory values won, while the meta table ends up with whichever
        # wrote last (C7). Only pay the lock-acquire cost when the
        # collection doesn't look already-created, so re-opening an
        # existing collection — the common, hot case, e.g. service.py opens
        # one per request — stays lock-free as before.
        looks_new = not os.path.exists(os.path.join(coll_dir, "store.sqlite3"))
        if looks_new:
            with self._locked():
                self._core = _CoreCollection(coll_dir, dim, bit_width, metric, embedder, lock_timeout)
        else:
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

    def _check_embedder_identity_matches(self, current_identity):
        """Raise EmbedderIdentityMismatchError if `current_identity` doesn't
        match the collection's stored embedder identity. Mirrors the Rust
        GAP-1 guard (``check_embedder_identity`` in collection.rs) — used
        both as a fast, lock-free pre-check before embedding and as the
        authoritative re-check right before the locked write (R3)."""
        stored = self._core.meta_get("embedder_identity")
        if stored is not None and current_identity != stored:
            raise EmbedderIdentityMismatchError(
                f"embedder identity mismatch: stored {stored!r} != current "
                f"{current_identity!r}; use reembed() to change embedders"
            )

    def _write(self, ids, documents, metadatas, vectors, *, upsert):
        self._warn_vectors_bypass(vectors)
        core_method = self._core.upsert if upsert else self._core.add
        if documents is not None and vectors is None and self._embedder is not None:
            # Embed *before* acquiring the cross-process write lock. A slow
            # (e.g. network-backed) embedder call used to run inside the
            # locked section (the Rust core embeds internally as part of
            # add/upsert), blocking every other writer in every other
            # process for the whole embedding duration, not just the actual
            # SQLite write (R3). The vectors-given branch of the Rust core's
            # resolve_vectors() L2-normalizes on the way in (the same
            # vecmath::l2_normalize the embedder path already used), so
            # passing the raw embedder output through as vectors= here is
            # equivalent to today's embed-inside-the-lock path.
            current_identity = embedder_identity(self._embedder)
            self._check_embedder_identity_matches(current_identity)  # fail fast, unlocked
            computed_vectors = self._embedder(documents)
            with self._locked():
                # Re-check under the lock: a concurrent reembed() elsewhere
                # could have changed the stored identity in the window
                # between the unlocked pre-check/embed above and acquiring
                # the lock here.
                self._check_embedder_identity_matches(current_identity)
                core_method(ids, documents, metadatas, computed_vectors)
        else:
            with self._locked():
                core_method(ids, documents, metadatas, vectors)

    def add(self, *, ids, documents=None, metadatas=None, vectors=None):
        self._write(ids, documents, metadatas, vectors, upsert=False)

    def upsert(self, *, ids, documents=None, metadatas=None, vectors=None):
        self._write(ids, documents, metadatas, vectors, upsert=True)

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
