"""Database — a directory of collections.

A ``Database`` is a lightweight handle over a directory; each collection is a
subdirectory. Construction does no I/O — work is deferred to
:meth:`Database.collection`.

Name validation, path resolution, listing, and directory removal are handled
by the Rust core (``_core.Database``). The collection-handle cache and its
locking stay here: each cached ``Collection`` wrapper owns a cross-process
``FileLock`` and must be identity-stable per name, so a second cache in Rust
would be redundant and never actually used (see
``docs/rust-core-database-plan.md``).
"""

import logging
import os
import threading

from filelock import FileLock, Timeout

from ._core import Database as _CoreDatabase
from .collection import Collection, _LOCK_TIMEOUT, embedder_identity, write_lock_path
from .errors import CollectionNotFoundError, TurboVecError
from .index import DEFAULT_BIT_WIDTH

_log = logging.getLogger(__name__)

# Sentinel distinguishing "caller didn't pass this option" from "caller
# explicitly passed the default value" — needed so Database.collection()'s
# conflicting-options check (C6) doesn't false-positive on a caller who
# never had an opinion about bit_width/metric/embedder.
_UNSET = object()


class Database:
    def __init__(self, path):
        self._path = path
        self._collections = {}
        self._lock = threading.Lock()
        self._core = _CoreDatabase(path)

    @property
    def path(self):
        return self._path

    def collection(self, name, *, dim=None, bit_width=_UNSET,
                   metric=_UNSET, embedder=_UNSET, create=True, lock_timeout=None):
        """Open (or create) a collection by name.

        With ``create=False`` a missing collection raises
        :class:`CollectionNotFoundError`. Handles are cached per name; a
        second call for an already-cached handle that explicitly requests a
        different ``dim``/``bit_width``/``metric``/``embedder`` raises
        :class:`TurboVecError` rather than silently reusing the first call's
        handle with the caller's options ignored (C6).

        ``name`` must match ``[A-Za-z0-9_-]{1,128}``.
        """
        with self._lock:
            cached = self._collections.get(name)
            if cached is not None:
                self._check_no_conflict(name, cached, dim, bit_width, metric, embedder)
                return cached
            coll_dir = self._core.collection_dir(name)
            if not create and not os.path.isdir(coll_dir):
                raise CollectionNotFoundError(f"collection {name!r} not found at {coll_dir}")
            kwargs = dict(
                dim=dim,
                bit_width=DEFAULT_BIT_WIDTH if bit_width is _UNSET else bit_width,
                metric="cosine" if metric is _UNSET else metric,
                embedder=None if embedder is _UNSET else embedder,
            )
            if lock_timeout is not None:
                kwargs["lock_timeout"] = lock_timeout
            col = Collection(coll_dir, **kwargs)
            self._collections[name] = col
            return col

    @staticmethod
    def _check_no_conflict(name, cached, dim, bit_width, metric, embedder):
        """Raise if any explicitly-requested option conflicts with the
        cached handle's actual configuration. Options the caller didn't
        specify (``None`` for dim, ``_UNSET`` for the rest) are never a
        conflict — this only catches an explicit, differing request."""
        conflicts = []
        if dim is not None and cached.dim is not None and dim != cached.dim:
            conflicts.append(f"dim={dim!r} (cached handle: {cached.dim!r})")
        if bit_width is not _UNSET and bit_width != cached._bit_width:
            conflicts.append(f"bit_width={bit_width!r} (cached handle: {cached._bit_width!r})")
        if metric is not _UNSET:
            cached_metric = cached._meta_get("metric", "cosine")
            if metric != cached_metric:
                conflicts.append(f"metric={metric!r} (cached handle: {cached_metric!r})")
        if embedder is not _UNSET:
            cached_identity = cached.embedder_identity
            requested_identity = None if embedder is None else embedder_identity(embedder)
            if requested_identity != cached_identity:
                conflicts.append(
                    f"embedder identity {requested_identity!r} (cached handle: {cached_identity!r})"
                )
        if conflicts:
            raise TurboVecError(
                f"collection {name!r} is already open with different options: {'; '.join(conflicts)}"
            )

    def list_collections(self):
        with self._lock:
            return self._core.list_collections()

    def delete_collection(self, name):
        """Delete a collection and all its data.

        Acquires the collection's write lock before removing the directory
        to prevent races with concurrent writers in other processes. The lock
        file lives outside the collection directory (see
        ``collection.write_lock_path``) precisely so that holding it survives
        the ``rmtree`` below — a lock file inside the directory being deleted
        would let a concurrent opener recreate a new lock file at the same
        path and believe it holds the lock too (R1).

        Args:
            name: Name of the collection to delete

        Raises:
            ValueError: If the name is invalid
            CollectionNotFoundError: If the collection does not exist
            TurboVecError: If the write lock cannot be acquired
        """
        coll_dir = self._core.ensure_collection(name)

        # Close cached handle if present
        with self._lock:
            if name in self._collections:
                try:
                    self._collections[name].close()
                except Exception as exc:
                    _log.warning("error closing collection %r during delete: %s", name, exc)
                del self._collections[name]

        # Acquire the write lock to serialize with concurrent writers.
        lock_path = write_lock_path(coll_dir)
        flock = FileLock(lock_path, timeout=_LOCK_TIMEOUT)
        try:
            flock.acquire()
        except Timeout:
            raise TurboVecError(
                f"could not acquire write lock on {coll_dir!r} within "
                f"{_LOCK_TIMEOUT}s to delete collection"
            )

        # Re-check cache under lock — a concurrent collection(create=True)
        # may have recreated the collection between the initial checks and
        # the lock acquisition. Close and evict the new handle so rmtree
        # does not delete a just-created collection out from under someone.
        with self._lock:
            if name in self._collections:
                try:
                    self._collections[name].close()
                except Exception as exc:
                    _log.warning("error closing collection %r during delete: %s", name, exc)
                del self._collections[name]

        try:
            self._core.remove_collection_dir(name)
        finally:
            flock.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False  # don't suppress exceptions

    def close(self):
        with self._lock:
            for name, col in list(self._collections.items()):
                try:
                    col.close()
                except Exception as exc:
                    _log.warning("error closing collection %r: %s", name, exc)
            self._collections.clear()


def connect(path):
    """Open a turbovecdb database rooted at ``path`` (a directory of collections)."""
    return Database(path)
