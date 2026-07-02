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
from .collection import Collection, _LOCK_TIMEOUT
from .errors import CollectionNotFoundError, TurboVecError
from .index import DEFAULT_BIT_WIDTH

_log = logging.getLogger(__name__)


class Database:
    def __init__(self, path):
        self._path = path
        self._collections = {}
        self._lock = threading.Lock()
        self._core = _CoreDatabase(path)

    @property
    def path(self):
        return self._path

    def collection(self, name, *, dim=None, bit_width=DEFAULT_BIT_WIDTH,
                   metric="cosine", embedder=None, create=True, lock_timeout=None):
        """Open (or create) a collection by name.

        With ``create=False`` a missing collection raises
        :class:`CollectionNotFoundError`. Handles are cached per name; the first
        call's options win for a cached handle.

        ``name`` must match ``[A-Za-z0-9_-]{1,128}``.
        """
        with self._lock:
            cached = self._collections.get(name)
            if cached is not None:
                return cached
            coll_dir = self._core.collection_dir(name)
            if not create and not os.path.isdir(coll_dir):
                raise CollectionNotFoundError(f"collection {name!r} not found at {coll_dir}")
            kwargs = dict(dim=dim, bit_width=bit_width,
                          metric=metric, embedder=embedder)
            if lock_timeout is not None:
                kwargs["lock_timeout"] = lock_timeout
            col = Collection(coll_dir, **kwargs)
            self._collections[name] = col
            return col

    def list_collections(self):
        with self._lock:
            return self._core.list_collections()

    def delete_collection(self, name):
        """Delete a collection and all its data.

        Acquires the collection's write lock before removing the directory
        to prevent races with concurrent writers in other processes.

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
        lock_path = os.path.join(coll_dir, "write.lock")
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
