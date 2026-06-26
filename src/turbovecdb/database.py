"""Database — a directory of collections.

A ``Database`` is a lightweight handle over a directory; each collection is a
subdirectory. Construction does no I/O — work is deferred to
:meth:`Database.collection`.
"""

import logging
import os
import re
import threading

from filelock import FileLock, Timeout

from .collection import Collection, _LOCK_TIMEOUT
from .errors import CollectionNotFoundError, TurboVecError
from .index import DEFAULT_BIT_WIDTH

_log = logging.getLogger(__name__)
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_\-]{1,128}$")


class Database:
    def __init__(self, path):
        self._path = path
        self._collections = {}
        self._lock = threading.Lock()

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
        if not _SAFE_NAME.fullmatch(name):
            raise ValueError(
                f"invalid collection name {name!r}: must match [A-Za-z0-9_-]{{1,128}}"
            )
        with self._lock:
            cached = self._collections.get(name)
            if cached is not None:
                return cached
            coll_dir = os.path.join(self._path, name)
            base = os.path.abspath(self._path) + os.sep
            if not os.path.abspath(coll_dir).startswith(base):
                raise ValueError(f"collection name {name!r} escapes database root")
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
        if not os.path.isdir(self._path):
            return []
        with self._lock:
            return sorted(
                d for d in os.listdir(self._path)
                if os.path.isdir(os.path.join(self._path, d, ""))
                and os.path.exists(os.path.join(self._path, d, "store.sqlite3"))
            )

    def delete_collection(self, name):
        """Delete a collection and all its data.

        Acquires the collection's write lock before removing the directory
        to prevent races with concurrent writers in other processes.

        Args:
            name: Name of the collection to delete

        Raises:
            CollectionNotFoundError: If the collection does not exist
            TurboVecError: If the write lock cannot be acquired
        """
        coll_dir = os.path.join(self._path, name)
        if not os.path.isdir(coll_dir):
            raise CollectionNotFoundError(f"collection {name!r} not found at {coll_dir}")
        if not os.path.exists(os.path.join(coll_dir, "store.sqlite3")):
            raise CollectionNotFoundError(f"collection {name!r} not found at {coll_dir}")

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
        try:
            import shutil
            shutil.rmtree(coll_dir)
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
