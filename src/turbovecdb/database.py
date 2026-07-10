"""Database — a directory of collections.

A ``Database`` is a lightweight handle over a directory; each collection is a
subdirectory. Construction does no I/O — work is deferred to
:meth:`Database.collection`.

Name validation, path resolution, listing, and — under the cross-process
write lock — directory removal are handled by the Rust core
(``_core.Database``). This wrapper keeps only the in-process name ->
``Collection`` handle cache, which must be identity-stable per name. That
cache needs no lock of its own: creation is serialized on disk by the core's
first-creation flock (C7), and handle identity is kept with a GIL-atomic
``dict.setdefault`` install (a concurrent creator's redundant handle is
simply closed and discarded).
"""

import logging
import os

from ._core import Database as _CoreDatabase
from .collection import Collection, _LOCK_TIMEOUT, embedder_identity
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
        self._path = os.path.abspath(path)
        self._collections = {}
        self._core = _CoreDatabase(self._path)

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
        cached = self._collections.get(name)  # GIL-atomic
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
        # The actual on-disk creation is serialized by the core's
        # first-creation flock (C7); the cache only needs identity stability.
        # ``dict.setdefault`` is a single GIL-atomic op: if a concurrent
        # caller installed a handle for this name while we were constructing
        # ours, we adopt theirs and close our now-redundant one — no lock
        # required, and callers always get the same object per name.
        col = Collection(coll_dir, **kwargs)
        installed = self._collections.setdefault(name, col)
        if installed is not col:
            try:
                col.close()
            except Exception as exc:  # noqa: BLE001
                _log.warning("error closing redundant handle for %r: %s", name, exc)
            self._check_no_conflict(name, installed, dim, bit_width, metric, embedder)
            return installed
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
        return self._core.list_collections()

    def _evict_and_close(self, name):
        """Pop a cached handle (GIL-atomic) and close it. Used outside the
        delete flock only — a ``close()`` acquires the collection's flock, so
        calling it while the Rust ``delete_collection`` holds that same lock
        would self-deadlock (second file description on the same lock file)."""
        col = self._collections.pop(name, None)
        if col is not None:
            try:
                col.close()
            except Exception as exc:  # noqa: BLE001
                _log.warning("error closing collection %r during delete: %s", name, exc)
                raise  # Re-raise the exception so callers know about close failures

    def delete_collection(self, name):
        """Delete a collection and all its data.

        The Rust core acquires the collection's cross-process write lock and
        removes the directory under it, serializing with concurrent writers
        in any process. The lock file lives *outside* the collection
        directory so holding it survives the ``rmtree`` (R1). This wrapper
        evicts and closes any cached handle *before* the delete (a handle's
        ``close()`` takes the same flock, so it must not run while the delete
        holds it), then re-evicts anything a racing ``collection(create=True)``
        re-installed while the delete ran.

        Args:
            name: Name of the collection to delete

        Raises:
            ValueError: If the name is invalid
            CollectionNotFoundError: If the collection does not exist
            TurboVecError: If the write lock cannot be acquired
        """
        # Validate + raise CollectionNotFoundError before touching the cache.
        self._core.ensure_collection(name)
        # Evict + close the cached handle outside the delete flock.
        self._evict_and_close(name)
        # Rust: acquire the flock, rmtree, release (I6). Uses the fixed
        # _LOCK_TIMEOUT with the "to delete collection" message variant (I4).
        self._core.delete_collection(name, _LOCK_TIMEOUT)
        # Re-evict anything a concurrent create re-installed during the delete.
        self._evict_and_close(name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False  # don't suppress exceptions

    def close(self):
        errors = []
        for name, col in list(self._collections.items()):
            try:
                col.close()
            except Exception as exc:  # noqa: BLE001
                _log.warning("error closing collection %r: %s", name, exc)
                errors.append((name, exc))
        self._collections.clear()
        if errors:
            # Raise an aggregated error if any collections failed to close
            error_msgs = [f"{name}: {exc}" for name, exc in errors]
            raise TurboVecError(f"Failed to close {len(errors)} collection(s): {'; '.join(error_msgs)}")


def connect(path):
    """Open a turbovecdb database rooted at ``path`` (a directory of collections)."""
    return Database(path)
