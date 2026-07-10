# S109: Bound Database Path Cache to Prevent Memory Leak

**Issue:** [#109](https://github.com/kostadis/turbovecdb/issues/109)
**Status:** Proposed
**Priority:** Medium

## Problem

The service stores one lock and one `Database` object for every raw `db_path` string and never removes either. `db_path` is supplied by each request and is neither canonicalized nor constrained to an allowed root.

A stream of unique paths or aliases grows `_locks` and `_databases` without bound. Once collections are opened, nested collection caches and SQLite descriptors can grow as well. Relative/alias paths also create duplicate locks/handles for the same physical database.

## Impact

Remote input controls long-lived process memory, file descriptors, and potentially filesystem creation. Even benign path spelling differences defeat reuse and request serialization.

## Code

- Process-global, unbounded maps: [service.py#L36-L60](https://github.com/kostadis/turbovecdb/blob/main/src/turbovecdb/service.py#L36-L60)
- Request routes use caller-provided paths as keys: [service.py#L84-L118](https://github.com/kostadis/turbovecdb/blob/main/src/turbovecdb/service.py#L84-L118)
- Count/clear also populate the maps: [service.py#L150-L170](https://github.com/kostadis/turbovecdb/blob/main/src/turbovecdb/service.py#L150-L170)
- There is no shutdown/eviction of cached databases: [service.py#L228-L241](https://github.com/kostadis/turbovecdb/blob/main/src/turbovecdb/service.py#L228-L241)

## Design: Add Cache Eviction with LRU Policy

### Changes to service.py

#### Replace unbounded dicts with LRU cache

```python
# Replace:
_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
_locks_guard = threading.Lock()
_databases: dict[str, turbovecdb.Database] = {}

# With:
from collections import OrderedDict
import threading

_MAX_CACHED_DATABASES = 100  # Configurable limit
_locks: OrderedDict[str, threading.Lock] = OrderedDict()
_locks_guard = threading.Lock()
_databases: OrderedDict[str, turbovecdb.Database] = OrderedDict()
_databases_lock = threading.Lock()

def _get_lock(db_path: str) -> threading.Lock:
    with _locks_guard:
        if db_path in _locks:
            # Move to end (most recently used)
            lock = _locks.pop(db_path)
            _locks[db_path] = lock
            return lock
        else:
            lock = threading.Lock()
            _locks[db_path] = lock
            # Enforce limit
            if len(_locks) > _MAX_CACHED_DATABASES:
                _locks.popitem(last=False)  # Remove least recently used
            return lock

def _get_db(db_path: str) -> turbovecdb.Database:
    with _databases_lock:
        if db_path in _databases:
            # Move to end (most recently used)
            db = _databases.pop(db_path)
            _databases[db_path] = db
            return db
        else:
            db = turbovecdb.connect(db_path)
            _databases[db_path] = db
            # Enforce limit
            if len(_databases) > _MAX_CACHED_DATABASES:
                _databases.popitem(last=False)  # Remove least recently used
            return db
```

#### Add cache statistics for monitoring

```python
def get_cache_stats() -> dict:
    with _locks_guard, _databases_lock:
        return {
            "cached_locks": len(_locks),
            "cached_databases": len(_databases),
            "max_cached": _MAX_CACHED_DATABASES
        }
```

#### Add cleanup on shutdown (optional)

```python
def clear_all_caches():
    """Clear all caches - useful for testing or shutdown."""
    with _locks_guard, _databases_lock:
        _locks.clear()
        _databases.clear()
```

### What this fixes

| Issue | Root cause | Status |
|-------|-----------|--------|
| S109 (#109) | Unbounded growth of `_locks` and `_databases` | **FIXED** |

### What this does NOT fix (related issues)

- **S3 (#70):** Synchronous http.server blocks thread per request. Still true — each request ties up a handler thread. Separate concern.
- **S2 (#72):** No auth / rate limiting. Unchanged.
- **S6 (#69):** Flat error strings. Unchanged.

### Test plan

1. Add test verifying `_get_db` returns the same object for repeated calls with same `db_path`
2. Add test verifying that when cache limit is exceeded, least recently used entries are evicted
3. Add test verifying that different `db_path` values create different cache entries
4. Add test verifying that `get_cache_stats()` returns correct counts
5. Add test verifying that very long streams of unique paths don't cause unbounded memory growth

### Implementation notes

- The LRU implementation uses `OrderedDict` with `popitem(last=False)` to remove least recently used items
- Separate locks are used for `_locks` and `_databases` to minimize contention
- The cache size limit (`_MAX_CACHED_DATABASES`) is configurable and defaults to 100
- No changes needed to the request handling logic - `_get_db` and `_get_lock` drop-in replacements for direct dict access