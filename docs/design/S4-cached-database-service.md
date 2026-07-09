# S4: Cache Database handles in service.py

**Issue:** [#74](https://github.com/kostadis/turbovecdb/issues/74)
**Status:** Implemented (2026-07-09)
**Priority:** Low

## Problem

Every request calls `_open()` → `turbovecdb.connect(db_path)`, creating
a fresh `Database` object. The Python `Database` class (database.py:37)
already caches `Collection` handles per name in `self._collections`, but
that cache lives on the instance — a new `Database` per request means
an empty cache every time.

Waste per request:
1. `Database.__init__` → `_CoreDatabase(path)` — creates a new Rust
   `Database` handle
2. `db.collection(name, ...)` — SQLite open, DDL, meta reads, index
   reload, flock acquire
3. `col.close()` + `db.close()` — write flock acquire

## Design: Cache `Database` objects per db_path

Add a module-level dict that caches `turbovecdb.Database` instances
keyed by `db_path`. Once opened, a `Database` persists for the server
lifetime. Its internal `_collections` cache keeps collection handles
warm across requests.

### Changes to service.py

#### Remove `_open` and `_close_db`

These are no longer called.

#### Add `_get_db(db_path)`

```python
_databases: dict[str, turbovecdb.Database] = {}

def _get_db(db_path: str) -> turbovecdb.Database:
    db = _databases.get(db_path)
    if db is None:
        db = turbovecdb.connect(db_path)
        _databases[db_path] = db
    return db
```

No lock needed — CPython's GIL serializes dict access.

#### Each `op_*` — use `_get_db` + the cached collection handle

```python
def op_upsert(req: dict) -> dict:
    db_path = req["db_path"]
    collection = req["collection"]
    items = req.get("items", [])
    if not items:
        return {"count": 0}
    by_id: dict[str, dict] = {}
    for it in items:
        by_id[it["id"]] = it
    items = list(by_id.values())
    dim = len(items[0]["vector"])
    with _lock_for(db_path):
        db = _get_db(db_path)
        col = db.collection(collection, dim=dim, create=True)
        col.upsert(...)
        return {"count": col.count()}
```

No `try/finally` — no close. The handle stays cached.

Similarly for `op_candidate_pairs`, `op_count`, `op_clear`.

#### `_lock_for` stays (for now)

The per-db_path Python lock serializes same-DB requests within the
process. This is no longer correctness-critical (the Rust core handles
concurrency via its own Mutex + flock), but it smooths latency under
write bursts. Can be removed as a follow-up.

### What this eliminates

| Issue | Root cause | Status |
|-------|-----------|--------|
| S4 (#74) | Reopen per request | **FIXED** |
| S8 (#73) | Close blocks under lock contention | **ELIMINATED** — no close per request |

### What this does NOT fix

- **S3 (#70):** Synchronous http.server blocks thread per request.
  Still true — each request ties up a handler thread. Separate concern.

- **S2 (#72):** No auth / rate limiting. Unchanged.

- **S6 (#69):** Flat error strings. Unchanged.

### Test plan

- Existing tests pass (they drive `op_*` directly; the `_get_db` cache
  is transparent to callers)
- Add test verifying the same `Database` object is returned for the
  same `db_path` on successive calls
- Add test verifying collection handles survive across requests
