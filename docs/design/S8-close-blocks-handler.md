# S8: close in finally block can block handler thread under lock contention

**Issue:** [#73](https://github.com/kostadis/turbovecdb/issues/73)
**Status:** Design
**Priority:** Low

## Problem

Every `op_*` function follows this pattern:

```python
with _lock_for(db_path):
    db, col = _open(db_path, collection, ...)
    try:
        return result
    finally:
        _close_db(db, col)
```

`_close_db` calls `col.close()` → `db.close()`. Both acquire the write
flock. If another process holds that lock (e.g. a concurrent writer),
close blocks for up to `lock_timeout` seconds (default 5). Because
`finally` runs before `return`, the response is **not sent** until
close finishes, meaning every caller pays the close-latency tax on
every request.

With a `ThreadingHTTPServer` serving N requests concurrently, N blocked
close threads means zero throughput.

## Design: Fire-and-forget close

Replace the synchronous `_close_db` with an asynchronous version that
spawns a daemon thread. The response is sent immediately; close runs
in the background and cannot block the handler.

### `_close_async`

```python
import threading

def _close_async(db, col):
    """Close connection in a daemon thread. Never blocks the caller."""
    t = threading.Thread(target=_close_db, args=(db, col), daemon=True)
    t.start()
```

Daemon threads are automatically cleaned up on process exit.

### Changes in each `op_*`

| Before                          | After                             |
|----------------------------------|-----------------------------------|
| `finally: _close_db(db, col)`   | `finally: _close_async(db, col)`  |

No other logic changes. The `with _lock_for(db_path)` scope is unchanged
— the Python lock is released immediately after `_close_async` returns
(which is ~microseconds), so the next request for the same `db_path`
can proceed without waiting for close.

### Safety

- **Flock is per-fd.** The background close thread releases the old
  file descriptors. A concurrent request opens new fds via `_open`.
  These are independent — no conflict.
- **`_lock_for` is a Python threading.Lock.** The close thread does
  not touch it, so no deadlock.
- **Exceptions during close** are caught by `_close_db` and logged.
  They never reach the handler.
- **Daemon threads** are fine for a single-user service. Under heavy
  load, many close threads may pile up, but each is just a blocking
  flock acquire on a unique fd — bounded by `lock_timeout`.

### Trade-offs

- **No back-pressure.** If close is genuinely slow (e.g. a large flush
  to disk), background threads accumulate. For a single-user service
  this is acceptable. Could add a bounded `concurrent.futures.ThreadPoolExecutor`
  if it becomes a problem.
- **Close errors are invisible to the caller.** The handler already
  returned 200. This matches the current logging-only error handling.

### Longer-term: CachedDatabase integration

Once S4 (CachedDatabase, PR #77) is plumbed into service.py, the
open/close-per-request pattern goes away entirely. Connections are
reused across requests; close only happens on shutdown. S8 is a
stopgap for the current reconnect-every-request pattern.

## Test plan

- Existing tests pass unchanged (close is still called, just async)
- Add a test that verifies rapid sequential requests to the same
  db_path do not block each other (implicitly tests that close
  doesn't hold the Python lock)
