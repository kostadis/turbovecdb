# Phase 4: Paths, delete, schema_version, HTTP body bounds

## Bugs

| ID | Title | Severity |
|----|-------|----------|
| #83 | `chdir` after `Database()` breaks all collection paths | HIGH |
| #82 | `delete_collection` race + stale handle returns garbage after delete | CRITICAL |
| #93 | `schema_version` in future silently corrupts on read | HIGH |
| #107 | HTTP service unbounded body read enables slow-loris DoS | HIGH |

---

## Bug analysis

### #83 — chdir breaks paths

`Database.__init__()` stores `root` relative to cwd at init time. If the process calls `os.chdir()` after init, `cwd.join(root)` in `collection_dir()` resolves against the new cwd — pointing to a different directory or a non-existent one.

`os.path.realpath()` resolves symlinks and produces an absolute path at init time, making the path immune to chdir.

```python
# database.py
root = os.path.realpath(root)
```

The Rust `CoreDatabase` receives the canonicalized path; no Rust-side change needed because `CoreDatabase` only stores the path and defers resolution to `collection_dir()`.

### #82 — delete_collection race + stale handle

**Race**: `delete_collection` checks `root.exists()` **before** acquiring the write lock. A concurrent `add()` can create a file between the check and the lock. If `add()` wins the race, `delete_collection` deletes a live collection from under a writer.

**Stale handle**: After deleting a collection, `ensure_collection` via `exists()` returns `false`. But if a new collection with the same name is created at a different path (symlink swap), a stale handle referencing the old path returns garbage. `Path::exists()` has a TOCTOU bug.

**Fix**:
1. `delete_collection`: acquire lock FIRST, then `try_exists` inside lock
2. `ensure_collection`: use `try_exists` (stat + match on NotFound/EACCES) instead of `Path::exists()`
3. `list_collections`: propagate permission errors instead of silently skipping
4. `Collection::new`: pre-check `store.sqlite3` exists before `create_dir_all`

### #93 — Future schema_version

`migrate_schema` handles upgrading from older versions to current, but if a database written by a hypothetical future version (e.g., schema=5, current=3) is opened, it proceeds as if schema=3 — corrupting the schema.

Fix: compare `stored_version > SCHEMA_VERSION` and raise `IncompatibleVersion` error. New `CoreError::IncompatibleVersion` variant maps to Python `IncompatibleVersionError`.

### #107 — HTTP body unbounded read

`service.py` reads the entire request body via `body = await request.read()` which reads until connection close. A slow client can send 1 byte/second, holding a worker thread indefinitely. With bounded thread pool (default 10 workers), 10 slow connections exhaust the pool.

**Fix structure**:
```python
async def _read_body(self, request: aiohttp.Request) -> bytes:
    if request.transport is None:
        raise BadRequest("no transport")

    if request.headers.get("Transfer-Encoding", "").lower() == "chunked":
        raise HTTPNotImplemented(text="chunked transfer not supported")

    cl = request.headers.get("Content-Length", "0")
    try:
        length = int(cl)
    except (ValueError, TypeError):
        raise HTTPBadRequest(text="invalid Content-Length")

    if length <= 0:
        raise HTTPBadRequest(text="Content-Length must be positive")

    MAX_BODY = 16 * 1024 * 1024  # 16 MiB
    if length > MAX_BODY:
        raise HTTPRequestEntityTooLarge(text=f"body exceeds {MAX_BODY} bytes")

    total_start = time.monotonic()
    TOTAL_TIMEOUT = 60.0
    CHUNK_SIZE = 8192
    CHUNK_TIMEOUT = 5.0

    body = bytearray()
    remaining = length
    while remaining > 0:
        if time.monotonic() - total_start > TOTAL_TIMEOUT:
            raise HTTPBadRequest(text="total read timeout")
        chunk = await asyncio.wait_for(
            request.content.readexactly(min(CHUNK_SIZE, remaining)),
            timeout=CHUNK_TIMEOUT
        )
        body.extend(chunk)
        remaining -= len(chunk)

    return bytes(body)
```

Key elements:
- Content-Length validated with `int()` — rejects negative, float, non-numeric
- Max body: 16 MiB → 413
- Transfer-Encoding: chunked → 501 Not Implemented
- Per-chunk timeout: 5s → prevents wire-hang
- Total timeout: 60s → prevents slow-drip (2048 chunks × 5s = 2.8h without total timeout)
- Edge cases: Content-Length=0 → 400, missing CL → treated as 0 → 400

---

## Conflict: 501 vs 400 for chunked

Critic flagged that Transfer-Encoding: chunked with 501 is semantically correct per RFC 7230 §3.3.1. The orchestrator agrees — the request is valid HTTP, just using a transfer coding the server doesn't implement. 501 is standard.

**Decision (2026-07-10)**: ✅ 501 it is.

## Conflict: slow-drip mitigation

The 5s per-chunk timeout alone allows 2.8h for 16 MiB. The 60s total-receive-timeout closes this. Some designers argue 60s is too generous (normal upsert < 1s).

**Orchestrator opinion**: 60s is fine. A single slow client can hold one worker for 60s, not 2.8h. With 10 workers, worst case is 10 × 60s = 600s of degraded throughput. Tightening to 30s would be better — 30s is still generous for a 16 MiB body at 8 KiB/chunk (30s / 5s per chunk = 6 chunks = 48 KiB theoretical max before timeout — wait, that doesn't work if chunks arrive fast). The total-timeout is separate from per-chunk timeout. If chunks arrive at 0.1ms intervals, 16 MiB takes ~2s. The 60s total timeout only matters when the client is deliberately slow. 60s is acceptable.

**Decision (2026-07-10)**: ✅ 60s total receive timeout.

## Conflict: try_exists vs exists in ensure_collection

Read path (`ensure_collection`) does NOT hold a lock. Its `try_exists` is inherently racy.

**Orchestrator opinion**: Acceptable. A false-negative on read path is harmless — `add()` will acquire the lock and create the directory if needed. A false-positive (think dir exists when it was just deleted) will hit a proper error when the SQLite open fails. The critical fix is write-path, where the lock IS held.

**Decision (2026-07-10)**: ✅ Accept racy read path; write-path lock fix is sufficient.

---

## Files touched

| File | Change |
|------|--------|
| `src/turbovecdb/database.py` | `os.path.realpath` in `__init__` |
| `crates/turbovecdb-core/src/database.rs` | Lock-first in `delete_collection`; `try_exists` in `ensure_collection`; propagate EACCES in `list_collections`; pre-check in `Collection::new` |
| `crates/turbovecdb-core/src/error.rs` | `IncompatibleVersion` variant |
| `crates/turbovecdb-py/src/lib.rs` | Map `IncompatibleVersion` to Python |
| `src/turbovecdb/errors.py` | `IncompatibleVersionError(TurboVecError)` |
| `src/turbovecdb/service.py` | `_read_body()` with bounded chunked read + total timeout |

---

## Verification

- `test_chdir_after_database_init` — expect collections still accessible (xfail→pass)
- `test_delete_collection_race` — expect no corruption under concurrent add (xfail→pass)
- `test_stale_after_delete` — expect clean error, not garbage (xfail→pass)
- `test_future_schema_version` — expect `IncompatibleVersionError` (xfail→pass)
- `test_http_max_body_size` — expect 413 for oversized body (xfail→pass)
- `test_http_chunked_not_implemented` — expect 501 (xfail→pass)
- `test_http_negative_content_length` — expect 400 (xfail→pass)
- `test_http_body_validation` — expect work for valid body (xfail→pass)
