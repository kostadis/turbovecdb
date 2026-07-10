# Phase 4 — High-Severity Integrity & Safety Bugs

## Issues

### #83 — Relative path retargets after `os.chdir()`
- **Root cause**: `Database.__init__` stores `path` as-is (could be relative). After `os.chdir()`, all derived paths (`.tvim`, WAL, lock) resolve against the new cwd, splitting the coordination domain.
- **Fix**: Canonicalize to absolute path at construction time.
  - `database.py:36` — `self._path = os.path.abspath(path)`
  - `database.rs:83-84` — canonicalize `root` in `Database::new()`
  - `database.rs:95-107` — after canonicalization, `collection_dir()` no longer needs CWD re-resolution
- **Test**: `test_phase4_bugs.py::test_relative_path_retargets_after_chdir`

### #82 — Stale handles write after `delete_collection()`
- **Root cause**: Unix keeps fd valid to unlinked inode. SQLite writes succeed on ghost files. No directory-existence check on operations.
- **Fix**: Extend `conn_is_alive()` to also check `self.dir` exists. In `check_conn_or_reconnect()`, don't reconnect if dir is gone (return error).
  - `collection.rs:266-268` — `conn_is_alive()` also checks `Path::new(&self.dir).exists()`
  - `collection.rs:293-299` — if dir missing, return `CoreError::Other("stale handle")` instead of reconnecting
  - `collection.rs:815-833` — ensure `count()` and `flush()` call `check_conn_or_reconnect()`
- **Test**: `test_phase4_bugs.py::test_stale_handle_write_after_delete`
- **Test**: `test_phase4_bugs.py::test_stale_handle_query_after_delete`

### #93 — Future `schema_version` is accepted
- **Root cause**: `migrate_schema()` uses `>= SCHEMA_VERSION`, so `schema_version=999` passes as "already current".
- **Fix**: Reject `stored > SCHEMA_VERSION` with an error. Only accept `==`.
  - `collection.rs:329-334` — change logic:
    ```
    stored = meta_get_i64("schema_version", 0)?
    if stored > SCHEMA_VERSION → error("future schema version")
    if stored == SCHEMA_VERSION → Ok(())
    if stored < SCHEMA_VERSION → run migrations
    ```
- **Test**: `test_phase4_bugs.py::test_future_schema_version_rejected`
- **Test**: `test_phase4_bugs.py::test_future_schema_version_allows_write`

### #107 — HTTP body unbounded, no read timeout
- **Root cause**: `do_POST()` calls `rfile.read(n)` with no max or negative validation. Negative → reads until EOF (DoS). Oversized → worker blocked indefinitely.
- **Fix**: Validate Content-Length before reading.
  - `service.py:213` — after `n = int(...)`, add:
    - if `n < 0` → return 400
    - if `n > MAX_BODY_SIZE` (1 MB) → return 413
    - Only then read body
- **Test**: `test_phase4_bugs.py::test_http_oversized_content_length`
- **Test**: `test_phase4_bugs.py::test_http_negative_content_length`
