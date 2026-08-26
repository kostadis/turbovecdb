# Resiliency Review — turbovecdb (2026-07-05)

**Purpose of this document:** input to a planning session. It is a *findings report*, not a plan.
The planner should turn the findings below into an ordered, executable work plan (tasks, tests,
acceptance criteria). Severity ranking and suggested directions are included, but the planner owns
sequencing and scoping decisions.

**How this was produced:** full knowledge-graph index of the repo (1,045 nodes / 4,996 edges) via
codebase-memory MCP, followed by source-level trace of the write, flush, recovery, delete, and
service paths. Every finding below was verified against actual source, with file:line references.
Line numbers are against the repo state at commit on `main` as of 2026-07-05.

**Codebase shape:** Python API layer (`src/turbovecdb/`) over a Rust core
(`crates/turbovecdb-core`, PyO3 bindings in `crates/turbovecdb-py`), mid-cutover per
`docs/rust-core-cutover-plan.md`. SQLite is the source of truth; the in-memory turbovec index is a
cache keyed by `store_gen`, persisted as `.tvim`. Cross-process writer serialization via flock on a
sibling `<name>.lock` file. Design invariants and prior fixes are tagged in code comments (R1–R7,
C1–C7, I4–I6, G-series gaps in `docs/mempalace-backend-gaps.md`) — the plan should keep using that
tagging convention.

---

## Overall verdict

The core is unusually resiliency-conscious (see "Already strong" at the end — do NOT re-fix those).
The real issues are at the edges: the HTTP service layer, and one cross-process corner the core's
guarantees don't cover.

One-line summary: **fix the service-layer handle leak (mechanical), decide whether the
cross-process delete-vs-stale-handle hole is inside or outside the guarantee contract (design
decision needed), everything else is polish.**

---

## Findings (ranked by severity)

### F1. HTTP service leaks DB handles on early returns — HIGH, mechanical fix

**Where:** `src/turbovecdb/service.py:103-105` (`op_candidate_pairs`) and `:147-149` (`op_clear`).

`op_candidate_pairs` and `op_clear` open the database and then early-return **before** entering the
`try/finally` that closes it:

```python
db, col = _open(db_path)
if col is None or col.count() == 0:
    return {"pairs": []}          # db never closed
try:
    ...
finally:
    _close_db(db, col)
```

- Every request against an empty/missing collection leaks a SQLite connection (plus WAL/shm fds).
  Long-running service → fd exhaustion → slow-death availability failure.
- `col.count()` on line 104 is also outside the guard; an exception there leaks too.
- `op_count` (`service.py:137-139`) does it correctly — its `try/finally` wraps the `col is None`
  branch. Fix is to make the other two match that shape.

**Acceptance criteria hint:** a test that hits `/candidate_pairs` and `/clear` against a missing
collection N times and asserts fd count (or open-connection count) does not grow.

### F2. Cross-process delete vs. stale open handle → silently lost writes — HIGH, needs a design decision

**Where:** `crates/turbovecdb-core/src/collection.rs:559` (`write_locked` — no re-stat of the
store), `crates/turbovecdb-core/src/database.rs:127-136` (`ensure_collection` — existence checks
only on delete/open paths), `src/turbovecdb/database.py:131-159` (cache eviction is
in-process only).

Scenario:

1. Process A holds an open `Collection` handle (idle).
2. Process B runs `delete_collection` — acquires the flock, `rmtree`s the collection dir. The
   sibling `.lock` file deliberately survives (tested at `database.rs:290-292`).
3. Process A's next `add()`: `acquire_write_lock` succeeds (lock file still exists),
   `ensure_current`/meta reads succeed (A's SQLite fd points at the **unlinked inode**), the INSERT
   commits — into a ghost file. The write is acknowledged and permanently lost.

`Database.delete_collection` evicts cached handles in its *own* process only; there is no
cross-process signal. This violates the documented guarantee "a committed write is durable in
SQLite before add()/delete() returns" (`docs/core/concurrency.md:88`) in exactly the multi-process
scenario that document is about.

**Suggested direction (planner to decide):** under the write lock, before committing, verify the
collection still exists — either re-stat `store.sqlite3` / the collection dir, or compare
`fstat(conn_fd)` inode against a fresh `stat(path)`. Surface `CollectionNotFound` on mismatch.
Alternative: explicitly document this as out of contract in `docs/core/concurrency.md` (weaker, but
honest). Note `flush()` already fails loudly in this scenario (tvim rename into a missing dir →
ENOENT) — it's the non-flushing `add`/`upsert`/`delete` acks that are silent.

**Acceptance criteria hint:** a two-process test — open handle in a subprocess, delete the
collection from the parent, have the subprocess `add()` — asserting the add raises rather than
returning success. Follows the existing pattern in `tests/test_crash_injection.py`.

### F3. `flock` EINTR: comment and code disagree — MEDIUM-LOW, tiny fix

**Where:** `crates/turbovecdb-core/src/flock.rs:113-118` (`FlockGuard::acquire`).

The comment says EINTR "is retried by the loop naturally on the next attempt," but the code returns
`Err(CoreError::Io)` for **any** errno other than `EWOULDBLOCK` — including EINTR. Practical risk
is low (`LOCK_NB` makes the call non-blocking, so EINTR is rare), but if it fires, a merely
interrupted lock attempt surfaces as a spurious hard I/O error. Either retry EINTR explicitly
(`err.raw_os_error() == Some(libc::EINTR)` → continue the loop) or fix the comment to match
behavior. Recommend the code fix; it's two lines.

### F4. `reembed` holds the cross-process write lock for its entire duration — MEDIUM, mostly documentation/operational

**Where:** `crates/turbovecdb-core/src/collection.rs:1237-1251`.

Documented as deliberate ("bulk maintenance op, not a hot write path"), but the resiliency
consequence should be named in a plan:

- With a network-backed embedder and a large collection, the lock is held for minutes. Every other
  process's `add`/`flush`/`close`/`delete_collection` fails with `LockTimeout`
  (default `_LOCK_TIMEOUT = 30`, `src/turbovecdb/collection.py:43`).
- `close()` failing during that window is the nasty case: the caller's `.tvim` flush is skipped
  and — per F5 — the failure is swallowed as a log line.
- The `on_progress` callback runs **under the lock** (`collection.rs:1367-1370`), so a slow
  callback extends the outage.

**Suggested direction:** at minimum, document "don't reembed live collections with active writers"
in `docs/core/concurrency.md`. Larger option (backlog-grade): chunked lock-release reembed. If
mempalace will reembed live collections, this graduates to HIGH.

### F5. Close/flush failures are swallowed everywhere above the core — MEDIUM-LOW, logging/signal fix

**Where:** `src/turbovecdb/database.py:170-176` (`Database.close`), `database.py:120-129`
(`_evict_and_close`), `src/turbovecdb/service.py:57-67` (`_close_db` — bare `except Exception: pass`).

Because SQLite commits at write time, store durability is safe; what's silently lost is the `.tvim`
flush and the WAL checkpoint (rebuild cost on next open, WAL growth). Defensible tradeoff — but a
`close()` that timed out on lock contention and one that hit ENOSPC currently look identical
(invisible or a warning log). At minimum, distinguish disk-error close failures from
lock-contention ones; consider letting genuine I/O errors propagate from `Database.close()`.

### F6. Full-collection materialization on rebuild and reembed — MEDIUM-LOW at current scale, memory-as-availability risk

**Where:**
- `reload_index` (`crates/turbovecdb-core/src/collection.rs:271-316`): loads every vector into a
  `Vec<f32>`, then copies into the index — ~2× resident spike, triggered **lock-free on any read
  path** whose handle is stale (`ensure_current` → `reload_index`).
- `reembed` (`collection.rs:~1330`): accumulates all new vectors in `updates` before Phase 2;
  `batch_size` bounds embedder calls, **not** memory.

At mempalace's current scale this is fine. At millions of vectors it becomes an OOM-kill vector,
which takes down the whole process and anything co-resident. `docs/core/concurrency.md:95-97`
already flags incremental reload as backlog — add "streaming/chunked rebuild" to the same entry
rather than treating this as a new workstream.

### F7. Minor items — LOW, batch as polish

- **`health()` can't report unhealthy** (`collection.rs:757-771`): a failed `quick_check` *raises*
  instead of returning `HealthResult { ok: false, .. }` — the `ok` field is always `true` when a
  result is returned (dead field), and monitoring probes must treat exception-vs-result as the real
  signal. Return the structured unhealthy result instead.
- **Service hardening** (`src/turbovecdb/service.py`): unbounded `Content-Length` read into memory
  (`do_POST`), no socket timeouts on `ThreadingHTTPServer`, `_locks` dict grows one entry per
  `db_path` forever. All mitigated by the 127.0.0.1 default bind; fix if it ever binds wider.
- **`wal_checkpoint` is fire-and-forget** (`let _ =`, `collection.rs:389-391`): persistent
  checkpoint failure = silent WAL growth. `tests/test_wal_checkpoint.py` covers successful flushes;
  nothing signals when checkpointing itself is failing repeatedly. Consider logging/counting
  failures.

---

## Already strong — verified; do NOT re-fix, and don't regress

The plan should treat these as protected invariants (several have regression tests keyed to the
R/C/I tags):

- **Torn-write defense (R2):** temp-file write → `sync_all` → rename → best-effort parent-dir fsync
  → `tvim_gen` stamped from `seen_gen` (not a re-read of `store_gen`) — `flush_impl`,
  `collection.rs:713-744`. The comments show the stale-stamp race was found and deliberately closed.
- **Crash safety (R7):** real SIGKILL injection mid-write and mid-flush
  (`tests/test_crash_injection.py`), plus corrupt/mismatched-shape `.tvim` rebuild tests that
  verify dim/bit_width before trusting the cache (`reload_index_rebuilds_when_loaded_dim_mismatches_meta`,
  `test_corrupt_tvim_rebuilds_from_store`).
- **Write path (I5/R3):** embedding runs outside the flock; embedder identity re-checked under it;
  index-mirror failure → rollback + cache invalidation (`index = None`, `seen_gen = -1`) so the
  next read rebuilds from committed store — never a divergent cache (`write_locked`).
- **Delete split-brain defense (R1):** lock file is a *sibling* of the collection dir specifically
  so it survives `rmtree`; scenario named and tested (`test_lock_survives_rmtree_no_split_brain`).
- **Snapshot-consistent queries (C3):** allowlist + re-rank SELECTs wrapped in one deferred
  transaction; regression test uses `Connection::trace` to inject a concurrent commit between
  statements.
- **Constructor race (C7):** first-creation flock scoped to meta-init only; reopen path is
  lock-free; cached-handle identity via GIL-atomic `dict.setdefault` with redundant-handle close.
- **Poisoned-mutex mapping** (PyO3 layer, `crates/turbovecdb-py/src/collection.rs:60-64`): panics
  while holding the in-process lock surface as `TurboVecError`, not interpreter aborts.
- **Honest scope docs:** `docs/core/concurrency.md:104-127` explicitly restricts guarantees to
  local POSIX filesystems (NFS/9p/WSL2-`/mnt/c` called out as unsafe).

## Notes for the planner

- F1 and F3 are independent, small, and test-backed — good first tasks.
- F2 requires a contract decision before implementation (fix vs. document-as-out-of-scope). If
  fixing: the check must live in the Rust core under the write lock, and the error must be
  `CollectionNotFound` for message-stability with the existing test conventions.
- Several existing tests assert exact error-message text (see `delete_lock_timeout_msg_is_byte_identical`,
  `database.rs:324`). Any change touching error strings must preserve byte-identical messages or
  update the paired tests deliberately.
- Repo conventions: invariant tags (R/C/I/G + number) in comments, regression test per fixed race,
  plan docs live in `docs/`.
