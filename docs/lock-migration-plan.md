# Lock migration: Python wrapper → Rust core — orchestration plan

> **ARCHIVED — executed and merged.** This orchestration plan was carried out
> in full (Phases 1–7) and merged to `main` via PR #48 (merge commit
> `9057ded`). Kept as a historical design record; the code, not this doc, is
> the source of truth. Follow-up filed as issue #47 (recover in-process
> read/write concurrency during a slow embed). See `CHANGELOG.md` (Unreleased)
> and `docs/ARCHITECTURE.md` for the resulting design.

Plan for moving **all locking** out of the Python wrapper and into the Rust
core, so `src/turbovecdb/*.py` becomes a lock-free shim (dataclasses,
argument shaping, deprecation-stable API) and `_core` is the sole
serialization point — in-process *and* cross-process.

Written to be executed by an **Opus orchestrator** delegating implementation
to **Sonnet sub-agents**. Opus owns the design decisions (§4), reviews every
diff at the phase checkpoints, and runs the verification gates (§8). Sonnet
agents implement one task card (§6) each; task cards are self-contained on
purpose — do not assume a sub-agent has read this whole document, paste the
relevant invariants (§3) into its prompt.

## 1. Goal / non-goals

**Goal.** After this migration:

- `_core.Collection` owns the in-process lock (today `Collection._tlock`,
  a `threading.RLock`) and the cross-process write lock (today
  `Collection._flock`, a `filelock.FileLock` on the sibling
  `<root>/<name>.lock`).
- `_core.Database` owns whatever locking `delete_collection` and the handle
  cache need.
- `import filelock` and `import threading` disappear from
  `src/turbovecdb/collection.py` and `database.py`. `filelock` moves from a
  runtime dependency (`pyproject.toml:29`) to a **test-only** dependency.
- Observable behavior is unchanged: same lock file path, same timeout
  semantics, same exception types **with byte-identical messages** (tests
  assert exact wording), same crash-safety (the SIGKILL harness in
  `test_crash_injection.py` must pass unmodified).

**Non-goals.** No new concurrency features. Do **not** upgrade the
in-process lock to a read-write lock (reads serialize in-process today via
`_tlock`; keep that — an RwLock is a separate follow-up issue). No Windows
lock-compat work beyond a documented note (see D1). `service.py`'s
per-`db_path` `threading.Lock` is application-level HTTP serialization, not
a collection lock — out of scope, but Phase 6 files a follow-up note on
whether it's still needed.

## 2. Current lock inventory

| Lock | Where | Guards | Notes |
|---|---|---|---|
| `Collection._tlock` (`threading.RLock`) | `collection.py:121` | every method incl. reads (`query/get/count/health` at `collection.py:246-260`) | exists partly because `_core.Collection` methods are `&mut self` — two unguarded Python threads would hit PyO3's runtime borrow check (`RuntimeError: Already borrowed`), not block |
| `Collection._flock` (`filelock.FileLock`) | `collection.py:128` | all writes via `_locked()` (`collection.py:155-170`) | sibling path from `write_lock_path()` (`collection.py:90`), default timeout 30s (`_LOCK_TIMEOUT`), `Timeout` → `TurboVecError` |
| constructor lock-on-create (C7) | `collection.py:141-146` | Rust constructor's read-then-write of meta on first creation | `looks_new = not exists(store.sqlite3)` gate keeps re-open lock-free |
| embed-before-lock (R3) | `collection.py:187-213` | identity pre-check unlocked → embed unlocked → **re-check identity under lock** → write | the under-lock re-check is why this can't stay in Python once the lock moves |
| `Database._lock` (`threading.Lock`) | `database.py:39` | `_collections` handle cache; C6 conflict check (`database.py:59-77`) | cache must stay identity-stable per name |
| `delete_collection` direct `FileLock` | `database.py:140-166` | close-evict → acquire flock → **re-evict under lock** → `remove_collection_dir` | interleaves the cache lock and the file lock; R1 sibling-path rationale |

Rust side today: `crates/turbovecdb-py/src/collection.rs` takes **no
locks**; `lock_timeout` is accepted and discarded (`collection.rs:57`).
Every method is `&mut self` and most wrap core calls in
`py.allow_threads`. `reembed` (`collection.rs:249-293`) is the exception —
it holds the GIL throughout because the `on_progress` callback captures
`py`. `turbovecdb-core` has no lock code and no `LockTimeout` error variant
(`error.rs`).

## 3. Load-bearing invariants

Paste the relevant ones into every sub-agent prompt.

- **I1 — GIL/Mutex ordering.** A thread must **never block on the
  in-process lock while holding the GIL**. Every PyO3 method does argument
  conversion under the GIL, then `py.allow_threads(|| { lock mutex; ... })`.
  Python callbacks (`PyEmbedder::embed`, `reembed`'s `on_progress`)
  re-enter via `Python::with_gil` from inside `allow_threads`. Violating
  this deadlocks: thread A holds mutex inside `allow_threads` and waits for
  the GIL (embedder); thread B holds the GIL and waits for the mutex.
  **This forces a `reembed` refactor** — it currently never releases the
  GIL; its progress callback must switch from a captured `py` to
  `Python::with_gil`.
- **I2 — lock order.** In-process mutex first, then cross-process flock
  (mirrors today's `_tlock` → `_flock` nesting in `_locked()`). Reads take
  only the mutex.
- **I3 — lock primitive compat.** Python `filelock` on Unix is
  `fcntl.flock(LOCK_EX)` on the lock file. The Rust implementation must use
  the same syscall on the same path (`<root>/<name>.lock`, still computed
  sibling-of-dir per R1) so a process running the old wheel and a process
  running the new wheel genuinely exclude each other. Timeout is a poll
  loop (`LOCK_NB` + sleep ~0.05s, matching filelock's poll interval), since
  `flock` has no native timeout.
- **I4 — message fidelity.** `CoreError` variants carry pre-formatted
  strings and `Display` is a passthrough precisely so messages stay
  byte-stable (`error.rs` header comment). The new timeout error must
  render exactly today's Python text, including the `repr()` of the path:
  `could not acquire write lock on '<dir>' within <timeout:.1f>s`
  (from `collection.py:167-170`; delete's variant at `database.py:146-149`
  appends `to delete collection`). Reproduce Python `!r` quoting in Rust.
- **I5 — embed outside the write lock (R3).** Identity pre-check
  (unlocked) → embed → acquire flock → identity re-check → write. This
  whole sequence moves into the core write path; the Python `_write`
  special case (`collection.py:187-213`) is deleted, not relocated.
- **I6 — R1 delete semantics.** The lock file stays a *sibling* of the
  collection dir; `delete_collection` holds it across the rmtree and
  re-evicts the handle cache under it.

## 4. Design decisions (Opus resolves before Phase 1)

Recommended answers below; Opus confirms or overrides at kickoff and
records the outcome at the top of the working branch's PR description.

- **D1 — file-lock implementation.** Recommend a small hand-rolled module
  in `turbovecdb-core` (`src/flock.rs`, ~100 lines) using `rustix::fs::flock`
  (or `libc::flock`) with an NB-poll timeout loop, rather than pulling in
  `fs4`. Rationale: we need exact control of primitive + poll interval +
  error text, and the crate surface we'd use is ~20 lines anyway. Unix
  only; on Windows compile a `LockFileEx` fallback or `compile_error!` —
  note that Python `filelock` uses `msvcrt.locking` on Windows, so old/new
  interop there is already lost; document, don't chase.
- **D2 — in-process lock shape.** `#[pyclass] Collection { inner:
  Mutex<CoreCollection<...>> }` (std `Mutex`; RLock reentrancy is not
  actually needed — audit in Phase 2 confirms no method re-enters another
  locked method). All `#[pymethods]` become `&self`; the mutex is acquired
  inside `allow_threads` per I1. This also retires the PyO3
  `&mut`-borrow-check hazard that `_tlock` was papering over.
- **D3 — where embed-before-lock lives.** In `turbovecdb-core`'s write
  path, generically over the `Embedder` trait: `check_identity (no flock)
  → embed → flock.acquire → check_identity → write`. The PyEmbedder
  re-acquires the GIL itself (I1), so this is sound under `allow_threads`.
  Python's `_write` collapses to a one-line delegation (keep only the
  `_warn_vectors_bypass` warning, which is Python-visible logging).
- **D4 — constructor C7.** Core `Collection::new` takes the flock when
  `store.sqlite3` does not exist (same `looks_new` gate so hot-path
  re-opens stay lock-free), using the caller-supplied `lock_timeout` —
  which finally becomes a real parameter instead of the discarded one at
  `collection.rs:57`.
- **D5 — handle cache + `delete_collection`.** Move the cache into
  `_core.Database` as `Mutex<HashMap<String, Py<PyAny>>>` storing the
  *Python wrapper* objects, with a `get_or_create(name, factory)` method
  that calls a Python factory callback for misses. `delete_collection`
  moves wholesale into Rust: evict+close → flock → evict+close again →
  rmtree → release (I6). The C6 conflicting-options check stays in Python
  (it inspects the cached wrapper's properties) but runs on the object
  returned from Rust — no Python lock needed because Rust guarantees
  identity-stability. **Deadlock caveat for the implementer:** the factory
  callback runs Python code; do not hold the cache mutex while a *different*
  GIL-holding thread could block on that same mutex — the factory must be
  invoked per I1 discipline (this is the trickiest task card; see Phase 4).
  *Fallback if this fights PyO3:* keep the dict in Python guarded by GIL
  atomicity for `get`, and route only the miss path through a Rust-side
  `Mutex` — Opus decides if the primary shape costs more than a day.
- **D6 — public shims.** `write_lock_path()` stays in `collection.py` as a
  pure path helper (three test files import it, and it documents R1); it
  just no longer backs a Python `FileLock`. `_LOCK_TIMEOUT = 30` stays
  exported (imported by `database.py` today; keep for API stability).

## 5. Orchestration shape

```
Phase 1 (core flock)        — 1 Sonnet agent, pure Rust, no deps
Phase 2 (core lock plumbing)— 1 Sonnet agent, depends on 1
Phase 3 (PyO3 layer)        — 1 Sonnet agent, depends on 2; lands as one
                              PR with the wrapper's flock removal (see card)
Phase 4 (Database/cache)    — 1 Sonnet agent, depends on 3
Phase 5 (Python strip-down) — 1 Sonnet agent, depends on 3 (4 for database.py);
                              the flock half already landed with Phase 3
Phase 6 (test migration)    — 2–3 Sonnet agents in parallel, depends on 5
Phase 7 (docs/deps/changelog)— 1 Sonnet agent, depends on 6
```

- Phases 1–5 are **sequential** — each builds on the previous crate/API
  surface. Don't parallelize them into worktrees; the merge conflicts in
  `collection.rs` would cost more than the wall-clock saved.
- **Opus checkpoint after every phase**: read the full diff, run that
  phase's gate (§8), and verify the invariants I1–I6 by inspection — the
  GIL-ordering invariant (I1) in particular is not something the test
  suite reliably catches (deadlocks appear as CI timeouts, or worse, only
  under load). This review is the human-checkpoint analog: lock ordering
  and message fidelity are precision decisions; do not let one agent's
  unreviewed output feed the next agent.
- Each task card below lists **Files**, **Do**, **Don't**, **Gate**. Give
  Sonnet the card plus invariants verbatim; require it to run the gate
  itself before reporting done.

## 6. Task cards

### Phase 1 — `turbovecdb-core`: cross-process flock module

- **Files:** new `crates/turbovecdb-core/src/flock.rs`; `lib.rs` (export);
  `error.rs` (new `CoreError::LockTimeout(String)` variant mapping →
  `TurboVecError`, follow the existing variant/doc pattern).
- **Do:** `FlockGuard::acquire(path, timeout_secs) -> Result<FlockGuard,
  CoreError>` — open/create lock file, `flock(LOCK_EX | LOCK_NB)` poll
  loop at 50ms, RAII release on drop (release the flock but **do not
  delete the lock file** — filelock leaves it in place and deleting races
  with other openers). Message per I4 — take the *formatted-with-repr*
  message text as a parameter or helper so Phase 2/4 call sites can produce
  the two historical variants. Unit tests: acquire/release, contention
  between two threads with separate opens, timeout fires with correct
  message, drop-releases.
- **Don't:** touch `collection.rs`/`database.rs` yet; add crate deps beyond
  `rustix` (or `libc`).
- **Gate:** `cargo test -p turbovecdb-core`.

### Phase 2 — `turbovecdb-core`: lock the write paths

- **Files:** `crates/turbovecdb-core/src/collection.rs` (1840 lines — the
  agent should read the write/reembed/constructor paths, not the whole
  file), `database.rs`.
- **Do:** every mutating core entry point (`add`, `upsert`, `delete`,
  `update_metadata`, `update_documents`, `clear`, `reembed`, `flush`,
  `close`) acquires the flock (sibling path derived from `dir`; port
  `write_lock_path` logic) for its duration. Constructor per D4. Write
  path restructured per D3/I5: identity pre-check → `Embedder::embed` →
  flock → identity re-check → write. `lock_timeout: f64` becomes a real
  field on `CoreCollection`. Reads (`query`/`get`/`count`/`health`/
  `meta_get`/`store_gen`) take **no** flock (unchanged today — SQLite WAL
  handles read/write concurrency; the comment at `collection.rs:103-107`
  about busy_timeout still applies).
- **Don't:** add the in-process mutex here — that's the PyO3 layer's
  (Phase 3) concern; core stays single-threaded-per-handle. Don't change
  any error message that existing tests match on.
- **Gate:** `cargo test -p turbovecdb-core` (extend the existing
  in-crate tests: two `CoreCollection` handles on one dir, writer B times
  out while A holds; reembed identity-swap race now covered in core).

### Phase 3 — `turbovecdb-py`: in-process mutex + GIL discipline

- **Files:** `crates/turbovecdb-py/src/collection.rs`, `convert.rs`
  (LockTimeout → `TurboVecError` mapping), `embedder.rs` (verify
  `with_gil` re-entry — should already be correct).
- **Do:** wrap per D2 (`Mutex<CoreCollection>`, `&self` methods, lock
  inside `allow_threads` — I1). **Refactor `reembed`** to `allow_threads`
  + `with_gil`-based progress callback (I1's forcing function); preserve
  its progress-error precedence logic (`collection.rs:288-292`). Plumb
  `lock_timeout` through instead of discarding it. Poisoned-mutex policy:
  map to `TurboVecError` mentioning a prior panic, don't unwrap.
- **Don't:** change Python-visible signatures. **Do not leave both lock
  layers active:** `flock` treats separately-opened fds on the same file
  as independent even within one process, so the wrapper's held
  `FileLock` (fd₁) makes the core's acquisition (fd₂) time out on every
  write. Phase 3 therefore lands **together with Phase 5's flock
  removal** in one PR: the same change that turns the Rust flock on
  strips `_flock`/`_locked()` from `collection.py` (removing `_tlock`
  and the rest of the strip-down can still follow in Phase 5). Structure
  the work as two commits on one branch so review stays legible.
- **Gate:** `cargo test` + `maturin develop` + full `pytest` (with the
  paired wrapper change applied) + a 30s two-process write-hammer smoke
  script.

### Phase 4 — `turbovecdb-py`/core: Database cache + delete_collection

- **Files:** `crates/turbovecdb-py/src/database.rs`, core `database.rs`,
  `src/turbovecdb/database.py`.
- **Do:** D5. Rust `Database` gains the handle cache and
  `get_or_create(name, factory)`; `delete_collection` moves in full
  (evict/close → flock with the `to delete collection` message variant →
  re-evict → `remove_collection_dir`). Python `Database` drops
  `threading`, `filelock` imports and `_lock`; `collection()` becomes:
  C6-check-if-cached (via a Rust `get(name)` peek) then
  `get_or_create`. Preserve the close-error warning logs (Python-side
  logging is fine — do it in the factory/close shims).
- **Don't:** break handle identity (`db.collection("c") is
  db.collection("c")` — `test_cached_handle_conflicts.py` and C6 depend
  on it).
- **Gate:** full pytest, with special attention to
  `test_delete_collection.py`, `test_delete_collection_race.py`,
  `test_cached_handle_conflicts.py`, `test_constructor_race.py`.

### Phase 5 — Python strip-down

- **Files:** `src/turbovecdb/collection.py`.
- **Do:** delete `_tlock`, `_flock`, `_locked()`, the `_write`
  embed-before-lock branch, the constructor `looks_new` lock, and the
  `filelock`/`threading` imports. Keep: dataclasses (Rust constructs them
  by name — `collection.py:9-10`), `write_lock_path` + `_LOCK_TIMEOUT`
  (D6), `embedder_identity`, `_warn_vectors_bypass`,
  `_check_embedder_identity_matches` *only if* the C6 pre-check path in
  `database.py` still uses it — otherwise delete. Rewrite the module
  docstring: the wrapper no longer owns any lock; locking is documented
  as a core concern.
- **Don't:** change any public signature, dataclass field, or `__all__`.
- **Gate:** full pytest; `grep -rn "filelock\|threading" src/turbovecdb/
  --include="*.py"` returns only `service.py` (out of scope) and comments.

### Phase 6 — test migration (parallelizable, 2–3 agents by file group)

The suite is the spec — most tests must pass **unchanged**. Only tests
that poke wrapper internals need rework. Contention tests should keep
using Python `filelock.FileLock` as the *opposing* lock holder — that is
now a feature, not a shim: it proves I3 (old-wheel/new-wheel
interoperability) on every CI run. `filelock` moves to the dev/test
dependency group.

| File | Change |
|---|---|
| `test_lock_timeout.py` | `col._flock.acquire()` (line 20) and `col._flock.timeout` asserts (41, 49, 57) no longer exist. Rework: hold an external `FileLock(write_lock_path(...))` to create contention; timeout-value asserts move to behavior (measure elapsed) or a new `_core` introspection getter (`lock_timeout` property) — prefer the getter. |
| `test_seen_gen.py` | `col2._flock.acquire()` (78) → external `FileLock`. |
| `test_delete_collection_race.py` | monkeypatches `dbmod.FileLock` (129, 179) to sequence the race — that seam is gone. Rework as a real two-process race (subprocess + barrier files) or a Rust test hook; Opus reviews which. |
| `test_constructor_race.py`, `test_embed_before_lock.py`, `test_write_lock_location.py`, `test_concurrency.py` | already use external `FileLock` + `write_lock_path` — expected to pass unchanged (they now double as I3 compat tests). Verify, don't rewrite. |
| `test_crash_injection.py` | must pass **unmodified** — it SIGKILLs mid-write; flock auto-releases on process death exactly like filelock's fd did. Any change needed here is a red flag, stop and escalate. |
| `test_performance.py` | run and compare against pre-migration numbers (flock-per-write adds a syscall pair; poll loop only on contention — expect noise-level change; >10% write regression escalates to Opus). |

- **Gate:** full pytest green; the two-process hammer smoke from Phase 3
  rerun; performance comparison recorded in the PR.

### Phase 7 — docs, deps, changelog

- **Files:** `pyproject.toml` (move `filelock>=3.12` to the test/dev
  group), `docs/ARCHITECTURE.md`, `AGENT.md` (its "Building" section is
  already stale re: `rust/` path — fix opportunistically), `CHANGELOG.md`,
  module docstrings not covered in Phase 5, and a follow-up note on
  `service.py`'s `_lock_for` (probably still wanted for HTTP request
  serialization, but no longer *correctness*-load-bearing).
- **Gate:** `grep -rn filelock pyproject.toml` shows dev-group only;
  docs build/read clean; fresh `maturin develop` + pytest from a clean
  venv without runtime `filelock` installed.

## 7. Sub-agent prompting notes (for Opus)

- Give each agent: its task card, invariants I1–I6, the file list, and
  the gate command. Require the agent to run its gate before reporting.
- Sonnet agents must **not** decide: lock ordering changes, error-message
  wording, public API changes, dependency additions. Card says "Don't" —
  if an agent believes a Don't is wrong, it reports back instead of
  proceeding.
- After each phase, Opus reads the diff *against the invariants*, not
  just the tests: I1 violations (mutex acquired under GIL, `py` captured
  into a locked region) and I4 drift (message punctuation) survive green
  test runs.

## 8. Verification gates (Opus runs at each checkpoint)

1. `cargo test` (both crates) and `cargo clippy -- -D warnings`.
2. `maturin develop && python -m pytest` — full suite.
3. Crash harness: `pytest tests/test_crash_injection.py` (unmodified).
4. Two-process contention smoke: process A holds a Python
   `filelock.FileLock` on `<name>.lock`; process B (new build) `add()`s
   with `lock_timeout=1` → must raise `TurboVecError` with the exact I4
   message; then reversed (Rust holds, Python filelock times out) — this
   is the I3 interop proof in both directions.
5. Deadlock canary: N threads × (writes with a slow embedder + reads +
   one `reembed` with an `on_progress` callback) for 60s — exercises the
   I1 GIL/mutex interleavings that unit tests don't.
6. `test_performance.py` before/after comparison (record in PR).

## 9. Risks

- **GIL/mutex deadlock (I1)** — highest-severity, lowest test
  visibility. Mitigation: the uniform allow_threads-then-lock rule, the
  reembed refactor called out explicitly, and gate 5.
- **No incremental double-lock state.** `flock` is per-open-file-
  description: the wrapper's held `FileLock` (fd₁) and the core's new
  flock (fd₂) on the same file conflict *within one process*, so a
  "both layers on" transition build fails every write with a timeout.
  Already folded into Phase 3 (lands paired with the wrapper's flock
  removal, one PR, two commits) — noted here so nobody "fixes" a red
  intermediate build by making the Rust lock reentrant.
- **Message drift (I4)** — several tests `pytest.raises(match=...)` on
  exact text; Rust `{:?}`-style quoting differs from Python `repr()`.
  Mitigation: a small `py_repr_str` helper + unit tests comparing against
  fixture strings captured from the current Python output.
- **`Py<PyAny>` cache in Rust (D5)** holding Python wrapper objects
  creates a reference cycle (`Database` ↔ cached `Collection` if the
  wrapper back-references) — audit; wrappers don't reference `Database`
  today, keep it that way.
- **Windows** — interop with Python filelock is lost (different
  primitives). Document in ARCHITECTURE.md; CI is Linux/WSL.
