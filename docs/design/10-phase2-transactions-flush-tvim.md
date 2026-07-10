# Phase 2: Transactions, flush race, .tvim content

## Bugs

| ID | Title | Severity |
|----|-------|----------|
| #95 | Leaked transaction leaves SQLite WAL growing unbounded | HIGH |
| #96 | Post-commit errors re-run `ensure_current()` and double-advance generation | HIGH |
| #91 | Flush race corrupts `.tvim` file under concurrent write + flush | CRITICAL |
| #90 | `.tvim` content between versions is garbage after downgrade | HIGH |

---

## Bug analysis

### #95 — Leaked transaction

Functions like `add`, `update`, `delete`, `clear` call `store_gen_val()` which issues `BEGIN` on the SQLite store. If the operation fails before `COMMIT`, the transaction stays open. An open but idle transaction prevents SQLite WAL checkpoints, so the WAL file grows unbounded until the connection is closed.

The fix: move `store_gen_val()` (which issues `BEGIN`) to **after** all pre-flight validation, and wrap in a deferred-execution pattern.

### #96 — Post-commit errors

After `store_gen.commit()`, if the index build or `.tvim` write fails, the caller retries, which calls `ensure_current()` again. This re-reads `seen_gen` from metadata, but `seen_gen` was already advanced by the failed commit, so `ensure_current()` skips the generation — losing the docs that were committed but not indexed.

The fix: move `seen_gen = sg` and the index build **before** `store_gen.commit()`. If the build fails, nothing was committed, so retry is clean.

### #91 — Flush race / cross-stamp

Two concurrent calls: one writes vector data (add), the other writes `.tvim` metadata (flush). The `.tvim` is written outside the write lock. If order of writes interleaves, the result is a cross-stamped `.tvim` whose content belongs to a different generation than its filename claims.

The Rust test `flush_cross_stamp_simulated_health_hides_it` proves this: it directly simulates the end state of a cross-stamp (`.tvim` filename says generation N, content says N-1). `health()` reports `coherent=true` because it trusts the filename over the content — hiding the corruption.

### #90 — .tvim content swapped

The `.tvim` file stores `(magic, gen_a, uid_a, gen_b, uid_b)` as two (generation, UID) pairs. When `FlushStamper` was modified, the byte order of the two pairs was swapped. Collections written by a new binary cannot be read by an old binary, and vice versa. Since there is no version marker in the format, the reader cannot detect the swap.

---

## Design

### Fix #95: Move BEGIN after validation

For each write function (`add`, `update`, `delete`, `clear`, `flush`):

1. Perform all validation and pre-flighting **before** calling `store_gen_val()` (which does `BEGIN`)
2. Call `store_gen_val()` only when ready to commit
3. If the operation fails after BEGIN, `ROLLBACK` in the error path

This requires no structural change — just reordering existing code.

### Fix #96: Move seen_gen + index build before COMMIT

In the five write functions:

```rust
// 1. BEGIN (store_gen_val)
let sg = store_gen_val(&mut store, Some(&gen))?;

// 2. Build index / update seen_gen
let seen_gen = sg;
index.ensure_generation(seen_gen)?;

// 3. Only then COMMIT
store_gen.commit()?;
```

If step 2 fails, the transaction is rolled back, `seen_gen` was never persisted, and retry works cleanly.

### Fix #91: Move .tvim write inside lock

**Core insight**: the cross-stamp race exists because `.tvim` write is outside the lock while `meta.put("dim", ...)` / `meta.put("doc_count", ...)` are inside.

Fix: move `FlushStamper::write()` call **inside** the lock scope, right after the meta writes. Keep `ensure_current()` (which reads `.tvim` but does not write it) outside.

This closes the race because:
- Write lock serializes all writers
- Inside lock: meta writes + `.tvim` write are atomic w.r.t. other writers
- Outside lock: `ensure_current()` can read .tvim without blocking — it's read-only

This partially reverses the #51 optimization (which moved I/O outside lock). The remaining I/O inside lock is only the `.tvim` metadata file (~few bytes), not the full SQLite or HNSW index.

### Fix #90: SipHash UID fingerprint + version-aware migration

1. Add a UID fingerprint derived from the UID bytes via SipHash to `.tvim` format
2. The fingerprint is written as a 4th field after the two (gen, uid) pairs
3. On read: compute expected fingerprint from the two UIDs; if mismatch → format mismatch → panic with migration instructions
4. Add a migration that re-stamps `.tvim` in the correct order on first read

Alternative considered: add a `tvim_format_version` to metadata. Rejected because metadata itself could be corrupt. The fingerprint is self-validating.

---

## Conflict: lock I/O vs #51

The critic noted this reverses the #51 "I/O outside lock" optimization.

**Orchestrator opinion**: Accept the partial reversal. #51's goal was to prevent `ensure_current()` from acquiring the lock during write. That still holds — `ensure_current()` is outside. Only the `.tvim` metadata write (~64 bytes) moves inside. The SQLite and HNSW writes remain outside. The performance impact is negligible (<100µs per write).

**Decision (2026-07-10)**: ✅ Accept the partial reversal.

## Files touched

| File | Change |
|------|--------|
| `crates/turbovecdb-core/src/collection.rs` | Reorder gen_begin/gen_commit/ensure_current in 5 functions; move `.tvim` write inside lock |
| `crates/turbovecdb-core/src/flush.rs` | Add UID fingerprint to `.tvim` format; version-aware read |
| `crates/turbovecdb-core/src/database.rs` | If applicable, adjust `flush_collection` |
| `src/turbovecdb/errors.py` | Add `CorruptedTvimError(TurboVecError)` |

---

## Verification

- `test_leaked_transaction_after_add_failure` — expect no WAL growth (xfail→pass)
- `test_committed_but_not_indexed` — expect doc retrievable after retry (xfail→pass)
- `flush_cross_stamp_simulated_health_hides_it` — expect `coherent=False` (xfail→pass, Rust test)
- `test_tvim_uid_swap_migration` — expect migration succeeds (xfail→pass)
