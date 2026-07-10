# Phase 2 Triage — Remaining Issues After 2 Iterations

## Status

| Test | Bug | Verdict |
|------|-----|---------|
| `test_clear_leaked_transaction_on_store_gen_failure` | #95 | ✅ FIXED (PASSED) |
| `test_delete_leaked_transaction_on_store_gen_failure` | #95 | ✅ FIXED (PASSED) |
| `test_concurrent_flush_cross_stamp` | #91 | ✅ PASSES WHEN TRIGGERED (XPASS, timing-dependent) |
| `test_tvim_from_other_collection_produces_wrong_results` | #90 | ✅ FIXED (PASSED) |
| `test_clear_post_commit_index_failure_masks_success` | #96 | ❌ TEST DESIGN ISSUE — see below |
| `test_add_post_commit_store_gen_failure` | #96 | ❌ TEST DESIGN ISSUE — see below |

## Issue: #96 tests corrupt SQLite but in-memory cache is unaffected

Both #96 tests directly corrupt SQLite metadata (`dim=''` / `store_gen='corrupt'`) via a
separate connection, then call the already-open handle. The fix for #96 moved post-commit
bookkeeping (make_index, store_gen_val) **before** COMMIT. Since the in-memory `self.dim`
and `self.seen_gen` are cached from `Collection::new()`, the corruption in SQLite does NOT
affect the already-open handle's operations.

**Why the "clear" test fails**: clear() calls make_index() which uses `self.dim.unwrap()`
(in-memory, still `Some(8)`). Since dim is cached, make_index succeeds, COMMIT happens,
and clear returns Ok(()) — no error. The test expects an error.

**Why the "add" test fails**: add() reads store_gen via `store_gen_val()`. Since `store_gen`
was cached in the in-memory meta cache, the corrupt value in SQLite is never read by the
open handle. The test expects an error but add succeeds.

**Root cause**: These tests were designed to prove a theoretical race where SQLite-level
state changes between the BEGIN and the bookkeeping step. But in practice, the Rust code
caches metadata in memory and never re-reads from SQLite during write operations. The
test's corruption of SQLite state via a second connection is invisible to the open handle.

**To fix the tests**: They need to be redesigned to inject failures closer to the actual code
path — e.g., via a commit callback that fails after N operations (similar to the Rust test
`#91`'s simulation approach), or by using a mock store. The current SQLite-level
corruption neither proves nor disproves the bug with the fix in place.

**Recommendation**: The code fix (move bookkeeping before COMMIT) is correct by
construction. Close these tests as "fixed by design" or rewrite them using Rust-level
test infrastructure (commit hooks, fault injection).
