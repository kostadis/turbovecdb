# Phase 1: Query dimension validation & corrupt metadata

## Bugs

| ID | Title | Severity |
|----|-------|----------|
| #87 | Wrong-shape query vector panics or returns garbage | HIGH |
| #85 | Corrupted dimension metadata returns norm-mismatch instead of clear error | HIGH |

---

## Bug analysis

### #87 — Query dim mismatch

`query` in `collection.rs` calls `collection_dir()` but validates vectors via `resolve_vectors` which checks `vec.dim() == dims`. If the caller passes a vector with wrong dimensionality, no explicit pre-check catches it early.

A query with `dim=768` against a collection built with `dim=384` either:
- Panics in BLAS ops (wrong stride)
- Returns garbage cosine similarities (silent wrong answers)
- Hits dimension mismatch deep in HNSW or quantization

The `tests/test_phase1_bugs.py::test_query_wrong_dim` proves this: passing `dim=1536` to a `dim=384` collection triggers a panic instead of a clean `DimensionMismatch` error.

### #85 — Corrupt dim metadata

Collection metadata stores dimensionality in SQLite. If the file is truncated or hand-edited, `ensure_current()` reads an unparseable value. Current code either returns a wrong `dim=0` (which passes all checks) or silently defaults.

The `CorruptedMetadata` error variant exists but is not thrown from the metadata parse path. The fix: guard `get_meta("dim")` with proper error handling and detect `doc_count > 0` via actual count, not `next_uid`.

---

## Design

### Fix #87: `validate_query_vector()`

Introduce a public helper in `collection.rs`:

```
fn validate_query_vector(vec: &[f32], dims: u32) -> Result<(), CoreError>
```

Checks:
1. `vec.len() == dims as usize` — wrong shape → `DimensionMismatch`
2. Any NaN or Inf — `InvalidArgument`
3. All-zero vector (norm ≈ 0) — `InvalidArgument` with message "zero-norm query vector"

Call site: at the top of `query()`, right after `ensure_current()`, before any compute.

Existing `resolve_vectors` already does (1) — the fix adds (1) earlier and adds (2)+(3).

### Fix #85: `CorruptedMetadata` on bad dim

**Change 1**: In `ensure_current()`, after `collection.meta.get("dim")`:

```rust
let dim_str = collection.meta.get("dim")
    .ok_or(CoreError::CorruptedMetadata("missing dim"))?;
let dim: u32 = dim_str.parse()
    .map_err(|_| CoreError::CorruptedMetadata("unparseable dim"))?;
```

**Change 2**: Instead of `next_uid - 1` as doc_count proxy, query `SELECT COUNT(*) FROM doc` from the store:

```rust
let doc_count = collection.store.doc_count()?;
```

Pre-existing `doc_count()` must exist on the store trait (add if missing). This avoids trusting a write-only counter for read validation.

**Change 3**: Route `CorruptedMetadata` through PyO3 to Python `CorruptedMetadataError(TurboVecError)`.

---

## Conflict: retry vs abort on corrupt metadata

The critic argued that throwing `CorruptedMetadata` is a breaking change for callers who tolerate `dim=0`. If a caller does `if col.dim() == 0: skip`, they must now catch `CorruptedMetadataError`.

**Orchestrator opinion**: Accept the break. Silent wrong answers are worse. Document in changelog: "collections with corrupt metadata now raise `CorruptedMetadataError`."

**Decision (2026-07-10)**: ✅ Accept the break.

---

## Files touched

| File | Change |
|------|--------|
| `crates/turbovecdb-core/src/collection.rs` | `validate_query_vector()` helper + call in `query()` |
| `crates/turbovecdb-core/src/error.rs` | `CorruptedMetadata` variant (exists, verify) |
| `crates/turbovecdb-py/src/lib.rs` | Map `CorruptedMetadata` to Python exception |
| `src/turbovecdb/errors.py` | Add `CorruptedMetadataError(TurboVecError)` |

---

## Verification

- `test_query_wrong_dim` — expect `DimensionMismatch` (xfail→pass)
- `test_validate_query_nan` — expect `InvalidArgument` (xfail→pass)
- `test_query_vector_zero_norm` — expect `InvalidArgument` (xfail→pass)
- `test_validate_query_zero_norm_length_1` — edge case (xfail→pass)
- `test_query_wrong_dim_regression_1d` — 1-D edge case (xfail→pass)
- `test_corrupt_meta_bad_dim` — expect `CorruptedMetadataError` (xfail→pass)
- `test_corrupt_meta_missing_dim` — expect `CorruptedMetadataError` (xfail→pass)
- `test_corrupt_meta_nan_inf_dim` — expect `CorruptedMetadataError` (xfail→pass)
