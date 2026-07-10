# Phase 3: Reembed locks, embedder identity, stale config

## Bugs

| ID | Title | Severity |
|----|-------|----------|
| #102 | Reembed holds write lock during slow embedding — blocks all reads | CRITICAL |
| #101 | Legacy embedder silently adopts wrong identity for existing data | HIGH |
| #100 | Embedder identity collision corrupts index after embedder change | CRITICAL |
| #99 | Stale collection config uses old model settings after file restore | HIGH |

---

## Bug analysis

### #102 — Reembed holds write lock

`reembed()` in `collection.rs` loops: for each batch of docs, retrieve from store, embed via Python, write new vectors. The entire loop holds the **write lock** (`state.write()`). If embedding takes 10ms per doc × 10K docs = 100s, no reads can execute for 100s. This is a denial-of-service for any concurrent reader.

The fix requires splitting reembed into phases:
1. **Classify** — which docs need re-embedding (lock held, fast)
2. **Embed** — call embedder on each doc (no lock held, slow)
3. **Apply** — write new vectors under lock (lock held, fast)

This is the "two-phase" pattern described in `AGENTS.md` for the general case.

### #101 — Legacy identity adoption

When `collection.embedder()` returns `None`, the embedder is "legacy" — any embedder that produces matching vector dimensions is accepted. This means you can silently switch from `text-embedding-3-small` (dim=1536) to `text-embedding-3-large` (dim=3072) and the collection accepts it, returning garbage for previously embedded docs whose vectors don't match the new embedder's output.

Fix: block identity adoption when `count > 0`. Once a collection has data, changing the embedder requires `reembed()`.

### #100 — Identity collision

Two different embedder models with the same dimensions (e.g., two `dim=384` models) produce different vectors for the same text. The current identity is just the class name (e.g., `"HuggingFaceEmbedder"`), so swapping one for the other silently corrupts query results.

Fix: identity includes model configuration (model ID, pooler, normalize). `reidentity()` allows users to explicitly acknowledge the change and trigger re-embedding.

### #99 — Stale config refresh

`Collection::new()` reads config from metadata and caches it. If the metadata file is replaced (restore from backup), the cached config is stale. `ensure_current()` does not re-read config, so the collection silently uses old settings.

Fix: `ensure_current()` calls `refresh_config_from_meta()` which re-reads the metadata file and updates the cached config struct.

---

## Design

### Fix #102: Three-phase reembed

```rust
// Phase 1: Classify (lock held)
let mut state = self.state.write().unwrap();
let doc_ids: Vec<u64> = state.index.all_doc_ids(); // fast
drop(state); // release lock

// Phase 2: Embed (no lock held)
let embedder = self.embedder(); // clone Arc<E>
let new_vecs: Vec<Vec<f32>> = doc_ids
    .par_chunks(BATCH_SIZE)
    .map(|batch| embedder.embed(self.get_texts(batch)))
    .flatten()
    .collect(); // slow, no lock

// Phase 3: Apply (lock held, check identity still matches)
let mut state = self.state.write().unwrap();
if state.embedder_identity != embedder.identity() {
    return Err(CoreError::EmbedderChanged);
}
state.index.update_vectors(&doc_ids, &new_vecs)?;
```

Critical detail: Phase 3 must verify the embedder identity still matches after re-acquiring the lock. This prevents a race where the embedder was changed between Phase 2 and Phase 3.

### Fix #101: Block identity adoption on non-empty

In `resolve_vectors` or at the start of `add()`:

```rust
if collection.doc_count() > 0 && embedder.identity() != collection.embedder_identity() {
    return Err(CoreError::EmbedderMismatch(
        "cannot change embedder on non-empty collection; use reembed()".into()
    ));
}
```

This blocks silent adoption without forcing a `reembed()`.

### Fix #100: Identity includes model config + reidentity()

**New identity format**: `"EmbedderName(model=..., pooler=..., normalize=...)"` constructed from the embedder's configuration. Stored in metadata as `embedder_identity`.

**New method `reidentity()`**:
```rust
pub fn reidentity(&self, new_identity: &str) -> Result<()>
```
- Does NOT re-embed any vectors
- Only updates the stored identity string
- Requires explicit user acknowledgment: "I know the embedder identity is changing"
- Logs a warning

### Fix #99: `refresh_config_from_meta()`

```rust
pub fn refresh_config_from_meta(&mut self) -> Result<()> {
    let meta = MetaStore::open(self.collection_dir())?;
    self.config = CollectionConfig {
        dim: meta.get("dim")?.parse()?,
        distance: meta.get("distance")?.parse()?,
        // ... other fields
    };
    Ok(())
}
```

Called at the start of `ensure_current()` before any generation check.

---

## Conflict: retry vs abort on concurrent modify during reembed

Designer chose 3 retries with exponential backoff (0.1–1.6s). Critic argued for set reconciliation (merge concurrent changes instead of retrying).

**Orchestrator opinion**: Backoff is sufficient. Reembed is an admin operation, not a hot path. Document "do not run concurrent writes during reembed." Full set reconciliation is engineering effort disproportionate to the risk.

**Decision (2026-07-10)**: ✅ Accept backoff. Include an upgrade script (`scripts/upgrade-phase3.py`) that automates reembed + identity migration for existing collections.

## Conflict: backward compat of new identity format

Existing collections have identities like `"HuggingFaceEmbedder"`. New format is `"HuggingFaceEmbedder(model=sentence-transformers/all-MiniLM-L6-v2)"`. Old and new won't match, causing `add()` to reject legitimate operations.

**Orchestrator opinion**: `reidentity()` exists for exactly this case. Ship a one-liner migration script. In `add()`, compare both old-format and new-format identity; if the format was the old one and the new one matches the same class, allow the operation with a deprecation warning.

**Decision (2026-07-10)**: ✅ Agreed. The upgrade script (`scripts/upgrade-phase3.py`) handles identity re-stamping + optional reembed for all existing collections. `add()` has a grace period where old-format identities are accepted with a deprecation warning.

---

## Files touched

| File | Change |
|------|--------|
| `crates/turbovecdb-core/src/collection.rs` | Three-phase reembed; `reidentity()`; `refresh_config_from_meta()`; identity adoption guard |
| `crates/turbovecdb-core/src/embedder.rs` | Identity includes model config field |
| `crates/turbovecdb-py/src/embedder.rs` | Construct identity from PyEmbedder config |
| `crates/turbovecdb-core/src/error.rs` | `EmbedderChanged` variant |
| `src/turbovecdb/collection.py` | Expose `reidentity()` |
| `scripts/upgrade-phase3.py` | Upgrade script: identity re-stamp + optional reembed for existing collections |

---

## Verification

- `test_reembed_blocks_reads_timeout` — expect add succeeds during reembed (xfail→pass)
- `test_reembed_concurrent_add_same_collection` — expect both complete (xfail→pass)
- `test_reembed_with_identity_guard` — expect `EmbedderMismatch` (xfail→pass)
- `test_reembed_rejects_changed_identity` — expect `EmbedderChanged` (xfail→pass)
- `test_legacy_embedder_identity` — expect `EmbedderMismatch` for new model (xfail→pass)
- `test_change_embedder_model_same_dim` — expect `EmbedderMismatch` (xfail→pass)
- `test_reidentity_updates_stored_identity` — verify identity change (xfail→pass)
- `test_stale_config_after_meta_restore` — expect correct config (xfail→pass)
