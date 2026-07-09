# #47: Recover in-process read/write concurrency during slow embed

**Issue:** [#47](https://github.com/kostadis/turbovecdb/issues/47)
**Status:** Implemented
**Priority:** Low

## Problem

Embedding runs inside the collection's in-process lock (`Mutex`):

```
lock()
  resolve_vectors()
    embedder.embed(docs)   ← slow, blocks concurrent reads
  write to sqlite
```

A concurrent `query()`/`get()`/`count()` on the same collection blocks
for the entire embed duration, even though reads don't need the embedder.

## Constraint

The fix must live in the **Rust core** so it survives replacing the
Python/PyO3 layer with a native CLI or any other binding (per AGENTS.md).

## Design: Arc-embedder + two-phase write

### Step 1: Embedder trait gets `Sync`

```rust
pub trait Embedder: Send + Sync {
    fn embed(&self, docs: &[String]) -> Result<Array2<f32>, CoreError>;
    fn identity(&self) -> String;
}
```

`Sync` is safe for all existing embedders:
- `NoEmbedder` — no state, trivially `Sync`
- `PyEmbedder` — wraps `PyObject`, which is `Send + Sync`
- `ConstantEmbedder` (test) — no shared state, trivially `Sync`

### Step 2: Collection stores embedder behind `Arc`

```rust
pub struct Collection<E: Embedder, I: VectorIndex> {
    ...
    embedder: Option<Arc<E>>,
    ...
}
```

The constructor takes `embedder: Option<E>` and wraps it in `Arc`.
`resolve_vectors` calls `self.embedder.as_ref().unwrap().embed(docs)`
(unchanged — `Arc<E>: Deref<Target=E>`).

### Step 3: Expose embedder + resolve_vectors publicly

```rust
impl<E: Embedder, I: VectorIndex> Collection<E, I> {
    /// Get a reference-counted handle to the embedder.
    /// The caller can clone the Arc and embed outside any collection lock.
    pub fn embedder(&self) -> Option<Arc<E>> {
        self.embedder.clone()
    }

    /// Resolve documents → L2-normalized vectors.
    /// If `vectors` is provided, just normalizes and returns.
    /// Now public so callers can embed outside their lock.
    pub fn resolve_vectors(
        &self,
        documents: Option<&[String]>,
        vectors: Option<Array2<f32>>,
    ) -> Result<Array2<f32>, CoreError>;
}
```

### Step 4: CachedDatabase two-phase `add_text`

```rust
impl<E: Embedder, I: VectorIndex> CachedDatabase<E, I> {
    /// Add text documents to a collection.
    /// Embedding runs OUTSIDE the collection lock so concurrent reads
    /// are not blocked. The embedder identity is re-checked under the
    /// write lock to guard against embedder swaps (R3).
    pub fn add_text(
        &self,
        name: &str,
        ids: Vec<String>,
        documents: Vec<String>,
        metadatas: Option<Vec<String>>,
        dim: Option<i64>,
        bit_width: i64,
        metric: Option<String>,
        lock_timeout: f64,
    ) -> Result<(), CoreError> {
        // Phase 1: grab the embedder reference (brief lock)
        let emb = {
            let handle = self.collection(name, dim, bit_width, metric.clone(), None, lock_timeout)?;
            let coll = handle.lock().unwrap();
            let emb = coll.embedder().ok_or_else(|| {
                CoreError::EmbedderRequired("...".into())
            })?;
            let identity = emb.identity();
            (emb, identity)
        };

        // Phase 2: embed OUTSIDE the collection lock
        let vectors = emb.embed(&documents)?;

        // Phase 3: re-acquire lock, re-check identity, write
        {
            let handle = self.collection(name, dim, bit_width, metric, None, lock_timeout)?;
            let mut coll = handle.lock().unwrap();
            // Re-check embedder identity under lock (R3)
            let current = coll.embedder().map(|e| e.identity()).unwrap_or_default();
            if current != emb.identity() {
                return Err(CoreError::EmbedderIdentityMismatch(
                    format!("embedder changed between embedding and write")
                ));
            }
            // resolve_vectors with pre-embedded vectors (fast: just normalizes)
            let vectors = coll.resolve_vectors(None, Some(vectors))?;
            coll.add(ids, None, metadatas, Some(vectors))?;
        }
        Ok(())
    }
}
```

### Step 5: Update callers

- **PyO3 bindings** — use `CachedDatabase::add_text` instead of
  locking + calling `add(embeddings=...)` directly. Or call
  `collection.embedder()` + embed + `collection.add(vectors=...)`.
- **CLI** (`turbovecdb-cli`) — unchanged (uses `NoEmbedder` + vectors
  directly, never hits the slow path).
- **Existing tests** — `add(documents=...)` tests still work through
  `Collection::add` directly (backward compat).

### Effect on concurrency

Before:                 After (two-phase):
```
Thread A    Thread B         Thread A              Thread B
  lock()                     |-- get embedder (brief lock)
  |-- embed (SLOW)           |-- release lock
  |             blocked      |-- embed (SLOW)      query() → lock (read) → OK
  |-- write                  |-- lock (write)
  unlock()                   |-- re-check identity
                             |-- write
                             unlock()
```

Concurrent reads are NOT blocked during embedding.

### Reentrant-deadlock side effect

This also fixes the accepted-but-documented reentrant-deadlock: an
embedder / `reembed` `on_progress` callback that re-enters the same
collection no longer deadlocks, because the embed runs outside the
collection's Mutex. The caller still needs to avoid re-entrancy in
their own lock, but the core no longer contributes to it.

## Test plan

- Unit test: embedder identity re-check catches a swap between
  phase 1 and phase 3
- Unit test: concurrent `query()` does not block during
  `add_text()`'s embed phase
- Existing `CachedDatabase` tests pass unchanged
- Existing `resolve_vectors` tests pass unchanged
