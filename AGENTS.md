# Design rules

## Lock / embed concurrency

Embedding must run OUTSIDE the write lock. The Rust core must provide
a two-phase write path so bindings (PyO3, CLI, etc.) can:

1. Embed documents → vectors (no lock held)
2. Lock → check embedder identity → write vectors

This ensures a slow embedder never blocks concurrent reads, regardless
of which binding layer is in use. Do NOT put the embed call inside the
lock at any layer — the fix must live in the Rust core, not in PyO3.

### How: the two-phase API

- **`Collection.embedder()`** returns `Option<Arc<E>>` — clone the `Arc`
  and embed outside the lock.
- **`Collection::new()`** takes `embedder: Option<E>` and wraps it in
  `Arc` internally.
- **`resolve_vectors`** is `pub` — call it with `resolve_vectors(None, Some(vecs))`
  to normalize pre-embedded vectors without re-embedding.
- **`CachedDatabase::add_text()`** is the reference two-phase implementation
  (embed outside lock, re-check identity under lock).

Callers should prefer `CachedDatabase::add_text()` or follow the same
three-phase pattern:
1. `collection.embedder()` → clone `Arc<E>`
2. `emb.embed(docs)` (no lock held)
3. Lock → `collection.resolve_vectors(None, Some(vecs))` → `add(ids, ...)`
