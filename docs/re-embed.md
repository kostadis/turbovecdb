# Design note: first-class collection re-embed / re-dimension

**Status:** proposal (design note, not yet implemented)
**Motivation:** a real migration done *by hand* against turbovecdb's current
API; this note records what that required, why it was ugly, and the small
in-library primitive that would replace it.

---

## 1. The incident that motivated this

MemPalace's embedding model was upgraded **`nomic-embed-text` (768-dim) →
`qwen3-embedding:0.6b` (1024-dim)** after an A/B showed a ~27% MRR win on real
data. That meant **re-embedding every stored vector** across 6 palaces —
**135,646 documents** in 11 turbovec collections — changing vector dimension
768 → 1024.

turbovecdb has **no API for this**. The migration was done with an external
script, `~/src/mempalace/scripts/reembed_turbovec.py`, that reaches around the
library. It worked, but every awkward step in it is a turbovecdb gap. This note
turns that script into a feature request.

## 2. What the external script had to do (and why each step is a smell)

Per collection:

```
col = db.collection(name, create=False)
res = col.get(include=("documents", "metadatas"))   # read ids, docs, metas
vectors = new_embedder(res.documents)                # recompute (new model/dim)
db.close()
shutil.rmtree(<db_path>/<name>/)                     # (!) no drop API — delete the dir
db = turbovecdb.connect(...)
new = db.collection(name, dim=1024, bit_width=4, metric="cosine", create=True)
new.add(ids=res.ids, documents=res.documents, metadatas=res.metadatas, vectors=vectors)
```

What's wrong with this as a pattern:

| step | smell | why it's a library gap |
|---|---|---|
| `shutil.rmtree(dir)` | reaching into on-disk layout | **no `delete_collection` API.** The only way to drop a collection is to delete its directory out from under the library, after `close()`. |
| recreate with `dim=1024` | dimension is immutable | `_commit_dim` sets `dim` **once**; `_write` then raises `DimensionMismatchError` forever. There is **no sanctioned re-dimension** path, so "change the model" forces a full drop. |
| drop **then** recreate | destructive, non-atomic | between `rmtree` and a successful re-`add`, the collection **does not exist**. A crash there loses it. This **contradicts turbovecdb's own crash-safety guarantee** (see §3) — the one place the library is destructive is the one place it has no primitive. |
| re-specify `bit_width=4, metric="cosine"` | params not carried over | the caller must *remember and restate* the collection's quantization/metric. The library already has them in `meta`; the external path can silently change them (a latent re-quantization bug). |
| round-trip `documents` + `metadatas` through the caller | needless data motion | the library already holds `document`, `metadata`, `str_id`, and `uid` in SQLite. Only `vector` actually changes. Pulling everything out and pushing it back is wasted I/O and a chance to corrupt `str_id`↔`uid` stability. |
| 135k docs, ~3.5 h, one shot | no progress / resume | a multi-hour remote-embedder job with no checkpoint, no progress callback, and an all-or-nothing failure mode (mitigated only by an external `cp -a` backup). |

## 3. The insight: SQLite is the source of truth, so re-embed is *in-place*

From `docs/core/data-model.md`: `docs.vector` (float32, L2-normalized) is the
**source of truth**; `index.tvim` is a **rebuildable cache** reconciled via
`store_gen` / `tvim_gen`. A stale or missing `.tvim` is never a correctness
problem — it rebuilds from `docs.vector` on next open.

That changes everything. Re-embedding does **not** need a drop:

> Recompute each row's `vector` from its stored `document`, write the new
> vectors back into `docs.vector`, re-commit `dim`, and let the index rebuild
> from the new vectors. The directory is never deleted; the collection never
> ceases to exist.

Everything the external script does the hard way, the library can do the safe
way, because it owns the schema:

- **Dim change** is just `_commit_dim`'s sibling — a `_recommit_dim(new_dim)`
  that updates `meta.dim` and re-shapes the index. (`_commit_dim` today only
  sets `dim` when it is `None`; re-embed is the *sanctioned* path to change it.)
- **`bit_width` / `metric`** are read from `meta` and preserved automatically.
- **`str_id` ↔ `uid`** mapping, `document`, `metadata` are untouched — only
  `vector` is rewritten — so identity and metadata can't drift.
- **Atomicity** comes from the existing model: do the `UPDATE docs SET vector`
  + `meta.dim` writes under the **write lock** in a single SQLite transaction,
  bump `store_gen`. A crash before the `.tvim` flush is harmless — the next
  open rebuilds the index from the already-committed new vectors. No external
  backup required for crash-safety (still wise for model-rollback).
- **Reader coherence** is free: the `store_gen` bump makes live readers detect
  staleness and rebuild from the new vectors, exactly as for any other write.

## 4. Proposed API

Two additions. The first is generally useful; the second is the real feature.

### 4.1 `Database.delete_collection(name)`

The missing drop primitive. Closes any cached handle, releases the DB, removes
the collection directory through the library (so lock files and the handle
cache are handled correctly) rather than the caller calling `shutil.rmtree`.

```python
db.delete_collection("mempalace_drawers")   # raises CollectionNotFoundError if absent
```

### 4.2 `Collection.reembed(embedder, *, dim=None, batch_size=256, on_progress=None, skip_empty="error")`

Recompute every vector in place from the stored `document`, using a **new**
embedder. Handles a dimension change. Preserves ids, documents, metadata,
`bit_width`, `metric`.

```python
col = db.collection("mempalace_drawers", create=False)
report = col.reembed(qwen3_embedder, on_progress=lambda d, n: ...)
# report: ReembedReport(n_docs, old_dim, new_dim, n_skipped, elapsed_s)
```

Semantics:

1. Stream rows from `docs` (ordered by `uid`), `batch_size` at a time.
2. For each batch, call `embedder(documents)` → new vectors; L2-normalize.
3. Under the **write lock**, in one transaction:
   - `UPDATE docs SET vector=? WHERE uid=?` for the batch;
   - on the first batch, if `len(vec) != stored dim`, `_recommit_dim(new_dim)`
     (must be a positive multiple of 8 — same guard as `_commit_dim`);
   - bump `store_gen`.
4. After all batches commit, rebuild the in-memory index at the new
   `(dim, bit_width)` from `docs.vector` and `flush()` (`tvim_gen = store_gen`).
5. Update the **embedder-identity** meta (see §5).

Parameters:

- **`embedder`** — required. The new embedding callable (the whole point is a
  *different* model). Same contract as the constructor's `embedder`.
- **`dim`** — optional assertion; if given and the embedder returns a different
  width, raise before mutating anything (catch a misconfigured embedder early).
- **`batch_size`** — embedder call size; tune for remote endpoints.
- **`on_progress(done, total)`** — for multi-hour remote embedders.
- **`skip_empty`** — policy for rows whose `document` is empty/NULL and thus
  can't be re-embedded from text: `"error"` (default), `"keep"` (leave the old
  vector — only valid when dim is unchanged), or `"drop"`. The external script
  never hit this because every mined row had text, but the schema allows empty
  `document`, so the library must take a position.

## 5. Interaction with the embedder-identity guard (GAP-1)

`docs/mempalace-backend-gaps.md` **GAP-1** wants turbovecdb to reject reads/writes
when the embedding model differs from the one a collection was built with
(`EmbedderIdentityMismatchError`). Re-embed is the **counterpart** to that guard:
the guard forbids an *accidental* model swap; `reembed` is how you perform a
*deliberate* one. So `reembed` must **rewrite the stored embedder-identity** to
the new model as part of its transaction — otherwise the identity guard (once it
exists) would reject every post-migration query. Implementing the two together
is cleaner than either alone: the guard gives safety, `reembed` gives the
sanctioned escape hatch.

## 6. Open questions

- **Resumability.** One transaction over 135k rows is crash-safe (it rolls back),
  but a mid-run failure wastes hours of remote embedding. Worth a `--resume`
  mode that commits per batch and records progress in `meta`? The all-or-nothing
  version is simpler and correct; resume is a v2.
- **Re-quantize too?** Should `reembed` optionally accept a new `bit_width`
  (re-quantize while re-embedding)? Cheap to allow, but widens scope; probably a
  separate `recompress()` primitive.
- **Who owns batching/HTTP?** turbovecdb's `embedder` is just a callable, so the
  caller still owns the network (Ollama/OpenAI). `reembed` only orchestrates
  read → call → write. That boundary feels right; keep the library transport-free.
- **Multi-process during migration.** `reembed` holds the write lock for the
  whole run (hours). Either accept that writers block, or commit per batch and
  release between batches (interleaves other writers but lengthens the window
  where some rows are new-dim and some old — a dimension-mixing hazard the guard
  in §5 should catch).

## 7. Back-compat & cleanup

Both additions are **purely additive** — existing collections and call sites are
unaffected. Once `delete_collection` + `reembed` land, the external
`reembed_turbovec.py` in MemPalace can be deleted and MemPalace can expose a
`mempalace reembed` command that calls the library primitive instead of reaching
into the on-disk layout.

## 8. References

- External workaround this note generalizes: `~/src/mempalace/scripts/reembed_turbovec.py`
- Data model / crash-safety / generations: `docs/core/data-model.md`
- Dimension guard & write path: `src/turbovecdb/collection.py` (`_commit_dim`, `_write`)
- Embedder-identity guard: `docs/mempalace-backend-gaps.md` GAP-1
- Motivating migration: MemPalace nomic→qwen3 cutover, 6 palaces / 135,646 docs,
  dim 768 → 1024, 2026-06-12
