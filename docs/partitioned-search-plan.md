# Partitioned Search Plan — Routed Per-Wing Shards

Status: **proposed**, 2026-07-24. Spans two repos (`turbovecdb`, `mempalace`)
plus one optional upstream PR to `turbovec`.

## Decision

Partition mempalace's vector search into **one turbovecdb collection per wing**,
routed on the `where` filter, rather than clustering slots inside a single
collection.

The rejected alternative (slot clustering — reorder the index build so a filter's
allowed slots land in contiguous 32-vector blocks and `block_has_allowed` can skip
them) is not dead; see [Deferred: slot clustering](#deferred-slot-clustering).

## Why

turbovec is a **flat quantized scan**, not a graph: query cost is O(N), not
log N. Partitioning is the only scaling lever the design has. Two measured facts
picked routing over clustering:

1. **A single search uses a single core.** turbovec's rayon parallelism is over
   *queries* — `(0..nq).into_par_iter()` for the LUT build (search.rs:1550) and
   `(0..nq).step_by(QBS)` for the scoring kernels (search.rs:1567, 1705, 1811).
   mempalace queries one text at a time, so `nq = 1`. Sharding plus a threaded
   fan-out makes one search M-way parallel.

   Verified reachable: every `turbovecdb.Collection` method releases the GIL via
   `py.allow_threads` and takes a **per-collection** `Mutex`
   (crates/turbovecdb-py/src/collection.rs:10-21, 90-171). Distinct shards are
   distinct objects with distinct mutexes, so a `ThreadPoolExecutor` fan-out gets
   real parallelism rather than GIL-serialized work.

2. **Write amplification dominates query cost.** `Collection::ensure_current()`
   (crates/turbovecdb-core/src/collection.rs:496) sees `store_gen != seen_gen`
   and rebuilds the *whole* index from SQLite, re-quantizing all N vectors. One
   drawer added to any wing forces every long-lived reader to re-encode the
   entire corpus. mempalace's real deployment is a long-lived MCP reader while
   `mempalace mine` writes wing by wing — this is the steady state. Sharding
   confines it to the written wing.

Clustering also maintains a *decaying* invariant: `IdMapIndex::remove` is a
**swap-remove** (id_map.rs:161) that moves the last slot into the hole, and
turbovecdb calls it on every delete and upsert-replace
(`mirror_write_to_index`, collection.rs:610). One delete teleports a tail
document into an arbitrary block, and nothing surfaces when the layout drifts.
Shards cannot decay.

## Where the work lands

| Piece | Repo | Risk |
|---|---|---|
| Public single-collection handle eviction | `turbovecdb` | low |
| Router + virtual sharded collection | `mempalace` (`backends/turbovec.py`) | medium — correctness surface |
| Migration script | `mempalace` | low, reversible |
| Per-dim rotation-matrix cache | upstream `turbovec` | optional, not blocking |
| Intra-collection striping | `turbovecdb` | optional — *alternative* to Phase 4 threading |

The mempalace side is confined to `mempalace/backends/turbovec.py`. No call site
changes: mempalace already passes the wing down as a `where` filter everywhere,
via `searcher.build_where_filter` (searcher.py:224) and
`_build_where_filter_multi` (searcher.py:244).

---

## Phase 0 — Measure before committing

Three numbers, none of which require any of the code below.

1. **Wing-filter ratio.** What fraction of real `mempalace_search` /
   `search_hierarchical` calls arrive with a wing, or with a closet-pruned wing
   set? A few lines in `searcher.py`. This sets whether the routed path or the
   fan-out path is the one to optimize.
2. **Shard-size distribution.** Drawers per wing on a real palace. Sets **M** and
   the `__small__` folding threshold. Read straight from the existing collection:
   `get(include=["metadatas"])`, group by `wing`.
3. **Baseline latency.** p50/p95 for a scoped and an unscoped query on the
   current single collection, so Phase 4 has something to be compared against.

**Exit:** M chosen, `__small__` threshold chosen, baseline recorded. If the
wing-filter ratio is low, stop and reconsider — the whole design assumes most
queries carry a wing.

## Phase 1 — turbovecdb: bounded shard handles

`Database._collections` is an **unbounded dict** (database.py:37, 57-87) with no
cap and no LRU. Sharding multiplies handles, each holding a SQLite connection, a
resident `.tvim`, a rotation matrix (dim² f32), and a blocked-codes cache — so
the cache needs a ceiling.

`_evict_and_close(name)` (database.py:119) is already exactly the primitive: pop
GIL-atomically, then close. It is private and its docstring is delete-specific.

- Add a public `Database.close_collection(name)` wrapping it.
- Document the contract explicitly: **the caller must not retain the handle.**
  `Database.collection()` guarantees identity stability per name (the C6
  conflicting-options check depends on it), so eviction is only safe when the
  evictor is the sole holder.
- Tests: reopen-after-close yields a working handle; close of an unknown name is
  a no-op; close of a handle with pending writes flushes (or documents that it
  does not).

**Do not** put the LRU inside `turbovecdb.Database` — it hands handles out to
callers who keep them, so it cannot know when eviction is safe. The cap belongs
in the sharded collection, which owns its shard handles and never exposes them.

**Exit:** turbovecdb can drop one shard handle without touching data.

## Phase 2 — mempalace: the router (pure, no I/O)

Three pure functions in `backends/turbovec.py`, fully testable without a palace:

```python
def _shard_name(wing) -> str
def _route(where) -> tuple[list[str] | None, dict | None]   # None = all shards
def _shard_for_id(drawer_id, known_shards) -> str | None    # None = fan out
```

- **`_shard_name`** must be **injective**. mempalace's `_SAFE_NAME_RE`
  (config.py:24) admits `.`, `'`, and space; turbovecdb collection names allow
  only `[A-Za-z0-9_-]{1,128}` (crates/turbovecdb-core/src/database.rs:48), and
  the name becomes a directory. Normalize, then escape the residue. A lossy
  encoding silently merges two wings into one shard.
- **`_route`** must handle the shapes mempalace actually emits, and nothing more:
  `{"wing": x}`, `{"$and": [{"wing": x}, {"room": y}]}`, `{"wing": {"$in": [...]}}`,
  `{"source_file": ...}` with no wing, `{}`, `None`.
- **Drop the wing clause from the residual filter only on exact equality with
  the shard.** This is the point of the exercise. Leave it in and
  `query_allowlist()` (collection.rs:652) returns every uid in the shard,
  turbovec builds an all-ones mask, zero blocks are skipped, and you have paid
  three O(N) allocation passes for nothing.
- **`_shard_for_id`** exists because `delete(ids=...)` carries no `where`. Drawer
  ids are `drawer_{wing}_{room}_{hash24}` (ids.py:62), so strip the prefix and
  the trailing 24 hex and longest-prefix-match against the known shard set. Do
  **not** try to split wing from room: `normalize_wing_name` (config.py:38)
  collapses separators *to* `_`, so the split is ambiguous. Return `None` and let
  the caller fan out.

Tests: exhaustive over the filter shapes, plus a property test that
`residual_where ∧ (wing = shard)` is equivalent to the original `where` for every
routed case. That property is the correctness core of the whole design.

**Exit:** full branch coverage on `_route`; property test green.

## Phase 3 — mempalace: virtual collection, fan-out only

`ShardedTurboVecCollection` implementing `BaseCollection`, with **fan-out for
every operation** — correct and slow, no routing yet. This is the differential
baseline for Phase 4.

- `query` — every shard, `n_results` each (not divided), concat, sort by
  distance, truncate. Merging is **exact**, not approximate: turbovecdb re-ranks
  with true cosine from stored float32, so distances are comparable across
  shards. Raw quantized scores would not be (per-shard rotation and codebook).
- `add`/`upsert` — group the batch by `metadatas[i]["wing"]`, one call per shard.
  No wing → `__unwinged__`.
- `delete(ids=...)` — fan out. Safe: it compiles to
  `DELETE ... WHERE str_id IN (...)` (collection.rs), so a shard without the id
  is a silent no-op.
- `get` — with `limit`/`offset`, request `limit + offset` from each shard, merge,
  then slice. Easy to get subtly wrong; test against the unsharded collection.
- `count` — sum.

Differential test: same corpus, same queries, sharded vs unsharded, identical
ids and distances. Then the mempalace suite green with the sharded backend.

**Exit:** results provably identical to unsharded on a real palace.

## Phase 4 — mempalace: routing, threading, bounded handles

- Route when `_route` returns shards; fan out otherwise.
- Thread the fan-out (`ThreadPoolExecutor`, bounded). Justified by the
  `allow_threads` + per-collection-`Mutex` finding above. Size the pool against
  the MCP server's own concurrency so one query fanning across M shards does not
  oversubscribe while other requests are in flight.
- LRU over shard handles using Phase 1's `close_collection`, plus lazy open.
- `__small__` folding at the Phase 0 threshold. A 12-drawer wing returns 12
  candidates (`effective_k = k.min(n_vectors).min(n_allowed)`) while paying a
  full rotation matrix, SQLite file, lock, and block padding.

**Watch for the LRU cliff.** An unscoped search touches every shard, so nothing
is evictable. If M exceeds the cap it does not degrade — it *thrashes*: evict,
reload `.tvim`, re-QR, re-repack, per query. Assert `M <= cap` at open and fail
loudly, or fold aggressively enough that it cannot happen.

**Exit:** a routed query touches exactly one shard (assert via open-handle
telemetry); latency compared against the Phase 0 baseline for both scoped and
unscoped.

## Phase 5 — migration

No re-embed. SQLite holds exact float32, so:

1. `get(include=["embeddings", "documents", "metadatas"])` from the current
   collection.
2. Group by `metadata["wing"]`.
3. `add` into shards. Each `.tvim` builds itself on first open.

Run on a **copy** of a real palace. Diff top-10 against the unsharded collection
over a query corpus; expect identical-or-better (the merged pool is M×50 rather
than 50, so recall should improve slightly). Rollback is keeping the old
collection directory — nothing is destroyed.

Leave `mempalace_closets` **unsharded**. It is small, and it is the collection
the hierarchical prune reads to *produce* the wing list — which makes
`search_hierarchical` the best-case path: prune, then route to a few shards.

**Exit:** migrated palace matches or beats unsharded top-10 on the query corpus.

## Phase 6 — optional, parallel: upstream rotation-matrix cache

`make_rotation_matrix(dim)` (turbovec rotation.rs:15) is a dim×dim Gaussian plus
a QR, materializing dim² f32 — 590KB at 384-d, 2.36MB at 768-d, resident per
index. The seed is fixed (`ROTATION_SEED`, ChaCha8), so **every shard computes a
bit-identical matrix**: M redundant QRs on cold start and M copies in RAM.

**Not reachable from turbovecdb.** `rotation` is a private `OnceLock`
(turbovec lib.rs:146) and `from_parts` is `pub(crate)` (lib.rs:573). Options:

1. PR to `RyanCodrai/turbovec` — a process-global per-dim cache. Strictly less
   work, bit-identical results, benefits any multi-index user. Cleanest.
2. Vendor turbovec via `[patch.crates-io]`. Feasible (the build already
   statically links OpenBLAS) but adds maintenance.
3. Do nothing. **Defensible for v1:** a routed query touches one shard and pays
   one QR, exactly as today. Only unscoped fan-out pays M, and the first
   unscoped query after a restart is already behind `_lazy_embedder()`'s model
   load, which costs more.

Start at (3), open (1) as a follow-up. This is explicitly **not** on the critical
path.

---

## Option S — intra-collection striping

**An alternative to Phase 4's Python thread pool, not an addition to it.** Both
exist to make one query use many cores; running both nests two levels of fan-out
and oversubscribes. Pick one.

`Collection` holds a single `index: Option<I>` (`I: VectorIndex`). Striping holds
**S** of them, splits the corpus across them, and fans out with rayon *inside
Rust*. The parallel win then requires no partition key, no routing, and no
mempalace change at all — and it benefits every turbovecdb consumer, including
unsharded and legacy collections.

Positioning against Phase 4:

| | Threaded fan-out (Phase 4) | Striping (Option S) |
|---|---|---|
| Where | `mempalace` backend, ~20 lines | `turbovecdb-core` index lifecycle |
| Needs a partition key | yes | **no** |
| Helps unsharded collections | no | yes |
| Helps other turbovecdb consumers | no | yes |
| Cost | trivial | real work — `.tvim` layout, invalidation, config |

Striping does **not** replace Phases 1–5. Those exist for *write isolation* —
confining `ensure_current()`'s full-corpus rebuild to one wing — which striping
does nothing about, since all S stripes live in one collection behind one
`store_gen`. Striping only addresses read parallelism.

### Design

**Stripe assignment: `uid % S`.** Stable per uid, so `add`/`remove` routing is
O(1) with no lookup; self-balancing; never needs rebalancing. Contiguous ranges
would need N up front and would rebalance on every insert — don't.

**Merge by candidate union, not by score.** Verified: TQ+ calibration is computed
from each index's *own* first batch and then locked (turbovec encode.rs:141,
lib.rs:257, 291), so `tqplus_shift`/`tqplus_scale` differ per stripe and raw
scores are **not comparable across stripes**. Do not sort the merged scores.
Instead concatenate all `S × pool` candidate uids and hand the union to the
existing exact-cosine re-rank, which is the only arbiter that matters. This is
the same argument that makes cross-shard merging safe, and it makes the merge
trivially correct — there is no comparison to get wrong.

Consequence: candidate cost is `S × max(k, RERANK_FLOOR)` = `S × 50` SQLite rows
and cosines per query, same shape as shard fan-out. Recall goes *up* (more
candidates), latency floor goes up with S.

**S is config, stored in meta, and must invalidate the cache.** Same trap as slot
clustering below: `.tvim` validity is `tvim_gen == store_gen` plus a dim /
bit_width shape check (collection.rs:416) and nothing about stripe count. Change
S without a `layout_gen` in meta and stale stripe files validate as current.
Auto-default S from `available_parallelism()` with a floor (don't stripe a small
collection — per-stripe fixed costs dominate), but persist the resolved value.

### Work items

All in `crates/turbovecdb-core/src/collection.rs` unless noted.

1. `index: Option<I>` → `stripes: Vec<I>`, preserving a fast path at S = 1 so
   existing single-file collections load unchanged.
2. `.tvim` layout: `tvim_path` is a hardcoded `{coll_dir}/index.tvim`
   (collection.rs:179) with a `.tmp` sibling for atomic write (line 187). Becomes
   `index.{i}.tvim`. Keep the atomic-rename discipline per file.
3. `compute_collection_fingerprint` (collection.rs:435, 960) binds to the single
   `.tvim`'s bytes — must cover all S files, or the copy-detection property it
   was built for silently weakens.
4. Rebuild (collection.rs:456): one `SELECT ... ORDER BY uid` pass, dealt into S
   `add_with_ids` calls by `uid % S`.
5. Route the incremental paths: `mirror_write_to_index` (collection.rs:610) and
   `remove_from_index` per uid.
6. `query` / `query_batch` (collection.rs:1307, 1451): rayon `par_iter` over
   stripes, union the candidates. Safe with respect to the Python layer — the GIL
   is already released and the per-collection `Mutex` held for the whole call
   (crates/turbovecdb-py/src/collection.rs:10-21), so there's no contention to
   introduce. In `query_batch`, bound the outer(queries) × inner(stripes) product
   so it doesn't oversubscribe.
7. `health()` (collection.rs:1027): report S and per-stripe `len()`. Without this
   there is no way to see that striping is even active.
8. `layout_gen` in meta, participating in `.tvim` validity.

### Exit criteria

- S = 1 is byte-identical in behaviour to today, and loads pre-striping `.tvim`
  files unchanged.
- Differential test: same corpus, S ∈ {1, 4, 16}, identical query results after
  re-rank (candidate union means results should match or improve, never regress).
- Wall-clock for a single unscoped query improves with S until memory bandwidth
  saturates — see the prediction in
  [scan-vs-hnsw-scaling.md](scan-vs-hnsw-scaling.md). If it keeps improving past
  the predicted ceiling, that model is wrong and worth knowing.
- `health()` shows S and per-stripe counts.

### Risks

| Risk | Mitigation |
|---|---|
| Stale stripe `.tvim` files accepted after S changes | `layout_gen` in meta (work item 8) — do not ship without it |
| Score-sorted merge silently degrades recall | Union candidates; never compare across stripes |
| Nested fan-out with Phase 4 oversubscribes | Option S and Phase 4 threading are mutually exclusive; assert one is off |
| S too high on a small collection | Floor on N per stripe; auto-default with the floor applied |
| Fingerprint weakened by only covering stripe 0 | Work item 3; regression test with a copied stripe file |

### When to prefer this over Phase 4

Choose Option S if unscoped full-palace search latency is the binding complaint,
if you want the win available to turbovecdb consumers other than mempalace, or if
sharding stalls for any reason. Choose Phase 4's thread pool if sharding is
already landing and you want the parallel win for ~20 lines. **Phases 1–5 remain
the priority either way** — they buy write isolation, which neither threading
option touches.

---

## Deferred: slot clustering

Build the index in partition order rather than insertion order — at
collection.rs:456, `ORDER BY json_extract(metadata,'$.wing'), uid` instead of
`ORDER BY uid` — so filters become block-contiguous.

It composes with sharding and gets *cheaper* once sharding exists: inside a wing
shard, `ORDER BY room, source_file, uid` adds room-scoped block skipping for one
ORDER BY, with per-shard N an order of magnitude smaller so the swap-remove decay
problem shrinks with it.

If it is ever picked up, one trap is already identified: **`.tvim` validity does
not depend on slot order.** It is `tvim_gen == store_gen` (collection.rs:416)
plus a dim/bit_width shape check, and `compute_collection_fingerprint`
(collection.rs:960) hashes `uid, vector ORDER BY uid`. Change the build order and
every existing `.tvim` still validates as current and is never re-clustered — the
feature ships, measures zero, and nothing looks broken. The cluster key must
participate in the validity check (a `layout_gen` in meta).

## Risks

| Risk | Mitigation |
|---|---|
| Router returns incomplete results for an unhandled filter shape | Phase 2 property test; fan out on anything unrecognized rather than guessing |
| LRU thrash when M > handle cap | Assert `M <= cap` at open; aggressive `__small__` folding |
| Cross-shard `get(offset=)` pagination bug | Phase 3 differential test against unsharded |
| Lossy shard-name encoding merges two wings | Injectivity test over the full `_SAFE_NAME_RE` charset |
| Handle evicted while still in use | Shard handles never leave the virtual collection |
| Threaded fan-out oversubscribes the MCP server | Bounded pool sized against server concurrency |

## Out of scope

- Cross-palace sharding.
- Sharding `mempalace_closets`.
- Any change to embedding, BM25, or `_hybrid_rank`.
- Making turbovecdb itself shard-aware. Sharding is a mempalace policy decision;
  turbovecdb stays a single-collection library. The only turbovecdb change is
  Phase 1's handle eviction.

## Open questions

1. Shard granularity: wing only, or wing+room for the largest wings? Phase 0's
   size distribution answers this.
2. Does `__unwinged__` exist in practice, or do all drawers carry a wing?
3. Should the migration run in place or write to a new palace path and swap?
