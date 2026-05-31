# Architecture

turbovecdb is a small embedded vector database. It combines two storage tiers:

- **[turbovec](https://github.com/RyanCodrai/turbovec)** — a CPU ANN index that
  maps `uint64` ids to 4-bit-quantized vectors and answers approximate
  nearest-neighbour queries quickly.
- **SQLite** — a durable sidecar holding the documents, metadata, the
  string-id ↔ `uint64` map, and the *exact* float32 vectors.

The guiding principle:

> **SQLite is the source of truth; the turbovec index is a rebuildable cache.**

Everything else follows from that. The quantized index can be lossy, lost, or
stale and the database never loses data or returns wrong text — at worst it
rebuilds the index from SQLite.

```
          ┌─────────────────────────── Collection ───────────────────────────┐
          │                                                                    │
 add()    │   documents + metadata + EXACT float32 vectors                     │
 ───────► │   ┌──────────────────────────── store.sqlite3 ────────────────┐   │
          │   │  docs(uid, str_id, document, metadata, vector BLOB)        │   │  ← source of truth
          │   │  meta(dim, bit_width, metric, next_uid, store_gen, tvim_gen)│   │
          │   └────────────────────────────────────────────────────────────┘   │
          │                         │ build / rebuild                            │
          │                         ▼                                            │
 query()  │   ┌──────────── index.tvim (turbovec, 4-bit) ──────────────┐        │  ← rebuildable cache
 ───────► │   │  uint64 id → quantized vector ; approximate top-k       │        │
          │   └────────────────────────────────────────────────────────┘        │
          │                         │ candidate uids                              │
          │                         ▼                                            │
          │   exact-cosine re-rank against SQLite float32 vectors  ──► QueryResult│
          └────────────────────────────────────────────────────────────────────┘
```

Source: `src/turbovecdb/collection.py` (the engine), `database.py` (collection
factory), `filters.py`, `index.py` (turbovec lifecycle), `errors.py`.

## The write path — `add` / `upsert`

1. Resolve vectors: use the caller's `vectors=`, or embed `documents=` with the
   collection's `embedder` (the *batteries-included* path). Vectors are
   **L2-normalized** so cosine similarity is a plain dot product.
2. Take the cross-process write lock, then under it: refresh `meta` (so a second
   writer sees the first writer's committed rows), commit the dimension on first
   write (turbovec requires `dim % 8 == 0`), allocate a `uint64 uid` per new
   `str_id` from the persisted `next_uid`.
3. Write the row to SQLite (`document`, `metadata` JSON, `vector` BLOB) — this is
   the durable step — and mirror it into the in-memory turbovec index
   (`add_with_ids`, or `remove` + re-add for an upsert).
4. Bump `store_gen`, commit. The `.tvim` file is **not** rewritten per call (that
   would be O(N) every batch); it is flushed on `flush()` / `close()`.

## The read path — `query`

1. Cheap **staleness check**: read `store_gen`; if another process advanced it,
   reload the index (load `.tvim` if it is current, else rebuild from SQLite).
2. If a `where` / `where_document` filter is present, compile it to SQL over the
   metadata JSON, select the matching `uid`s, and hand them to turbovec as an
   `allowlist` (empty → empty result).
3. Ask turbovec for a **candidate pool** of `max(k, 50)` ids.
4. **Exact re-rank**: fetch those candidates' float32 vectors from SQLite,
   compute true cosine, set `distance = 1 - cosine ∈ [0, 2]`, sort, return top
   `k`.

### Why re-rank?

turbovec returns *unnormalized, approximate* scores. Two problems that the
re-rank solves at once:

- **Correct distances.** Callers (and hybrid rankers layered on top) usually
  expect a real cosine distance on a fixed scale. Recomputing from the exact
  stored vectors gives a true `[0, 2]` value, not a quantization artifact.
- **Recovered precision.** Re-ranking a candidate pool with exact math fixes the
  near-tie ordering errors 4-bit quantization introduces, so end-to-end quality
  matches an exact index even though the *index* is approximate.

The cost is one SQLite fetch of ~`max(k, 50)` rows per query — small, and still
far cheaper than the alternative graph indexes' query path in practice (see
[performance](../performance/README.md)).

## Layers

| Concern | Where | Notes |
|---|---|---|
| Public API | `__init__.py`, `database.py` | `connect()`, `Database.collection()` |
| Engine | `collection.py` | the read/write paths above; `QueryResult`/`GetResult` |
| Index lifecycle | `index.py` | build / load / atomic write + L2 normalize |
| Filters | `filters.py` | filter dict → SQL (see [data-model](data-model.md)) |
| Errors | `errors.py` | `UnsupportedFilter`, `DimensionMismatch`, … |

## What turbovecdb is *not*

- **Not a server.** It's an embedded library — one process (or several) opening a
  directory. No daemon, no network protocol.
- **Not an embedder.** It stores and searches vectors. It will *call* an embedder
  you give it, but it ships none.
- **Not a graph index.** It uses turbovec's quantized flat index plus exact
  re-rank, not HNSW. That trades a different set of properties — see
  [performance](../performance/README.md) and [concurrency](concurrency.md).

## Design constraints worth knowing

- **`dim % 8 == 0`.** A turbovec requirement (768, 384, … are fine). Violations
  raise `DimensionMismatchError`.
- **`bit_width ∈ {2, 3, 4}`**, default `4` (the recall ceiling).
- **`metric="cosine"`** only, today. Vectors are normalized on the way in.
- **Single embedding model per collection.** Changing models changes the vector
  space; mixing them silently corrupts results. (A stored-model-name guard is on
  the backlog.)
