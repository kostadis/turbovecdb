# Why a Parallel Flat Scan Can Keep Pace With HNSW

Analysis, 2026-07-24. Companion to
[partitioned-search-plan.md](partitioned-search-plan.md) — that document says
*what to build*; this one says *why the approach has headroom* and *where it runs
out*.

> **Naming.** `turbovec` and `turbovecdb` are two different codebases and this
> document refers to both. **`turbovec`** is the upstream ANN engine crate
> (`RyanCodrai/turbovec`, pinned at 0.9.0) — SIMD kernels, rayon, the rotation
> matrix, `IdMapIndex`; **not ours**, changes there mean a PR or a vendored fork.
> **`turbovecdb`** is this repo, which depends on it and adds SQLite, filters,
> locking, and the exact-cosine re-rank. Our only view of the engine is the
> `VectorIndex` trait (crates/turbovecdb-core/src/index.rs:19) — `add_with_ids`,
> `remove`, `search`, `write`. Anything inside the kernel is out of reach from
> here. Where this document says "upstream", it means `turbovec`.

> **Epistemic status.** Three tiers, kept separate deliberately:
> - **Verified against source** — the code citations below (turbovec 0.9.0,
>   turbovecdb `main`). Reliable.
> - **Measured** — the 15.8k-doc benchmark (build 12.4× faster, query p50 3.3×
>   faster, disk 2.3× smaller vs ChromaDB/HNSW; hybrid retrieval tied on
>   quality). One corpus, one embedder, one machine.
> - **Arithmetic** — the crossover table. Derived from byte counts and published
>   bandwidth specs, **not measured**. Treat as an order-of-magnitude prediction
>   to be tested, not a result. The experiment that would settle it is specified
>   at the end and **has not been run**.

## The core asymmetry

turbovec is a flat quantized scan: `O(N)` work per query, streaming `dim/2` bytes
per vector through a LUT-based SIMD kernel. HNSW is `O(log N)` work. On paper the
scan loses at any interesting N.

What the paper comparison misses is that **HNSW cannot use multiple cores for a
single query either.** Its search is a dependent chain of pointer chases — each
hop's target depends on the previous hop's distance comparisons. There is modest
intra-query parallelism in scoring one node's neighbour list, but the hop
*sequence* is serial and latency-bound on random DRAM access. hnswlib and FAISS
both parallelise across queries, never within one.

turbovec today is also single-core-per-query, for a different reason: its rayon
parallelism is over *queries* — `(0..nq).into_par_iter()` for the LUT build
(turbovec search.rs:1550) and `(0..nq).step_by(QBS)` for the scoring kernels
(search.rs:1567, 1705, 1811). mempalace issues one query text at a time, so
`nq = 1` and exactly one core works.

The asymmetry is that **a flat scan is trivially decomposable and an HNSW
traversal is not.** Partition the corpus, search the shards concurrently, merge —
and one query now uses P cores. There is no equivalent move for a graph.

Verified reachable in this stack: every `turbovecdb.Collection` method releases
the GIL via `py.allow_threads` and takes a **per-collection** `Mutex`
(crates/turbovecdb-py/src/collection.rs:10-21, 90-171). Distinct shards are
distinct objects with distinct mutexes, so a Python `ThreadPoolExecutor` fan-out
yields real parallelism rather than GIL-serialised work.

## Parallelism converts work into latency — it does not reduce work

Per query the scan does `W = N × dim/2` bytes; HNSW does
`w = O(log N × dim × 4)`. With P cores you can run one query P× faster **or** P
queries at 1× — either way system throughput is `P/W`. The `W/w` gap is untouched.

**You cannot close a work gap with cores. You can only hide it as latency, and
only while cores are otherwise idle.**

This makes the conclusion workload-dependent, and the distinction is the whole
ballgame:

- **Single-user, one query at a time, cores idle** (mempalace's local MCP server):
  latency is the only metric, throughput is free, and the trade is pure profit.
- **Multi-tenant server under concurrent load**: there are no idle cores to
  donate, HNSW's work advantage reasserts immediately, and this entire line of
  reasoning collapses.

Do not carry this conclusion into a server context. It is a claim about an idle
machine.

## The ceiling is memory bandwidth, not core count

**Arithmetic, not measurement.** At 768-d (nomic), 4-bit codes are 384
bytes/vector. Bandwidth figures: DGX Spark / GB10 unified LPDDR5X ≈ 273 GB/s
(published spec, not measured here); a dual-channel DDR laptop ≈ 50 GB/s
(estimate).

| N | codes streamed per query | regime |
|---|---|---|
| 16k | 6 MB | L3-resident; cores scale near-linearly, and it is sub-ms regardless |
| 100k | 38 MB | around the L3 boundary |
| 1M | 384 MB | DRAM streaming; ≈1.4 ms if bandwidth is fully saturated at 273 GB/s |
| 10M | 3.8 GB | ≈14 ms even fully saturated |

Typical HNSW p50 at 1M/768-d is roughly 0.5–2 ms single-threaded. So the
prediction is that a **bandwidth-saturating parallel scan is genuinely
competitive at ~1M and loses outright by ~10M**, where the work gap wins
regardless of core count.

Two consequences worth internalising:

- **Cores stop helping before they run out.** Scaling is
  `O(N / min(P, bandwidth_limit))`, not `O(N/P)`. On the laptop that limit
  arrives around a dozen cores' worth of scan throughput; on the Spark there is
  substantially more headroom.
- **Small N parallelises *better* than large N**, which inverts the naive
  intuition — cache-resident codes are not bandwidth-limited. It is also
  irrelevant, because at small N everything is already faster than the embedding
  call.

The Spark is an unusually good target for this specific shape of work: many
cores, high-bandwidth unified memory, and turbovec already ships an aarch64 NEON
kernel with 4-query fused scoring. (Note that the NEON path also parallelises
over queries, `QBS = 4`, so `nq = 1` is single-core there too — the sharding
argument applies identically on ARM.)

## Two effects that favour the scan more than complexity classes suggest

**Sequential 4-bit versus random float32.** The scan streams `dim/2` bytes
sequentially; HNSW touches `ef × dim × 4` bytes randomly. That is roughly an 8×
compression advantage *plus* a sequential-versus-random access-pattern advantage,
and it is why the empirical crossover sits far further out than a flops count
predicts. turbovec's real weapon is not the scan — it is that the scan is over an
8×-compressed representation with perfect locality.

**Recall is better-behaved, not merely comparable.** An exhaustive scan over
quantised codes scores *every* vector, so recall loss is purely quantisation
error: uniform, predictable, no unreachable-node pathology, no degradation on
clustered or adversarial data. HNSW recall depends on graph connectivity and `ef`
and carries a long tail. turbovecdb then re-ranks the candidate pool with exact
cosine from stored float32, recovering most of the quantisation loss — the
mechanism behind the measured result that hybrid retrieval tied ChromaDB
end-to-end despite a 3–4 pt pure-vector recall gap.

## Where this stops working

At ~10M+ unscoped, the scan loses and no amount of parallelism helps. Getting
sublinear behaviour *without* a partition key requires an actual coarse
quantiser — IVF-style centroid pruning, i.e. scan a fraction of the cells rather
than all of them. That is a different feature, not a tuning of this one.

## Implication for the design

**Fan-out parallelism buys a constant factor, capped by memory bandwidth.
Routing buys an asymptotic reduction** — it shrinks N itself rather than
distributing it.

This is an independent argument for the priority ordering already chosen in
[partitioned-search-plan.md](partitioned-search-plan.md): the routed path is the
durable win because it changes *what you touch*; threaded fan-out is a
bandwidth-capped constant-factor bonus for the unscoped case. Unscoped search at
very large N is the one regime the partitioning design does not address at all.

## The experiment that would settle it — NOT YET RUN

The crossover table above is arithmetic and should not be cited as a result.
Extending the existing benchmark (the one that produced the 15.8k figures) would
test it directly:

- **Sweep** N ∈ {16k, 100k, 1M} × shard count M ∈ {1, 4, 16} × fan-out threads
  ∈ {1, 4, 16}, against an HNSW baseline on identical vectors.
- **Measure** query p50/p95, and separately the *first* query after process start
  (that one pays M rotation-matrix QRs plus M `.tvim` loads plus M block
  repacks — see Phase 6 of the plan).
- **Check the bandwidth prediction** by confirming that wall-clock stops
  improving once `M × per-shard-stream-rate` approaches the machine's measured
  bandwidth. If cores keep helping past that point, the model is wrong.
- **Report end-to-end hit@k, not index recall@k** — per the standing rule that
  index-level recall overstates quantisation cost for hybrid retrieval. Use
  natural-language queries, the regime where both modalities actually contribute.
- **Include the embedding call** in at least one end-to-end column. If query
  embedding dominates every configuration, most of this analysis is
  academic for the current corpus size, and that is a useful thing to know
  early.

Not started. Whether to build the harness is an open question, not a decision.

## What would change the conclusion

- Concurrent query load (multi-tenant, batch reindexing) — the idle-cores
  premise fails and HNSW's work advantage returns.
- `nq > 1` becoming the common case — turbovec already parallelises over queries,
  so the sharding argument weakens considerably.
- Upstream turbovec parallelising *within* a query (over blocks) — that would
  deliver most of the multi-core win with no sharding at all, and would make
  routing purely a write-isolation play.
- Corpus growth past ~10M unscoped — needs IVF-style coarse quantisation, not
  partitioning.
