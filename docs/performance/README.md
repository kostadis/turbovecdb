# Performance

Two complementary measurements:

1. A **standalone, reproducible benchmark** ([`benchmark.py`](benchmark.py)) —
   synthetic vectors, no embedding model, turbovecdb vs ChromaDB vs exact kNN.
2. An **integration measurement** from turbovecdb's first customer (MemPalace) on
   ~15.8k real text documents — the document-heavy regime the synthetic test
   doesn't cover.

The honest summary up front: **turbovecdb builds much faster, queries comparably
-to-faster, and matches exact recall via its re-rank; on-disk size depends on the
workload** — smaller than ChromaDB when documents dominate, slightly larger on a
pure-vector workload. None of these are universal laws; reproduce on your data.

## 1. Standalone benchmark

```bash
pip install -e '.[dev]' chromadb     # chromadb optional; skipped if absent
python docs/performance/benchmark.py --n 20000 --dim 768 --queries 300
```

It generates clustered, L2-normalized vectors, feeds the **same** vectors to
every index, and measures build time, single-query latency (p50/p95/mean),
recall@k vs **exact cosine kNN**, and on-disk size. `recall@k` is set-overlap
with the exact top-k; turbovecdb's number reflects its exact re-rank, not the raw
quantized index. (The `noise` is deliberately high so intra-cluster points are
*distinct* neighbours — otherwise top-k is a tie-lottery and recall stops being a
fair metric for any index.)

### Result (20,000 × 768-d, 300 queries, k=10; WSL2 x86_64)

| index | build s | q p50 ms | q p95 ms | q mean ms | recall@10 | size MB |
|---|---|---|---|---|---|---|
| **turbovecdb** | **1.42** | **0.874** | **1.081** | 1.087 | **1.000** | 86.1 |
| chromadb | 7.00 | 1.149 | 1.350 | 1.165 | 0.673 | 72.5 |

> turbovecdb vs chromadb: **build 4.9× faster**, q p50 **1.3× faster**, q p95
> 1.2× faster, size **1.19× (turbovecdb larger here)**.

Reading it honestly:

- **Build** — turbovecdb is decisively faster (~5×); quantizing + a SQLite write
  beats building an HNSW graph.
- **Query** — comparable, modestly in turbovecdb's favour out of the box. Both are
  sub-millisecond-ish at this scale.
- **Recall** — turbovecdb returns the exact top-10 (1.000) because it over-fetches
  a pool of `max(k, 50)` and re-ranks with exact cosine. ChromaDB's 0.673 is its
  **default** HNSW `search_ef`; raising `ef` trades latency for recall. So this is
  a default-vs-default comparison, and turbovecdb's over-fetch+re-rank is part of
  why its default recall is higher.
- **Size** — here turbovecdb is **larger**. With no documents, the exact float32
  vectors (20k×768×4 ≈ 59 MB) dominate turbovecdb's footprint, while ChromaDB's
  graph overhead is comparatively modest. This flips when documents are present
  (below).

## 2. Integration measurement (MemPalace, real text)

Measured end-to-end through MemPalace's backend adapter (so turbovecdb's query
includes its SQLite fetch + exact re-rank), on **15,805 real text drawers**,
local ONNX MiniLM (384-d), same vectors fed to both backends:

| backend | build s | q p50 ms | q p95 ms | size MB |
|---|---|---|---|---|
| **turbovecdb** | **0.80** | **0.59** | **0.76** | **62.0** |
| chromadb | 10.74 | 2.08 | 2.61 | 142.1 |

> ~**13× faster build**, ~**3.5× faster p50** / **3.4× p95**, **2.3× smaller on
> disk** — at indistinguishable retrieval quality (hybrid hit@k tied within
> noise).

Why the size result inverts vs the synthetic test: here every record carries
text. Both stores hold the documents and the float32 vectors; the differentiator
is the **index** — turbovecdb's 4-bit `.tvim` (a few MB) vs ChromaDB's HNSW graph
(`link_lists.bin`, ~100 MB). When documents are in the mix, the graph overhead is
what makes ChromaDB larger.

## What to take away

- **Build speed** is a consistent, large turbovecdb win.
- **Query latency** is comparable to several-× faster depending on workload and
  ChromaDB's `ef` setting.
- **Quality** matches exact kNN thanks to the re-rank; the raw quantized index
  alone would not.
- **Size** is workload-dependent — measure it on *your* data (document-heavy →
  turbovecdb tends smaller; pure vectors → comparable to slightly larger).

## Caveats

- One machine (WSL2 x86_64), synthetic vectors in (1); ratios shift with N, dim,
  hardware, and ChromaDB tuning.
- recall@k is a default-vs-default comparison; ChromaDB recall is tunable via
  `hnsw:search_ef` at a latency cost.
- These numbers are a snapshot; the point of shipping `benchmark.py` is that you
  can reproduce and adjust them.
