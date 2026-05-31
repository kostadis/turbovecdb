#!/usr/bin/env python3
"""Standalone benchmark: turbovecdb vs ChromaDB vs exact kNN.

Self-contained — no embedding model, no external corpus. It generates synthetic
*clustered* vectors (so nearest-neighbours are meaningful), feeds the SAME
vectors to every index, and measures, on this machine:

  * build time          — time to index all vectors
  * query latency        — p50 / p95 / mean over many single-vector queries
  * recall@k            — overlap of each index's top-k with the EXACT cosine
                          top-k (the answer key); turbovecdb's number reflects
                          its exact re-rank, not the raw quantized index
  * on-disk size         — bytes the index persists

ChromaDB is optional: if it isn't importable the comparison row is skipped and
turbovecdb is still measured against exact kNN.

Usage:
    python docs/performance/benchmark.py --n 20000 --dim 768 --queries 300
"""

import argparse
import os
import shutil
import statistics
import tempfile
import time

import numpy as np

import turbovecdb


def make_vectors(n, dim, clusters, seed, noise=0.9):
    """Clustered, L2-normalized float32 vectors.

    ``noise`` is the per-coordinate stddev added around a cluster centre,
    relative to the centre's own unit-Gaussian scale. It must be large enough
    that intra-cluster points are *distinct* neighbours, not near-duplicates —
    otherwise the exact top-k is a tie-lottery and recall@k stops being a fair
    fidelity metric for any index.
    """
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((clusters, dim)).astype(np.float32)
    assign = rng.integers(0, clusters, size=n)
    v = centers[assign] + noise * rng.standard_normal((n, dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return np.ascontiguousarray(v, dtype=np.float32)


def dir_size_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / (1024 * 1024)


def exact_topk(C, Q, k):
    """Brute-force cosine top-k (vectors are normalized → dot product)."""
    out = []
    for q in Q:
        sims = C @ q
        part = np.argpartition(-sims, k)[:k]
        out.append(set(int(i) for i in part[np.argsort(-sims[part])]))
    return out


def recall_at_k(approx_ids, exact_sets, k):
    tot = 0.0
    for got, gold in zip(approx_ids, exact_sets):
        tot += len(set(got[:k]) & gold) / k
    return tot / len(exact_sets)


def percentiles(times_ms):
    s = sorted(times_ms)
    p50 = statistics.median(s)
    p95 = s[min(len(s) - 1, int(0.95 * len(s)))]
    return p50, p95, statistics.mean(s)


def bench_turbovecdb(tmp, C, Q, k, ids):
    path = os.path.join(tmp, "turbovecdb")
    db = turbovecdb.connect(path)
    col = db.collection("bench", dim=C.shape[1], create=True)
    t0 = time.perf_counter()
    B = 2000
    for i in range(0, len(C), B):
        sl = slice(i, i + B)
        col.add(ids=ids[sl], vectors=C[sl])
    build_s = time.perf_counter() - t0
    col.flush()

    got, lat = [], []
    for q in Q:
        t = time.perf_counter()
        res = col.query(vector=q, k=k, include=[])  # ids only → minimal work
        lat.append((time.perf_counter() - t) * 1000)
        got.append([int(x) for x in res.ids])
    db.close()
    return build_s, got, lat, dir_size_mb(path)


def bench_chroma(tmp, C, Q, k, ids):
    import chromadb

    path = os.path.join(tmp, "chroma")
    client = chromadb.PersistentClient(path=path)
    col = client.create_collection("bench", metadata={"hnsw:space": "cosine"})
    t0 = time.perf_counter()
    B = 2000
    for i in range(0, len(C), B):
        sl = slice(i, i + B)
        col.add(ids=ids[sl], embeddings=C[sl].tolist())
    build_s = time.perf_counter() - t0

    got, lat = [], []
    for q in Q:
        t = time.perf_counter()
        res = col.query(query_embeddings=[q.tolist()], n_results=k,
                        include=[])  # ids only
        lat.append((time.perf_counter() - t) * 1000)
        got.append([int(x) for x in res["ids"][0]])
    return build_s, got, lat, dir_size_mb(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--clusters", type=int, default=200)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.dim % 8 != 0:
        ap.error("dim must be a multiple of 8 (turbovec requirement)")

    print(f"n={args.n} dim={args.dim} queries={args.queries} clusters={args.clusters} "
          f"k={args.k}\n(synthetic clustered vectors; identical vectors to every index)\n")

    C = make_vectors(args.n, args.dim, args.clusters, args.seed)
    Q = make_vectors(args.queries, args.dim, args.clusters, args.seed + 1)
    ids = [str(i) for i in range(args.n)]

    print("computing exact kNN ground truth...")
    exact = exact_topk(C, Q, args.k)

    rows = []
    tmp = tempfile.mkdtemp(prefix="tvdb-bench-")
    try:
        print("benchmarking turbovecdb...")
        b, got, lat, size = bench_turbovecdb(tmp, C, Q, args.k, ids)
        p50, p95, mean = percentiles(lat)
        rows.append(("turbovecdb", b, p50, p95, mean, recall_at_k(got, exact, args.k), size))

        try:
            print("benchmarking chromadb...")
            b, got, lat, size = bench_chroma(tmp, C, Q, args.k, ids)
            p50, p95, mean = percentiles(lat)
            rows.append(("chromadb", b, p50, p95, mean, recall_at_k(got, exact, args.k), size))
        except ImportError:
            print("  chromadb not installed — skipping comparison row\n")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 78)
    print(f"{'index':<12}{'build s':>9}{'q p50 ms':>10}{'q p95 ms':>10}"
          f"{'q mean ms':>10}{'recall@'+str(args.k):>11}{'size MB':>10}")
    print("-" * 78)
    for name, b, p50, p95, mean, rec, size in rows:
        print(f"{name:<12}{b:>9.2f}{p50:>10.3f}{p95:>10.3f}{mean:>10.3f}{rec:>11.3f}{size:>10.1f}")
    print("=" * 78)
    if len(rows) == 2:
        t, c = rows[0], rows[1]
        print(f"\nturbovecdb vs chromadb: build {c[1]/t[1]:.1f}x faster, "
              f"q p50 {c[2]/t[2]:.1f}x faster, q p95 {c[3]/t[3]:.1f}x faster, "
              f"size {t[6]/c[6]:.2f}x (<1 = smaller).")
    print(f"\nrecall@{args.k} is vs exact cosine kNN on identical vectors; turbovecdb's "
          f"number reflects its exact re-rank. q time = index query only.")


if __name__ == "__main__":
    main()
