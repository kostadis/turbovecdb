# turbovecdb docs

- **[core/architecture.md](core/architecture.md)** — how turbovecdb is put
  together: the two-tier store, the read/write paths, exact-cosine re-rank.
- **[core/data-model.md](core/data-model.md)** — on-disk layout, SQLite schema,
  the `.tvim` cache, the id map, and the generation counters.
- **[core/concurrency.md](core/concurrency.md)** — the multi-process model:
  the cross-process write lock and lock-free reader cache coherence.
- **[performance/README.md](performance/README.md)** — benchmark methodology and
  measured results (build time, query latency, recall, on-disk size) vs ChromaDB
  and exact kNN, with a reproducible harness.
