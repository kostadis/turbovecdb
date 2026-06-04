# turbovecdb

An embedded vector database. [turbovec](https://github.com/RyanCodrai/turbovec)'s
4-bit TurboQuant ANN for fast approximate search, paired with a durable SQLite
sidecar that holds the documents, metadata, id map, and the exact float32
vectors. Metadata filters, persistence, exact-cosine re-rank, and multi-process
safety are built in.

It's the kind of thing you reach for when you want a local, CPU-resident vector
store with a small footprint and no server to run — and you're happy to bring
your own embedding model (or hand it one).

## Why

- **SQLite is the source of truth.** Every vector is stored exactly (float32) and
  durably. The turbovec `.tvim` index is a *rebuildable cache* — if it's missing
  or stale it's rebuilt from SQLite on open, so a crash never loses data.
- **Exact answers from an approximate index.** turbovec finds a candidate pool
  fast; turbovecdb re-ranks it with true cosine, so callers get a correct
  `distance ∈ [0, 2]`, not turbovec's raw quantized score.
- **Multi-process safe.** Writes take a cross-process file lock; readers detect
  another process's writes and reload. Run a writer and readers against the same
  database directory.
- **Small and quick.** In a local benchmark (15.8k docs, 384-d), turbovecdb built
  ~12× faster, queried ~3× faster (p50/p95), and used ~2.3× less disk than an
  HNSW-based store — at indistinguishable retrieval quality.

## Install

```bash
pip install turbovecdb        # pulls turbovec, numpy, filelock
```

Requires Python ≥ 3.9. The vector dimension must be a positive multiple of 8
(e.g. 384, 768) — a turbovec requirement.

## Integrate with Mempalace

Mempalace doesn't really support drop-in replacement of ChromaDB. 

What to do
1. Try out my personal branch (kostadis-dev)
2. Ask your favorite AI system to patch mempalce. To make that easier, I have asked Cursor to provide instructions that can be used by a human or an LLM to do the patch. 


[docs/MEMPALACE_TURBOVEC_MIGRATION_HANDOFF.md](docs/MEMPALACE_TURBOVEC_MIGRATION_HANDOFF.md) is a document that is human readable and appopriate for a coding agent to use to patch an instance of mempalace to use turbovecdb. 


## Usage

```python
import turbovecdb

db = turbovecdb.connect("/path/to/db")

# Bring your own vectors:
col = db.collection("docs", create=True)
col.add(
    ids=["a", "b"],
    documents=["the quick brown fox", "lorem ipsum"],
    metadatas=[{"lang": "en"}, {"lang": "la"}],
    vectors=[[...384 floats...], [...384 floats...]],
)
hits = col.query(vector=[...384 floats...], k=5, where={"lang": "en"})
print(hits.ids, hits.distances, hits.documents)

# ...or hand it an embedder (list[str] -> list[list[float]]):
col = db.collection("docs2", embedder=my_embed_fn, create=True)
col.add(ids=["a"], documents=["hello world"])
hits = col.query(text="a greeting", k=5)
```

### Filters

`where` supports `$eq` (bare scalar too), `$ne`, `$in`, `$nin`, `$gt`, `$gte`,
`$lt`, `$lte`, and `$and` / `$or` (recursive). `where_document` supports
`$contains`. Unsupported operators raise `UnsupportedFilterError` — filters
never silently fail.

```python
col.query(vector=v, k=10, where={"$and": [{"lang": "en"}, {"year": {"$gte": 2020}}]})
col.get(where={"lang": {"$in": ["en", "fr"]}}, where_document={"$contains": "fox"})
```

## API

`connect(path) -> Database` · `Database.collection(name, *, dim=None,
bit_width=4, metric="cosine", embedder=None, create=True) -> Collection`

`Collection`: `add` / `upsert` (`ids`, `documents`/`vectors`, `metadatas`),
`query` (`text`|`vector`, `k`, `where`, `where_document`, `include`), `get`,
`delete`, `count`, `flush`, `close`. Results are `QueryResult` / `GetResult`
dataclasses with flat lists.

## Documentation

- [docs/core/architecture.md](docs/core/architecture.md) — how it's built (two-tier
  store, read/write paths, exact re-rank)
- [docs/core/data-model.md](docs/core/data-model.md) — on-disk layout, SQLite
  schema, the `.tvim` cache, generation counters
- [docs/core/concurrency.md](docs/core/concurrency.md) — the multi-process model
- [docs/performance/](docs/performance/) — benchmark harness + measured results

## License

MIT.
