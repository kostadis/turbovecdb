# Data model

## On-disk layout

A `Database` is a directory; each collection is a subdirectory under it:

```
<db_path>/
└── <collection_name>/
    ├── store.sqlite3     # durable source of truth (WAL mode)
    ├── store.sqlite3-wal # WAL sidecar (SQLite-managed)
    ├── store.sqlite3-shm # shared-memory index (SQLite-managed)
    ├── index.tvim        # turbovec index — rebuildable cache
    └── write.lock        # cross-process write lock (filelock)
```

`connect(path)` does no I/O; `Database.collection(name, create=...)` creates or
opens one subdirectory. `create=False` on a missing collection raises
`CollectionNotFoundError`.

## SQLite schema

```sql
CREATE TABLE docs (
    uid      INTEGER PRIMARY KEY,   -- turbovec external id (uint64)
    str_id   TEXT UNIQUE NOT NULL,  -- the caller's string id
    document TEXT,                  -- verbatim text (may be empty)
    metadata TEXT,                  -- JSON object
    vector   BLOB                   -- float32 bytes, L2-normalized
);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
```

- **`vector` is the exact copy.** `np.float32` bytes via `tobytes()` /
  `np.frombuffer`. This is what makes the `.tvim` disposable and what powers the
  exact-cosine re-rank.
- **`metadata` is JSON**, queried with SQLite's `json_extract` — no fixed columns,
  so arbitrary metadata keys work (see [filters](#filters)).
- **WAL mode** allows concurrent readers alongside a single writer; a
  `busy_timeout` smooths brief contention.

### `meta` keys

| key | meaning |
|---|---|
| `dim` | vector dimensionality (committed on first add; must be a multiple of 8) |
| `bit_width` | turbovec quantization bits (2/3/4) |
| `metric` | distance metric (`cosine`) |
| `next_uid` | monotonic allocator for new `uid`s |
| `store_gen` | bumped on **every committed write** |
| `tvim_gen` | the `store_gen` the on-disk `index.tvim` reflects |

## The id map

Callers use **string ids**; turbovec needs **`uint64`**. The `docs` table *is*
the bidirectional map (`str_id` ↔ `uid`). New ids draw a fresh `uid` from
`next_uid` (persisted, never an in-memory guess — important for multi-writer
correctness). An `upsert` of an existing `str_id` reuses its `uid` (turbovec
`remove` + re-add), so ids stay stable across updates.

## The `.tvim` cache and generation counters

`index.tvim` is turbovec's serialized index. It is a **cache**, kept consistent
with the source of truth via two counters:

- `store_gen` advances whenever SQLite is mutated and committed.
- `tvim_gen` records which `store_gen` the on-disk `.tvim` was written at.

On open (or on a reader's staleness check):

```
if index.tvim exists and tvim_gen == store_gen:
    load index.tvim            # fast path
else:
    rebuild from docs.vector   # ~sub-second for tens of thousands of rows
```

So a missing, partial, or stale `.tvim` is never a correctness problem — only a
cold-start cost. This is the crash-safety guarantee: a process can die any time;
SQLite holds every committed row, and the next open rebuilds the index.

`flush()` (and `close()`) write the `.tvim` and set `tvim_gen = store_gen`. They
take the write lock because that `tvim_gen` update is a SQLite write that must be
serialized with other processes.

## Filters

`where` (metadata) and `where_document` (text) compile to parameterized SQL over
the `docs` table; the resulting `uid` set is passed to turbovec as a search
`allowlist`. The JSON path and every operand are **bound parameters**, never
string-interpolated, so arbitrary field names and values can't inject SQL.

| form | example | SQL |
|---|---|---|
| bare equality | `{"lang": "en"}` | `json_extract(metadata,'$.lang') = ?` |
| `$eq`/`$ne` | `{"lang": {"$ne": "en"}}` | `... != ?` |
| `$gt`/`$gte`/`$lt`/`$lte` | `{"year": {"$gte": 2021}}` | `... >= ?` |
| `$in`/`$nin` | `{"lang": {"$in": ["en","fr"]}}` | `... IN (?, ?)` |
| `$and`/`$or` | `{"$and": [c1, c2]}` | recursive, `AND`/`OR` joined |
| `where_document $contains` | `{"$contains": "fox"}` | `document LIKE ? ESCAPE '\'` |

Any other operator raises `UnsupportedFilterError` — filters never silently
degrade to "match everything". See `src/turbovecdb/filters.py`.

## Result types

`query` → `QueryResult(ids, distances, documents, metadatas, vectors)`;
`get` → `GetResult(ids, documents, metadatas, vectors)`. Flat lists for a single
query. Fields not named in `include=` come back empty (`vectors` is `None` when
not requested), so you only pay to materialize what you ask for.
