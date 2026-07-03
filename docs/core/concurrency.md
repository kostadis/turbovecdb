# Concurrency

turbovecdb is built to be opened by **several processes at once** against the
same database directory — the common shape being one writer (an ingest job) and
one or more readers (a server, a CLI), all sharing a collection.

Two mechanisms make that safe:

1. a **cross-process write lock** that serializes writers, and
2. **lock-free reader cache coherence** so a long-lived reader notices another
   process's writes and refreshes.

Both lean on the same invariant as the rest of the system: **SQLite is the
source of truth; the in-memory turbovec index is a cache keyed by `store_gen`.**

## Writers — serialized by a file lock

Every `add` / `upsert` / `delete` (and `flush` / `close`) takes a
[`filelock`](https://py-filelock.readthedocs.io/) lock on
`<db_path>/<collection_name>.lock` (cross-platform: Linux, macOS, Windows).
The lock file is deliberately a *sibling* of the collection directory, not a
file inside it: `delete_collection` holds this lock while removing the
collection directory, and a lock file living inside that directory would let
a concurrent opener recreate a *new* lock file at the same path mid-delete —
two processes then both believe they hold the collection's write lock, one of
them writing into a directory being torn down. Under the lock a writer:

1. re-reads `meta` (`next_uid`, `dim`, `store_gen`);
2. if `store_gen` advanced past what this handle last applied, **reloads the
   index first** — so writer *B* sees writer *A*'s committed rows before adding
   its own;
3. allocates `uid`s from the persisted `next_uid` (never an in-memory guess),
   writes rows + mirrors them into the index;
4. bumps `store_gen` and commits.

Because all SQLite writes happen under this one lock, there is exactly one writer
to the database at a time — no lost updates, no `uid` collisions, no
`SQLITE_BUSY` from competing writers. The test
`tests/test_concurrency.py::test_concurrent_writers_no_loss_or_collision` spawns
two writer processes and asserts the final `count()` equals the sum and every id
is present.

Within a single process a `threading.RLock` guards the connection and index, and
the per-collection `FileLock` is re-entrant, so multi-threaded callers are safe
too.

## Readers — lock-free, coherent via `store_gen`

Reads take **no file lock** (SQLite WAL already lets readers run concurrently
with the single writer). Instead, every `query` / `get` first does a cheap
`SELECT store_gen` and compares it to the value this handle last saw:

```
def _ensure_current():
    if store_gen_in_sqlite != self._seen_gen:
        reload_index()        # load index.tvim if tvim_gen == store_gen,
                              # else rebuild from docs.vector
        self._seen_gen = store_gen_in_sqlite
```

So a reader that has been open for hours will, on its next query after another
process writes, transparently pick up the new rows. The cost is paid only when
something actually changed:

- if the writer flushed (`tvim_gen == store_gen`), the reader **loads** the
  `.tvim` (fast);
- otherwise it **rebuilds** from SQLite — sub-second for tens of thousands of
  rows, and the source of truth is always complete.

`tests/test_concurrency.py::test_reader_sees_other_process_writes` opens a
collection, has a *separate process* add rows, and asserts the live reader's next
query returns them.

## Why this design (vs. a shared mutable index)

A graph index like HNSW is mutated in place and is delicate under concurrent
access — which is why systems built on it lean on single-writer daemons, file
locks around the segment files, and corruption-recovery code. turbovecdb sidesteps
that: the durable state is plain SQLite rows, and the ANN index is a derived
artifact each process holds **in its own memory**, reconciled through a single
integer counter. There is no shared mutable index to corrupt; a crashed process
leaves only committed SQLite rows behind, and the next open rebuilds.

## Guarantees and limits

**Guarantees**
- Writers never lose or collide on writes (serialized by the file lock).
- A committed write is durable in SQLite before `add`/`delete` returns.
- Readers eventually (on their next query) observe any committed write.
- A crash never corrupts the database; at worst the `.tvim` is stale and rebuilt.

**Limits / current scope**
- Reader refresh granularity is per-query: a reader observes a writer's changes
  at its next `query`/`get`, not mid-call.
- Refresh does a **full** index reload when stale; an *incremental* reload (apply
  only the changed rows) is on the backlog — relevant if writes are very frequent
  against a very large collection.
- The write lock is **per collection** (`<name>.lock` lives in the database
  root, alongside the collection directory it guards). Different collections
  write independently.
- `busy_timeout` is set as insurance, but correct use shouldn't rely on it since
  writers are already serialized by the file lock.
