# Changelog

All notable changes to turbovecdb are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Upgraded the turbovec quantization engine from 0.9 to 1.0.**
  - *On-disk format:* turbovec 1.0 reads and writes only format v7. A `.tvim`
    written by 0.9 is v3, which predates the v5 rotation change and cannot be
    decoded by any current build. Such a cache is now rejected on load and the
    index is **rebuilt from the SQLite vectors**, which have always been the
    source of truth — so no data is lost, but the first open of a
    pre-upgrade collection pays a one-time rebuild and logs why.
  - *No more BLAS.* turbovec 1.0 drops `faer` and `ndarray`; its
    block-Hadamard rotation needs no matrix multiply. The `openblas-src`
    dependency added in #121 is gone, removing the vendored static OpenBLAS
    compile from a clean build. This also collapses the two `ndarray`
    majors that used to coexist in the dependency graph.
  - *Fallible search.* An empty allowlist or an allowlist id absent from the
    index used to `panic!` out of turbovec 0.9; 1.0 reports them as errors,
    and the internal `VectorIndex::search` is now fallible to match.
  - *Dimension ceiling.* turbovec lowered `MAX_DIM` from 65536 to 16384; an
    oversized dim now raises `DimensionMismatchError` naming the limit.
  - Requires a Rust toolchain of 1.89 or newer to build.

- **All locking moved from the Python wrapper into the Rust core.**
  `_core.Collection` now owns both the in-process lock (a `Mutex` around the
  core, acquired inside `allow_threads` so a thread never blocks on it while
  holding the GIL) and the cross-process write lock (an `flock` on the same
  sibling `<root>/<name>.lock`, held for every write, for `delete_collection`'s
  rmtree, and for a brand-new collection's first-creation meta init).
  `delete_collection`'s lock-and-remove moved into the Rust `Database`. The
  Python `src/turbovecdb/*.py` is now a lock-free shim. Observable behavior is
  unchanged: same lock file path, same timeout semantics, same exception types
  with byte-identical messages, same crash-safety.
- **Embedding stays outside the write lock.** `add`/`upsert` embed (and run the
  identity pre-check) before acquiring the flock, then re-check the embedder
  identity under the lock — a slow embedder no longer blocks other processes'
  writers.

### Removed

- **`filelock` is no longer a runtime dependency** — it moved to the test/dev
  group. The cross-process lock is now the Rust core's `flock`; the test suite
  keeps using Python `filelock` as the *opposing* lock holder to prove the two
  implementations exclude each other (old-wheel/new-wheel interop).

### Notes

- On Unix the core uses the same `flock(2)` primitive on the same path as
  Python `filelock`, so a process on the old wheel and one on the new wheel
  still exclude each other. On Windows this interop is lost (Python `filelock`
  uses `msvcrt.locking`, a different primitive); CI is Linux/WSL.
- **Two behavioral exceptions to "unchanged," both from embedding now running
  under the in-process `Mutex` (previously it ran in Python outside the lock):**
  (1) an embedder — or a `reembed` `on_progress` callback — that *re-enters the
  same collection* (e.g. calls `count()`) now deadlocks the non-reentrant
  `Mutex`; embedders are expected to map text→vectors without touching the
  collection. (2) A slow `documents=` embed now holds the in-process `Mutex`
  for its whole duration, so other *in-process* reads/writes on that collection
  wait for it (they could overlap it before). The cross-process guarantee is
  unchanged and better — the embed still runs *outside* the cross-process
  `flock`, so it never blocks writers in *other* processes (R3). Multi-threaded
  same-process callers of a slow embedder (e.g. the HTTP service) may want to
  keep `service.py`'s per-db lock or shard collections; a follow-up may revisit
  in-process embed concurrency.

## [0.5.0] - 2026-07-03

The engine moved from a pure-Python implementation to a native Rust core. The
public Python API (`connect`, collections, query/get/add/upsert/delete,
metadata filters, re-embed) is unchanged; the implementation underneath it was
rewritten and hardened.

### Changed

- **Rust core rewrite.** The storage engine, index lifecycle, filter compiler,
  and read/write paths now live in a Cargo workspace — `turbovecdb-core` (the
  pure engine) and `turbovecdb-py` (the PyO3 adapter compiled into
  `turbovecdb._core`). `src/turbovecdb/*.py` is now a thin wrapper that owns the
  process locks and the public dataclasses. The two-tier store, invariants,
  generations, and locking carried over 1:1 from the v0.2.0 design.
- **Native turbovec engine, no wheel dependency.** The turbovec 4-bit ANN index
  is a statically linked Rust crate (OpenBLAS bundled). The `turbovec` PyPI
  wheel is no longer a runtime dependency.
- Writes and reads release the GIL; embedding now happens before the write lock
  is acquired.

### Added

- HTTP service layer (`turbovecdb-service`) over the embedded database.
- SIGKILL-mid-write crash-injection test harness.
- Rust (`cargo test`) coverage for `turbovecdb-core`, plus performance and scale
  tests across the full API.

### Fixed

Correctness and durability hardening (PR #45):

- Corrupt `meta` values now raise instead of silently defaulting.
- Collection first-creation and cached-handle option conflicts are serialized
  under the write lock; conflicting options on a cached handle now raise.
- `query()` reads run inside a single deferred transaction; a failed `COMMIT`
  rolls back and invalidates the index.
- `.tvim` index is fsync'd before rename; loaded index shape and vector-blob
  length are validated before use.
- `next_uid` stays monotonic across `clear()`.
- `write.lock` moved outside the collection directory; `close()` takes the write
  lock and stamps `tvim_gen` from `seen_gen`.
- The embedder-identity guard is applied to text queries.

### Notes

- filelock/WAL correctness requires a local filesystem (documented).

## [0.2.0]

Pure-Python implementation: two-tier SQLite + turbovec store, metadata filters,
persistence, exact-cosine re-rank, multi-process safety, and an HTTP service
layer.
