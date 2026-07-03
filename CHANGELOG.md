# Changelog

All notable changes to turbovecdb are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
