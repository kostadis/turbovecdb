# Rust core/adapter split — design

**Status: done.** All 8 phases (#19-#26) landed on `feat/rust-core`; see the
[Phased migration](#phased-migration) section below for what shipped in each
commit. This doc is kept as the design record for the split, not a live plan.

Companion to `docs/rust-core-plan.md`. Tracks the fix for
[#18](https://github.com/kostadis/turbovecdb/issues/18): the Rust engine was
not a pure, standalone crate — it reached into Python for its own domain
types, error types, and ANN index. This doc designed (and now records) the
split into a pure `turbovecdb-core` crate and a thin `turbovecdb-py` PyO3
adapter crate.

## Sequencing: before #16, before #17

This work should land **before** #16 (Database in Rust) and #17 (cutover to
main), not after, reversing #18's originally-filed "depends on #17" note.
Reasoning: #16 hasn't started, and if it's built against today's single-crate
shape it will likely repeat the same by-name Python construction #18 fixes —
better to build it once, cleanly, against the split. Only `Collection` needs
migrating today (`Database` is still pure Python), which keeps this slice
self-contained.

## Problem recap

- `rust/src/collection.rs` constructs Python result objects by name at 5+
  call sites (`QueryResult`/`GetResult`/`HealthResult`/`ReembedReport` via
  `py.import_bound("turbovecdb.collection")`) and raises Python exceptions by
  name at ~10 call sites (`turbovec_error()` → `turbovecdb.errors.*`).
- `rust/src/lib.rs`'s `new_index`/`build_index`/`load_index`/
  `write_index_atomic` drive the ANN index only by calling Python's
  `turbovec` wheel via PyO3, on the belief that no Rust crate exists for it.
  **That belief is wrong** — see below.
- The crate builds only as a `cdylib`; there is no `rlib` target, so none of
  this logic can be unit-tested without the full Python/maturin build.

## The `turbovec` crate is real and pure

`turbovec` is published on crates.io as a pure-Rust crate (v0.9.0, MIT, no
pyo3/numpy dependency) *in addition to* the PyO3-wrapped Python wheel
(`turbovec-python`, a separate crate in the same upstream workspace that
produces the `_turbovec` cdylib). The pure crate exports `IdMapIndex` with
external u64 ids — `add_with_ids`, `remove`, `search` — the same surface
`collection.rs` already drives through PyO3. turbovecdb's Rust core can
depend on `turbovec` directly as a normal Cargo dependency instead of calling
back into the Python wheel.

**Version drift — spiked and confirmed compatible (2026-07-01):** the Python
wheel pinned in `uv.lock`/the venv is `turbovec==0.8.0`; crates.io's pure
crate is at `0.9.0`. A standalone spike crate (`cargo add turbovec@0.9`)
confirmed:
- **API matches exactly** what `collection.rs` needs: `IdMapIndex::new(dim,
  bit_width)`, `add_with_ids(&[f32], &[u64])`, `remove(u64) -> bool`,
  `search(&[f32], k)`, `search_with_allowlist(&[f32], k, Option<&[u64]>)`
  (the real allowlist API — even richer than assumed), `write`/`load`,
  `dim`/`bit_width`/`contains`/`len`/`is_empty`/`prepare`.
- **`IdMapIndex` is `Send + Sync`** (compile-time `assert_send`/`assert_sync`
  passed) — all fields are plain `Vec`/`HashMap`/`OnceLock<Vec<u8>>`-style
  data, no interior-mutability-without-sync or raw pointers.
- **`.tvim` format is bit-identical across 0.8.0 ↔ 0.9.0**, verified
  bidirectionally, not just "loads without error": a `.tvim` file written by
  the Python wheel (`turbovec==0.8.0`) was loaded by the native 0.9.0 crate
  and produced identical `search`/`search_with_allowlist` scores and ids; a
  file written by the native crate was loaded by the Python wheel with the
  same result. Confirms the format is genuinely shared, not just
  superficially similar.

**Risk found in the spike, resolved in Phase C2 (2026-07-01): `turbovec`
requires a BLAS provider to build.** `turbovec`'s own `Cargo.toml`
unconditionally enables `ndarray`'s `blas` feature on `cfg(target_os =
"linux")` and `"macos"`, which pulls in `cblas-sys` and requires linking
`-lopenblas` (or another BLAS provider) at build time. No system OpenBLAS was
available in the spike sandbox and no root access to install one.
**Resolution:** add `openblas-src` (`features = ["static"]`) as a direct
dependency of `turbovecdb-core`. Its build script compiles a vendored OpenBLAS
from source and statically links it — no system package, no root, and (contrary
to the spike's worry) **no working `gfortran` binary was needed either**; only
the `libgfortran5` *runtime* was present in this sandbox and the build still
succeeded. Verified end-to-end: a real `#[test]` in `turbovecdb-core`
constructing a `TurbovecIndex` (wrapping `turbovec::IdMapIndex` directly, no
PyO3 callback), running add/remove/write/load/search, passes standalone via
`cargo test -p turbovecdb-core`; `maturin develop` + the full 163-test suite
also stayed green with `turbovec` now a real dependency. Cost: ~4 minutes
added to a clean build (one-time; Cargo caches the compiled OpenBLAS in
`target/` afterward) — acceptable for both local dev and CI.

**Update (turbovec 1.0, 2026-09-05): this risk is dissolved and the
resolution has been reverted.** turbovec 1.0 dropped `faer` and `ndarray`
outright — its rotation is now a block-Hadamard transform, which is
O(n log n) and needs no matrix multiply, so the crate has no BLAS provider
requirement at all. Its whole dependency set is `ordered-float`, `rand`,
`rand_chacha`, `rayon`, `statrs`. The `openblas-src` dependency described
above has been removed from `turbovecdb-core`, taking the ~4-minute vendored
OpenBLAS compile out of a clean build. Note `turbovecdb-core` still depends
on `ndarray` directly (`vecmath.rs`, `embedder.rs`, `collection.rs`) — but
without the `blas` feature, so nothing links BLAS. Removing turbovec's own
`ndarray 0.17` also collapses the two `ndarray` majors that used to coexist
in the graph.

**Scope correction, found while implementing C2:** the original issue
description assumed `rust/src/lib.rs`'s `new_index`/`build_index`/
`load_index`/`write_index_atomic` functions were what drove the Python
callback — they're actually **dead code**, re-exported by
`turbovecdb.index` but called from nowhere in `collection.rs`.
`Collection`'s own methods (`make_index`/`reload_index`/
`mirror_write_to_index`/`remove_from_index`/the search call in `query`)
have their *own* separate inline `py.import_bound("turbovec")` calls — that's
the real, live Python-callback code, and it lives in `Collection`'s fields
(`index: Option<PyObject>`) and business logic, not in freestanding `lib.rs`
functions. Actually swapping it out means changing what `Collection` owns and
calls — which is exactly split phase 5/8's job ("port `Collection`'s business
logic... using the new abstractions"), not this phase's. So C2's real,
self-contained scope is: prove the native `turbovec` dependency and
`TurbovecIndex` (implementing `VectorIndex`) work end-to-end, available for
Phase D to consume. Phase D deletes the dead `lib.rs` functions and
`Collection`'s inline PyO3-callback calls once it rewires `Collection` to use
`TurbovecIndex`.

## Target layout

```
Cargo.toml                          # becomes [workspace]
crates/
  turbovecdb-core/                  # pure Rust — NO pyo3, NO numpy
    Cargo.toml                      # deps: rusqlite (bundled), turbovec = "0.9", serde_json, thiserror
    src/
      lib.rs
      collection.rs                 # ported business logic (SQLite, generations, reembed)
      filters.rs                    # where/where_document compiler, operates on serde_json::Value
      error.rs                      # CoreError (thiserror)
      types.rs                      # native QueryResult/GetResult/HealthResult/ReembedReport
      embedder.rs                   # Embedder trait
      index.rs                      # VectorIndex trait + turbovec::IdMapIndex-backed impl
    tests/                          # cargo test — new capability, doesn't exist today
  turbovecdb-py/                    # thin PyO3 adapter — the ONLY crate touching Python by name
    Cargo.toml                      # deps: turbovecdb-core (path), pyo3, numpy
    src/
      lib.rs                        # #[pymodule]; registers PyCollection, FilterError, l2_normalize
      collection.rs                 # #[pyclass] PyCollection wrapping turbovecdb_core::Collection
      convert.rs                    # CoreError -> PyErr (one match); native results -> Python dataclasses
      embedder.rs                   # PyEmbedder: wraps a PyObject, implements core::Embedder
pyproject.toml                      # [tool.maturin] manifest-path = "crates/turbovecdb-py/Cargo.toml"
```

Two crates, not more: nothing outside `turbovecdb-py` will ever consume
`filters`/`error`/`index` independently, so these stay as modules inside
`turbovecdb-core` rather than separate crates — extra crates would only add
Cargo bookkeeping, no real isolation benefit at ~1,900 lines of source.

`turbovecdb-core`'s `Cargo.toml` must omit `crate-type` (defaulting to
`rlib`) or set it explicitly — this is the literal mechanism that makes
`cargo test` possible, which is impossible today.

## Core abstractions (`turbovecdb-core`)

- **`CoreError`** (`thiserror`-based enum: `DimensionMismatch`,
  `EmbedderRequired`, `EmbedderIdentityMismatch { stored, current }`,
  `UnsupportedFilter`, `Sql(rusqlite::Error)`, `Io`, `Other`). No PyO3
  dependency. **`Display` text must reproduce today's exact exception
  message strings**, not just map to the right class — the Python test suite
  uses `pytest.raises(..., match=...)` against specific wording in several
  places, so message parity is a hard acceptance check, not a nicety.
- **Native result structs** (`QueryResult`/`GetResult`/`HealthResult`/
  `ReembedReport`) using `serde_json::Value` for metadata instead of
  `PyObject`.
- **`Embedder` trait** (`fn embed(&self, docs: &[String]) -> Result<Array2<f32>, CoreError>`).
  Core logic (`resolve_vectors`, `query`, `reembed`) becomes generic/dyn over
  this instead of hardcoding a Python callable. `PyEmbedder` (in
  `turbovecdb-py`) is the only implementation, wrapping a `PyObject` — this
  is the one Python callback that's inherent and permanent (embedders are
  arbitrary user code), not something this design tries to eliminate.
- **`VectorIndex` trait** — narrow, only the methods `collection.rs` actually
  calls (`add_with_ids`/`remove`/`search`/`write`/`load`/`dim`/`bit_width`),
  not a general "swappable ANN backend" surface. Its only justification is
  unit-testability of `turbovecdb-core` without linking the real
  quantization/`faer` path (a fake in-memory impl for `cargo test`); resist
  adding methods "for completeness." Production impl wraps
  `turbovec::IdMapIndex` (native crate).
- **`Send` bounds are load-bearing, not optional.** `#[pyclass]` requires the
  wrapped struct to be `Send` (unless marked `unsendable`). Today's
  `Option<PyObject>` fields are fine because `Py<T>` is unconditionally
  `Send + Sync`. Once these become `Box<dyn Embedder>` / `Box<dyn VectorIndex>`,
  `Send` does **not** propagate to trait objects automatically — declare
  `Embedder: Send` / `VectorIndex: Send` (or type fields as
  `Box<dyn Embedder + Send>`), or `turbovecdb-py` fails to compile. Also
  verify `turbovec::IdMapIndex` itself is `Send` (fold into the Phase 0
  spike). `PyEmbedder::embed` will need to take the GIL internally
  (`Python::with_gil`) since the core trait signature carries no `Python<'_>`
  token — safe, but means no future `allow_threads` optimization may wrap a
  code path that can call into `dyn Embedder`.

## Filter/metadata JSON conversion

The filter compiler moves into `turbovecdb-core::filters`, taking
`serde_json::Value` instead of `Bound<PyAny>` — `turbovecdb-py` does the one
`PyAny -> serde_json::Value` conversion at the boundary via the `pythonize`
crate. **Scope this narrowly**: use `pythonize` only for `where`/
`where_document` (already JSON-shaped by the filter grammar). Keep the
existing hand-rolled `py_to_sql_value`-style converter for free-form
`metadatas`, because:
- `serde_json::Value::Object` defaults to a `BTreeMap` (alphabetical key
  order) unless the `preserve_order` feature is enabled — a behavior change
  vs. Python dict insertion order if anything downstream depends on it.
- `serde_json` cannot represent NaN/Infinity; Python's `json.dumps` allows
  them by default — metadata containing them would hard-fail instead of
  serializing.
- Today's converter silently stringifies unrecognized Python types via
  `.str()`; `pythonize`'s depythonize hard-errors on non-JSON-native types.
  Free-form metadata is more likely to hit this than the constrained filter
  grammar.

## Adapter crate (`turbovecdb-py`) responsibilities

This is the **only** place allowed to touch Python objects by name:
- `PyCollection` (`#[pyclass]`) wraps `turbovecdb_core::Collection`; each
  `#[pymethods]` fn converts incoming PyO3 args to core types, calls core,
  converts the returned `Result<T, CoreError>` via `convert.rs`.
- `convert.rs`: one `impl From<CoreError> for PyErr` (imports
  `turbovecdb.errors` classes by name **once**, in one match statement,
  replacing today's ~10 scattered call sites); one set of functions
  converting the 4 native result types into the existing Python dataclasses
  in `turbovecdb.collection` (replacing today's 5 scattered call sites).
  Collapsing scattered by-name construction into one file is the actual fix
  — Rust still needs a seam to hand back a Python dataclass, but the seam
  becomes one file instead of being woven through 1,500 lines of business
  logic.
- `PyEmbedder`: wraps a `PyObject`, implements `core::Embedder`.

`src/turbovecdb/collection.py` barely changes: it keeps its dataclasses and
locking wrapper; its docstring claim ("the Rust core constructs them by
name") becomes accurate in a narrower, contained sense — only
`turbovecdb-py/src/convert.rs` does that now.

## Phased migration

Each phase keeps the existing pytest suite green, matching the project's
established "flip once, at parity" incremental-slice style
(`docs/rust-core-plan.md`).

1. **[Done, #19] Phase 0 — spike.** Verified `turbovec::IdMapIndex`'s exact
   method signatures, confirmed it's `Send`, and ran a falsifiable
   cross-format test: a `.tvim` index written by the Python wheel
   (`turbovec==0.8.0`) loaded and searched correctly from the native 0.9.0
   crate, and vice versa — format is genuinely shared, not just superficially
   similar.
2. **[Done, #20] Phase A — mechanical workspace shell, zero logic change.**
   Moved `rust/src/*.rs` as-is into `crates/turbovecdb-py/src/*`, added
   `crates/turbovecdb-core`, updated `Cargo.toml`/`pyproject.toml`. Build +
   full suite green with no behavior change.
3. **[Done, #21] Phase B — extracted the filter compiler** into
   `turbovecdb-core::filters`, retyped over `serde_json::Value`.
   `test_filters.py` green.
4. **[Done, #22] Phase C1 — introduced `CoreError`/`Embedder`/`VectorIndex`.**
   Each with `#[cfg(test)]`-only fakes; `Collection`'s actual index handling
   was untouched in this phase (still its own inline PyO3 calls into Python's
   `turbovec` wheel).
5. **[Done, #23] Phase C2 — added the native `turbovec` crate dependency**
   and `TurbovecIndex` (wraps `turbovec::IdMapIndex` directly, no PyO3
   callback) as `VectorIndex`'s production impl. Resolved the BLAS-build risk
   via `openblas-src` (`features = ["static"]`) — no system package, no root,
   no `gfortran` needed. Scope-corrected while implementing: the live
   Python-callback code turned out to be `Collection`'s own inline calls, not
   the (already-dead) `new_index`/`build_index`/`load_index`/
   `write_index_atomic` functions this issue originally named.
6. **[Done, #24] Phase D — ported `Collection`'s business logic** (SQLite
   reads/writes, generation bookkeeping, reembed) into
   `turbovecdb-core::collection::Collection<E: Embedder, I: VectorIndex>`
   (generic, not `dyn`-boxed). `turbovecdb-py::collection::Collection` is now
   a ~230-line thin wrapper (down from 1500). `Collection` switched from
   inline PyO3-callback index handling to `TurbovecIndex`; the dead
   `new_index`/`build_index`/`load_index`/`write_index_atomic` functions and
   the stale "turbovec has no Rust crate" comment are gone. One test needed
   updating: `test_reembed_rolls_back_when_index_rebuild_fails` monkeypatched
   Python's `turbovec.IdMapIndex`, which the native index no longer calls —
   replaced with a real failure (embedder output dim not a multiple of 8).
7. **[Done, #25] Phase E — added `cargo test` coverage**: 17 filter-compiler
   tests plus 13 `Collection` integration tests against
   `Collection<ConstantEmbedder, FakeIndex>` and a temp SQLite dir. Landed as
   `#[cfg(test)]` modules in `filters.rs`/`collection.rs` (matching the
   pattern in `error.rs`/`embedder.rs`/`index.rs`) rather than a separate
   `tests/` directory. `turbovecdb-core` has 37 tests total, zero Python.
8. **[Done, #26] Phase F — cleanup.** This doc, `docs/rust-core-plan.md`, and
   issue #18 updated to record the completed split.

## Out of scope

- Migrating `Database`/`database.py` into Rust — that's #16, sequenced
  *after* this split (see Sequencing).
- Changing the on-disk SQLite schema or `.tvim` binary format.
- Eliminating the embedder Python callback — inherent, permanent boundary.
