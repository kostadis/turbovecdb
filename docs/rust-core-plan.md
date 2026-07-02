# Rust core rewrite — plan & roadmap

**Status: complete.** All nine slices (#9–#17) plus the interlude core/adapter
split (#18–#26) are done; the rewrite landed on `main` via PR #27. This doc is
the historical record of the plan and its evolution.

Tracking doc for migrating turbovecdb's Python "core" engine to Rust. All work
happens on the long-lived branch **`feat/rust-core`**; `main` stays Rust-free
until the rewrite is complete, then lands as a single PR.

## Goal & constraints (decided)

- **Storage:** keep SQLite as the durable source of truth, accessed from Rust via `rusqlite`. (SQLite is already solid; the win is moving the *glue* — concurrency, cache coherence, index/store atomicity — into Rust, without re-owning durability.)
- **Bindings:** a PyO3 extension module `turbovecdb._core` that preserves the exact `turbovecdb` Python API, so the existing **163-test suite + `service.py`** remain the acceptance harness.
- **Delivery:** one vertical slice at a time on `feat/rust-core`; every slice keeps the Python API identical and leaves the full suite green.

## Done (baseline, on branch)

- PyO3 crate scaffold (`Cargo.toml`, `rust/src/lib.rs`, maturin build backend).
- `where` / `where_document` **filter compiler** ported to Rust; `turbovecdb.filters` is now a thin shim re-raising `_core.FilterError` as `UnsupportedFilterError`.

## Interlude: core/adapter split (done, before #16)

[#18](https://github.com/kostadis/turbovecdb/issues/18) found that the Rust
engine built so far wasn't a pure, standalone crate — it reached back into
Python for its own result/error types and its ANN index. See
`docs/rust-core-split-design.md` for the design and phase-by-phase record
(issues #19-#26, all closed): the single `rust/` crate became a Cargo
workspace — `crates/turbovecdb-core` (pure Rust: SQLite store, filter
compiler, generation bookkeeping, reembed, native `turbovec` crate
dependency for the ANN index, 37 `cargo test`s, zero Python) plus
`crates/turbovecdb-py` (thin PyO3 adapter — `convert.rs` is now the only
place that constructs Python objects/exceptions by name). Landed **before**
slice #16 below, so Database-in-Rust is authored directly against the clean
shape instead of the tangled one. `rust/src/lib.rs` referenced elsewhere in
this doc no longer exists — see the workspace layout in
`docs/rust-core-split-design.md`.

## Roadmap

Ordered; later slices build on earlier ones. Each is a GitHub issue (label `rust-core`).

| Slice | Issue | Scope |
|-------|-------|-------|
| 1. numpy boundary + `l2_normalize` | [#9](https://github.com/kostadis/turbovecdb/issues/9)  | `index.l2_normalize` → Rust; establish numpy↔Rust interop |
| 2. index lifecycle | [#10](https://github.com/kostadis/turbovecdb/issues/10) | `index.py` new/build/load/write; resolve turbovec interop |
| 3. SQLite storage via rusqlite | [#11](https://github.com/kostadis/turbovecdb/issues/11) | conn+PRAGMAs, docs/meta schema, meta, generations, next_uid, migration |
| 4. write paths | [#12](https://github.com/kostadis/turbovecdb/issues/12) | `_write` (add/upsert), delete, update_metadata, update_documents, clear |
| 5. read paths | [#13](https://github.com/kostadis/turbovecdb/issues/13) | query (+ exact-cosine rerank), get, count |
| 6. reembed (atomic) | [#14](https://github.com/kostadis/turbovecdb/issues/14) | atomic two-phase reembed; carry the drop/keep + bug-A/B fixes |
| 7. concurrency & coherence | [#15](https://github.com/kostadis/turbovecdb/issues/15) | file lock, in-proc lock, generation cache reload, WAL checkpoint, health() |
| 8. Database layer | [#16](https://github.com/kostadis/turbovecdb/issues/16) (done) | `database.py`: connect, collection cache, list/delete_collection |
| 9. cutover & cleanup | [#17](https://github.com/kostadis/turbovecdb/issues/17) (done) | thin-shim/remove dead Python engine; full green; completion PR to `main` — see `docs/rust-core-cutover-plan.md`: the flip had already happened incrementally (#14–#16), so this became dependency/docs cleanup + a clean-venv wheel parity proof |

Rust deps added along the way: `rusqlite` (bundled SQLite), `numpy`/`ndarray`, a file-lock crate (e.g. `fs2`).

## Open design question (resolve at slice 2 / #10) — resolved in the core/adapter split

`turbovec` ships as a PyO3 Python extension, **and** (contrary to what was
assumed here) is also published independently as a pure Rust crate on
crates.io with no PyO3/numpy dependency. The Rust core depends on it
directly (`turbovec = "0.9"` in `crates/turbovecdb-core/Cargo.toml`) rather
than calling back through PyO3 — see `docs/rust-core-split-design.md`'s
"The `turbovec` crate is real and pure" section for how this was confirmed
and the BLAS-build-dependency risk it surfaced (resolved via a vendored
static OpenBLAS build).

## Update (post-slice-2): slices 3–8 build one Rust `Collection`/`Database` class

Slices 1–2 were stateless leaf functions, so they flipped cleanly. Slices 3–8
cannot: the **SQLite connection is atomic** — `_write`/`reembed` write `docs` +
`meta` (store_gen) in a single transaction on one connection, and
`query`/`get`/`count` read that same connection. Moving only "storage" to a
Rust-owned `rusqlite` connection while write/read/reembed stay on Python's
`sqlite3` would mean two connections fighting over one file, breaking
single-transaction atomicity and `store_gen`/`tvim_gen` coherence.

So slices 3–8 become **internal milestones of building a Rust `Collection` (+
`Database`) PyO3 class** that owns the `rusqlite` connection, the turbovec index
(via PyO3), and the locks:

- **#11 storage foundation** — conn + PRAGMAs, docs/meta schema, meta, store_gen/tvim_gen, next_uid, schema migration, `count`, `dim`
- **#12 write paths** — `_write` (add/upsert), delete, update_metadata/documents, clear
- **#13 read paths** — query (+ exact-cosine rerank), get, count
- **#14 reembed** — atomic two-phase (drop/keep + bug-A/B fixes)
- **#15 concurrency** — file lock, in-proc lock, generation cache reload, WAL checkpoint, health()
- **#16 Database** — connect, collection cache, list/delete_collection

The Rust class is built and tested incrementally (its methods exercised directly
via `_core.Collection` while the Python `Collection` still runs the suite), but
the **Python-facing flip happens once, at parity** — Python `Collection`/`Database`
become thin delegators to `_core` (folded into **#17**). Expect fewer, larger
green commits than one-per-slice.

**#16 landed differently than sketched above** (see
`docs/rust-core-database-plan.md`): the collection-handle cache did **not**
move into a Rust `Database` class. `Collection`'s Python wrapper owns the
cross-process `FileLock` and must stay identity-stable per name, so the
cache has to live in Python regardless — a second cache in Rust would need
`Arc<Mutex<...>>` sharing just to exist and would never actually be read.
What moved to Rust is the pure part: name validation, path resolution,
listing, and directory removal (`turbovecdb_core::database::Database`, no
generics, no cache), via a thin `_core.Database` adapter. `database.py`'s
cache, locking, and the `delete_collection` race-guard dance are unchanged.

## Working agreement

- Build on `feat/rust-core`; `main` stays Rust-free until the cutover (#17).
- Each milestone lands when it builds, its own smoke/tests pass, and the **full
  163-suite stays green** (green is trivial pre-flip since Python is unchanged;
  the flip in #17 is where the suite proves parity).
- On done: commit → `git push origin feat/rust-core` → close the issue → mark the todo complete. (No pre-commit checkpoint.)
- [PR #27](https://github.com/kostadis/turbovecdb/pull/27) to `main` was opened as a **draft** during the core/adapter split, accumulating `feat/rust-core`'s commits as they landed so each phase's diff was reviewable as it went; it stayed draft until #16 and #17 were done, then merged as the completion PR.
