# Rust core rewrite — plan & roadmap

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
| 8. Database layer | [#16](https://github.com/kostadis/turbovecdb/issues/16) | `database.py`: connect, collection cache, list/delete_collection |
| 9. cutover & cleanup | [#17](https://github.com/kostadis/turbovecdb/issues/17) | thin-shim/remove dead Python engine; full green; completion PR to `main` |

Rust deps added along the way: `rusqlite` (bundled SQLite), `numpy`/`ndarray`, a file-lock crate (e.g. `fs2`).

## Open design question (resolve at slice 2 / #10)

`turbovec` ships as a PyO3 Python extension. The Rust core must reach the ANN
index either (a) via a native turbovec Rust crate if one is published, or (b) by
calling the turbovec Python module back through PyO3. This affects slices 2/5/6.

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

## Working agreement

- Build on `feat/rust-core`; `main` stays Rust-free until the cutover (#17).
- Each milestone lands when it builds, its own smoke/tests pass, and the **full
  163-suite stays green** (green is trivial pre-flip since Python is unchanged;
  the flip in #17 is where the suite proves parity).
- On done: commit → `git push origin feat/rust-core` → close the issue → mark the todo complete. (No pre-commit checkpoint.)
- No PR to `main` until the cutover (#17).
