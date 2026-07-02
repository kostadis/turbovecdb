# Cutover & cleanup (#17) — implementation plan

Plan for [#17](https://github.com/kostadis/turbovecdb/issues/17), the last
slice of the Rust core rewrite. Note the issue body's `Files:
rust/src/lib.rs` reference is stale (the workspace is
`crates/turbovecdb-core` + `crates/turbovecdb-py`).

## Scope reality-check: the "flip" already happened

The issue was written when slices 3–8 were expected to land behind a
still-Python engine, with one big Python-facing flip at the end. That flip
happened incrementally instead (#14/#15 cut `Collection` over; #16 cut
`Database` over). Surveying `src/turbovecdb/` today: 702 lines total, all of
it either thin delegation, lock/cache wrapper logic that deliberately stays
in Python, or the public dataclasses/exceptions. **There is no dead Python
engine code left to remove.** What #17 actually is: dependency + docs
cleanup, a fresh-install parity proof, and the completion PR.

## Findings driving the cleanup

- **The `turbovec` PyPI wheel is no longer used.** Zero `import turbovec`
  in `src/` or `tests/`; the native crate is compiled into `_core.abi3.so`,
  and `ldd` confirms the extension is self-contained (only
  libc/libm/libgcc — OpenBLAS statically linked via `openblas-src`). Yet
  `pyproject.toml` still declares `turbovec>=0.7` as a runtime dependency
  and `uv.lock` pins the 0.8.0 wheel. Every install pulls a dead wheel.
- **README** says `pip install turbovecdb  # pulls turbovec, numpy,
  filelock` and describes turbovec as the Python-wheel engine; needs
  updating (deps are now just `numpy` + `filelock`; source builds need a
  Rust toolchain, with OpenBLAS vendored/static — no system BLAS).
- **`docs/ARCHITECTURE.md`** (14K, pre-rewrite) likely still describes the
  Python engine — review; either update its engine sections or banner it as
  historical with a pointer to `docs/rust-core-split-design.md`.
- **`turbovecdb.filters` / `turbovecdb.index`** are thin shims with no
  internal callers (`test_filters.py` exercises filtering through the
  `Collection` API) and no external ones (mempalace, the only known
  consumer, imports only `connect`/`Database`/errors). **Keep them** — they
  are documented public surface and cost nothing. Same for the three
  `_core` filter pyfunctions backing `filters.py`.

## Steps

1. **Sync the worktree.** Work happens in `~/src/turbovecdb-issue-17`
   (branch `rust-core-issue-17-cutover`, created for this issue); it
   branched before this plan landed, so first `git merge feat/rust-core`.
   Setup: `uv venv && uv pip install maturin pytest numpy filelock`, then
   `maturin develop --release` from that tree.
2. **Drop the dead dependency.** Remove `turbovec>=0.7` from
   `[project].dependencies`; regenerate `uv.lock`. Grep-verify nothing
   imports the wheel.
3. **Docs pass.** README install/requirements + engine description;
   ARCHITECTURE.md review; fix the stale `rust/src/lib.rs`-era references
   if any remain (`grep -rn "rust/src" docs/ README.md`).
4. **Fresh-install parity proof** (the acceptance test for step 2):
   `maturin build --release`, install the wheel into a **clean venv without
   the turbovec wheel**, run the full 163-test suite + import smoke there.
   Also `cargo test --workspace` green.
5. **Land it.** Commit(s) on `rust-core-issue-17-cutover` → merge
   `--ff-only` into `feat/rust-core` → push. Update
   `docs/rust-core-plan.md` (roadmap complete) and PR #27's description.
6. **Stop for review** (per the issue's workflow), then: un-draft PR #27,
   merge to `main`, close #17. `main` gets the Rust engine in one PR, as
   agreed.

## Acceptance

- Full pytest suite green **from a clean-venv wheel install with no
  `turbovec` wheel present**.
- `cargo test --workspace` green.
- `pip install turbovecdb` dependency set is `numpy`, `filelock` only.
- No stale "engine in Python" / "pulls turbovec" claims in README or docs.
- PR #27 merged to `main`; #17 closed; roadmap (#9–#17) fully done.

## Out of scope

- CI / wheel-publishing automation (no CI exists on this repo; building
  release wheels for PyPI is its own follow-up if wanted).
- Removing `filters.py`/`index.py` shims or the `_core` filter functions
  (public surface, kept).
- Version bump / changelog for a 0.3.0 release — decide at merge time.
