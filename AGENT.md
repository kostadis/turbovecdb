# Rules

We are building the design specified in docs/re-embed.md. The Architecture of this code is ARCHITECTURE.md

# Building

The engine is a Rust/PyO3 extension module (`turbovecdb._core`), a Cargo
workspace under `crates/` — `turbovecdb-core` (the pure engine: SQLite store,
turbovec index, query/reembed, filter compiler, **and all locking**) and
`turbovecdb-py` (the PyO3 adapter maturin builds into `turbovecdb._core`). A
fresh checkout needs a Rust toolchain plus maturin, and an editable install
compiles the extension:

    pip install "maturin>=1.7,<2"
    maturin develop            # builds turbovecdb._core and installs the package

Then `python -m pytest` as usual. `src/turbovecdb/*.py` is now a thin,
lock-free shim over `_core`: it shapes arguments, holds the public result
dataclasses, and delegates. Cross-process serialization (an `flock` on the
sibling `<root>/<name>.lock`) and in-process serialization (a `Mutex` around
the core, acquired inside `allow_threads`) both live in the Rust core.

