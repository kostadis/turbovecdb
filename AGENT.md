# Rules

We are building the design specified in docs/re-embed.md. The Architecture of this code is ARCHITECTURE.md

# Building

The core is migrating to a Rust/PyO3 extension module (`turbovecdb._core`, source
under `rust/`, built by maturin). A fresh checkout therefore needs a Rust
toolchain plus maturin, and an editable install compiles the extension:

    pip install "maturin>=1.7,<2"
    maturin develop            # builds turbovecdb._core and installs the package

Then `python -m pytest` as usual. Currently `_core` implements the `where` /
`where_document` filter compiler (behind the unchanged `turbovecdb.filters`
API); the storage/query/reembed engine is still Python and is being ported
incrementally.

