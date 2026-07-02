//! Pure Rust engine for turbovecdb — no PyO3, no Python dependency.
//!
//! Logic moves here incrementally from `turbovecdb-py` (see
//! `docs/rust-core-split-design.md`); this crate stays buildable/testable via
//! plain `cargo build`/`cargo test` throughout.

pub mod filters;
