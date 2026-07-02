//! Pure Rust engine for turbovecdb — no PyO3, no Python dependency.
//!
//! Logic moves here incrementally from `turbovecdb-py` (see
//! `docs/rust-core-split-design.md`); this crate stays buildable/testable via
//! plain `cargo build`/`cargo test` throughout.

pub mod collection;
pub mod embedder;
pub mod error;
pub mod filters;
pub mod index;
pub mod types;
pub mod vecmath;
