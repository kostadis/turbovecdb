//! Engine-level errors — no PyO3 dependency.
//!
//! Each variant maps 1:1 to a public `turbovecdb.errors.*` exception class
//! (mapping lives in `turbovecdb-py::convert`). Variants carry an
//! already-formatted message rather than structured fields: the call sites
//! that construct them (ported from `turbovecdb-py::collection` in split
//! phase 5/8, see `docs/rust-core-split-design.md`) build the exact same
//! `format!()` text the historical PyO3 code used, so `Display` is a
//! passthrough and message text stays byte-for-byte stable across the port.
//! The existing Python test suite asserts on that exact wording in several
//! places (`pytest.raises(..., match=...)`).

use std::fmt;

#[derive(Debug)]
pub enum CoreError {
    /// -> `turbovecdb.errors.DimensionMismatchError`
    DimensionMismatch(String),
    /// -> `turbovecdb.errors.EmbedderRequiredError`
    EmbedderRequired(String),
    /// -> `turbovecdb.errors.EmbedderIdentityMismatchError`
    EmbedderIdentityMismatch(String),
    /// -> `turbovecdb.errors.UnsupportedFilterError`
    UnsupportedFilter(String),
    /// -> a plain `ValueError` (bad argument shape/value — length mismatches,
    /// "already exists", "not found", etc.). Matches the many historical
    /// `PyValueError::new_err(...)` call sites that never raised a
    /// `turbovecdb.errors.*` class.
    InvalidArgument(String),
    /// -> a plain `RuntimeError` (matches the historical "X: meta update
    /// failed" call sites, which deliberately swallowed the underlying
    /// error and raised a generic one instead).
    Runtime(String),
    /// -> a plain `RuntimeError` (matches the historical `sql_err()` helper,
    /// which never raised a `turbovecdb.errors.*` class for SQL failures).
    Sql(rusqlite::Error),
    /// -> a plain `RuntimeError`.
    Io(std::io::Error),
    /// -> `turbovecdb.errors.TurboVecError` (the root/catch-all class).
    Other(String),
    /// -> `turbovecdb.errors.CollectionNotFoundError`.
    CollectionNotFound(String),
    /// -> `turbovecdb.errors.TurboVecError` (the root class, matching the
    /// historical Python wrapper, which raised a plain `TurboVecError` when
    /// the cross-process write lock could not be acquired within the
    /// timeout). Carries the already-formatted, `repr()`-quoted message so
    /// `Display` stays a passthrough (invariant I4).
    LockTimeout(String),
    /// -> `turbovecdb.errors.TurboVecError`. The SQLite connection died
    /// (filesystem error, NFS hiccup, Docker volume remount) and needs a
    /// `reconnect()` before any further I/O. Carries the already-formatted
    /// message so `Display` stays a passthrough.
    ConnectionClosed(String),
}

impl fmt::Display for CoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoreError::DimensionMismatch(m)
            | CoreError::EmbedderRequired(m)
            | CoreError::EmbedderIdentityMismatch(m)
            | CoreError::UnsupportedFilter(m)
            | CoreError::InvalidArgument(m)
            | CoreError::Runtime(m)
            | CoreError::Other(m)
            | CoreError::CollectionNotFound(m)
            | CoreError::LockTimeout(m)
            | CoreError::ConnectionClosed(m) => write!(f, "{m}"),
            CoreError::Sql(e) => write!(f, "sqlite error: {e}"),
            CoreError::Io(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for CoreError {}

impl From<rusqlite::Error> for CoreError {
    fn from(e: rusqlite::Error) -> Self {
        CoreError::Sql(e)
    }
}

impl From<std::io::Error> for CoreError {
    fn from(e: std::io::Error) -> Self {
        CoreError::Io(e)
    }
}

impl From<crate::filters::FilterError> for CoreError {
    fn from(e: crate::filters::FilterError) -> Self {
        CoreError::UnsupportedFilter(e.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_is_passthrough_for_message_variants() {
        let msg = "turbovec requires dim to be a positive multiple of 8, got 5";
        assert_eq!(CoreError::DimensionMismatch(msg.to_string()).to_string(), msg);
        assert_eq!(CoreError::EmbedderRequired(msg.to_string()).to_string(), msg);
        assert_eq!(CoreError::EmbedderIdentityMismatch(msg.to_string()).to_string(), msg);
        assert_eq!(CoreError::UnsupportedFilter(msg.to_string()).to_string(), msg);
        assert_eq!(CoreError::Other(msg.to_string()).to_string(), msg);
    }

    #[test]
    fn sql_error_reproduces_historical_prefix() {
        let e = rusqlite::Connection::open_in_memory()
            .unwrap()
            .execute("SELECT * FROM nonexistent", [])
            .unwrap_err();
        let msg = CoreError::from(e).to_string();
        assert!(msg.starts_with("sqlite error: "), "got {msg:?}");
    }

    #[test]
    fn connection_closed_is_passthrough_display() {
        let msg = "SQLite connection is closed; call reconnect()";
        assert_eq!(CoreError::ConnectionClosed(msg.to_string()).to_string(), msg);
    }

    #[test]
    fn filter_error_maps_to_unsupported_filter() {
        let fe = crate::filters::compile_where(
            &serde_json::json!({"a": {"$bogus": 1}}),
            0,
        )
        .unwrap_err();
        match CoreError::from(fe) {
            CoreError::UnsupportedFilter(m) => assert!(m.contains("$bogus")),
            other => panic!("expected UnsupportedFilter, got {other:?}"),
        }
    }
}
