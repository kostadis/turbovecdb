//! The one place `turbovecdb-py` constructs Python exceptions by name.
//!
//! Historically this logic (`turbovec_error()`) lived inline in
//! `collection.rs`, called from ~10 sites scattered through business logic.
//! Consolidating it here is the actual fix for issue #18: `CoreError`
//! (ported from `turbovecdb-core` in split phase 5/8, see
//! `docs/rust-core-split-design.md`) maps to the public
//! `turbovecdb.errors.*` classes in exactly one match statement.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use turbovecdb_core::error::CoreError;

/// Raise one of the public `turbovecdb.errors.*` classes by name.
pub(crate) fn turbovec_error(py: Python<'_>, cls: &str, msg: String) -> PyErr {
    let build = || -> PyResult<PyErr> {
        let c = py.import_bound("turbovecdb.errors")?.getattr(cls)?;
        Ok(PyErr::from_value_bound(c.call1((msg,))?))
    };
    build().unwrap_or_else(|e| e)
}

/// `CoreError -> PyErr`, matching `Collection`'s historical exception
/// types exactly (`Sql`/`Io` never raised a `turbovecdb.errors.*` class —
/// the historical `sql_err()` helper always raised a plain `RuntimeError`).
#[allow(dead_code)] // wired into Collection's ported methods in split phase 5/8 (#24)
pub(crate) fn core_err_to_py(py: Python<'_>, e: CoreError) -> PyErr {
    match e {
        CoreError::DimensionMismatch(m) => turbovec_error(py, "DimensionMismatchError", m),
        CoreError::EmbedderRequired(m) => turbovec_error(py, "EmbedderRequiredError", m),
        CoreError::EmbedderIdentityMismatch(m) => {
            turbovec_error(py, "EmbedderIdentityMismatchError", m)
        }
        CoreError::UnsupportedFilter(m) => turbovec_error(py, "UnsupportedFilterError", m),
        CoreError::Other(m) => turbovec_error(py, "TurboVecError", m),
        CoreError::Sql(err) => PyRuntimeError::new_err(format!("sqlite error: {err}")),
        CoreError::Io(err) => PyRuntimeError::new_err(err.to_string()),
    }
}
