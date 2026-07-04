//! `Database` — thin PyO3 wrapper around `turbovecdb_core::database::Database`.
//!
//! No collection cache here: `database.py` keeps the name -> `Collection`
//! wrapper cache (and its close/lock/re-check-under-lock dance in
//! `delete_collection`) in Python, since `Collection`'s handle identity and
//! `FileLock` ownership already live there. This wrapper only exposes the
//! pure path/name/list/delete operations. See
//! `docs/rust-core-database-plan.md`.

use pyo3::prelude::*;
use turbovecdb_core::database::Database as CoreDatabase;
use turbovecdb_core::error::CoreError;

use crate::convert;

#[pyclass]
pub struct Database {
    inner: CoreDatabase,
}

#[pymethods]
impl Database {
    #[new]
    fn new(root: String) -> Self {
        Database { inner: CoreDatabase::new(root) }
    }

    fn collection_dir(&self, name: &str) -> PyResult<String> {
        Python::with_gil(|py| {
            self.inner
                .collection_dir(name)
                .map(|p| p.to_string_lossy().into_owned())
                .map_err(|e| convert::core_err_to_py(py, e))
        })
    }

    fn list_collections(&self) -> PyResult<Vec<String>> {
        Python::with_gil(|py| self.inner.list_collections().map_err(|e| convert::core_err_to_py(py, e)))
    }

    fn ensure_collection(&self, name: &str) -> PyResult<String> {
        Python::with_gil(|py| {
            self.inner
                .ensure_collection(name)
                .map(|p| p.to_string_lossy().into_owned())
                .map_err(|e| convert::core_err_to_py(py, e))
        })
    }

    /// Remove a collection's directory. Caller must already hold the write
    /// lock and have closed any cached handle for `name` (see
    /// `database.py`'s `delete_collection`). I/O failures surface as the raw
    /// `OSError` subclass `shutil.rmtree` would have raised (e.g.
    /// `PermissionError`), not the generic `RuntimeError` `Collection`'s I/O
    /// errors map to — `delete_collection` never wrapped `rmtree`'s
    /// exceptions.
    fn remove_collection_dir(&self, name: &str) -> PyResult<()> {
        Python::with_gil(|py| {
            self.inner.remove_collection_dir(name).map_err(|e| match e {
                CoreError::Io(io_err) => PyErr::from(io_err),
                other => convert::core_err_to_py(py, other),
            })
        })
    }

    /// Delete a collection's directory under the core's cross-process write
    /// lock (I6). The GIL is released for the duration: this may block on the
    /// flock for up to `lock_timeout` seconds, and it touches no Python
    /// objects. `database.py` evicts and closes any cached wrapper *before*
    /// calling this — a handle's `close()` would deadlock against the guard
    /// held here (second file description on the same lock). rmtree I/O
    /// failures surface as the raw `OSError` subclass (e.g. `PermissionError`)
    /// rather than a generic `RuntimeError`, matching the historical
    /// `delete_collection`, which never wrapped `rmtree`'s exceptions.
    #[pyo3(signature = (name, lock_timeout=30.0))]
    fn delete_collection(&self, py: Python<'_>, name: &str, lock_timeout: f64) -> PyResult<()> {
        let inner = &self.inner;
        py.allow_threads(|| inner.delete_collection(name, lock_timeout)).map_err(|e| match e {
            CoreError::Io(io_err) => PyErr::from(io_err),
            other => convert::core_err_to_py(py, other),
        })
    }
}
