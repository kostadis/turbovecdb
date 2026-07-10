//! The one place `turbovecdb-py` constructs Python objects/exceptions by
//! name.
//!
//! Historically this logic was scattered through `collection.rs`: exception
//! construction (`turbovec_error()`) at ~10 call sites, and
//! `QueryResult`/`GetResult`/`HealthResult`/`ReembedReport` construction at
//! 5 more. Consolidating both here is the actual fix for issue #18:
//! `turbovecdb-core`'s native types (ported in split phase 5/8, see
//! `docs/rust-core-split-design.md`) map to the public Python types in
//! exactly one file.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use turbovecdb_core::error::CoreError;
use turbovecdb_core::types::{GetResult, HealthResult, QueryResult, ReembedReport};

/// Raise one of the public `turbovecdb.errors.*` classes by name.
pub(crate) fn turbovec_error(py: Python<'_>, cls: &str, msg: String) -> PyErr {
    let build = || -> PyResult<PyErr> {
        let c = py.import_bound("turbovecdb.errors")?.getattr(cls)?;
        Ok(PyErr::from_value_bound(c.call1((msg,))?))
    };
    build().unwrap_or_else(|e| e)
}

/// `CoreError -> PyErr`, matching `Collection`'s historical exception
/// types exactly (`InvalidArgument` never raised a `turbovecdb.errors.*`
/// class, always a plain `ValueError`; `Runtime`/`Sql`/`Io` likewise always
/// raised a plain `RuntimeError`).
pub(crate) fn core_err_to_py(py: Python<'_>, e: CoreError) -> PyErr {
    match e {
        CoreError::DimensionMismatch(m) => turbovec_error(py, "DimensionMismatchError", m),
        CoreError::EmbedderRequired(m) => turbovec_error(py, "EmbedderRequiredError", m),
        CoreError::EmbedderIdentityMismatch(m) => {
            turbovec_error(py, "EmbedderIdentityMismatchError", m)
        }
        CoreError::EmbedderMismatch(m) => {
            turbovec_error(py, "EmbedderMismatchError", m)
        }
        CoreError::UnsupportedFilter(m) => turbovec_error(py, "UnsupportedFilterError", m),
        CoreError::CollectionNotFound(m) => turbovec_error(py, "CollectionNotFoundError", m),
        CoreError::CorruptedMetadata(m) => turbovec_error(py, "CorruptedMetadataError", m),
        CoreError::Other(m) => turbovec_error(py, "TurboVecError", m),
        // The cross-process write-lock timeout — historically a plain
        // `TurboVecError` raised by the Python wrapper's `_locked()`.
        CoreError::LockTimeout(m) => turbovec_error(py, "TurboVecError", m),
        CoreError::ConnectionClosed(m) => turbovec_error(py, "TurboVecError", m),
        CoreError::CorruptedTvim(m) => turbovec_error(py, "TurboVecError", m),
        CoreError::InvalidArgument(m) => PyValueError::new_err(m),
        CoreError::Runtime(m) => PyRuntimeError::new_err(m),
        CoreError::Sql(err) => PyRuntimeError::new_err(format!("sqlite error: {err}")),
        CoreError::Io(err) => PyRuntimeError::new_err(err.to_string()),
    }
}

/// Raw JSON text (or `""` for "no metadata") -> a Python dict, matching the
/// historical `if meta.is_empty() { {} } else { json.loads(meta) }` pattern
/// used for both `QueryResult` and `GetResult`.
fn metadata_to_py<'py>(py: Python<'py>, meta: &str) -> PyResult<Bound<'py, PyAny>> {
    if meta.is_empty() {
        Ok(PyDict::new_bound(py).into_any())
    } else {
        py.import_bound("json")?.getattr("loads")?.call1((meta,))
    }
}

fn vectors_to_py<'py>(py: Python<'py>, vectors: &Option<Vec<Vec<f32>>>) -> PyResult<PyObject> {
    match vectors {
        Some(rows) => {
            let out = PyList::empty_bound(py);
            for row in rows {
                let vlist = PyList::empty_bound(py);
                for f in row {
                    vlist.append(*f as f64)?;
                }
                out.append(vlist)?;
            }
            Ok(out.into_any().unbind())
        }
        None => Ok(py.None()),
    }
}

pub(crate) fn query_result_to_py(py: Python<'_>, r: QueryResult) -> PyResult<PyObject> {
    let metadatas = r
        .metadatas
        .iter()
        .map(|m| metadata_to_py(py, m))
        .collect::<PyResult<Vec<_>>>()?;
    let vectors = vectors_to_py(py, &r.vectors)?;
    let qr = py.import_bound("turbovecdb.collection")?.getattr("QueryResult")?;
    Ok(qr
        .call1((r.ids, r.distances, r.documents, metadatas, vectors))?
        .unbind())
}

pub(crate) fn get_result_to_py(py: Python<'_>, r: GetResult) -> PyResult<PyObject> {
    let metadatas = r
        .metadatas
        .iter()
        .map(|m| metadata_to_py(py, m))
        .collect::<PyResult<Vec<_>>>()?;
    let vectors = vectors_to_py(py, &r.vectors)?;
    let gr = py.import_bound("turbovecdb.collection")?.getattr("GetResult")?;
    Ok(gr.call1((r.ids, r.documents, metadatas, vectors))?.unbind())
}

pub(crate) fn health_result_to_py(py: Python<'_>, r: HealthResult) -> PyResult<PyObject> {
    let hr = py.import_bound("turbovecdb.collection")?.getattr("HealthResult")?;
    Ok(hr
        .call1((r.ok, r.quick_check, r.store_gen, r.tvim_gen, r.coherent, r.doc_count, r.wal_size_bytes))?
        .unbind())
}

pub(crate) fn query_batch_result_to_py(py: Python<'_>, results: Vec<QueryResult>) -> PyResult<PyObject> {
    let list = PyList::empty_bound(py);
    for r in results {
        let item = query_result_to_py(py, r)?;
        list.append(item)?;
    }
    Ok(list.into_any().unbind())
}

pub(crate) fn reembed_report_to_py(py: Python<'_>, r: ReembedReport) -> PyResult<PyObject> {
    let rr = py.import_bound("turbovecdb.collection")?.getattr("ReembedReport")?;
    Ok(rr
        .call1((r.total_docs, r.old_dim, r.new_dim, r.skipped, r.elapsed_seconds))?
        .unbind())
}
