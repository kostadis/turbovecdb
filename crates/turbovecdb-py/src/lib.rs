//! Rust core for turbovecdb — first slice: the metadata/document filter compiler.
//!
//! This is a faithful port of the former pure-Python `turbovecdb.filters`. It
//! translates Chroma/Mongo-style `where` / `where_document` dicts into a
//! parameterised SQL fragment `(sql, params)` over the JSON `metadata` column.
//! The JSON path and every operand are returned as *bound parameters* (never
//! interpolated), so arbitrary field names and values cannot inject SQL.
//!
//! Unsupported input raises `turbovecdb._core.FilterError`; the thin Python
//! shim in `turbovecdb.filters` re-raises it as `UnsupportedFilterError` so the
//! public contract (and every existing test) is unchanged.

use numpy::ndarray::{Array2, Ix2};
use numpy::{PyArray2, PyReadonlyArrayDyn, ToPyArray};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pythonize::{depythonize, pythonize};
use serde_json::Value as Json;
use turbovecdb_core::filters;

mod collection;
mod convert;
mod embedder;

create_exception!(_core, FilterError, PyException);

/// `PyAny -> serde_json::Value`, treating a Python `None` object as JSON
/// null (matches the historical PyAny-based compiler's `is_none()` checks).
pub(crate) fn py_to_json(obj: &Bound<'_, PyAny>) -> PyResult<Json> {
    depythonize(obj).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn json_to_py<'py>(py: Python<'py>, value: &Json) -> PyResult<Bound<'py, PyAny>> {
    pythonize(py, value).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// The core compiler's `FilterError` has no Python dependency; raise it here
/// as `_core.FilterError` so `turbovecdb.filters`'s catch-and-reraise into
/// `UnsupportedFilterError` is unchanged.
pub(crate) fn map_core_filter_err(e: filters::FilterError) -> PyErr {
    FilterError::new_err(e.0)
}

fn to_result<'py>(py: Python<'py>, sql: String, params: Vec<Json>) -> PyResult<(String, Bound<'py, PyList>)> {
    let items = params
        .iter()
        .map(|v| json_to_py(py, v))
        .collect::<PyResult<Vec<_>>>()?;
    Ok((sql, PyList::new_bound(py, &items)))
}

#[pyfunction]
#[pyo3(signature = (where_obj, _depth=0))]
fn where_to_sql<'py>(
    py: Python<'py>,
    where_obj: &Bound<'py, PyAny>,
    _depth: i32,
) -> PyResult<(String, Bound<'py, PyList>)> {
    let where_json = py_to_json(where_obj)?;
    let (sql, params) = filters::compile_where(&where_json, _depth).map_err(map_core_filter_err)?;
    to_result(py, sql, params)
}

#[pyfunction]
fn where_document_to_sql<'py>(
    py: Python<'py>,
    where_document: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyList>)> {
    let wd_json = py_to_json(where_document)?;
    let (sql, params) = filters::compile_where_document(&wd_json).map_err(map_core_filter_err)?;
    to_result(py, sql, params)
}

#[pyfunction]
fn combined_sql<'py>(
    py: Python<'py>,
    where_obj: &Bound<'py, PyAny>,
    where_document: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyList>)> {
    let where_json = py_to_json(where_obj)?;
    let wd_json = py_to_json(where_document)?;
    let mut frags: Vec<String> = Vec::new();
    let mut params: Vec<Json> = Vec::new();
    for (frag, p) in [
        filters::compile_where(&where_json, 0).map_err(map_core_filter_err)?,
        filters::compile_where_document(&wd_json).map_err(map_core_filter_err)?,
    ] {
        if !frag.is_empty() {
            frags.push(format!("({})", frag));
            params.extend(p);
        }
    }
    to_result(py, frags.join(" AND "), params)
}

/// Row-wise L2 normalization → C-contiguous float32, with a zero guard.
///
/// Faithful port of the former `turbovecdb.index.l2_normalize`: accepts any
/// array-like (list, 1-D or 2-D array), returns a 2-D C-contiguous float32
/// array (1-D input is treated as a single row). Rows with zero norm are left
/// unchanged (divided by 1.0) rather than producing NaNs.
#[pyfunction]
fn l2_normalize<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    l2_normalize_impl(py, matrix)
}

/// Coerce arbitrary array-like Python input (list, 1-D or 2-D array) into an
/// owned 2-D float32 `ndarray` — the numpy-interop half of `l2_normalize`.
/// The actual normalization math is pure Rust, in
/// `turbovecdb_core::vecmath::l2_normalize`; this is reused directly by
/// `Collection`'s add/upsert/query paths for raw (non-embedder) vectors.
pub(crate) fn coerce_array2(py: Python<'_>, matrix: &Bound<'_, PyAny>) -> PyResult<Array2<f32>> {
    let np = py.import_bound("numpy")?;
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("dtype", np.getattr("float32")?)?;
    let arr_any = np.getattr("asarray")?.call((matrix,), Some(&kwargs))?;
    let readonly: PyReadonlyArrayDyn<f32> = arr_any.extract()?;
    let view = readonly.as_array();

    match view.ndim() {
        1 => {
            let n = view.len();
            view.to_owned()
                .into_shape_with_order((1, n))
                .map_err(|e| PyValueError::new_err(e.to_string()))
        }
        2 => view
            .to_owned()
            .into_dimensionality::<Ix2>()
            .map_err(|e| PyValueError::new_err(e.to_string())),
        other => Err(PyValueError::new_err(format!(
            "l2_normalize expects a 1-D or 2-D array, got {}-D",
            other
        ))),
    }
}

/// Shared implementation backing the public `_core.l2_normalize` pyfunction
/// and `PyEmbedder`'s output normalization.
pub(crate) fn l2_normalize_impl<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let mut out = coerce_array2(py, matrix)?;
    turbovecdb_core::vecmath::l2_normalize(&mut out);
    Ok(out.to_pyarray_bound(py))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("FilterError", m.py().get_type_bound::<FilterError>())?;
    m.add_function(wrap_pyfunction!(where_to_sql, m)?)?;
    m.add_function(wrap_pyfunction!(where_document_to_sql, m)?)?;
    m.add_function(wrap_pyfunction!(combined_sql, m)?)?;
    m.add_function(wrap_pyfunction!(l2_normalize, m)?)?;
    m.add_class::<collection::Collection>()?;
    Ok(())
}
