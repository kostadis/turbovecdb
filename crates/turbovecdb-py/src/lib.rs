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

use numpy::ndarray::{Array2, Axis, Ix2};
use numpy::{PyArray2, PyReadonlyArrayDyn, ToPyArray};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

mod collection;

create_exception!(_core, FilterError, PyException);

const MAX_DEPTH: i32 = 10;
const MAX_IN_LIST: usize = 900;

/// Map a comparison operator to its SQL symbol.
fn cmp_symbol(op: &str) -> Option<&'static str> {
    match op {
        "$eq" => Some("="),
        "$ne" => Some("!="),
        "$gt" => Some(">"),
        "$gte" => Some(">="),
        "$lt" => Some("<"),
        "$lte" => Some("<="),
        _ => None,
    }
}

/// Coerce a Python object that is expected to be a non-empty list/tuple into a
/// vector of its elements, or `None` if it is neither.
fn as_sequence<'py>(obj: &Bound<'py, PyAny>) -> Option<Vec<Bound<'py, PyAny>>> {
    if let Ok(list) = obj.downcast::<PyList>() {
        Some(list.iter().collect())
    } else if let Ok(tuple) = obj.downcast::<PyTuple>() {
        Some(tuple.iter().collect())
    } else {
        None
    }
}

/// LIKE-escape backslash, percent, and underscore (order matters: backslash first).
fn like_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('%', "\\%").replace('_', "\\_")
}

/// Compile a single `field: value` predicate. Pushes its bound params (path
/// first, then operands) onto `params` and returns the SQL fragment.
fn field_clause(
    py: Python<'_>,
    field: &str,
    value: &Bound<'_, PyAny>,
    params: &mut Vec<PyObject>,
) -> PyResult<String> {
    let path = format!("$.{}", field);

    if let Ok(dict) = value.downcast::<PyDict>() {
        if dict.len() != 1 {
            return Err(FilterError::new_err(format!(
                "field '{}' predicate must have exactly one operator",
                field
            )));
        }
        let (key, operand) = dict.iter().next().unwrap();
        let op = key.extract::<String>().unwrap_or_default();

        if op == "$in" || op == "$nin" {
            let items = as_sequence(&operand).filter(|v| !v.is_empty()).ok_or_else(|| {
                FilterError::new_err(format!("{} on '{}' requires a non-empty list", op, field))
            })?;
            if items.len() > MAX_IN_LIST {
                return Err(FilterError::new_err(format!(
                    "{} on '{}': list length {} exceeds maximum of {}",
                    op,
                    field,
                    items.len(),
                    MAX_IN_LIST
                )));
            }
            let placeholders = vec!["?"; items.len()].join(",");
            let negate = if op == "$nin" { "NOT " } else { "" };
            params.push(path.to_object(py));
            for item in items {
                params.push(item.unbind());
            }
            return Ok(format!(
                "json_extract(metadata, ?) {}IN ({})",
                negate, placeholders
            ));
        }

        if let Some(sym) = cmp_symbol(&op) {
            params.push(path.to_object(py));
            params.push(operand.unbind());
            return Ok(format!("json_extract(metadata, ?) {} ?", sym));
        }

        return Err(FilterError::new_err(format!(
            "unsupported operator '{}' on field '{}'",
            op, field
        )));
    }

    // Bare scalar → equality.
    params.push(path.to_object(py));
    params.push(value.clone().unbind());
    Ok("json_extract(metadata, ?) = ?".to_string())
}

/// Core recursive compiler for a `where` mapping.
pub(crate) fn compile_where(
    py: Python<'_>,
    where_obj: &Bound<'_, PyAny>,
    depth: i32,
) -> PyResult<(String, Vec<PyObject>)> {
    if depth > MAX_DEPTH {
        return Err(FilterError::new_err(format!(
            "filter nesting depth exceeds maximum of {}",
            MAX_DEPTH
        )));
    }
    if where_obj.is_none() {
        return Ok((String::new(), Vec::new()));
    }
    let dict = where_obj
        .downcast::<PyDict>()
        .map_err(|_| FilterError::new_err("where must be a mapping"))?;
    if dict.is_empty() {
        return Ok((String::new(), Vec::new()));
    }

    let has_and = dict.contains("$and")?;
    let has_or = dict.contains("$or")?;
    if has_and || has_or {
        if dict.len() != 1 {
            return Err(FilterError::new_err(
                "a logical operator ($and/$or) cannot have sibling keys",
            ));
        }
        let (key, subs) = dict.iter().next().unwrap();
        let op = key.extract::<String>().unwrap_or_default();
        let sub_items = as_sequence(&subs).filter(|v| !v.is_empty()).ok_or_else(|| {
            FilterError::new_err(format!("{} requires a non-empty list of clauses", op))
        })?;
        let joiner = if op == "$and" { " AND " } else { " OR " };
        let mut frags: Vec<String> = Vec::new();
        let mut params: Vec<PyObject> = Vec::new();
        for sub in sub_items {
            let (frag, sub_params) = compile_where(py, &sub, depth + 1)?;
            if !frag.is_empty() {
                frags.push(format!("({})", frag));
                params.extend(sub_params);
            }
        }
        return Ok((frags.join(joiner), params));
    }

    // Reject any other top-level operator ($not, $nor, ...).
    for key in dict.keys() {
        if let Ok(s) = key.extract::<String>() {
            if s.starts_with('$') {
                return Err(FilterError::new_err(format!(
                    "unsupported top-level operator '{}'",
                    s
                )));
            }
        }
    }

    let mut frags: Vec<String> = Vec::new();
    let mut params: Vec<PyObject> = Vec::new();
    for (key, value) in dict.iter() {
        let field = key.str()?.to_string();
        frags.push(field_clause(py, &field, &value, &mut params)?);
    }
    Ok((frags.join(" AND "), params))
}

/// Core compiler for a `where_document` mapping (only `$contains`).
pub(crate) fn compile_where_document(
    py: Python<'_>,
    wd: &Bound<'_, PyAny>,
) -> PyResult<(String, Vec<PyObject>)> {
    if wd.is_none() {
        return Ok((String::new(), Vec::new()));
    }
    let dict = wd
        .downcast::<PyDict>()
        .map_err(|_| FilterError::new_err("where_document must be a mapping"))?;
    if dict.is_empty() {
        return Ok((String::new(), Vec::new()));
    }
    if dict.len() != 1 || !dict.contains("$contains")? {
        return Err(FilterError::new_err(
            "where_document supports only $contains",
        ));
    }
    let needle: String = dict.get_item("$contains")?.unwrap().extract()?;
    let params = vec![format!("%{}%", like_escape(&needle)).to_object(py)];
    Ok(("document LIKE ? ESCAPE '\\'".to_string(), params))
}

fn to_result<'py>(
    py: Python<'py>,
    sql: String,
    params: Vec<PyObject>,
) -> (String, Bound<'py, PyList>) {
    (sql, PyList::new_bound(py, &params))
}

#[pyfunction]
#[pyo3(signature = (where_obj, _depth=0))]
fn where_to_sql<'py>(
    py: Python<'py>,
    where_obj: &Bound<'py, PyAny>,
    _depth: i32,
) -> PyResult<(String, Bound<'py, PyList>)> {
    let (sql, params) = compile_where(py, where_obj, _depth)?;
    Ok(to_result(py, sql, params))
}

#[pyfunction]
fn where_document_to_sql<'py>(
    py: Python<'py>,
    where_document: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyList>)> {
    let (sql, params) = compile_where_document(py, where_document)?;
    Ok(to_result(py, sql, params))
}

#[pyfunction]
fn combined_sql<'py>(
    py: Python<'py>,
    where_obj: &Bound<'py, PyAny>,
    where_document: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyList>)> {
    let mut frags: Vec<String> = Vec::new();
    let mut params: Vec<PyObject> = Vec::new();
    for (frag, p) in [
        compile_where(py, where_obj, 0)?,
        compile_where_document(py, where_document)?,
    ] {
        if !frag.is_empty() {
            frags.push(format!("({})", frag));
            params.extend(p);
        }
    }
    Ok(to_result(py, frags.join(" AND "), params))
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

/// Shared implementation, reused by the Rust `Collection` write path.
pub(crate) fn l2_normalize_impl<'py>(
    py: Python<'py>,
    matrix: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    // Coerce arbitrary array-like input to a float32 array, exactly like the
    // former `np.asarray(matrix, dtype=np.float32)`.
    let np = py.import_bound("numpy")?;
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("dtype", np.getattr("float32")?)?;
    let arr_any = np.getattr("asarray")?.call((matrix,), Some(&kwargs))?;
    let readonly: PyReadonlyArrayDyn<f32> = arr_any.extract()?;
    let view = readonly.as_array();

    let mut out: Array2<f32> = match view.ndim() {
        1 => {
            let n = view.len();
            view.to_owned()
                .into_shape_with_order((1, n))
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        }
        2 => view
            .to_owned()
            .into_dimensionality::<Ix2>()
            .map_err(|e| PyValueError::new_err(e.to_string()))?,
        other => {
            return Err(PyValueError::new_err(format!(
                "l2_normalize expects a 1-D or 2-D array, got {}-D",
                other
            )))
        }
    };

    for mut row in out.axis_iter_mut(Axis(0)) {
        let norm = row.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let denom = if norm == 0.0 { 1.0 } else { norm };
        row.mapv_inplace(|x| x / denom);
    }

    Ok(out.to_pyarray_bound(py))
}

// ── turbovec index lifecycle ────────────────────────────────────────────────
//
// turbovec ships only as a PyO3 Python extension (no Rust crate), so the core
// drives its `IdMapIndex` through PyO3: we construct / load / persist the index
// and hand the Python object back to the caller, which keeps calling
// `.add_with_ids` / `.remove` / `.search` on it directly. This establishes the
// turbovec calling pattern the later query/reembed slices depend on.

/// `turbovec.IdMapIndex(dim=dim, bit_width=bit_width)`.
#[pyfunction]
fn new_index<'py>(
    py: Python<'py>,
    dim: &Bound<'py, PyAny>,
    bit_width: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let turbovec = py.import_bound("turbovec")?;
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("dim", dim)?;
    kwargs.set_item("bit_width", bit_width)?;
    turbovec.getattr("IdMapIndex")?.call((), Some(&kwargs))
}

/// Build an index from parallel `uids` (1-D) and `vecs` (list of 1-D).
#[pyfunction]
fn build_index<'py>(
    py: Python<'py>,
    dim: &Bound<'py, PyAny>,
    bit_width: &Bound<'py, PyAny>,
    uids: &Bound<'py, PyAny>,
    vecs: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let turbovec = py.import_bound("turbovec")?;
    let kwargs = PyDict::new_bound(py);
    kwargs.set_item("dim", dim)?;
    kwargs.set_item("bit_width", bit_width)?;
    let idx = turbovec.getattr("IdMapIndex")?.call((), Some(&kwargs))?;
    if uids.len()? > 0 {
        let np = py.import_bound("numpy")?;
        let stacked = np.getattr("stack")?.call1((vecs,))?;
        let ckw = PyDict::new_bound(py);
        ckw.set_item("dtype", np.getattr("float32")?)?;
        let mat = np.getattr("ascontiguousarray")?.call((stacked,), Some(&ckw))?;
        let ukw = PyDict::new_bound(py);
        ukw.set_item("dtype", np.getattr("uint64")?)?;
        let uid_arr = np.getattr("asarray")?.call((uids,), Some(&ukw))?;
        idx.call_method1("add_with_ids", (mat, uid_arr))?;
    }
    Ok(idx)
}

/// `turbovec.IdMapIndex.load(path)`.
#[pyfunction]
fn load_index<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyAny>> {
    let turbovec = py.import_bound("turbovec")?;
    turbovec.getattr("IdMapIndex")?.getattr("load")?.call1((path,))
}

/// Serialize `index` to `path` atomically (temp file + os.replace).
#[pyfunction]
fn write_index_atomic(py: Python<'_>, index: &Bound<'_, PyAny>, path: &str) -> PyResult<()> {
    let tmp = format!("{}.tmp", path);
    index.call_method1("write", (tmp.as_str(),))?;
    py.import_bound("os")?
        .getattr("replace")?
        .call1((tmp.as_str(), path))?;
    Ok(())
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("FilterError", m.py().get_type_bound::<FilterError>())?;
    m.add_function(wrap_pyfunction!(where_to_sql, m)?)?;
    m.add_function(wrap_pyfunction!(where_document_to_sql, m)?)?;
    m.add_function(wrap_pyfunction!(combined_sql, m)?)?;
    m.add_function(wrap_pyfunction!(l2_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(new_index, m)?)?;
    m.add_function(wrap_pyfunction!(build_index, m)?)?;
    m.add_function(wrap_pyfunction!(load_index, m)?)?;
    m.add_function(wrap_pyfunction!(write_index_atomic, m)?)?;
    m.add_class::<collection::Collection>()?;
    Ok(())
}
