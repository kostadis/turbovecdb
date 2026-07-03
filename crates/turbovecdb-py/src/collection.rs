//! `Collection` — thin PyO3 wrapper around
//! `turbovecdb_core::collection::Collection<PyEmbedder, TurbovecIndex>`.
//!
//! All business logic (SQLite reads/writes, generation bookkeeping, reembed)
//! lives in `turbovecdb-core`; this file's only job is converting PyO3
//! arguments into core types, calling core, and converting the result back
//! (via `convert.rs`, the one place that touches Python objects by name).

use numpy::ndarray::Array2;
use pyo3::prelude::*;
use turbovecdb_core::collection::Collection as CoreCollection;
use turbovecdb_core::error::CoreError;
use turbovecdb_core::index::TurbovecIndex;

use crate::convert;
use crate::embedder::PyEmbedder;

pub const DEFAULT_BIT_WIDTH: i64 = turbovecdb_core::collection::DEFAULT_BIT_WIDTH;

fn json_dumps(py: Python<'_>, obj: &PyObject) -> PyResult<String> {
    py.import_bound("json")?.getattr("dumps")?.call1((obj,))?.extract::<String>()
}

fn where_to_json(py: Python<'_>, w: &Option<PyObject>) -> PyResult<Option<serde_json::Value>> {
    w.as_ref().map(|v| crate::py_to_json(v.bind(py))).transpose()
}

fn vectors_to_array(py: Python<'_>, v: &Option<PyObject>) -> PyResult<Option<Array2<f32>>> {
    v.as_ref().map(|v| crate::coerce_array2(py, v.bind(py))).transpose()
}

/// Dump each metadata item as-is (no truthy check — that's only applied by
/// `update_metadata`); `None` (the whole argument omitted) is left as
/// `None`, and `Collection::write` defaults each row to `"{}"`.
fn metadatas_to_json(py: Python<'_>, m: Option<Vec<PyObject>>) -> PyResult<Option<Vec<String>>> {
    m.map(|list| list.iter().map(|item| json_dumps(py, item)).collect::<PyResult<Vec<String>>>())
        .transpose()
}

#[pyclass]
pub struct Collection {
    inner: CoreCollection<PyEmbedder, TurbovecIndex>,
}

#[pymethods]
impl Collection {
    #[new]
    #[pyo3(signature = (coll_dir, dim=None, bit_width=DEFAULT_BIT_WIDTH, metric=None, embedder=None, lock_timeout=30.0))]
    fn new(
        coll_dir: String,
        dim: Option<i64>,
        bit_width: i64,
        metric: Option<String>,
        embedder: Option<PyObject>,
        lock_timeout: f64,
    ) -> PyResult<Self> {
        Python::with_gil(|py| {
            let inner = CoreCollection::new(
                coll_dir,
                dim,
                bit_width,
                metric,
                embedder.map(PyEmbedder::new),
                lock_timeout,
            )
            .map_err(|e| convert::core_err_to_py(py, e))?;
            Ok(Collection { inner })
        })
    }

    #[getter]
    fn dir(&self) -> &str {
        self.inner.dir()
    }

    #[getter]
    fn dim(&self) -> Option<i64> {
        self.inner.dim()
    }

    #[getter]
    fn bit_width(&self) -> i64 {
        self.inner.bit_width()
    }

    #[getter]
    fn tvim_path(&self) -> &str {
        self.inner.tvim_path()
    }

    fn count(&self) -> PyResult<i64> {
        Python::with_gil(|py| self.inner.count().map_err(|e| convert::core_err_to_py(py, e)))
    }

    fn store_gen(&self) -> PyResult<i64> {
        Python::with_gil(|py| self.inner.store_gen().map_err(|e| convert::core_err_to_py(py, e)))
    }

    #[pyo3(name = "meta_get")]
    fn py_meta_get(&self, key: &str) -> PyResult<Option<String>> {
        Python::with_gil(|py| self.inner.meta_get(key).map_err(|e| convert::core_err_to_py(py, e)))
    }

    fn flush(&mut self, py: Python<'_>) -> PyResult<()> {
        let inner = &mut self.inner;
        py.allow_threads(|| inner.flush()).map_err(|e| convert::core_err_to_py(py, e))
    }

    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        let inner = &mut self.inner;
        py.allow_threads(|| inner.close()).map_err(|e| convert::core_err_to_py(py, e))
    }

    fn health(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        // &mut, not &self: rusqlite::Connection holds RefCell-based statement
        // caches that aren't Sync, so a shared reference can't cross the
        // allow_threads boundary (health()'s core method only needs &self;
        // this is purely about satisfying allow_threads' Send bound).
        let inner = &mut self.inner;
        // `move`: without it, Rust's disjoint-closure-capture analysis
        // captures `*inner` by shared reference (since health() only needs
        // &self), which would require Collection<..> to be Sync (it isn't —
        // rusqlite::Connection's caches use RefCell). `move` forces
        // capturing the outer `&mut` binding itself instead.
        let r = py.allow_threads(move || inner.health()).map_err(|e| convert::core_err_to_py(py, e))?;
        convert::health_result_to_py(py, r)
    }

    #[pyo3(signature = (ids, documents=None, metadatas=None, vectors=None))]
    fn add(
        &mut self,
        py: Python<'_>,
        ids: Vec<String>,
        documents: Option<Vec<String>>,
        metadatas: Option<Vec<PyObject>>,
        vectors: Option<PyObject>,
    ) -> PyResult<()> {
        let metadatas = metadatas_to_json(py, metadatas)?;
        let vectors = vectors_to_array(py, &vectors)?;
        // Release the GIL for the actual write: SQLite I/O + index
        // quantization/mirroring don't touch Python, so holding the GIL for
        // their duration needlessly blocks every other Python thread in the
        // process (R3). The embedder itself now runs in the Python wrapper
        // (collection.py) before this call, outside the cross-process write
        // lock — if it were still invoked from here, PyEmbedder::embed's own
        // Python::with_gil would simply re-acquire, which is sound but
        // wasteful; add()/upsert() no longer take that path in practice.
        let inner = &mut self.inner;
        py.allow_threads(|| inner.add(ids, documents, metadatas, vectors))
            .map_err(|e| convert::core_err_to_py(py, e))
    }

    #[pyo3(signature = (ids, documents=None, metadatas=None, vectors=None))]
    fn upsert(
        &mut self,
        py: Python<'_>,
        ids: Vec<String>,
        documents: Option<Vec<String>>,
        metadatas: Option<Vec<PyObject>>,
        vectors: Option<PyObject>,
    ) -> PyResult<()> {
        let metadatas = metadatas_to_json(py, metadatas)?;
        let vectors = vectors_to_array(py, &vectors)?;
        let inner = &mut self.inner;
        py.allow_threads(|| inner.upsert(ids, documents, metadatas, vectors))
            .map_err(|e| convert::core_err_to_py(py, e))
    }

    fn clear(&mut self, py: Python<'_>) -> PyResult<()> {
        let inner = &mut self.inner;
        py.allow_threads(|| inner.clear()).map_err(|e| convert::core_err_to_py(py, e))
    }

    #[pyo3(signature = (ids=None, where_=None))]
    fn delete(&mut self, py: Python<'_>, ids: Option<Vec<String>>, where_: Option<PyObject>) -> PyResult<()> {
        let where_json = where_to_json(py, &where_)?;
        let inner = &mut self.inner;
        py.allow_threads(|| inner.delete(ids, where_json.as_ref()))
            .map_err(|e| convert::core_err_to_py(py, e))
    }

    fn update_metadata(&mut self, py: Python<'_>, ids: Vec<String>, metadatas: Vec<PyObject>) -> PyResult<()> {
        let dumped: PyResult<Vec<String>> = metadatas
            .iter()
            .map(|m| {
                let truthy = m.bind(py).is_truthy().unwrap_or(true);
                if truthy {
                    json_dumps(py, m)
                } else {
                    Ok("{}".to_string())
                }
            })
            .collect();
        let dumped = dumped?;
        let inner = &mut self.inner;
        py.allow_threads(|| inner.update_metadata(ids, dumped))
            .map_err(|e| convert::core_err_to_py(py, e))
    }

    fn update_documents(&mut self, py: Python<'_>, ids: Vec<String>, documents: Vec<String>) -> PyResult<()> {
        let inner = &mut self.inner;
        py.allow_threads(|| inner.update_documents(ids, documents))
            .map_err(|e| convert::core_err_to_py(py, e))
    }

    #[pyo3(signature = (text=None, vector=None, k=10, where_=None, where_document=None, include=None))]
    fn query(
        &mut self,
        py: Python<'_>,
        text: Option<String>,
        vector: Option<PyObject>,
        k: usize,
        where_: Option<PyObject>,
        where_document: Option<PyObject>,
        include: Option<Vec<String>>,
    ) -> PyResult<PyObject> {
        let vector_arr = vectors_to_array(py, &vector)?;
        let where_json = where_to_json(py, &where_)?;
        let wd_json = where_to_json(py, &where_document)?;
        // text queries still call the embedder internally (query() applies
        // the GAP-1 guard itself, C5) — allow_threads is still sound here:
        // PyEmbedder::embed re-acquires the GIL itself when it needs it.
        let inner = &mut self.inner;
        let r = py
            .allow_threads(|| {
                inner.query(text, vector_arr, k, where_json.as_ref(), wd_json.as_ref(), include.as_deref())
            })
            .map_err(|e| convert::core_err_to_py(py, e))?;
        convert::query_result_to_py(py, r)
    }

    #[pyo3(signature = (ids=None, where_=None, where_document=None, limit=None, offset=None, include=None))]
    fn get(
        &mut self,
        py: Python<'_>,
        ids: Option<Vec<String>>,
        where_: Option<PyObject>,
        where_document: Option<PyObject>,
        limit: Option<i64>,
        offset: Option<i64>,
        include: Option<Vec<String>>,
    ) -> PyResult<PyObject> {
        let where_json = where_to_json(py, &where_)?;
        let wd_json = where_to_json(py, &where_document)?;
        // &mut + `move`: see health() above for why.
        let inner = &mut self.inner;
        let r = py
            .allow_threads(move || {
                inner.get(ids, where_json.as_ref(), wd_json.as_ref(), limit, offset, include.as_deref())
            })
            .map_err(|e| convert::core_err_to_py(py, e))?;
        convert::get_result_to_py(py, r)
    }

    #[pyo3(signature = (embedder, dim=None, bit_width=None, batch_size=256, on_progress=None, skip_empty=None))]
    fn reembed(
        &mut self,
        py: Python<'_>,
        embedder: PyObject,
        dim: Option<i64>,
        bit_width: Option<i64>,
        batch_size: usize,
        on_progress: Option<PyObject>,
        skip_empty: Option<String>,
    ) -> PyResult<PyObject> {
        if !embedder.bind(py).is_callable() {
            return Err(pyo3::exceptions::PyValueError::new_err("embedder must be a callable"));
        }
        let skip_empty = skip_empty.unwrap_or_else(|| "error".to_string());
        let core_embedder = PyEmbedder::new(embedder);

        let mut progress_err: Option<PyErr> = None;
        let mut progress_cb = |processed: i64, total: i64| -> Result<(), CoreError> {
            if let Some(cb) = &on_progress {
                if let Err(e) = cb.bind(py).call1((processed, total)) {
                    let msg = e.to_string();
                    progress_err = Some(e);
                    return Err(CoreError::Other(msg));
                }
            }
            Ok(())
        };
        let progress_ref: Option<&mut dyn FnMut(i64, i64) -> Result<(), CoreError>> =
            Some(&mut progress_cb);

        let result = self.inner.reembed(
            core_embedder,
            dim,
            bit_width,
            batch_size,
            progress_ref,
            &skip_empty,
        );
        match result {
            Ok(r) => convert::reembed_report_to_py(py, r),
            Err(_) if progress_err.is_some() => Err(progress_err.unwrap()),
            Err(e) => Err(convert::core_err_to_py(py, e)),
        }
    }
}
