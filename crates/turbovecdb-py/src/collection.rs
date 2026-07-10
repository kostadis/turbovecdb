//! `Collection` — thin PyO3 wrapper around
//! `turbovecdb_core::collection::Collection<PyEmbedder, TurbovecIndex>`.
//!
//! All business logic (SQLite reads/writes, generation bookkeeping, reembed,
//! *and now all locking* — the cross-process flock and, via the `Mutex`
//! below, the in-process serialization) lives in `turbovecdb-core`; this
//! file's only job is converting PyO3 arguments into core types, calling
//! core, and converting the result back (via `convert.rs`).
//!
//! ## Locking discipline (invariant I1 — GIL/Mutex ordering)
//!
//! The in-process lock is the `Mutex` wrapping the core collection. It must
//! be acquired **only inside `py.allow_threads`**, never while holding the
//! GIL. Otherwise: thread A takes the mutex inside `allow_threads` and, to
//! run a Python embedder, re-acquires the GIL; thread B holds the GIL and
//! blocks on the mutex — classic deadlock. Every `#[pymethods]` therefore
//! (1) converts arguments while holding the GIL, (2) drops into
//! `allow_threads`, locks the mutex, and calls core, then (3) converts the
//! result back after `allow_threads` has re-acquired the GIL. The mutex is
//! taken *before* the core's cross-process flock (invariant I2). Because the
//! mutex is held across `allow_threads`, all methods can be `&self` — which
//! also retires the PyO3 `&mut`-borrow-check hazard the old `_tlock` papered
//! over.

use numpy::ndarray::Array2;
use pyo3::prelude::*;
use std::sync::Mutex;
use turbovecdb_core::collection::Collection as CoreCollection;
use turbovecdb_core::embedder::Embedder;
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

/// The CoreError raised when the in-process mutex is poisoned — i.e. a prior
/// operation panicked while holding it. We never `.unwrap()` a poisoned lock
/// (that would abort the interpreter); it maps to a `TurboVecError`.
fn lock_poisoned() -> CoreError {
    CoreError::Other(
        "collection is unusable: a previous operation panicked while holding its lock".to_string(),
    )
}

#[pyclass]
pub struct Collection {
    inner: Mutex<CoreCollection<PyEmbedder, TurbovecIndex>>,
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
            // Release the GIL: constructing a collection does SQLite I/O and,
            // for a brand-new collection, may block on the cross-process
            // write lock (C7). The embedder's identity() call inside core
            // re-acquires the GIL itself (I1).
            let inner = py
                .allow_threads(|| {
                    CoreCollection::new(coll_dir, dim, bit_width, metric, embedder.map(PyEmbedder::new), lock_timeout)
                })
                .map_err(|e| convert::core_err_to_py(py, e))?;
            Ok(Collection { inner: Mutex::new(inner) })
        })
    }

    #[getter]
    fn dir(&self, py: Python<'_>) -> PyResult<String> {
        let inner = &self.inner;
        py.allow_threads(|| Ok::<_, CoreError>(inner.lock().map_err(|_| lock_poisoned())?.dir().to_string()))
            .map_err(|e| convert::core_err_to_py(py, e))
    }

    #[getter]
    fn dim(&self, py: Python<'_>) -> PyResult<Option<i64>> {
        let inner = &self.inner;
        py.allow_threads(|| inner.lock().map_err(|_| lock_poisoned())?.dim())
            .map_err(|e| convert::core_err_to_py(py, e))
    }

    #[getter]
    fn bit_width(&self, py: Python<'_>) -> PyResult<i64> {
        let inner = &self.inner;
        py.allow_threads(|| {
            let mut guard = inner.lock().map_err(|_| lock_poisoned())?;
            guard.bit_width()
        }).map_err(|e| convert::core_err_to_py(py, e))
    }

    #[getter]
    fn tvim_path(&self, py: Python<'_>) -> PyResult<String> {
        let inner = &self.inner;
        py.allow_threads(|| Ok::<_, CoreError>(inner.lock().map_err(|_| lock_poisoned())?.tvim_path().to_string()))
            .map_err(|e| convert::core_err_to_py(py, e))
    }

    /// Introspection getter for the cross-process write-lock timeout — the
    /// value tests used to read off the wrapper's `FileLock.timeout`.
    #[getter]
    fn lock_timeout(&self, py: Python<'_>) -> PyResult<f64> {
        let inner = &self.inner;
        py.allow_threads(|| Ok::<_, CoreError>(inner.lock().map_err(|_| lock_poisoned())?.lock_timeout()))
            .map_err(|e| convert::core_err_to_py(py, e))
    }

    fn count(&self, py: Python<'_>) -> PyResult<i64> {
        let inner = &self.inner;
        py.allow_threads(|| inner.lock().map_err(|_| lock_poisoned())?.count())
            .map_err(|e| convert::core_err_to_py(py, e))
    }

    fn store_gen(&self, py: Python<'_>) -> PyResult<i64> {
        let inner = &self.inner;
        py.allow_threads(|| inner.lock().map_err(|_| lock_poisoned())?.store_gen())
            .map_err(|e| convert::core_err_to_py(py, e))
    }

    #[pyo3(name = "meta_get")]
    fn py_meta_get(&self, py: Python<'_>, key: &str) -> PyResult<Option<String>> {
        let inner = &self.inner;
        py.allow_threads(|| inner.lock().map_err(|_| lock_poisoned())?.meta_get(key))
            .map_err(|e| convert::core_err_to_py(py, e))
    }

    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        let inner = &self.inner;
        py.allow_threads(|| inner.lock().map_err(|_| lock_poisoned())?.flush())
            .map_err(|e| convert::core_err_to_py(py, e))
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let inner = &self.inner;
        py.allow_threads(|| inner.lock().map_err(|_| lock_poisoned())?.close())
            .map_err(|e| convert::core_err_to_py(py, e))
    }

    fn health(&self, py: Python<'_>) -> PyResult<PyObject> {
        let inner = &self.inner;
        let r = py
            .allow_threads(|| inner.lock().map_err(|_| lock_poisoned())?.health())
            .map_err(|e| convert::core_err_to_py(py, e))?;
        convert::health_result_to_py(py, r)
    }

    /// Open a fresh SQLite connection to the same store file, recovering
    /// from a lost connection (e.g. transient filesystem error).
    fn reconnect(&self, py: Python<'_>) -> PyResult<()> {
        let inner = &self.inner;
        py.allow_threads(|| inner.lock().map_err(|_| lock_poisoned())?.reconnect())
            .map_err(|e| convert::core_err_to_py(py, e))
    }

    #[pyo3(signature = (ids, documents=None, metadatas=None, vectors=None))]
    fn add(
        &self,
        py: Python<'_>,
        ids: Vec<String>,
        documents: Option<Vec<String>>,
        metadatas: Option<Vec<PyObject>>,
        vectors: Option<PyObject>,
    ) -> PyResult<()> {
        let metadatas = metadatas_to_json(py, metadatas)?;
        let vectors = vectors_to_array(py, &vectors)?;
        // The embedder (for the documents= path) runs inside core, under
        // allow_threads but *outside* the cross-process write lock (I5); its
        // PyEmbedder::embed re-acquires the GIL itself.
        let inner = &self.inner;
        py.allow_threads(|| inner.lock().map_err(|_| lock_poisoned())?.add(ids, documents, metadatas, vectors))
            .map_err(|e| convert::core_err_to_py(py, e))
    }

    #[pyo3(signature = (ids, documents=None, metadatas=None, vectors=None))]
    fn upsert(
        &self,
        py: Python<'_>,
        ids: Vec<String>,
        documents: Option<Vec<String>>,
        metadatas: Option<Vec<PyObject>>,
        vectors: Option<PyObject>,
    ) -> PyResult<()> {
        let metadatas = metadatas_to_json(py, metadatas)?;
        let vectors = vectors_to_array(py, &vectors)?;
        let inner = &self.inner;
        py.allow_threads(|| inner.lock().map_err(|_| lock_poisoned())?.upsert(ids, documents, metadatas, vectors))
            .map_err(|e| convert::core_err_to_py(py, e))
    }

    fn clear(&self, py: Python<'_>) -> PyResult<()> {
        let inner = &self.inner;
        py.allow_threads(|| inner.lock().map_err(|_| lock_poisoned())?.clear())
            .map_err(|e| convert::core_err_to_py(py, e))
    }

    #[pyo3(signature = (ids=None, where_=None))]
    fn delete(&self, py: Python<'_>, ids: Option<Vec<String>>, where_: Option<PyObject>) -> PyResult<()> {
        let where_json = where_to_json(py, &where_)?;
        let inner = &self.inner;
        py.allow_threads(|| inner.lock().map_err(|_| lock_poisoned())?.delete(ids, where_json.as_ref()))
            .map_err(|e| convert::core_err_to_py(py, e))
    }

    fn update_metadata(&self, py: Python<'_>, ids: Vec<String>, metadatas: Vec<PyObject>) -> PyResult<()> {
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
        let inner = &self.inner;
        py.allow_threads(|| inner.lock().map_err(|_| lock_poisoned())?.update_metadata(ids, dumped))
            .map_err(|e| convert::core_err_to_py(py, e))
    }

    fn update_documents(&self, py: Python<'_>, ids: Vec<String>, documents: Vec<String>) -> PyResult<()> {
        let inner = &self.inner;
        py.allow_threads(|| inner.lock().map_err(|_| lock_poisoned())?.update_documents(ids, documents))
            .map_err(|e| convert::core_err_to_py(py, e))
    }

    #[pyo3(signature = (text=None, vector=None, k=10, where_=None, where_document=None, include=None))]
    fn query(
        &self,
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
        // the GAP-1 guard itself, C5); PyEmbedder::embed re-acquires the GIL.
        let inner = &self.inner;
        let r = py
            .allow_threads(|| {
                inner
                    .lock()
                    .map_err(|_| lock_poisoned())?
                    .query(text, vector_arr, k, where_json.as_ref(), wd_json.as_ref(), include.as_deref())
            })
            .map_err(|e| convert::core_err_to_py(py, e))?;
        convert::query_result_to_py(py, r)
    }

    #[pyo3(signature = (vectors, k=10, where_=None, where_document=None, include=None))]
    fn query_batch(
        &self,
        py: Python<'_>,
        vectors: Vec<Vec<f32>>,
        k: usize,
        where_: Option<PyObject>,
        where_document: Option<PyObject>,
        include: Option<Vec<String>>,
    ) -> PyResult<PyObject> {
        let maybe_dim: Option<i64> = {
            let inner = &self.inner;
            py.allow_threads(|| -> Result<Option<i64>, CoreError> {
                Ok(inner.lock().map_err(|_| lock_poisoned())?.dim()?)
            })
            .map_err(|e| convert::core_err_to_py(py, e))?
        };
        let mut arrs = Vec::with_capacity(vectors.len());
        if let Some(dim) = maybe_dim {
            for v in &vectors {
                if v.len() as i64 != dim {
                    return Err(convert::core_err_to_py(py, CoreError::InvalidArgument(format!(
                        "vector has dimension {} but collection expects {}",
                        v.len(), dim
                    ))));
                }
                arrs.push(Array2::from_shape_vec((1, dim as usize), v.clone())
                    .map_err(|e| convert::core_err_to_py(py, CoreError::Other(e.to_string())))?);
            }
        }
        let where_json = where_to_json(py, &where_)?;
        let wd_json = where_to_json(py, &where_document)?;
        let inner = &self.inner;
        let r = py
            .allow_threads(|| {
                inner
                    .lock()
                    .map_err(|_| lock_poisoned())?
                    .query_batch(arrs, k, where_json.as_ref(), wd_json.as_ref(), include.as_deref())
            })
            .map_err(|e| convert::core_err_to_py(py, e))?;
        convert::query_batch_result_to_py(py, r)
    }

    #[pyo3(signature = (ids=None, where_=None, where_document=None, limit=None, offset=None, include=None))]
    fn get(
        &self,
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
        let inner = &self.inner;
        let r = py
            .allow_threads(|| {
                inner
                    .lock()
                    .map_err(|_| lock_poisoned())?
                    .get(ids, where_json.as_ref(), wd_json.as_ref(), limit, offset, include.as_deref())
            })
            .map_err(|e| convert::core_err_to_py(py, e))?;
        convert::get_result_to_py(py, r)
    }

    #[pyo3(signature = (embedder, dim=None, bit_width=None, batch_size=256, on_progress=None, skip_empty=None))]
    fn reembed(
        &self,
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
        let inner = &self.inner;

        // Phase 1: prepare under lock — read docs, validate params.
        let embedder_identity = core_embedder.identity();
        let prep = py.allow_threads(|| {
            let mut guard = inner.lock().map_err(|_| lock_poisoned())?;
            guard.prepare_reembed(&embedder_identity, dim, bit_width, &skip_empty)
        }).map_err(|e| convert::core_err_to_py(py, e))?;

        if prep.total_docs == 0 {
            return convert::reembed_report_to_py(py, turbovecdb_core::types::ReembedReport {
                total_docs: 0,
                old_dim: Some(prep.old_dim),
                new_dim: Some(prep.old_dim),
                skipped: 0,
                elapsed_seconds: 0.0,
            });
        }

        // Phase 2: embed OUTSIDE the lock (no Mutex, no flock). The
        // embedder is just a PyO3 callable; the progress callback runs
        // outside the lock too so it can re-enter the collection safely.
        let mut progress_err: Option<PyErr> = None;
        let mut updates: Vec<(Vec<u8>, i64)> = Vec::new();
        let mut new_dim: Option<i64> = None;
        let mut processed = 0i64;
        let bs = batch_size.max(1);
        let total = prep.total_docs;
        let old_dim = prep.old_dim;
        let mut start_i = 0;
        while start_i < prep.embed_docs.len() {
            let end = std::cmp::min(start_i + bs, prep.embed_docs.len());
            let batch = &prep.embed_docs[start_i..end];
            let batch_uids = &prep.embed_uids[start_i..end];

            let out = core_embedder.embed(batch).map_err(|e| {
                let first = batch_uids[0];
                let last = batch_uids[batch_uids.len() - 1];
                let msg = format!("Embedder failed on batch uids [{first}..{last}]: {e}");
                convert::core_err_to_py(py, CoreError::Other(msg))
            })?;

            let bdim = out.ncols() as i64;
            match new_dim {
                None => new_dim = Some(bdim),
                Some(nd) if bdim != nd => {
                    return Err(convert::core_err_to_py(py, CoreError::DimensionMismatch(
                        format!("embedder returned inconsistent dimensions: {nd} then {bdim}")
                    )));
                }
                _ => {}
            }
            for (i, &uid) in batch_uids.iter().enumerate() {
                let bytes: Vec<u8> = out.row(i).iter()
                    .flat_map(|x| x.to_ne_bytes())
                    .collect();
                updates.push((bytes, uid));
            }
            processed += batch.len() as i64;
            if let Some(ref cb) = on_progress {
                let result = cb.call1(py, (processed, total));
                match result {
                    Ok(_) => {}
                    Err(e) => {
                        progress_err = Some(e);
                        break;
                    }
                }
            }
            start_i = end;
        }

        if let Some(err) = progress_err.take() {
            return Err(err);
        }

        let new_dim_v = new_dim.unwrap_or(old_dim);
        if let Some(dd) = dim {
            if new_dim_v != dd {
                return Err(convert::core_err_to_py(py, CoreError::DimensionMismatch(
                    format!("Embedder returned dim {new_dim_v}, but dim was set to {dd}")
                )));
            }
        }
        let new_bit_width = bit_width.unwrap_or(prep.old_bit_width);

        // Phase 3: re-acquire lock, write vectors, update meta.
        let result = py.allow_threads(|| {
            let mut guard = inner.lock().map_err(|_| lock_poisoned())?;
            guard.finish_reembed(prep, updates, new_dim_v, new_bit_width)
        });
        match result {
            Ok(r) => convert::reembed_report_to_py(py, r),
            Err(e) => Err(convert::core_err_to_py(py, e)),
        }
    }
}
