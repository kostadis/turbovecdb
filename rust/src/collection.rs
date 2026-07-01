//! Rust `Collection` — storage foundation (#11) + write paths (#12).
//!
//! Owns the durable `rusqlite` store, the meta/generation bookkeeping, and the
//! turbovec index (held as a Python object, driven via PyO3). Built and tested
//! directly via `_core.Collection` while the Python engine still runs the suite;
//! the Python-facing flip happens once at parity (#17).
//!
//! Index coherence (this milestone): writes commit to SQLite and bump
//! `store_gen`; the index is then rebuilt lazily from the committed store on the
//! next read (`ensure_current`). Incremental index mirroring + the cross-process
//! file lock are deferred to #15.

use numpy::ndarray::{Array1, Array2};
use numpy::{PyArrayMethods, ToPyArray};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rusqlite::{params, Connection, OptionalExtension};

const SCHEMA_VERSION: i64 = 1;
pub const DEFAULT_BIT_WIDTH: i64 = 4;

fn sql_err(e: rusqlite::Error) -> PyErr {
    PyRuntimeError::new_err(format!("sqlite error: {e}"))
}

/// Raise one of the public `turbovecdb.errors.*` classes so behavior (and
/// tests) match the Python engine once flipped.
fn turbovec_error(py: Python<'_>, cls: &str, msg: String) -> PyErr {
    let build = || -> PyResult<PyErr> {
        let c = py.import_bound("turbovecdb.errors")?.getattr(cls)?;
        Ok(PyErr::from_value_bound(c.call1((msg,))?))
    };
    build().unwrap_or_else(|e| e)
}

fn blob_to_f32(blob: &[u8]) -> impl Iterator<Item = f32> + '_ {
    blob.chunks_exact(4)
        .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
}

/// Bridge a Python filter operand into a rusqlite bound value. Metadata JSON
/// scalars are str / int / float / bool / null; bool falls through to Integer.
fn py_to_sql_value(obj: &Bound<'_, PyAny>) -> rusqlite::types::Value {
    use rusqlite::types::Value;
    if obj.is_none() {
        return Value::Null;
    }
    if let Ok(i) = obj.extract::<i64>() {
        return Value::Integer(i);
    }
    if let Ok(f) = obj.extract::<f64>() {
        return Value::Real(f);
    }
    if let Ok(s) = obj.extract::<String>() {
        return Value::Text(s);
    }
    match obj.str() {
        Ok(s) => Value::Text(s.to_string()),
        Err(_) => Value::Null,
    }
}

#[pyclass]
pub struct Collection {
    #[pyo3(get)]
    dir: String,
    tvim_path: String,
    metric: String,
    bit_width: i64,
    dim: Option<i64>,
    next_uid: i64,
    conn: Connection,
    embedder: Option<PyObject>,
    index: Option<PyObject>,
    seen_gen: i64,
    dirty: bool,
}

impl Collection {
    fn meta_get(&self, key: &str) -> PyResult<Option<String>> {
        self.conn
            .query_row("SELECT value FROM meta WHERE key=?1", params![key], |r| {
                r.get::<_, String>(0)
            })
            .optional()
            .map_err(sql_err)
    }

    fn meta_set(&self, key: &str, value: &str) -> PyResult<()> {
        self.conn
            .execute(
                "INSERT INTO meta(key, value) VALUES(?1, ?2) \
                 ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                params![key, value],
            )
            .map_err(sql_err)?;
        Ok(())
    }

    fn meta_get_i64(&self, key: &str, default: i64) -> PyResult<i64> {
        Ok(self
            .meta_get(key)?
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(default))
    }

    fn store_gen_val(&self) -> PyResult<i64> {
        self.meta_get_i64("store_gen", 0)
    }

    fn commit_dim(&mut self, py: Python<'_>, dim: i64) -> PyResult<()> {
        if dim <= 0 || dim % 8 != 0 {
            return Err(turbovec_error(
                py,
                "DimensionMismatchError",
                format!("turbovec requires dim to be a positive multiple of 8, got {dim}"),
            ));
        }
        self.dim = Some(dim);
        self.meta_set("dim", &dim.to_string())
    }

    fn migrate_schema(&self) -> PyResult<()> {
        if self.meta_get_i64("schema_version", 0)? >= SCHEMA_VERSION {
            return Ok(());
        }
        self.meta_set("schema_version", &SCHEMA_VERSION.to_string())
    }

    fn make_index(&self, py: Python<'_>) -> PyResult<PyObject> {
        let kwargs = PyDict::new_bound(py);
        kwargs.set_item("dim", self.dim)?;
        kwargs.set_item("bit_width", self.bit_width)?;
        let idx = py
            .import_bound("turbovec")?
            .getattr("IdMapIndex")?
            .call((), Some(&kwargs))?;
        Ok(idx.unbind())
    }

    /// (Re)build the in-memory index to reflect the current store_gen.
    fn reload_index(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.dim.is_none() {
            self.index = None;
            self.seen_gen = self.store_gen_val()?;
            return Ok(());
        }
        let store_gen = self.store_gen_val()?;
        let tvim_gen = self.meta_get_i64("tvim_gen", -1)?;
        if std::path::Path::new(&self.tvim_path).exists() && tvim_gen == store_gen {
            let loaded = py
                .import_bound("turbovec")
                .and_then(|t| t.getattr("IdMapIndex"))
                .and_then(|c| c.getattr("load"))
                .and_then(|f| f.call1((self.tvim_path.as_str(),)));
            if let Ok(idx) = loaded {
                self.index = Some(idx.unbind());
                self.seen_gen = store_gen;
                return Ok(());
            }
        }
        // Rebuild from the SQLite vectors (source of truth).
        let (uids, flat) = {
            let mut stmt = self
                .conn
                .prepare("SELECT uid, vector FROM docs ORDER BY uid")
                .map_err(sql_err)?;
            let mut rows = stmt
                .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?)))
                .map_err(sql_err)?;
            let mut uids: Vec<u64> = Vec::new();
            let mut flat: Vec<f32> = Vec::new();
            while let Some(row) = rows.next() {
                let (uid, blob) = row.map_err(sql_err)?;
                uids.push(uid as u64);
                flat.extend(blob_to_f32(&blob));
            }
            (uids, flat)
        };
        let idx = self.make_index(py)?;
        if !uids.is_empty() {
            let d = self.dim.unwrap() as usize;
            let mat = Array2::from_shape_vec((uids.len(), d), flat)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
                .to_pyarray_bound(py);
            let uid_arr = Array1::from(uids).to_pyarray_bound(py);
            idx.bind(py).call_method1("add_with_ids", (mat, uid_arr))?;
        }
        self.index = Some(idx);
        self.seen_gen = store_gen;
        self.dirty = true;
        Ok(())
    }

    fn ensure_current(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.store_gen_val()? != self.seen_gen {
            self.reload_index(py)?;
        }
        Ok(())
    }

    /// Mirror of collection.py `_get_embedder_identity`.
    fn embedder_identity(&self, py: Python<'_>, emb: &PyObject) -> PyResult<String> {
        let b = emb.bind(py);
        if let Ok(name) = b.getattr("__name__") {
            if let Ok(s) = name.extract::<String>() {
                return Ok(s);
            }
        }
        if let Ok(cls) = b.getattr("__class__") {
            let module = cls
                .getattr("__module__")
                .and_then(|m| m.extract::<String>())
                .unwrap_or_default();
            let qn = cls
                .getattr("__name__")
                .and_then(|m| m.extract::<String>())
                .unwrap_or_default();
            return Ok(format!("{module}.{qn}"));
        }
        Ok("unknown_embedder".to_string())
    }

    /// Return the normalized (2-D, C-contiguous float32) vectors for a write.
    fn resolve_vectors<'py>(
        &self,
        py: Python<'py>,
        documents: Option<&Vec<String>>,
        vectors: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f32>>> {
        if let Some(v) = vectors {
            return crate::l2_normalize_impl(py, v);
        }
        let docs = documents.ok_or_else(|| {
            PyValueError::new_err(
                "add/upsert requires either vectors= or documents= (with an embedder)",
            )
        })?;
        let emb = self.embedder.as_ref().ok_or_else(|| {
            turbovec_error(
                py,
                "EmbedderRequiredError",
                "text was provided but this collection has no embedder; pass vectors instead, \
                 or create the collection with embedder=..."
                    .to_string(),
            )
        })?;
        // GAP-1 guard: prevent accidental embedder swap on the write path.
        if let Some(stored) = self.meta_get("embedder_identity")? {
            let current = self.embedder_identity(py, emb)?;
            if current != stored {
                return Err(turbovec_error(
                    py,
                    "EmbedderIdentityMismatchError",
                    format!(
                        "embedder identity mismatch: stored {stored:?} != current {current:?}; \
                         use reembed() to change embedders"
                    ),
                ));
            }
        }
        let docs_list = PyList::new_bound(py, docs);
        let out = emb.bind(py).call1((docs_list,))?;
        crate::l2_normalize_impl(py, &out)
    }

    fn dumps_meta(&self, py: Python<'_>, meta: Option<&PyObject>) -> PyResult<String> {
        let json = py.import_bound("json")?;
        let obj: Bound<PyAny> = match meta {
            Some(m) => m.bind(py).clone(),
            None => PyDict::new_bound(py).into_any(),
        };
        json.getattr("dumps")?.call1((obj,))?.extract::<String>()
    }

    fn rollback(&self) {
        let _ = self.conn.execute_batch("ROLLBACK");
    }

    /// The filter compiler raises `_core.FilterError`; the public engine raises
    /// `UnsupportedFilterError`. Re-wrap so behavior matches.
    fn map_filter_err(&self, py: Python<'_>, e: PyErr) -> PyErr {
        let msg = e
            .value_bound(py)
            .str()
            .map(|s| s.to_string())
            .unwrap_or_else(|_| e.to_string());
        turbovec_error(py, "UnsupportedFilterError", msg)
    }
}

#[pymethods]
impl Collection {
    #[new]
    #[pyo3(signature = (coll_dir, dim=None, bit_width=DEFAULT_BIT_WIDTH, metric=None, embedder=None, lock_timeout=30.0))]
    fn new(
        py: Python<'_>,
        coll_dir: String,
        dim: Option<i64>,
        bit_width: i64,
        metric: Option<String>,
        embedder: Option<PyObject>,
        lock_timeout: f64,
    ) -> PyResult<Self> {
        let metric = metric.unwrap_or_else(|| "cosine".to_string());
        if metric != "cosine" {
            return Err(PyValueError::new_err(format!(
                "unsupported metric {metric:?} (only 'cosine' for now)"
            )));
        }
        std::fs::create_dir_all(&coll_dir).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let db_path = format!("{coll_dir}/store.sqlite3");
        let tvim_path = format!("{coll_dir}/index.tvim");

        let conn = Connection::open(&db_path).map_err(sql_err)?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL; \
             PRAGMA busy_timeout=5000; \
             CREATE TABLE IF NOT EXISTS docs (uid INTEGER PRIMARY KEY, str_id TEXT UNIQUE NOT NULL, document TEXT, metadata TEXT, vector BLOB); \
             CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);",
        )
        .map_err(sql_err)?;

        let mut c = Collection {
            dir: coll_dir,
            tvim_path,
            metric,
            bit_width,
            dim: None,
            next_uid: 0,
            conn,
            embedder,
            index: None,
            seen_gen: -1,
            dirty: false,
        };

        c.bit_width = c.meta_get_i64("bit_width", bit_width)?;
        c.meta_set("bit_width", &c.bit_width.to_string())?;
        let metric = c.metric.clone();
        c.meta_set("metric", &metric)?;

        c.dim = c.meta_get("dim")?.and_then(|s| s.parse::<i64>().ok());
        if c.dim.is_none() {
            if let Some(d) = dim {
                c.commit_dim(py, d)?;
            }
        }
        c.next_uid = c.meta_get_i64("next_uid", 0)?;
        c.migrate_schema()?;

        // Record embedder identity on first creation so later opens with a
        // different embedder are caught by the write-path guard.
        if c.embedder.is_some() && c.meta_get("embedder_identity")?.is_none() {
            let id = c.embedder_identity(py, c.embedder.as_ref().unwrap())?;
            c.meta_set("embedder_identity", &id)?;
        }

        let _ = lock_timeout; // cross-process lock: #15
        c.reload_index(py)?;
        Ok(c)
    }

    #[getter]
    fn dim(&self) -> Option<i64> {
        self.dim
    }

    #[getter]
    fn bit_width(&self) -> i64 {
        self.bit_width
    }

    #[getter]
    fn tvim_path(&self) -> &str {
        &self.tvim_path
    }

    fn count(&self) -> PyResult<i64> {
        self.conn
            .query_row("SELECT COUNT(*) FROM docs", [], |r| r.get::<_, i64>(0))
            .map_err(sql_err)
    }

    fn store_gen(&self) -> PyResult<i64> {
        self.store_gen_val()
    }

    #[pyo3(name = "meta_get")]
    fn py_meta_get(&self, key: &str) -> PyResult<Option<String>> {
        self.meta_get(key)
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
        self.write(py, ids, documents, metadatas, vectors, false)
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
        self.write(py, ids, documents, metadatas, vectors, true)
    }

    /// Remove all documents; keeps configuration. Mirrors Collection.clear().
    fn clear(&mut self, py: Python<'_>) -> PyResult<()> {
        self.ensure_current(py)?;
        if self.count()? == 0 {
            return Ok(());
        }
        self.conn.execute_batch("BEGIN").map_err(sql_err)?;
        if let Err(e) = self.conn.execute("DELETE FROM docs", []) {
            self.rollback();
            return Err(sql_err(e));
        }
        let sg = self.store_gen_val()? + 1;
        if self.meta_set("next_uid", "0").is_err() || self.meta_set("store_gen", &sg.to_string()).is_err() {
            self.rollback();
            return Err(PyRuntimeError::new_err("clear: meta update failed"));
        }
        self.conn.execute_batch("COMMIT").map_err(sql_err)?;
        self.next_uid = 0;
        self.seen_gen = sg;
        self.dirty = true;
        self.index = if self.dim.is_some() {
            Some(self.make_index(py)?)
        } else {
            None
        };
        Ok(())
    }

    #[pyo3(signature = (ids=None, where_=None))]
    fn delete(
        &mut self,
        py: Python<'_>,
        ids: Option<Vec<String>>,
        where_: Option<PyObject>,
    ) -> PyResult<()> {
        use rusqlite::types::Value;
        let mut frags: Vec<String> = Vec::new();
        let mut values: Vec<Value> = Vec::new();
        if let Some(idlist) = &ids {
            if idlist.is_empty() {
                return Ok(());
            }
            let qs = vec!["?"; idlist.len()].join(",");
            frags.push(format!("str_id IN ({qs})"));
            for id in idlist {
                values.push(Value::Text(id.clone()));
            }
        }
        if let Some(w) = &where_ {
            let (wsql, wparams) =
                crate::compile_where(py, w.bind(py), 0).map_err(|e| self.map_filter_err(py, e))?;
            if !wsql.is_empty() {
                frags.push(format!("({wsql})"));
                for p in wparams {
                    values.push(py_to_sql_value(p.bind(py)));
                }
            }
        }
        self.ensure_current(py)?;
        let clause = if frags.is_empty() {
            String::new()
        } else {
            format!(" WHERE {}", frags.join(" AND "))
        };
        self.conn.execute_batch("BEGIN").map_err(sql_err)?;
        let sql = format!("DELETE FROM docs{clause}");
        let changed = match self
            .conn
            .execute(&sql, rusqlite::params_from_iter(values.iter()))
        {
            Ok(n) => n,
            Err(e) => {
                self.rollback();
                return Err(sql_err(e));
            }
        };
        if changed == 0 {
            self.conn.execute_batch("COMMIT").map_err(sql_err)?;
            return Ok(());
        }
        let sg = self.store_gen_val()? + 1;
        if self.meta_set("store_gen", &sg.to_string()).is_err() {
            self.rollback();
            return Err(PyRuntimeError::new_err("delete: meta update failed"));
        }
        self.conn.execute_batch("COMMIT").map_err(sql_err)?;
        // Rows removed → index rebuilds lazily from the committed store.
        self.dirty = true;
        Ok(())
    }

    fn update_metadata(
        &mut self,
        py: Python<'_>,
        ids: Vec<String>,
        metadatas: Vec<PyObject>,
    ) -> PyResult<()> {
        if metadatas.len() != ids.len() {
            return Err(PyValueError::new_err(format!(
                "metadatas length {} != ids length {}",
                metadatas.len(),
                ids.len()
            )));
        }
        if ids.is_empty() {
            return Ok(());
        }
        self.ensure_current(py)?;
        self.conn.execute_batch("BEGIN").map_err(sql_err)?;
        for i in 0..ids.len() {
            let exists: Option<i64> = match self
                .conn
                .query_row("SELECT uid FROM docs WHERE str_id=?1", params![ids[i]], |r| r.get(0))
                .optional()
            {
                Ok(v) => v,
                Err(e) => {
                    self.rollback();
                    return Err(sql_err(e));
                }
            };
            if exists.is_none() {
                self.rollback();
                return Err(PyValueError::new_err(format!("id {:?} not found", ids[i])));
            }
            let truthy = metadatas[i].bind(py).is_truthy().unwrap_or(true);
            let arg = if truthy { Some(&metadatas[i]) } else { None };
            let meta_json = match self.dumps_meta(py, arg) {
                Ok(s) => s,
                Err(e) => {
                    self.rollback();
                    return Err(e);
                }
            };
            if let Err(e) = self.conn.execute(
                "UPDATE docs SET metadata=?1 WHERE str_id=?2",
                params![meta_json, ids[i]],
            ) {
                self.rollback();
                return Err(sql_err(e));
            }
        }
        let sg = self.store_gen_val()? + 1;
        if self.meta_set("store_gen", &sg.to_string()).is_err() {
            self.rollback();
            return Err(PyRuntimeError::new_err("update_metadata: meta update failed"));
        }
        self.conn.execute_batch("COMMIT").map_err(sql_err)?;
        self.seen_gen = sg; // vectors unchanged → index stays valid
        Ok(())
    }

    fn update_documents(
        &mut self,
        py: Python<'_>,
        ids: Vec<String>,
        documents: Vec<String>,
    ) -> PyResult<()> {
        if documents.len() != ids.len() {
            return Err(PyValueError::new_err(format!(
                "documents length {} != ids length {}",
                documents.len(),
                ids.len()
            )));
        }
        if ids.is_empty() {
            return Ok(());
        }
        self.ensure_current(py)?;
        self.conn.execute_batch("BEGIN").map_err(sql_err)?;
        for i in 0..ids.len() {
            let exists: Option<i64> = match self
                .conn
                .query_row("SELECT uid FROM docs WHERE str_id=?1", params![ids[i]], |r| r.get(0))
                .optional()
            {
                Ok(v) => v,
                Err(e) => {
                    self.rollback();
                    return Err(sql_err(e));
                }
            };
            if exists.is_none() {
                self.rollback();
                return Err(PyValueError::new_err(format!("id {:?} not found", ids[i])));
            }
            if let Err(e) = self.conn.execute(
                "UPDATE docs SET document=?1 WHERE str_id=?2",
                params![documents[i], ids[i]],
            ) {
                self.rollback();
                return Err(sql_err(e));
            }
        }
        let sg = self.store_gen_val()? + 1;
        if self.meta_set("store_gen", &sg.to_string()).is_err() {
            self.rollback();
            return Err(PyRuntimeError::new_err("update_documents: meta update failed"));
        }
        self.conn.execute_batch("COMMIT").map_err(sql_err)?;
        self.seen_gen = sg; // vectors unchanged → index stays valid
        Ok(())
    }
}

impl Collection {
    fn write(
        &mut self,
        py: Python<'_>,
        ids: Vec<String>,
        documents: Option<Vec<String>>,
        metadatas: Option<Vec<PyObject>>,
        vectors: Option<PyObject>,
        replace: bool,
    ) -> PyResult<()> {
        let n = ids.len();
        if let Some(d) = &documents {
            if d.len() != n {
                return Err(PyValueError::new_err(format!(
                    "documents length {} != ids length {}",
                    d.len(),
                    n
                )));
            }
        }
        if let Some(m) = &metadatas {
            if m.len() != n {
                return Err(PyValueError::new_err(format!(
                    "metadatas length {} != ids length {}",
                    m.len(),
                    n
                )));
            }
        }
        let vectors_bound = vectors.as_ref().map(|o| o.bind(py));
        if let Some(v) = &vectors_bound {
            let vlen = v.len()?;
            if vlen != n {
                return Err(PyValueError::new_err(format!(
                    "vectors length {vlen} != ids length {n}"
                )));
            }
        }
        if n == 0 {
            return Ok(());
        }

        let vecs = self.resolve_vectors(py, documents.as_ref(), vectors_bound)?;
        let docs: Vec<String> = documents.unwrap_or_else(|| vec![String::new(); n]);

        // Refresh next_uid + index staleness under (eventual) lock.
        self.next_uid = self.meta_get_i64("next_uid", 0)?;
        self.ensure_current(py)?;

        let vro = vecs.readonly();
        let view = vro.as_array();
        let vdim = view.ncols() as i64;
        if self.dim.is_none() {
            self.commit_dim(py, vdim)?;
        } else if Some(vdim) != self.dim {
            return Err(turbovec_error(
                py,
                "DimensionMismatchError",
                format!("vector dim {} != collection dim {}", vdim, self.dim.unwrap()),
            ));
        }

        self.conn.execute_batch("BEGIN").map_err(sql_err)?;
        for i in 0..n {
            let meta_json = match self.dumps_meta(py, metadatas.as_ref().map(|m| &m[i])) {
                Ok(s) => s,
                Err(e) => {
                    self.rollback();
                    return Err(e);
                }
            };
            let bytes: Vec<u8> = view.row(i).iter().flat_map(|x| x.to_ne_bytes()).collect();

            let existing: Option<i64> = match self
                .conn
                .query_row("SELECT uid FROM docs WHERE str_id=?1", params![ids[i]], |r| {
                    r.get(0)
                })
                .optional()
            {
                Ok(v) => v,
                Err(e) => {
                    self.rollback();
                    return Err(sql_err(e));
                }
            };

            let res = if let Some(_uid) = existing {
                if !replace {
                    self.rollback();
                    return Err(PyValueError::new_err(format!(
                        "id {:?} already exists (use upsert)",
                        ids[i]
                    )));
                }
                self.conn.execute(
                    "UPDATE docs SET document=?1, metadata=?2, vector=?3 WHERE str_id=?4",
                    params![docs[i], meta_json, bytes, ids[i]],
                )
            } else {
                let uid = self.next_uid;
                self.next_uid += 1;
                self.conn.execute(
                    "INSERT INTO docs(uid, str_id, document, metadata, vector) VALUES(?1,?2,?3,?4,?5)",
                    params![uid, ids[i], docs[i], meta_json, bytes],
                )
            };
            if let Err(e) = res {
                self.rollback();
                return Err(sql_err(e));
            }
        }

        let sg = self.store_gen_val()? + 1;
        if self
            .meta_set("next_uid", &self.next_uid.to_string())
            .and_then(|_| self.meta_set("store_gen", &sg.to_string()))
            .is_err()
        {
            self.rollback();
            return Err(PyRuntimeError::new_err("write: meta update failed"));
        }
        self.conn.execute_batch("COMMIT").map_err(sql_err)?;

        // Index rebuilds lazily from the committed store on the next read.
        self.dirty = true;
        Ok(())
    }
}
