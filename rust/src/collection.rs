//! Rust `Collection` — storage foundation (milestone #11).
//!
//! This grows into the full engine (writes/reads/reembed/concurrency in later
//! milestones) and is flipped in for the Python `Collection` once at parity
//! (#17). For now it owns the durable `rusqlite` store and the meta/generation
//! bookkeeping, mirroring `collection.py`'s constructor + meta helpers. It is
//! exercised directly via `_core.Collection` while the Python engine still runs
//! the suite.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use rusqlite::{params, Connection, OptionalExtension};

const SCHEMA_VERSION: i64 = 1;
pub const DEFAULT_BIT_WIDTH: i64 = 4;

fn sql_err(e: rusqlite::Error) -> PyErr {
    PyRuntimeError::new_err(format!("sqlite error: {e}"))
}

/// Raise the public `turbovecdb.errors.DimensionMismatchError` so behavior
/// (and tests) match the Python engine once flipped.
fn dimension_error(py: Python<'_>, msg: String) -> PyErr {
    let build = || -> PyResult<PyErr> {
        let cls = py
            .import_bound("turbovecdb.errors")?
            .getattr("DimensionMismatchError")?;
        Ok(PyErr::from_value_bound(cls.call1((msg,))?))
    };
    build().unwrap_or_else(|e| e)
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

    fn commit_dim(&mut self, py: Python<'_>, dim: i64) -> PyResult<()> {
        if dim <= 0 || dim % 8 != 0 {
            return Err(dimension_error(
                py,
                format!("turbovec requires dim to be a positive multiple of 8, got {dim}"),
            ));
        }
        self.dim = Some(dim);
        self.meta_set("dim", &dim.to_string())
    }

    fn migrate_schema(&self) -> PyResult<()> {
        // Forward-only migration; v1 has no data changes, just records the version.
        if self.meta_get_i64("schema_version", 0)? >= SCHEMA_VERSION {
            return Ok(());
        }
        self.meta_set("schema_version", &SCHEMA_VERSION.to_string())
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
        };

        // bit_width: an already-stored value wins over the constructor arg.
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

        // Held for later milestones (index build, cross-process lock timeout).
        let _ = (embedder, lock_timeout);
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

    /// Current committed-write counter (0 until the first write).
    fn store_gen(&self) -> PyResult<i64> {
        self.meta_get_i64("store_gen", 0)
    }

    /// Read a raw meta value (used by smoke tests and later milestones).
    #[pyo3(name = "meta_get")]
    fn py_meta_get(&self, key: &str) -> PyResult<Option<String>> {
        self.meta_get(key)
    }
}
