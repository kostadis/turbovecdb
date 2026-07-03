//! Pure Rust `Collection` — SQLite store, generation bookkeeping, and
//! reembed, generic over `Embedder`/`VectorIndex` so it doesn't hardcode
//! "the embedder/index are Python objects." Ported from
//! `turbovecdb-py::collection` (split phase 5/8, see
//! `docs/rust-core-split-design.md`); `turbovecdb-py::collection::Collection`
//! is now a thin `#[pyclass]` wrapper around `Collection<PyEmbedder,
//! TurbovecIndex>`.

use ndarray::Array2;
use rusqlite::{params, Connection, OptionalExtension};
use std::collections::HashMap;

use crate::embedder::Embedder;
use crate::error::CoreError;
use crate::filters;
use crate::index::VectorIndex;
use crate::types::{GetResult, HealthResult, Include, QueryResult, ReembedReport};
use crate::vecmath::l2_normalize;

const RERANK_FLOOR: i64 = 50;
const SCHEMA_VERSION: i64 = 1;
pub const DEFAULT_BIT_WIDTH: i64 = 4;

fn blob_to_f32(blob: &[u8]) -> impl Iterator<Item = f32> + '_ {
    blob.chunks_exact(4).map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
}

fn f32_to_blob(row: impl Iterator<Item = f32>) -> Vec<u8> {
    row.flat_map(|x| x.to_ne_bytes()).collect()
}

/// Bridge a filter-compiler operand into a rusqlite bound value. Metadata
/// JSON scalars are str / int / float / bool / null; bool falls through to
/// Integer — JSON's `Bool` is a distinct variant from `Number`, unlike
/// Python where `bool` is an `int` subclass, so this must special-case it to
/// match the historical PyAny-based converter's behavior.
fn json_to_sql_value(v: &serde_json::Value) -> rusqlite::types::Value {
    use rusqlite::types::Value;
    match v {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Integer(if *b { 1 } else { 0 }),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Integer(i)
            } else if let Some(f) = n.as_f64() {
                Value::Real(f)
            } else {
                Value::Null
            }
        }
        serde_json::Value::String(s) => Value::Text(s.clone()),
        other => Value::Text(other.to_string()),
    }
}

pub struct Collection<E: Embedder, I: VectorIndex> {
    dir: String,
    tvim_path: String,
    metric: String,
    bit_width: i64,
    dim: Option<i64>,
    next_uid: i64,
    conn: Connection,
    embedder: Option<E>,
    index: Option<I>,
    seen_gen: i64,
    dirty: bool,
}

impl<E: Embedder, I: VectorIndex> Collection<E, I> {
    pub fn new(
        coll_dir: String,
        dim: Option<i64>,
        bit_width: i64,
        metric: Option<String>,
        embedder: Option<E>,
    ) -> Result<Self, CoreError> {
        let metric = metric.unwrap_or_else(|| "cosine".to_string());
        if metric != "cosine" {
            return Err(CoreError::InvalidArgument(format!(
                "unsupported metric {metric:?} (only 'cosine' for now)"
            )));
        }
        std::fs::create_dir_all(&coll_dir)?;
        let db_path = format!("{coll_dir}/store.sqlite3");
        let tvim_path = format!("{coll_dir}/index.tvim");

        let conn = Connection::open(&db_path)?;
        // Set the busy timeout FIRST — before the WAL-mode switch and DDL
        // below, which briefly need a write lock. The constructor runs
        // outside the cross-process file lock, so concurrent opens must
        // wait here rather than fail with SQLITE_BUSY ("database is locked").
        conn.busy_timeout(std::time::Duration::from_millis(5000))?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL; \
             CREATE TABLE IF NOT EXISTS docs (uid INTEGER PRIMARY KEY, str_id TEXT UNIQUE NOT NULL, document TEXT, metadata TEXT, vector BLOB); \
             CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);",
        )?;

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
        let metric_val = c.metric.clone();
        c.meta_set("metric", &metric_val)?;

        c.dim = c.meta_get("dim")?.and_then(|s| s.parse::<i64>().ok());
        if c.dim.is_none() {
            if let Some(d) = dim {
                c.commit_dim(d)?;
            }
        }
        c.next_uid = c.meta_get_i64("next_uid", 0)?;
        c.migrate_schema()?;

        // Record embedder identity on first creation so later opens with a
        // different embedder are caught by the write-path guard.
        if let Some(emb) = c.embedder.as_ref() {
            if c.meta_get("embedder_identity")?.is_none() {
                let id = emb.identity();
                c.meta_set("embedder_identity", &id)?;
            }
        }

        c.reload_index()?;
        Ok(c)
    }

    // -- private helpers ------------------------------------------------

    pub fn meta_get(&self, key: &str) -> Result<Option<String>, CoreError> {
        Ok(self
            .conn
            .query_row("SELECT value FROM meta WHERE key=?1", params![key], |r| {
                r.get::<_, String>(0)
            })
            .optional()?)
    }

    fn meta_set(&self, key: &str, value: &str) -> Result<(), CoreError> {
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?1, ?2) \
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    fn meta_get_i64(&self, key: &str, default: i64) -> Result<i64, CoreError> {
        Ok(self.meta_get(key)?.and_then(|s| s.parse::<i64>().ok()).unwrap_or(default))
    }

    fn store_gen_val(&self) -> Result<i64, CoreError> {
        self.meta_get_i64("store_gen", 0)
    }

    fn commit_dim(&mut self, dim: i64) -> Result<(), CoreError> {
        if dim <= 0 || dim % 8 != 0 {
            return Err(CoreError::DimensionMismatch(format!(
                "turbovec requires dim to be a positive multiple of 8, got {dim}"
            )));
        }
        self.dim = Some(dim);
        self.meta_set("dim", &dim.to_string())
    }

    fn migrate_schema(&self) -> Result<(), CoreError> {
        if self.meta_get_i64("schema_version", 0)? >= SCHEMA_VERSION {
            return Ok(());
        }
        self.meta_set("schema_version", &SCHEMA_VERSION.to_string())
    }

    fn make_index(&self) -> Result<I, CoreError> {
        I::new(self.dim.unwrap() as usize, self.bit_width as usize)
    }

    /// (Re)build the in-memory index to reflect the current store_gen.
    fn reload_index(&mut self) -> Result<(), CoreError> {
        if self.dim.is_none() {
            self.index = None;
            self.seen_gen = self.store_gen_val()?;
            return Ok(());
        }
        let store_gen = self.store_gen_val()?;
        let tvim_gen = self.meta_get_i64("tvim_gen", -1)?;
        if std::path::Path::new(&self.tvim_path).exists() && tvim_gen == store_gen {
            if let Ok(idx) = I::load(&self.tvim_path) {
                self.index = Some(idx);
                self.seen_gen = store_gen;
                return Ok(());
            }
        }
        // Rebuild from the SQLite vectors (source of truth).
        let (uids, flat) = {
            let mut stmt = self.conn.prepare("SELECT uid, vector FROM docs ORDER BY uid")?;
            let mut rows =
                stmt.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?)))?;
            let mut uids: Vec<u64> = Vec::new();
            let mut flat: Vec<f32> = Vec::new();
            while let Some(row) = rows.next() {
                let (uid, blob) = row?;
                uids.push(uid as u64);
                flat.extend(blob_to_f32(&blob));
            }
            (uids, flat)
        };
        let mut idx = self.make_index()?;
        if !uids.is_empty() {
            idx.add_with_ids(&flat, &uids)?;
        }
        self.index = Some(idx);
        self.seen_gen = store_gen;
        self.dirty = true;
        Ok(())
    }

    fn ensure_current(&mut self) -> Result<(), CoreError> {
        if self.store_gen_val()? != self.seen_gen {
            self.reload_index()?;
        }
        Ok(())
    }

    /// Return the normalized vectors for a write: either the caller's raw
    /// vectors (normalized here), or the embedder's output on `documents`
    /// (already normalized — see `Embedder::embed`'s contract).
    fn resolve_vectors(
        &self,
        documents: Option<&[String]>,
        vectors: Option<Array2<f32>>,
    ) -> Result<Array2<f32>, CoreError> {
        if let Some(mut v) = vectors {
            l2_normalize(&mut v);
            return Ok(v);
        }
        let docs = documents.ok_or_else(|| {
            CoreError::InvalidArgument(
                "add/upsert requires either vectors= or documents= (with an embedder)".to_string(),
            )
        })?;
        let emb = self.embedder.as_ref().ok_or_else(|| {
            CoreError::EmbedderRequired(
                "text was provided but this collection has no embedder; pass vectors instead, \
                 or create the collection with embedder=..."
                    .to_string(),
            )
        })?;
        // GAP-1 guard: prevent accidental embedder swap on the write path.
        if let Some(stored) = self.meta_get("embedder_identity")? {
            let current = emb.identity();
            if current != stored {
                return Err(CoreError::EmbedderIdentityMismatch(format!(
                    "embedder identity mismatch: stored {stored:?} != current {current:?}; \
                     use reembed() to change embedders"
                )));
            }
        }
        emb.embed(docs)
    }

    fn rollback(&self) {
        let _ = self.conn.execute_batch("ROLLBACK");
    }

    /// Truncate the WAL to bound its growth (best-effort).
    fn wal_checkpoint(&self) {
        let _ = self.conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE)");
    }

    /// Mirror a write into the in-memory index: drop the replaced uids, then
    /// add every row's (uid, vector). Keeps the index O(batch) per write
    /// instead of rebuilding from the whole store.
    fn mirror_write_to_index(
        &mut self,
        replaced: &[i64],
        uids: &[i64],
        vecs: &Array2<f32>,
    ) -> Result<(), CoreError> {
        let index = self.index.as_mut().unwrap();
        for &uid in replaced {
            index.remove(uid as u64);
        }
        let flat: Vec<f32> = vecs.iter().copied().collect();
        let uid_u64: Vec<u64> = uids.iter().map(|&x| x as u64).collect();
        index.add_with_ids(&flat, &uid_u64)?;
        Ok(())
    }

    /// Remove uids from the in-memory index (used by delete).
    fn remove_from_index(&mut self, uids: &[i64]) -> Result<(), CoreError> {
        if uids.is_empty() {
            return Ok(());
        }
        let index = self.index.as_mut().unwrap();
        for &uid in uids {
            index.remove(uid as u64);
        }
        Ok(())
    }

    fn set_reembed_meta(&self, new_dim: i64, new_bit_width: i64, identity: &str) -> Result<(), CoreError> {
        self.meta_set("dim", &new_dim.to_string())?;
        self.meta_set("bit_width", &new_bit_width.to_string())?;
        let sg = self.store_gen_val()? + 1;
        self.meta_set("store_gen", &sg.to_string())?;
        self.meta_set("embedder_identity", identity)?;
        Ok(())
    }

    fn empty_query_result(inc_vectors: bool) -> QueryResult {
        QueryResult {
            vectors: if inc_vectors { Some(Vec::new()) } else { None },
            ..Default::default()
        }
    }

    /// uid allowlist for a query filter: None → search everything; Some(uids)
    /// (possibly empty) → restrict to those uids.
    fn query_allowlist(
        &self,
        where_: Option<&serde_json::Value>,
        where_document: Option<&serde_json::Value>,
    ) -> Result<Option<Vec<i64>>, CoreError> {
        if where_.is_none() && where_document.is_none() {
            return Ok(None);
        }
        let mut frags: Vec<String> = Vec::new();
        let mut values: Vec<rusqlite::types::Value> = Vec::new();
        if let Some(w) = where_ {
            let (s, ps) = filters::compile_where(w, 0)?;
            if !s.is_empty() {
                frags.push(format!("({s})"));
                for p in &ps {
                    values.push(json_to_sql_value(p));
                }
            }
        }
        if let Some(wd) = where_document {
            let (s, ps) = filters::compile_where_document(wd)?;
            if !s.is_empty() {
                frags.push(format!("({s})"));
                for p in &ps {
                    values.push(json_to_sql_value(p));
                }
            }
        }
        if frags.is_empty() {
            return Ok(None);
        }
        let sql = format!("SELECT uid FROM docs WHERE {}", frags.join(" AND "));
        let mut stmt = self.conn.prepare(&sql)?;
        let uids = stmt
            .query_map(rusqlite::params_from_iter(values.iter()), |r| r.get::<_, i64>(0))?
            .collect::<Result<Vec<i64>, _>>()?;
        Ok(Some(uids))
    }

    fn write(
        &mut self,
        ids: Vec<String>,
        documents: Option<Vec<String>>,
        metadatas: Option<Vec<String>>,
        vectors: Option<Array2<f32>>,
        replace: bool,
    ) -> Result<(), CoreError> {
        let n = ids.len();
        if let Some(d) = &documents {
            if d.len() != n {
                return Err(CoreError::InvalidArgument(format!(
                    "documents length {} != ids length {}",
                    d.len(),
                    n
                )));
            }
        }
        if let Some(m) = &metadatas {
            if m.len() != n {
                return Err(CoreError::InvalidArgument(format!(
                    "metadatas length {} != ids length {}",
                    m.len(),
                    n
                )));
            }
        }
        if let Some(v) = &vectors {
            let vlen = v.nrows();
            if vlen != n {
                return Err(CoreError::InvalidArgument(format!(
                    "vectors length {vlen} != ids length {n}"
                )));
            }
        }
        if n == 0 {
            return Ok(());
        }

        let vecs = self.resolve_vectors(documents.as_deref(), vectors)?;
        let docs: Vec<String> = documents.unwrap_or_else(|| vec![String::new(); n]);

        // Refresh next_uid + index staleness under (eventual) lock.
        self.next_uid = self.meta_get_i64("next_uid", 0)?;
        self.ensure_current()?;

        let vdim = vecs.ncols() as i64;
        if self.dim.is_none() {
            self.commit_dim(vdim)?;
            self.index = Some(self.make_index()?);
        } else if Some(vdim) != self.dim {
            return Err(CoreError::DimensionMismatch(format!(
                "vector dim {} != collection dim {}",
                vdim,
                self.dim.unwrap()
            )));
        }

        self.conn.execute_batch("BEGIN")?;
        let mut batch_uids: Vec<i64> = Vec::with_capacity(n);
        let mut replaced_uids: Vec<i64> = Vec::new();
        for i in 0..n {
            let meta_json = metadatas.as_ref().map(|m| m[i].clone()).unwrap_or_else(|| "{}".to_string());
            let bytes = f32_to_blob(vecs.row(i).iter().copied());

            let existing: Option<i64> = match self
                .conn
                .query_row("SELECT uid FROM docs WHERE str_id=?1", params![ids[i]], |r| r.get(0))
                .optional()
            {
                Ok(v) => v,
                Err(e) => {
                    self.rollback();
                    return Err(e.into());
                }
            };

            let uid = if let Some(u) = existing {
                if !replace {
                    self.rollback();
                    return Err(CoreError::InvalidArgument(format!(
                        "id {:?} already exists (use upsert)",
                        ids[i]
                    )));
                }
                if let Err(e) = self.conn.execute(
                    "UPDATE docs SET document=?1, metadata=?2, vector=?3 WHERE str_id=?4",
                    params![docs[i], meta_json, bytes, ids[i]],
                ) {
                    self.rollback();
                    return Err(e.into());
                }
                replaced_uids.push(u);
                u
            } else {
                let u = self.next_uid;
                self.next_uid += 1;
                if let Err(e) = self.conn.execute(
                    "INSERT INTO docs(uid, str_id, document, metadata, vector) VALUES(?1,?2,?3,?4,?5)",
                    params![u, ids[i], docs[i], meta_json, bytes],
                ) {
                    self.rollback();
                    return Err(e.into());
                }
                u
            };
            batch_uids.push(uid);
        }

        let sg = self.store_gen_val()? + 1;
        if self
            .meta_set("next_uid", &self.next_uid.to_string())
            .and_then(|_| self.meta_set("store_gen", &sg.to_string()))
            .is_err()
        {
            self.rollback();
            return Err(CoreError::Runtime("write: meta update failed".to_string()));
        }

        // Mirror the write into the in-memory index, then commit. If either
        // fails, roll back and invalidate the index so the next read rebuilds
        // it from the committed store — never a divergent cache.
        let mirrored = self.mirror_write_to_index(&replaced_uids, &batch_uids, &vecs);
        match mirrored.and_then(|_| self.conn.execute_batch("COMMIT").map_err(CoreError::from)) {
            Ok(()) => {
                self.seen_gen = self.store_gen_val()?;
                self.dirty = true;
                Ok(())
            }
            Err(e) => {
                self.rollback();
                self.index = None;
                self.seen_gen = -1;
                Err(e)
            }
        }
    }

    // -- public API -------------------------------------------------------

    pub fn dir(&self) -> &str {
        &self.dir
    }

    pub fn dim(&self) -> Option<i64> {
        self.dim
    }

    pub fn bit_width(&self) -> i64 {
        self.bit_width
    }

    pub fn tvim_path(&self) -> &str {
        &self.tvim_path
    }

    pub fn count(&self) -> Result<i64, CoreError> {
        Ok(self.conn.query_row("SELECT COUNT(*) FROM docs", [], |r| r.get::<_, i64>(0))?)
    }

    pub fn store_gen(&self) -> Result<i64, CoreError> {
        self.store_gen_val()
    }

    /// Persist the in-memory index to `index.tvim` (a cold-start cache) and
    /// checkpoint the WAL. The cross-process write lock is held by the
    /// Python wrapper around this call.
    pub fn flush(&mut self) -> Result<(), CoreError> {
        self.ensure_current()?;
        if self.index.is_none() || !self.dirty {
            return Ok(());
        }
        let tmp = format!("{}.tmp", self.tvim_path);
        self.index.as_ref().unwrap().write(&tmp)?;
        std::fs::rename(&tmp, &self.tvim_path)?;
        // Stamp tvim_gen from seen_gen — the generation the in-memory index
        // actually reflects — not a fresh re-read of store_gen. Without the
        // caller's file lock held, a concurrent writer can bump store_gen
        // between the rename above and a re-read here, which would mark a
        // stale .tvim as coherent (wrong query results) instead of merely
        // stale (harmless rebuild on next open).
        self.meta_set("tvim_gen", &self.seen_gen.to_string())?; // autocommits
        self.wal_checkpoint();
        self.dirty = false;
        Ok(())
    }

    pub fn close(&mut self) -> Result<(), CoreError> {
        self.flush()?;
        self.wal_checkpoint();
        Ok(())
    }

    /// SQLite integrity + cache-coherence report.
    pub fn health(&self) -> Result<HealthResult, CoreError> {
        let qc: String = self.conn.query_row("PRAGMA quick_check", [], |r| r.get(0))?;
        let store_gen = self.store_gen_val()?;
        let tvim_gen = self.meta_get_i64("tvim_gen", -1)?;
        let doc_count = self.count()?;
        let coherent = store_gen == tvim_gen;
        let ok = qc == "ok";
        if !ok {
            return Err(CoreError::Other(format!(
                "SQLite integrity check failed for {:?}: {}",
                self.dir, qc
            )));
        }
        Ok(HealthResult { ok, quick_check: qc, store_gen, tvim_gen, coherent, doc_count })
    }

    pub fn add(
        &mut self,
        ids: Vec<String>,
        documents: Option<Vec<String>>,
        metadatas: Option<Vec<String>>,
        vectors: Option<Array2<f32>>,
    ) -> Result<(), CoreError> {
        self.write(ids, documents, metadatas, vectors, false)
    }

    pub fn upsert(
        &mut self,
        ids: Vec<String>,
        documents: Option<Vec<String>>,
        metadatas: Option<Vec<String>>,
        vectors: Option<Array2<f32>>,
    ) -> Result<(), CoreError> {
        self.write(ids, documents, metadatas, vectors, true)
    }

    /// Remove all documents; keeps configuration.
    pub fn clear(&mut self) -> Result<(), CoreError> {
        self.ensure_current()?;
        if self.count()? == 0 {
            return Ok(());
        }
        self.conn.execute_batch("BEGIN")?;
        if let Err(e) = self.conn.execute("DELETE FROM docs", []) {
            self.rollback();
            return Err(e.into());
        }
        let sg = self.store_gen_val()? + 1;
        if self.meta_set("next_uid", "0").is_err() || self.meta_set("store_gen", &sg.to_string()).is_err() {
            self.rollback();
            return Err(CoreError::Runtime("clear: meta update failed".to_string()));
        }
        if let Err(e) = self.conn.execute_batch("COMMIT") {
            // A failed COMMIT (e.g. SQLITE_BUSY) leaves the transaction open
            // in SQLite; roll it back explicitly so the next write doesn't
            // fail with "cannot start a transaction within a transaction",
            // and invalidate the index cache rather than risk a divergent
            // in-memory view of an uncommitted clear.
            self.rollback();
            self.index = None;
            self.seen_gen = -1;
            return Err(e.into());
        }
        self.next_uid = 0;
        self.seen_gen = sg;
        self.dirty = true;
        self.index = if self.dim.is_some() { Some(self.make_index()?) } else { None };
        Ok(())
    }

    pub fn delete(
        &mut self,
        ids: Option<Vec<String>>,
        where_: Option<&serde_json::Value>,
    ) -> Result<(), CoreError> {
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
        if let Some(w) = where_ {
            let (wsql, wparams) = filters::compile_where(w, 0)?;
            if !wsql.is_empty() {
                frags.push(format!("({wsql})"));
                for p in &wparams {
                    values.push(json_to_sql_value(p));
                }
            }
        }
        self.ensure_current()?;
        let clause = if frags.is_empty() {
            String::new()
        } else {
            format!(" WHERE {}", frags.join(" AND "))
        };

        // Collect the uids we're about to delete so we can drop them from
        // the in-memory index incrementally (no full rebuild).
        let del_uids: Vec<i64> = {
            let sel = format!("SELECT uid FROM docs{clause}");
            let mut stmt = self.conn.prepare(&sel)?;
            let rows = stmt.query_map(rusqlite::params_from_iter(values.iter()), |r| r.get::<_, i64>(0))?;
            let mut v = Vec::new();
            for row in rows {
                v.push(row?);
            }
            v
        };
        if del_uids.is_empty() {
            return Ok(());
        }

        self.conn.execute_batch("BEGIN")?;
        let sql = format!("DELETE FROM docs{clause}");
        if let Err(e) = self.conn.execute(&sql, rusqlite::params_from_iter(values.iter())) {
            self.rollback();
            return Err(e.into());
        }
        let sg = self.store_gen_val()? + 1;
        if self.meta_set("store_gen", &sg.to_string()).is_err() {
            self.rollback();
            return Err(CoreError::Runtime("delete: meta update failed".to_string()));
        }
        match self
            .remove_from_index(&del_uids)
            .and_then(|_| self.conn.execute_batch("COMMIT").map_err(CoreError::from))
        {
            Ok(()) => {
                self.seen_gen = self.store_gen_val()?;
                self.dirty = true;
                Ok(())
            }
            Err(e) => {
                self.rollback();
                self.index = None;
                self.seen_gen = -1;
                Err(e)
            }
        }
    }

    pub fn update_metadata(&mut self, ids: Vec<String>, metadatas: Vec<String>) -> Result<(), CoreError> {
        if metadatas.len() != ids.len() {
            return Err(CoreError::InvalidArgument(format!(
                "metadatas length {} != ids length {}",
                metadatas.len(),
                ids.len()
            )));
        }
        if ids.is_empty() {
            return Ok(());
        }
        self.ensure_current()?;
        self.conn.execute_batch("BEGIN")?;
        for i in 0..ids.len() {
            let exists: Option<i64> = match self
                .conn
                .query_row("SELECT uid FROM docs WHERE str_id=?1", params![ids[i]], |r| r.get(0))
                .optional()
            {
                Ok(v) => v,
                Err(e) => {
                    self.rollback();
                    return Err(e.into());
                }
            };
            if exists.is_none() {
                self.rollback();
                return Err(CoreError::InvalidArgument(format!("id {:?} not found", ids[i])));
            }
            if let Err(e) = self.conn.execute(
                "UPDATE docs SET metadata=?1 WHERE str_id=?2",
                params![metadatas[i], ids[i]],
            ) {
                self.rollback();
                return Err(e.into());
            }
        }
        let sg = self.store_gen_val()? + 1;
        if self.meta_set("store_gen", &sg.to_string()).is_err() {
            self.rollback();
            return Err(CoreError::Runtime("update_metadata: meta update failed".to_string()));
        }
        if let Err(e) = self.conn.execute_batch("COMMIT") {
            self.rollback();
            self.index = None;
            self.seen_gen = -1;
            return Err(e.into());
        }
        self.seen_gen = sg; // vectors unchanged → index stays valid
        Ok(())
    }

    pub fn update_documents(&mut self, ids: Vec<String>, documents: Vec<String>) -> Result<(), CoreError> {
        if documents.len() != ids.len() {
            return Err(CoreError::InvalidArgument(format!(
                "documents length {} != ids length {}",
                documents.len(),
                ids.len()
            )));
        }
        if ids.is_empty() {
            return Ok(());
        }
        self.ensure_current()?;
        self.conn.execute_batch("BEGIN")?;
        for i in 0..ids.len() {
            let exists: Option<i64> = match self
                .conn
                .query_row("SELECT uid FROM docs WHERE str_id=?1", params![ids[i]], |r| r.get(0))
                .optional()
            {
                Ok(v) => v,
                Err(e) => {
                    self.rollback();
                    return Err(e.into());
                }
            };
            if exists.is_none() {
                self.rollback();
                return Err(CoreError::InvalidArgument(format!("id {:?} not found", ids[i])));
            }
            if let Err(e) = self.conn.execute(
                "UPDATE docs SET document=?1 WHERE str_id=?2",
                params![documents[i], ids[i]],
            ) {
                self.rollback();
                return Err(e.into());
            }
        }
        let sg = self.store_gen_val()? + 1;
        if self.meta_set("store_gen", &sg.to_string()).is_err() {
            self.rollback();
            return Err(CoreError::Runtime("update_documents: meta update failed".to_string()));
        }
        if let Err(e) = self.conn.execute_batch("COMMIT") {
            self.rollback();
            self.index = None;
            self.seen_gen = -1;
            return Err(e.into());
        }
        self.seen_gen = sg; // vectors unchanged → index stays valid
        Ok(())
    }

    pub fn query(
        &mut self,
        text: Option<String>,
        vector: Option<Array2<f32>>,
        k: usize,
        where_: Option<&serde_json::Value>,
        where_document: Option<&serde_json::Value>,
        include: Option<&[String]>,
    ) -> Result<QueryResult, CoreError> {
        if text.is_some() == vector.is_some() {
            return Err(CoreError::InvalidArgument("exactly one of text / vector is required".to_string()));
        }
        let inc = Include::resolve(include, true);

        // Normalized query vector, shape (1, dim). Note: unlike
        // resolve_vectors (add/upsert), this does NOT apply the GAP-1
        // embedder-identity guard — matches the historical query() path,
        // which called the embedder directly rather than through
        // resolve_vectors.
        let q: Array2<f32> = if let Some(t) = text {
            let emb = self.embedder.as_ref().ok_or_else(|| {
                CoreError::EmbedderRequired(
                    "text was provided but this collection has no embedder; pass vectors \
                     instead, or create the collection with embedder=..."
                        .to_string(),
                )
            })?;
            emb.embed(std::slice::from_ref(&t))?
        } else {
            let mut v = vector.unwrap();
            l2_normalize(&mut v);
            v
        };

        self.ensure_current()?;
        if self.index.is_none() || self.dim.is_none() {
            return Ok(Self::empty_query_result(inc.vectors));
        }

        let allow = self.query_allowlist(where_, where_document)?;
        let allow_ids: Option<Vec<u64>> = match &allow {
            Some(uids) => {
                if uids.is_empty() {
                    return Ok(Self::empty_query_result(inc.vectors));
                }
                Some(uids.iter().map(|&x| x as u64).collect())
            }
            None => None,
        };

        let pool = std::cmp::max(k as i64, RERANK_FLOOR) as usize;
        let qrow: Vec<f32> = q.row(0).to_vec();
        let (_, cand_uids_u64) = self.index.as_ref().unwrap().search(&qrow, pool, allow_ids.as_deref());
        let cand_uids: Vec<i64> = cand_uids_u64.iter().map(|&x| x as i64).collect();

        // Exact-cosine re-rank of the candidate pool.
        let mut map: HashMap<i64, (String, String, String, Vec<u8>)> = HashMap::new();
        if !cand_uids.is_empty() {
            let qs = vec!["?"; cand_uids.len()].join(",");
            let sql = format!(
                "SELECT uid, str_id, document, metadata, vector FROM docs WHERE uid IN ({qs})"
            );
            let mut stmt = self.conn.prepare(&sql)?;
            let rows = stmt.query_map(rusqlite::params_from_iter(cand_uids.iter()), |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, Option<String>>(2)?.unwrap_or_default(),
                    r.get::<_, Option<String>>(3)?.unwrap_or_default(),
                    r.get::<_, Vec<u8>>(4)?,
                ))
            })?;
            for row in rows {
                let (u, sid, doc, meta, vec) = row?;
                map.insert(u, (sid, doc, meta, vec));
            }
        }
        let mut scored: Vec<(f32, i64)> = Vec::new();
        for &uid in &cand_uids {
            if let Some((_, _, _, vecb)) = map.get(&uid) {
                let dot: f32 = qrow.iter().zip(blob_to_f32(vecb)).map(|(a, b)| a * b).sum();
                scored.push((1.0 - dot, uid));
            }
        }
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        let mut result = QueryResult {
            vectors: if inc.vectors { Some(Vec::new()) } else { None },
            ..Default::default()
        };
        for (dist, uid) in &scored {
            let (sid, doc, meta, vecb) = map.get(uid).unwrap();
            result.ids.push(sid.clone());
            if inc.distances {
                result.distances.push(*dist as f64);
            }
            if inc.documents {
                result.documents.push(doc.clone());
            }
            if inc.metadatas {
                result.metadatas.push(meta.clone());
            }
            if inc.vectors {
                result.vectors.as_mut().unwrap().push(blob_to_f32(vecb).collect());
            }
        }
        Ok(result)
    }

    pub fn get(
        &self,
        ids: Option<Vec<String>>,
        where_: Option<&serde_json::Value>,
        where_document: Option<&serde_json::Value>,
        limit: Option<i64>,
        offset: Option<i64>,
        include: Option<&[String]>,
    ) -> Result<GetResult, CoreError> {
        let inc = Include::resolve(include, false);
        let mut frags: Vec<String> = Vec::new();
        let mut values: Vec<rusqlite::types::Value> = Vec::new();
        if let Some(idlist) = &ids {
            let qs = vec!["?"; idlist.len()].join(",");
            frags.push(format!("str_id IN ({qs})"));
            for id in idlist {
                values.push(rusqlite::types::Value::Text(id.clone()));
            }
        }
        if let Some(w) = where_ {
            let (s, ps) = filters::compile_where(w, 0)?;
            if !s.is_empty() {
                frags.push(s);
                for p in &ps {
                    values.push(json_to_sql_value(p));
                }
            }
        }
        if let Some(wd) = where_document {
            let (s, ps) = filters::compile_where_document(wd)?;
            if !s.is_empty() {
                frags.push(s);
                for p in &ps {
                    values.push(json_to_sql_value(p));
                }
            }
        }
        let clause = if frags.is_empty() {
            String::new()
        } else {
            format!(" WHERE {}", frags.join(" AND "))
        };
        let mut sql = format!("SELECT str_id, document, metadata, vector FROM docs{clause} ORDER BY uid");
        if let Some(l) = limit {
            sql.push_str(" LIMIT ?");
            values.push(rusqlite::types::Value::Integer(l));
            if let Some(o) = offset {
                sql.push_str(" OFFSET ?");
                values.push(rusqlite::types::Value::Integer(o));
            }
        }

        let mut result = GetResult {
            vectors: if inc.vectors { Some(Vec::new()) } else { None },
            ..Default::default()
        };
        {
            let mut stmt = self.conn.prepare(&sql)?;
            let rows = stmt.query_map(rusqlite::params_from_iter(values.iter()), |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, Option<String>>(1)?.unwrap_or_default(),
                    r.get::<_, Option<String>>(2)?.unwrap_or_default(),
                    r.get::<_, Vec<u8>>(3)?,
                ))
            })?;
            for row in rows {
                let (sid, doc, meta, vecb) = row?;
                result.ids.push(sid);
                result.documents.push(if inc.documents { Some(doc) } else { None });
                result.metadatas.push(if inc.metadatas { meta } else { String::new() });
                if inc.vectors {
                    result.vectors.as_mut().unwrap().push(blob_to_f32(&vecb).collect());
                }
            }
        }
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn reembed(
        &mut self,
        embedder: E,
        dim: Option<i64>,
        bit_width: Option<i64>,
        batch_size: usize,
        mut on_progress: Option<&mut dyn FnMut(i64, i64) -> Result<(), CoreError>>,
        skip_empty: &str,
    ) -> Result<ReembedReport, CoreError> {
        if !matches!(skip_empty, "error" | "keep" | "drop") {
            return Err(CoreError::InvalidArgument(format!(
                "skip_empty must be 'error', 'keep', or 'drop', got {skip_empty:?}"
            )));
        }
        if let Some(bw) = bit_width {
            if !matches!(bw, 2 | 3 | 4) {
                return Err(CoreError::InvalidArgument(format!("bit_width must be 2, 3, or 4, got {bw}")));
            }
        }
        if let Some(dd) = dim {
            if dd <= 0 || dd % 8 != 0 {
                return Err(CoreError::DimensionMismatch(format!(
                    "turbovec requires dim to be a positive multiple of 8, got {dd}"
                )));
            }
        }

        let embedder_identity = embedder.identity();
        let total_docs = self.count()?;
        if total_docs == 0 {
            return Ok(ReembedReport {
                total_docs: 0,
                old_dim: self.dim,
                new_dim: self.dim,
                skipped: 0,
                elapsed_seconds: 0.0,
            });
        }

        self.ensure_current()?;
        let start = std::time::Instant::now();
        let old_dim = self.dim.expect("dim is set when documents exist");
        let old_bit_width = self.bit_width;

        // Phase 1 — classify every doc; nothing is written yet, so any
        // error here (the 'error' policy, an embedder failure, a dim
        // mismatch) leaves the collection untouched.
        let mut embed_uids: Vec<i64> = Vec::new();
        let mut embed_docs: Vec<String> = Vec::new();
        let mut keep: Vec<(i64, Vec<u8>)> = Vec::new();
        let mut drop_uids: Vec<i64> = Vec::new();
        let mut skipped = 0i64;
        {
            let mut stmt = self.conn.prepare("SELECT uid, document, vector FROM docs ORDER BY uid")?;
            let rows = stmt.query_map([], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, Option<String>>(1)?.unwrap_or_default(),
                    r.get::<_, Vec<u8>>(2)?,
                ))
            })?;
            for row in rows {
                let (uid, doc, vecb) = row?;
                if !doc.is_empty() {
                    embed_uids.push(uid);
                    embed_docs.push(doc);
                } else if skip_empty == "error" {
                    return Err(CoreError::InvalidArgument(format!(
                        "Cannot re-embed empty document with uid {uid}"
                    )));
                } else if skip_empty == "drop" {
                    drop_uids.push(uid);
                    skipped += 1;
                } else {
                    keep.push((uid, vecb));
                    skipped += 1;
                }
            }
        }

        // Embed the non-empty docs; new_dim comes from the real embedder
        // output. Any embedder-call failure collapses to a single
        // InvalidArgument (matching the historical `emb_call` wrapper,
        // which always raised a plain ValueError regardless of cause).
        let mut updates: Vec<(Vec<u8>, i64)> = Vec::new();
        let mut new_dim: Option<i64> = None;
        let mut processed = 0i64;
        let bs = batch_size.max(1);
        let mut start_i = 0;
        while start_i < embed_docs.len() {
            let end = std::cmp::min(start_i + bs, embed_docs.len());
            let b_docs = &embed_docs[start_i..end];
            let b_uids = &embed_uids[start_i..end];
            let out = embedder.embed(b_docs).map_err(|e| {
                CoreError::InvalidArgument(format!(
                    "Embedder failed on batch uids [{}..{}]: {}",
                    b_uids[0],
                    b_uids[b_uids.len() - 1],
                    e
                ))
            })?;
            if out.nrows() != b_uids.len() {
                return Err(CoreError::DimensionMismatch(format!(
                    "embedder must return one 2-D vector per document; got array of shape {:?} for {} documents",
                    out.shape(),
                    b_uids.len()
                )));
            }
            let bdim = out.ncols() as i64;
            match new_dim {
                None => new_dim = Some(bdim),
                Some(nd) if bdim != nd => {
                    return Err(CoreError::DimensionMismatch(format!(
                        "embedder returned inconsistent dimensions across batches: {nd} then {bdim}"
                    )));
                }
                _ => {}
            }
            for (i, &uid) in b_uids.iter().enumerate() {
                let bytes = f32_to_blob(out.row(i).iter().copied());
                updates.push((bytes, uid));
            }
            processed += b_docs.len() as i64;
            if let Some(cb) = on_progress.as_deref_mut() {
                cb(processed, total_docs)?;
            }
            start_i = end;
        }

        let new_dim_v = new_dim.unwrap_or(old_dim);
        if let Some(dd) = dim {
            if new_dim_v != dd {
                return Err(CoreError::DimensionMismatch(format!(
                    "Embedder returned vectors of dimension {new_dim_v}, but dim was explicitly set to {dd}"
                )));
            }
        }
        if !keep.is_empty() && new_dim_v != old_dim {
            return Err(CoreError::DimensionMismatch(format!(
                "skip_empty='keep' cannot retain {old_dim}-dim vectors in a {new_dim_v}-dim \
                 collection; use skip_empty='drop' or give the empty documents text to embed"
            )));
        }
        // Retained docs keep their old vector (dims match new_dim here); re-normalize.
        for (uid, vecb) in &keep {
            let v: Vec<f32> = blob_to_f32(vecb).collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            let denom = if norm == 0.0 { 1.0 } else { norm };
            let bytes: Vec<u8> = v.iter().flat_map(|x| (x / denom).to_ne_bytes()).collect();
            updates.push((bytes, *uid));
        }
        if let Some(cb) = on_progress.as_deref_mut() {
            if skipped > 0 {
                cb(processed + skipped, total_docs)?;
            }
        }
        let new_bit_width = bit_width.unwrap_or(old_bit_width);

        // Phase 2 — apply in one transaction; rebuild the index from the
        // uncommitted rows, commit only if that succeeds, else roll back.
        self.conn.execute_batch("BEGIN")?;
        for chunk in drop_uids.chunks(900) {
            let qs = vec!["?"; chunk.len()].join(",");
            if let Err(e) = self.conn.execute(
                &format!("DELETE FROM docs WHERE uid IN ({qs})"),
                rusqlite::params_from_iter(chunk.iter()),
            ) {
                self.rollback();
                self.dim = Some(old_dim);
                self.bit_width = old_bit_width;
                self.index = None;
                self.seen_gen = -1;
                return Err(e.into());
            }
        }
        for (bytes, uid) in &updates {
            if let Err(e) = self.conn.execute("UPDATE docs SET vector=?1 WHERE uid=?2", params![bytes, uid]) {
                self.rollback();
                self.dim = Some(old_dim);
                self.bit_width = old_bit_width;
                self.index = None;
                self.seen_gen = -1;
                return Err(e.into());
            }
        }
        self.dim = Some(new_dim_v);
        self.bit_width = new_bit_width;
        if let Err(e) = self.set_reembed_meta(new_dim_v, new_bit_width, &embedder_identity) {
            self.rollback();
            self.dim = Some(old_dim);
            self.bit_width = old_bit_width;
            self.index = None;
            self.seen_gen = -1;
            return Err(e);
        }
        if let Err(e) = self.reload_index() {
            self.rollback();
            self.dim = Some(old_dim);
            self.bit_width = old_bit_width;
            self.index = None;
            self.seen_gen = -1;
            return Err(e);
        }
        if let Err(e) = self.conn.execute_batch("COMMIT") {
            self.rollback();
            self.dim = Some(old_dim);
            self.bit_width = old_bit_width;
            self.index = None;
            self.seen_gen = -1;
            return Err(e.into());
        }

        let elapsed = start.elapsed().as_secs_f64();
        Ok(ReembedReport {
            total_docs,
            old_dim: Some(old_dim),
            new_dim: Some(new_dim_v),
            skipped,
            elapsed_seconds: elapsed,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::ConstantEmbedder;
    use crate::index::FakeIndex;

    fn temp_dir(name: &str) -> String {
        let dir = std::env::temp_dir()
            .join(format!("turbovecdb-core-collection-test-{}-{}", std::process::id(), name));
        let _ = std::fs::remove_dir_all(&dir);
        dir.to_str().unwrap().to_string()
    }

    struct NamedEmbedder {
        name: String,
        dim: usize,
    }

    impl Embedder for NamedEmbedder {
        fn embed(&self, docs: &[String]) -> Result<Array2<f32>, CoreError> {
            Ok(Array2::from_elem((docs.len(), self.dim), 1.0 / (self.dim as f32).sqrt()))
        }

        fn identity(&self) -> String {
            self.name.clone()
        }
    }

    fn new_collection(dir: &str, dim: Option<i64>) -> Collection<ConstantEmbedder, FakeIndex> {
        Collection::new(dir.to_string(), dim, 4, None, Some(ConstantEmbedder { dim: 8 })).unwrap()
    }

    fn vecs(rows: &[[f32; 8]]) -> Array2<f32> {
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        Array2::from_shape_vec((rows.len(), 8), flat).unwrap()
    }

    #[test]
    fn add_and_get_roundtrip() {
        let dir = temp_dir("add_get");
        let mut c = new_collection(&dir, Some(8));
        c.add(
            vec!["a".into(), "b".into()],
            Some(vec!["doc-a".into(), "doc-b".into()]),
            None,
            Some(vecs(&[[1., 0., 0., 0., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 0., 0., 0.]])),
        )
        .unwrap();
        assert_eq!(c.count().unwrap(), 2);
        let r = c.get(None, None, None, None, None, None).unwrap();
        assert_eq!(r.ids, vec!["a", "b"]);
        assert_eq!(r.documents, vec![Some("doc-a".to_string()), Some("doc-b".to_string())]);
    }

    #[test]
    fn add_duplicate_id_errors() {
        let dir = temp_dir("dup");
        let mut c = new_collection(&dir, Some(8));
        c.add(vec!["a".into()], None, None, Some(vecs(&[[1., 0., 0., 0., 0., 0., 0., 0.]]))).unwrap();
        let e = c
            .add(vec!["a".into()], None, None, Some(vecs(&[[0., 1., 0., 0., 0., 0., 0., 0.]])))
            .unwrap_err();
        match e {
            CoreError::InvalidArgument(m) => assert!(m.contains("already exists")),
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn upsert_replaces_existing() {
        let dir = temp_dir("upsert");
        let mut c = new_collection(&dir, Some(8));
        c.add(vec!["a".into()], Some(vec!["v1".into()]), None, Some(vecs(&[[1., 0., 0., 0., 0., 0., 0., 0.]])))
            .unwrap();
        c.upsert(vec!["a".into()], Some(vec!["v2".into()]), None, Some(vecs(&[[0., 1., 0., 0., 0., 0., 0., 0.]])))
            .unwrap();
        assert_eq!(c.count().unwrap(), 1);
        let r = c.get(None, None, None, None, None, None).unwrap();
        assert_eq!(r.documents, vec![Some("v2".to_string())]);
    }

    #[test]
    fn dimension_mismatch_is_rejected() {
        let dir = temp_dir("dimmismatch");
        let mut c = new_collection(&dir, Some(8));
        c.add(vec!["a".into()], None, None, Some(vecs(&[[1., 0., 0., 0., 0., 0., 0., 0.]]))).unwrap();
        let bad = Array2::from_shape_vec((1, 4), vec![1.0, 0.0, 0.0, 0.0]).unwrap();
        let e = c.add(vec!["b".into()], None, None, Some(bad)).unwrap_err();
        assert!(matches!(e, CoreError::DimensionMismatch(_)));
    }

    #[test]
    fn delete_by_ids_and_where() {
        let dir = temp_dir("delete");
        let mut c = new_collection(&dir, Some(8));
        c.add(
            vec!["a".into(), "b".into(), "c".into()],
            None,
            Some(vec![r#"{"k":1}"#.into(), r#"{"k":2}"#.into(), r#"{"k":1}"#.into()]),
            Some(vecs(&[
                [1., 0., 0., 0., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0., 0., 0., 0.],
            ])),
        )
        .unwrap();
        c.delete(Some(vec!["a".into()]), None).unwrap();
        assert_eq!(c.count().unwrap(), 2);
        let filter = serde_json::json!({"k": 1});
        c.delete(None, Some(&filter)).unwrap();
        assert_eq!(c.count().unwrap(), 1);
    }

    #[test]
    fn clear_removes_everything_but_keeps_config() {
        let dir = temp_dir("clear");
        let mut c = new_collection(&dir, Some(8));
        c.add(vec!["a".into()], None, None, Some(vecs(&[[1., 0., 0., 0., 0., 0., 0., 0.]]))).unwrap();
        c.clear().unwrap();
        assert_eq!(c.count().unwrap(), 0);
        assert_eq!(c.dim(), Some(8));
    }

    /// Force the next COMMIT to fail deterministically: SQLite converts a
    /// commit_hook that returns `true` into a rollback, so `execute_batch
    /// ("COMMIT")` itself returns an error. Regression for C2: clear() /
    /// update_metadata() / update_documents() used to propagate that with a
    /// bare `?`, leaving the transaction open on the connection so every
    /// subsequent write failed with "cannot start a transaction within a
    /// transaction".
    fn fail_next_commit<E: Embedder, I: VectorIndex>(c: &mut Collection<E, I>) {
        c.conn.commit_hook(Some(|| true));
    }

    #[test]
    fn commit_failure_in_clear_rolls_back_and_recovers() {
        let dir = temp_dir("commit_fail_clear");
        let mut c = new_collection(&dir, Some(8));
        c.add(vec!["a".into()], None, None, Some(vecs(&[[1., 0., 0., 0., 0., 0., 0., 0.]]))).unwrap();

        fail_next_commit(&mut c);
        let err = c.clear().unwrap_err();
        assert!(matches!(err, CoreError::Sql(_)), "expected CoreError::Sql, got {err:?}");
        c.conn.commit_hook(None::<fn() -> bool>);

        // The rolled-back clear must not have removed the row...
        assert_eq!(c.count().unwrap(), 1);
        // ...and the connection must not be stuck mid-transaction.
        c.add(vec!["b".into()], None, None, Some(vecs(&[[0., 1., 0., 0., 0., 0., 0., 0.]]))).unwrap();
        assert_eq!(c.count().unwrap(), 2);
    }

    #[test]
    fn commit_failure_in_update_metadata_rolls_back_and_recovers() {
        let dir = temp_dir("commit_fail_update_metadata");
        let mut c = new_collection(&dir, Some(8));
        c.add(vec!["a".into()], None, Some(vec![r#"{"x":1}"#.into()]), Some(vecs(&[[1., 0., 0., 0., 0., 0., 0., 0.]])))
            .unwrap();

        fail_next_commit(&mut c);
        let err = c.update_metadata(vec!["a".into()], vec![r#"{"x":2}"#.into()]).unwrap_err();
        assert!(matches!(err, CoreError::Sql(_)), "expected CoreError::Sql, got {err:?}");
        c.conn.commit_hook(None::<fn() -> bool>);

        let include = ["metadatas".to_string()];
        let r = c.get(None, None, None, None, None, Some(&include)).unwrap();
        assert_eq!(r.metadatas, vec![r#"{"x":1}"#.to_string()], "rolled-back update must not stick");

        c.update_metadata(vec!["a".into()], vec![r#"{"x":3}"#.into()]).unwrap();
        let r2 = c.get(None, None, None, None, None, Some(&include)).unwrap();
        assert_eq!(r2.metadatas, vec![r#"{"x":3}"#.to_string()]);
    }

    #[test]
    fn commit_failure_in_update_documents_rolls_back_and_recovers() {
        let dir = temp_dir("commit_fail_update_documents");
        let mut c = new_collection(&dir, Some(8));
        c.add(vec!["a".into()], Some(vec!["orig".into()]), None, Some(vecs(&[[1., 0., 0., 0., 0., 0., 0., 0.]])))
            .unwrap();

        fail_next_commit(&mut c);
        let err = c.update_documents(vec!["a".into()], vec!["v2".into()]).unwrap_err();
        assert!(matches!(err, CoreError::Sql(_)), "expected CoreError::Sql, got {err:?}");
        c.conn.commit_hook(None::<fn() -> bool>);

        let include = ["documents".to_string()];
        let r = c.get(None, None, None, None, None, Some(&include)).unwrap();
        assert_eq!(r.documents, vec![Some("orig".to_string())], "rolled-back update must not stick");

        c.update_documents(vec!["a".into()], vec!["v3".into()]).unwrap();
        let r2 = c.get(None, None, None, None, None, Some(&include)).unwrap();
        assert_eq!(r2.documents, vec![Some("v3".to_string())]);
    }

    #[test]
    fn update_metadata_and_documents() {
        let dir = temp_dir("update");
        let mut c = new_collection(&dir, Some(8));
        c.add(vec!["a".into()], Some(vec!["orig".into()]), None, Some(vecs(&[[1., 0., 0., 0., 0., 0., 0., 0.]])))
            .unwrap();
        c.update_documents(vec!["a".into()], vec!["updated".into()]).unwrap();
        c.update_metadata(vec!["a".into()], vec![r#"{"x":1}"#.into()]).unwrap();
        let include = ["documents".to_string(), "metadatas".to_string()];
        let r = c.get(None, None, None, None, None, Some(&include)).unwrap();
        assert_eq!(r.documents, vec![Some("updated".to_string())]);
        assert_eq!(r.metadatas, vec![r#"{"x":1}"#.to_string()]);
    }

    #[test]
    fn update_missing_id_errors() {
        let dir = temp_dir("update_missing");
        let mut c = new_collection(&dir, Some(8));
        let e = c.update_documents(vec!["nope".into()], vec!["x".into()]).unwrap_err();
        assert!(matches!(e, CoreError::InvalidArgument(_)));
    }

    #[test]
    fn query_returns_nearest_by_cosine() {
        let dir = temp_dir("query");
        let mut c = new_collection(&dir, Some(8));
        c.add(
            vec!["a".into(), "b".into(), "c".into()],
            None,
            None,
            Some(vecs(&[
                [1., 0., 0., 0., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0., 0., 0., 0.],
                [0.7071, 0.7071, 0., 0., 0., 0., 0., 0.],
            ])),
        )
        .unwrap();
        let q = Array2::from_shape_vec((1, 8), vec![1., 0., 0., 0., 0., 0., 0., 0.]).unwrap();
        let r = c.query(None, Some(q), 1, None, None, None).unwrap();
        assert_eq!(r.ids, vec!["a"]);
    }

    #[test]
    fn health_reports_ok_and_coherence() {
        let dir = temp_dir("health");
        let mut c = new_collection(&dir, Some(8));
        c.add(vec!["a".into()], None, None, Some(vecs(&[[1., 0., 0., 0., 0., 0., 0., 0.]]))).unwrap();
        let h = c.health().unwrap();
        assert!(h.ok);
        assert_eq!(h.doc_count, 1);
    }

    #[test]
    fn flush_close_and_reopen_preserve_data() {
        let dir = temp_dir("reopen");
        {
            let mut c = new_collection(&dir, Some(8));
            c.add(vec!["a".into()], Some(vec!["hi".into()]), None, Some(vecs(&[[1., 0., 0., 0., 0., 0., 0., 0.]])))
                .unwrap();
            c.close().unwrap();
        }
        let c2 = new_collection(&dir, None);
        assert_eq!(c2.count().unwrap(), 1);
        let h = c2.health().unwrap();
        assert!(h.coherent);
    }

    #[test]
    fn reembed_changes_dim_and_records_new_identity() {
        let dir = temp_dir("reembed");
        let mut c: Collection<NamedEmbedder, FakeIndex> = Collection::new(
            dir,
            Some(8),
            4,
            None,
            Some(NamedEmbedder { name: "old-embedder".into(), dim: 8 }),
        )
        .unwrap();
        c.add(
            vec!["a".into(), "b".into()],
            Some(vec!["x".into(), "y".into()]),
            None,
            Some(vecs(&[[1., 0., 0., 0., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 0., 0., 0.]])),
        )
        .unwrap();
        let report = c
            .reembed(NamedEmbedder { name: "new-embedder".into(), dim: 8 }, None, None, 10, None, "error")
            .unwrap();
        assert_eq!(report.total_docs, 2);
        assert_eq!(report.skipped, 0);
        assert_eq!(c.meta_get("embedder_identity").unwrap(), Some("new-embedder".to_string()));
    }

    #[test]
    fn embedder_identity_mismatch_blocks_write() {
        let dir = temp_dir("identity");
        let mut c: Collection<NamedEmbedder, FakeIndex> = Collection::new(
            dir.clone(),
            Some(8),
            4,
            None,
            Some(NamedEmbedder { name: "embedder-a".into(), dim: 8 }),
        )
        .unwrap();
        c.add(vec!["a".into()], Some(vec!["x".into()]), None, None).unwrap();
        drop(c);

        let mut c2: Collection<NamedEmbedder, FakeIndex> = Collection::new(
            dir,
            Some(8),
            4,
            None,
            Some(NamedEmbedder { name: "embedder-b".into(), dim: 8 }),
        )
        .unwrap();
        let e = c2.add(vec!["b".into()], Some(vec!["y".into()]), None, None).unwrap_err();
        assert!(matches!(e, CoreError::EmbedderIdentityMismatch(_)));
    }
}
