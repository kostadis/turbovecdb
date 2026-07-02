//! Native result types, converted to the public `turbovecdb.collection`
//! Python dataclasses in exactly one place: `turbovecdb-py::convert`.
//!
//! Metadata travels as raw JSON *text* (`String`), not `serde_json::Value`,
//! deliberately: Python's `json.dumps`/`json.loads` (still used at the
//! `turbovecdb-py` boundary) accept non-standard tokens Python's json module
//! supports but `serde_json` doesn't (e.g. `NaN`/`Infinity` floats).
//! Round-tripping through `serde_json::Value` would silently change that
//! behavior for free-form metadata — seeing `docs/rust-core-split-design.md`'s
//! own caution about this. Filters (`turbovecdb-core::filters`) are exempt:
//! `where`/`where_document` are already JSON-shaped by the filter grammar,
//! so `serde_json::Value` there is exact, not lossy.

/// Which fields to materialize in a query/get result.
#[derive(Clone, Copy)]
pub struct Include {
    pub documents: bool,
    pub metadatas: bool,
    pub distances: bool,
    pub vectors: bool,
}

impl Include {
    pub fn resolve(include: Option<&[String]>, default_distances: bool) -> Self {
        match include {
            None => Include {
                documents: true,
                metadatas: true,
                distances: default_distances,
                vectors: false,
            },
            Some(keys) => {
                let has = |k: &str| keys.iter().any(|s| s == k);
                Include {
                    documents: has("documents"),
                    metadatas: has("metadatas"),
                    distances: has("distances"),
                    vectors: has("vectors"),
                }
            }
        }
    }
}

/// Flat, single-query result. Fields excluded by `include` are empty
/// vectors; `vectors` is `None` when not requested (matches the historical
/// Python `QueryResult` dataclass exactly). `metadatas` holds raw JSON text
/// (or `""` for "no metadata") — `turbovecdb-py::convert` does the final
/// `json.loads`.
#[derive(Default)]
pub struct QueryResult {
    pub ids: Vec<String>,
    pub distances: Vec<f64>,
    pub documents: Vec<String>,
    pub metadatas: Vec<String>,
    pub vectors: Option<Vec<Vec<f32>>>,
}

/// `get()` result. Unlike `QueryResult`, `documents`/`metadatas` are always
/// present with one entry per id — `None`/`""` per row when excluded by
/// `include`, matching the historical per-row placeholder behavior.
/// `vectors` is whole-field `None` when not requested.
#[derive(Default)]
pub struct GetResult {
    pub ids: Vec<String>,
    pub documents: Vec<Option<String>>,
    pub metadatas: Vec<String>,
    pub vectors: Option<Vec<Vec<f32>>>,
}

pub struct HealthResult {
    pub ok: bool,
    pub quick_check: String,
    pub store_gen: i64,
    pub tvim_gen: i64,
    pub coherent: bool,
    pub doc_count: i64,
}

/// `old_dim`/`new_dim` are `Option` to faithfully mirror the historical
/// early-return-on-zero-docs case, which passed `Collection.dim` (itself
/// `Option<i64>`, `None` on a never-written collection) straight through to
/// the Python dataclass's `int`-typed fields — Python doesn't enforce
/// dataclass field types at runtime, so that `None` was always a latent
/// possibility; this makes it explicit instead of unwrapping and risking a
/// panic on that edge case.
pub struct ReembedReport {
    pub total_docs: i64,
    pub old_dim: Option<i64>,
    pub new_dim: Option<i64>,
    pub skipped: i64,
    pub elapsed_seconds: f64,
}
