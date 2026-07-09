//! Abstraction over "turn documents into vectors."
//!
//! Embedders are inherently user-supplied code (arbitrary Python callables
//! in practice), so calling into Python to run one is a legitimate,
//! permanent boundary — this trait doesn't try to eliminate that. It exists
//! so `turbovecdb-core`'s write/query/reembed logic doesn't hardcode "the
//! embedder is a Python object," and so it's exercisable via `cargo test`
//! with a fake implementation. `turbovecdb-py::PyEmbedder` (wrapping a
//! `PyObject`) is the only production implementation.

use ndarray::Array2;

use crate::error::CoreError;

pub trait Embedder: Send + Sync {
    /// Embed a batch of documents, returning one L2-normalized row per
    /// document (row-normalization is the caller's contract — `Collection`
    /// relies on embedder output already being unit-norm, matching the
    /// historical `resolve_vectors`, which normalized the embedder's output
    /// exactly once at the call site).
    fn embed(&self, docs: &[String]) -> Result<Array2<f32>, CoreError>;

    /// A stable string identifying this embedder (its Python
    /// `__name__`/`__class__` today). Persisted so `Collection` can detect
    /// an accidental embedder swap on the write path (the "GAP-1 guard").
    fn identity(&self) -> String;
}

/// An embedder that only supports vector input (panics if asked to embed text).
/// Useful for CLI tools and non-Python consumers that pass vectors directly.
pub struct NoEmbedder;

impl Embedder for NoEmbedder {
    fn embed(&self, _docs: &[String]) -> Result<Array2<f32>, CoreError> {
        Err(CoreError::InvalidArgument(
            "NoEmbedder cannot embed text; pass vectors directly".to_string()
        ))
    }

    fn identity(&self) -> String {
        "NoEmbedder".to_string()
    }
}

#[cfg(test)]
pub(crate) struct ConstantEmbedder {
    pub dim: usize,
}

#[cfg(test)]
impl Embedder for ConstantEmbedder {
    fn embed(&self, docs: &[String]) -> Result<Array2<f32>, CoreError> {
        Ok(Array2::from_elem((docs.len(), self.dim), 1.0f32 / (self.dim as f32).sqrt()))
    }

    fn identity(&self) -> String {
        "ConstantEmbedder".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fake_embedder_returns_one_row_per_doc() {
        let e = ConstantEmbedder { dim: 4 };
        let out = e.embed(&["a".to_string(), "b".to_string()]).unwrap();
        assert_eq!(out.shape(), &[2, 4]);
    }
}
