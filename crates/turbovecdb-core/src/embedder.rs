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

pub trait Embedder: Send {
    /// Embed a batch of documents, returning one row per document.
    fn embed(&self, docs: &[String]) -> Result<Array2<f32>, CoreError>;
}

#[cfg(test)]
pub(crate) struct ConstantEmbedder {
    pub dim: usize,
}

#[cfg(test)]
impl Embedder for ConstantEmbedder {
    fn embed(&self, docs: &[String]) -> Result<Array2<f32>, CoreError> {
        Ok(Array2::from_elem((docs.len(), self.dim), 1.0f32))
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
