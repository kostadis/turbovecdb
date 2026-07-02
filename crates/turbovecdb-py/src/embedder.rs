//! `PyEmbedder` — the one production implementation of
//! `turbovecdb_core::embedder::Embedder`, wrapping an arbitrary Python
//! callable. This is the legitimate, permanent Python callback the split
//! design doesn't try to eliminate (see `docs/rust-core-split-design.md`):
//! embedders are inherently user-supplied code.

use numpy::ndarray::{Array2, Ix2};
use numpy::PyReadonlyArrayDyn;
use pyo3::prelude::*;
use pyo3::types::PyList;
use turbovecdb_core::embedder::Embedder;
use turbovecdb_core::error::CoreError;

pub(crate) struct PyEmbedder {
    callable: PyObject,
}

impl PyEmbedder {
    pub(crate) fn new(callable: PyObject) -> Self {
        Self { callable }
    }
}

impl Embedder for PyEmbedder {
    fn embed(&self, docs: &[String]) -> Result<Array2<f32>, CoreError> {
        Python::with_gil(|py| {
            let out = self
                .callable
                .bind(py)
                .call1((PyList::new_bound(py, docs),))
                .map_err(|e| CoreError::Other(format!("embedder call failed: {e}")))?;
            let normed = crate::l2_normalize_impl(py, &out)
                .map_err(|e| CoreError::Other(format!("embedder output invalid: {e}")))?;
            let readonly: PyReadonlyArrayDyn<f32> = normed
                .extract()
                .map_err(|e| CoreError::Other(format!("embedder output invalid: {e}")))?;
            let arr = readonly
                .as_array()
                .to_owned()
                .into_dimensionality::<Ix2>()
                .map_err(|e| CoreError::Other(e.to_string()))?;
            Ok(arr)
        })
    }

    /// Mirror of the historical `Collection::embedder_identity`.
    fn identity(&self) -> String {
        Python::with_gil(|py| {
            let b = self.callable.bind(py);
            if let Ok(name) = b.getattr("__name__") {
                if let Ok(s) = name.extract::<String>() {
                    return s;
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
                return format!("{module}.{qn}");
            }
            "unknown_embedder".to_string()
        })
    }
}
