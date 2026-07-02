//! Row-wise vector math shared by the embedder and raw-vector write paths.

use ndarray::{Array2, Axis};

/// L2-normalize each row in place. Rows with zero norm are left unchanged
/// (divided by 1.0) rather than producing NaNs. Faithful port of the
/// per-row loop in the historical `turbovecdb-py::l2_normalize_impl` — the
/// array-coercion half of that function (accepting arbitrary Python
/// array-likes) stays in `turbovecdb-py`, since it's inherently a PyO3/numpy
/// concern; this is the pure math it delegates to.
pub fn l2_normalize(matrix: &mut Array2<f32>) {
    for mut row in matrix.axis_iter_mut(Axis(0)) {
        let norm = row.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let denom = if norm == 0.0 { 1.0 } else { norm };
        row.mapv_inplace(|x| x / denom);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_rows_to_unit_length() {
        let mut m = Array2::from_shape_vec((2, 2), vec![3.0, 4.0, 0.0, 0.0]).unwrap();
        l2_normalize(&mut m);
        let n0: f32 = m.row(0).iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n0 - 1.0).abs() < 1e-6);
        assert_eq!(m.row(1).to_vec(), vec![0.0, 0.0]);
    }
}
