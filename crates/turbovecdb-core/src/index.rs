//! Narrow ANN-index abstraction.
//!
//! Only the operations `Collection` actually calls on the index object
//! today (`add_with_ids`/`remove`/`search`/`write` — confirmed by grepping
//! every `index.call_method*` site in the pre-split `collection.rs`). This
//! is NOT a general swappable-backend surface; its only reason to exist is
//! so `turbovecdb-core` can be exercised via `cargo test` with a fake
//! in-memory impl, without linking the real `turbovec` crate (and its BLAS
//! build dependency, see `docs/rust-core-split-design.md`). Resist adding
//! methods "for completeness." The production implementation — first a
//! PyO3 callback into the Python `turbovec` wheel (split phase 5/8), later
//! `turbovec::IdMapIndex` directly (split phase 4/8) — lives outside this
//! crate.

use crate::error::CoreError;

pub trait VectorIndex: Send {
    fn add_with_ids(&mut self, vectors: &[f32], ids: &[u64]) -> Result<(), CoreError>;
    fn remove(&mut self, id: u64) -> bool;
    /// Row-major `(scores, ids)` for the top-`k` matches per query row.
    /// `allowlist`, when `Some`, restricts results to those external ids.
    fn search(&self, queries: &[f32], k: usize, allowlist: Option<&[u64]>) -> (Vec<f32>, Vec<u64>);
    fn write(&self, path: &str) -> Result<(), CoreError>;
}

#[cfg(test)]
pub(crate) struct FakeIndex {
    dim: usize,
    entries: Vec<(u64, Vec<f32>)>,
}

#[cfg(test)]
impl FakeIndex {
    pub fn new(dim: usize) -> Self {
        Self { dim, entries: Vec::new() }
    }
}

#[cfg(test)]
impl VectorIndex for FakeIndex {
    fn add_with_ids(&mut self, vectors: &[f32], ids: &[u64]) -> Result<(), CoreError> {
        for (i, &id) in ids.iter().enumerate() {
            let row = vectors[i * self.dim..(i + 1) * self.dim].to_vec();
            self.entries.push((id, row));
        }
        Ok(())
    }

    fn remove(&mut self, id: u64) -> bool {
        let before = self.entries.len();
        self.entries.retain(|(eid, _)| *eid != id);
        self.entries.len() != before
    }

    fn search(&self, queries: &[f32], k: usize, allowlist: Option<&[u64]>) -> (Vec<f32>, Vec<u64>) {
        let mut candidates: Vec<&(u64, Vec<f32>)> = self
            .entries
            .iter()
            .filter(|(id, _)| allowlist.is_none_or(|a| a.contains(id)))
            .collect();
        // Squared-distance nearest-neighbor — good enough for a test double.
        candidates.sort_by(|a, b| {
            let da: f32 = a.1.iter().zip(queries).map(|(x, y)| (x - y).powi(2)).sum();
            let db: f32 = b.1.iter().zip(queries).map(|(x, y)| (x - y).powi(2)).sum();
            da.partial_cmp(&db).unwrap()
        });
        candidates.truncate(k);
        let scores = candidates
            .iter()
            .map(|(_, v)| v.iter().zip(queries).map(|(x, y)| (x - y).powi(2)).sum())
            .collect();
        let ids = candidates.iter().map(|(id, _)| *id).collect();
        (scores, ids)
    }

    fn write(&self, _path: &str) -> Result<(), CoreError> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_search_remove_roundtrip() {
        let mut idx = FakeIndex::new(2);
        idx.add_with_ids(&[0.0, 0.0, 1.0, 1.0, 2.0, 2.0], &[10, 20, 30]).unwrap();
        let (_, ids) = idx.search(&[0.1, 0.1], 1, None);
        assert_eq!(ids, vec![10]);

        assert!(idx.remove(10));
        let (_, ids) = idx.search(&[0.1, 0.1], 1, None);
        assert_eq!(ids, vec![20]);

        let (_, ids) = idx.search(&[0.0, 0.0], 5, Some(&[30]));
        assert_eq!(ids, vec![30]);
    }
}
