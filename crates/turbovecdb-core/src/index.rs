//! Narrow ANN-index abstraction.
//!
//! Only the operations `Collection` actually calls on the index object
//! today (`add_with_ids`/`remove`/`search`/`write` — confirmed by grepping
//! every `index.call_method*` site in the pre-split `collection.rs`). This
//! is NOT a general swappable-backend surface; its only reason to exist is
//! so `turbovecdb-core` can be exercised via `cargo test` with a fake
//! in-memory impl, without always paying for the real `turbovec` crate (and
//! its BLAS build dependency, see `docs/rust-core-split-design.md`). Resist
//! adding methods "for completeness."
//!
//! `TurbovecIndex` is the production implementation, wrapping
//! `turbovec::IdMapIndex` (the native crate confirmed API/Send/.tvim-format
//! compatible with the Python wheel in the split-phase-0 spike) directly —
//! no PyO3 callback into the Python `turbovec` wheel.

use crate::error::CoreError;

pub trait VectorIndex: Send
where
    Self: Sized,
{
    fn new(dim: usize, bit_width: usize) -> Result<Self, CoreError>;
    fn load(path: &str) -> Result<Self, CoreError>;
    fn add_with_ids(&mut self, vectors: &[f32], ids: &[u64]) -> Result<(), CoreError>;
    fn remove(&mut self, id: u64) -> bool;
    /// Row-major `(scores, ids)` for the top-`k` matches per query row.
    /// `allowlist`, when `Some`, restricts results to those external ids.
    fn search(&self, queries: &[f32], k: usize, allowlist: Option<&[u64]>) -> (Vec<f32>, Vec<u64>);
    fn write(&self, path: &str) -> Result<(), CoreError>;
}

pub struct TurbovecIndex(turbovec::IdMapIndex);

impl TurbovecIndex {
    pub fn dim(&self) -> usize {
        self.0.dim()
    }

    pub fn bit_width(&self) -> usize {
        self.0.bit_width()
    }
}

impl VectorIndex for TurbovecIndex {
    fn new(dim: usize, bit_width: usize) -> Result<Self, CoreError> {
        turbovec::IdMapIndex::new(dim, bit_width)
            .map(Self)
            .map_err(|e| CoreError::Other(e.to_string()))
    }

    fn load(path: &str) -> Result<Self, CoreError> {
        turbovec::IdMapIndex::load(path).map(Self).map_err(CoreError::from)
    }

    fn add_with_ids(&mut self, vectors: &[f32], ids: &[u64]) -> Result<(), CoreError> {
        self.0
            .add_with_ids(vectors, ids)
            .map_err(|e| CoreError::Other(e.to_string()))
    }

    fn remove(&mut self, id: u64) -> bool {
        self.0.remove(id)
    }

    fn search(&self, queries: &[f32], k: usize, allowlist: Option<&[u64]>) -> (Vec<f32>, Vec<u64>) {
        self.0.search_with_allowlist(queries, k, allowlist)
    }

    fn write(&self, path: &str) -> Result<(), CoreError> {
        self.0.write(path).map_err(CoreError::from)
    }
}

#[cfg(test)]
pub(crate) struct FakeIndex {
    dim: usize,
    entries: Vec<(u64, Vec<f32>)>,
}

#[cfg(test)]
impl VectorIndex for FakeIndex {
    fn new(dim: usize, _bit_width: usize) -> Result<Self, CoreError> {
        Ok(Self { dim, entries: Vec::new() })
    }

    fn load(path: &str) -> Result<Self, CoreError> {
        let data = std::fs::read_to_string(path).map_err(CoreError::from)?;
        let mut lines = data.lines();
        let dim: usize = lines.next().unwrap_or("0").parse().unwrap_or(0);
        let mut entries = Vec::new();
        for line in lines {
            let mut parts = line.split(',');
            let id: u64 = parts.next().unwrap().parse().unwrap();
            let row: Vec<f32> = parts.map(|p| p.parse().unwrap()).collect();
            entries.push((id, row));
        }
        Ok(Self { dim, entries })
    }

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

    fn write(&self, path: &str) -> Result<(), CoreError> {
        let mut out = format!("{}\n", self.dim);
        for (id, row) in &self.entries {
            let row_str: Vec<String> = row.iter().map(|f| f.to_string()).collect();
            out.push_str(&format!("{},{}\n", id, row_str.join(",")));
        }
        std::fs::write(path, out).map_err(CoreError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_search_remove_roundtrip() {
        let mut idx = FakeIndex::new(2, 4).unwrap();
        idx.add_with_ids(&[0.0, 0.0, 1.0, 1.0, 2.0, 2.0], &[10, 20, 30]).unwrap();
        let (_, ids) = idx.search(&[0.1, 0.1], 1, None);
        assert_eq!(ids, vec![10]);

        assert!(idx.remove(10));
        let (_, ids) = idx.search(&[0.1, 0.1], 1, None);
        assert_eq!(ids, vec![20]);

        let (_, ids) = idx.search(&[0.0, 0.0], 5, Some(&[30]));
        assert_eq!(ids, vec![30]);
    }

    #[test]
    fn turbovec_index_add_search_remove_write_load_roundtrip() {
        let dir = std::env::temp_dir().join(format!(
            "turbovecdb-core-test-{}-{}",
            std::process::id(),
            "turbovec_index_add_search_remove_write_load_roundtrip"
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.tvim");

        let mut idx = TurbovecIndex::new(8, 4).unwrap();
        let vectors: Vec<f32> = (0..8 * 3).map(|i| i as f32 * 0.1).collect();
        idx.add_with_ids(&vectors, &[10, 20, 30]).unwrap();
        assert!(idx.remove(20));

        idx.write(path.to_str().unwrap()).unwrap();
        let loaded = TurbovecIndex::load(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.dim(), 8);
        assert_eq!(loaded.bit_width(), 4);

        let (_, ids) = loaded.search(&vectors[0..8], 2, None);
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&10) || ids.contains(&30));

        std::fs::remove_dir_all(&dir).ok();
    }
}
