//! Narrow ANN-index abstraction.
//!
//! Only the operations `Collection` actually calls on the index object
//! today (`add_with_ids`/`remove`/`search`/`write` — confirmed by grepping
//! every `index.call_method*` site in the pre-split `collection.rs`). This
//! is NOT a general swappable-backend surface; its only reason to exist is
//! so `turbovecdb-core` can be exercised via `cargo test` with a fake
//! in-memory impl, without always paying to build and encode against the
//! real `turbovec` crate. Resist adding methods "for completeness."
//!
//! `TurbovecIndex` is the production implementation, wrapping
//! `turbovec::IdMapIndex` directly — no PyO3 callback into the Python
//! `turbovec` wheel.
//!
//! Since turbovec 1.0 the only on-disk format is v7. A `.tvim` written by
//! turbovec 0.9 (format v3) predates the v5 rotation change and cannot be
//! decoded by any current build, so `load` rejects it with a message that
//! says so; `Collection::reload_index` logs that and rebuilds from the
//! SQLite vectors, which are the source of truth.

use crate::error::CoreError;

/// Upper bound on vector dimensionality accepted by the backing engine.
///
/// turbovec 1.0 lowered this from 65536 to 16384. Re-exported (rather
/// than hardcoded at the call sites) so the limit tracks the engine and
/// `collection.rs` stays free of direct `turbovec::` references.
pub const MAX_DIM: usize = turbovec::MAX_DIM;

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
    ///
    /// Fallible since turbovec 1.0: an empty allowlist, an allowlist id
    /// not present in the index, and a malformed query buffer used to
    /// `panic!`/`assert!` out of the crate (0.9 `id_map.rs:209,214`) and
    /// are now reported as errors. Callers must still avoid passing an
    /// empty allowlist — `Collection::query` short-circuits that case
    /// before it reaches here.
    fn search(
        &self,
        queries: &[f32],
        k: usize,
        allowlist: Option<&[u64]>,
    ) -> Result<(Vec<f32>, Vec<u64>), CoreError>;
    fn write(&self, path: &str) -> Result<(), CoreError>;
    /// Used by `Collection::reload_index` to verify a loaded `.tvim` still
    /// matches the collection's current dim/bit_width before trusting it
    /// (a `tvim_gen` match only proves the file wasn't stale when written,
    /// not that its shape still matches — e.g. after a reembed).
    fn dim(&self) -> usize;
    fn bit_width(&self) -> usize;
    fn len(&self) -> usize;
}

pub struct TurbovecIndex(turbovec::IdMapIndex);

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

    fn search(
        &self,
        queries: &[f32],
        k: usize,
        allowlist: Option<&[u64]>,
    ) -> Result<(Vec<f32>, Vec<u64>), CoreError> {
        self.0
            .search_with_allowlist(queries, k, allowlist)
            .map_err(|e| CoreError::Other(e.to_string()))
    }

    fn write(&self, path: &str) -> Result<(), CoreError> {
        self.0.write(path).map_err(CoreError::from)
    }

    fn dim(&self) -> usize {
        self.0.dim_opt().unwrap_or(0)
    }

    fn bit_width(&self) -> usize {
        self.0.bit_width()
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

#[cfg(test)]
pub(crate) struct FakeIndex {
    dim: usize,
    bit_width: usize,
    entries: Vec<(u64, Vec<f32>)>,
}

#[cfg(test)]
impl VectorIndex for FakeIndex {
    fn new(dim: usize, bit_width: usize) -> Result<Self, CoreError> {
        Ok(Self { dim, bit_width, entries: Vec::new() })
    }

    fn load(path: &str) -> Result<Self, CoreError> {
        let data = std::fs::read_to_string(path).map_err(CoreError::from)?;
        let mut lines = data.lines();
        let dim: usize = lines.next().unwrap_or("0").parse().unwrap_or(0);
        let bit_width: usize = lines.next().unwrap_or("0").parse().unwrap_or(0);
        let mut entries = Vec::new();
        for line in lines {
            let mut parts = line.split(',');
            let id: u64 = parts.next().unwrap().parse().unwrap();
            let row: Vec<f32> = parts.map(|p| p.parse().unwrap()).collect();
            entries.push((id, row));
        }
        Ok(Self { dim, bit_width, entries })
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

    fn search(
        &self,
        queries: &[f32],
        k: usize,
        allowlist: Option<&[u64]>,
    ) -> Result<(Vec<f32>, Vec<u64>), CoreError> {
        if allowlist.is_some_and(|a| a.is_empty()) {
            return Err(CoreError::Other("allowlist is empty".into()));
        }
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
        Ok((scores, ids))
    }

    fn write(&self, path: &str) -> Result<(), CoreError> {
        let mut out = format!("{}\n{}\n", self.dim, self.bit_width);
        for (id, row) in &self.entries {
            let row_str: Vec<String> = row.iter().map(|f| f.to_string()).collect();
            out.push_str(&format!("{},{}\n", id, row_str.join(",")));
        }
        std::fs::write(path, out).map_err(CoreError::from)
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn bit_width(&self) -> usize {
        self.bit_width
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_search_remove_roundtrip() {
        let mut idx = FakeIndex::new(2, 4).unwrap();
        idx.add_with_ids(&[0.0, 0.0, 1.0, 1.0, 2.0, 2.0], &[10, 20, 30]).unwrap();
        let (_, ids) = idx.search(&[0.1, 0.1], 1, None).unwrap();
        assert_eq!(ids, vec![10]);

        assert!(idx.remove(10));
        let (_, ids) = idx.search(&[0.1, 0.1], 1, None).unwrap();
        assert_eq!(ids, vec![20]);

        let (_, ids) = idx.search(&[0.0, 0.0], 5, Some(&[30])).unwrap();
        assert_eq!(ids, vec![30]);

        // Fallible since turbovec 1.0: an empty allowlist is an error,
        // not an empty result. `Collection::query` short-circuits before
        // reaching the index, so this guard must stay in place.
        assert!(idx.search(&[0.0, 0.0], 5, Some(&[])).is_err());
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

        let (_, ids) = loaded.search(&vectors[0..8], 2, None).unwrap();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&10) || ids.contains(&30));

        std::fs::remove_dir_all(&dir).ok();
    }
}
