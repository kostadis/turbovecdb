//! `Database` — path/name/filesystem logic for a directory of collections.
//!
//! No PyO3 dependency, and deliberately **no collection cache and no
//! `Embedder`/`VectorIndex` generics** here: `Collection`'s handle (the
//! Python wrapper in `collection.py`) owns the cross-process `FileLock` and
//! must be identity-stable per name, so that cache has to live in the Python
//! wrapper layer regardless of what else moves to Rust — see
//! `docs/rust-core-database-plan.md` for why an earlier sketch that put a
//! generic `Database<E, I>` + cache here was wrong. What's left, and what
//! this module owns, is the pure bit: validating names, resolving a
//! collection's directory, listing collections, and removing one's
//! directory. As with `Collection`, callers are responsible for any locking
//! (`remove_collection_dir` assumes the caller already holds the write lock
//! and has closed any cached handle).

use crate::collection::{write_lock_path, Collection};
use crate::embedder::Embedder;
use crate::error::CoreError;
use crate::flock::{py_repr_str, FlockGuard};
use crate::index::VectorIndex;
use std::collections::HashMap;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};

const NAME_MAX_LEN: usize = 128;

/// The delete-path lock-timeout message, byte-identical to `database.py`'s
/// historical text (invariant I4): `could not acquire write lock on {dir!r}
/// within {N}s to delete collection`. Unlike the write-path variant
/// (`collection::write_lock_timeout_msg`, which formats `{:.1}` → `30.0s`),
/// the delete path formats the timeout as a *bare integer* — Python used the
/// int constant `_LOCK_TIMEOUT`, so `{_LOCK_TIMEOUT}s` rendered `30s`. Rust's
/// `f64` `Display` renders `30.0` as `30`, reproducing that exactly.
pub(crate) fn delete_lock_timeout_msg(dir: &str, timeout: f64) -> String {
    format!(
        "could not acquire write lock on {} within {}s to delete collection",
        py_repr_str(dir),
        timeout
    )
}

fn is_name_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-'
}

/// Validate a collection name against `[A-Za-z0-9_-]{1,128}`.
pub fn validate_name(name: &str) -> Result<(), CoreError> {
    let len = name.chars().count();
    if (1..=NAME_MAX_LEN).contains(&len) && name.chars().all(is_name_char) {
        Ok(())
    } else {
        Err(CoreError::InvalidArgument(format!(
            "invalid collection name '{name}': must match [A-Za-z0-9_-]{{1,128}}"
        )))
    }
}

/// Lexically normalize (resolve `.`/`..` components without touching the
/// filesystem) — mirrors `os.path.normpath`'s behavior, since the escape
/// check below must work for directories that don't exist yet and can't use
/// `Path::canonicalize` (which requires the path to exist and also resolves
/// symlinks, which `os.path.abspath` does not).
fn normalize(base: &Path, p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in base.join(p).components() {
        match comp {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

pub struct Database {
    root: PathBuf,
}

impl Database {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        let root = if root.is_relative() {
            std::env::current_dir().unwrap_or_default().join(&root)
        } else {
            root
        };
        Database { root }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Validate `name` and resolve it to `<root>/<name>`. The escape check
    /// is belt-and-braces (the name charset already excludes `/` and `..`),
    /// mirroring the historical `os.path.abspath(...).startswith(...)`
    /// guard in `database.py`.
    pub fn collection_dir(&self, name: &str) -> Result<PathBuf, CoreError> {
        validate_name(name)?;
        let dir = self.root.join(name);
        // Root is canonicalized at construction; escape check is belt-and-braces
        // (name charset already excludes `/` and `..`).
        let norm_dir = normalize(&self.root, &dir);
        if !norm_dir.starts_with(&self.root) {
            return Err(CoreError::InvalidArgument(format!(
                "collection name '{name}' escapes database root"
            )));
        }
        Ok(dir)
    }

    /// Sorted names of subdirectories that look like collections (contain a
    /// `store.sqlite3` file). Empty (not an error) when `root` doesn't exist.
    pub fn list_collections(&self) -> Result<Vec<String>, CoreError> {
        if !self.root.is_dir() {
            return Ok(Vec::new());
        }
        let mut names = Vec::new();
        for entry in fs::read_dir(&self.root)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() && path.join("store.sqlite3").exists() {
                if let Some(name) = entry.file_name().to_str() {
                    names.push(name.to_string());
                }
            }
        }
        names.sort();
        Ok(names)
    }

    /// Resolve `name` to its directory, erroring if the collection doesn't
    /// exist (no directory, or missing `store.sqlite3`).
    pub fn ensure_collection(&self, name: &str) -> Result<PathBuf, CoreError> {
        let dir = self.collection_dir(name)?;
        if !dir.is_dir() || !dir.join("store.sqlite3").exists() {
            return Err(CoreError::CollectionNotFound(format!(
                "collection '{name}' not found at {}",
                dir.display()
            )));
        }
        Ok(dir)
    }

    /// Remove a collection's entire directory. Contract: the caller already
    /// holds the write lock and has closed any cached handle for `name` —
    /// this does not check existence (call `ensure_collection` first) or
    /// lock anything itself.
    pub fn remove_collection_dir(&self, name: &str) -> Result<(), CoreError> {
        let dir = self.collection_dir(name)?;
        fs::remove_dir_all(&dir)?;
        Ok(())
    }

    /// Delete a collection's directory under the *same* cross-process write
    /// lock the core takes for writes (I6/R1): acquire the sibling
    /// `<name>.lock`, `remove_dir_all` the collection directory, release. The
    /// lock file is a sibling of the directory precisely so holding it
    /// survives the rmtree. Errors with `CollectionNotFound` if the
    /// collection doesn't exist, or `LockTimeout` (the `to delete collection`
    /// message variant) if the lock can't be acquired in `lock_timeout`
    /// seconds. The caller (`database.py`) is responsible for evicting and
    /// closing any cached Python handle *outside* this lock — a handle's
    /// `close()` would acquire the same flock on a second file description
    /// and self-deadlock against the guard held here.
    pub fn delete_collection(&self, name: &str, lock_timeout: f64) -> Result<(), CoreError> {
        let dir = self.ensure_collection(name)?;
        let dir_str = dir.to_string_lossy().into_owned();
        let lock_path = write_lock_path(&dir_str);
        let _guard = FlockGuard::acquire(&lock_path, lock_timeout, || {
            delete_lock_timeout_msg(&dir_str, lock_timeout)
        })?;
        self.remove_collection_dir(name)
    }
}

/// A `Database` wrapper that caches opened `Collection` handles per name.
///
/// Subsequent calls to `collection()` with the same name return the same
/// cached handle, avoiding repeated SQLite open / DDL / meta reads / index
/// reload / flock acquire. The cache is entirely in Rust — generic over
/// `E: Embedder` and `I: VectorIndex` so that Python (PyEmbedder), CLI tools
/// (NoEmbedder), or other bindings can each instantiate the concrete type
/// they need without carrying a Python-specific cache.
///
/// The cache is thread-safe: `collection()` serializes creation under a
/// `Mutex` with double-checked locking, and returned handles are
/// `Arc<Mutex<Collection<E, I>>>` — callers lock the `Mutex` to access the
/// collection's `&mut self` methods.
pub struct CachedDatabase<E: Embedder, I: VectorIndex> {
    inner: Database,
    cache: Mutex<HashMap<String, Arc<Mutex<Collection<E, I>>>>>,
}

// ---------------------------------------------------------------------------
// Implementation stub — methods will be filled in later
// ---------------------------------------------------------------------------
impl<E: Embedder, I: VectorIndex> CachedDatabase<E, I> {
    /// Create a new `CachedDatabase` rooted at `root`.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        CachedDatabase {
            inner: Database::new(root),
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Return a cached handle for collection `name`, creating one if absent.
    ///
    /// The first call for a given `name` creates the collection via
    /// `Collection::new(...)`; subsequent calls return the same
    /// `Arc<Mutex<Collection<E, I>>>` without any I/O.
    ///
    /// Parameters are forwarded to `Collection::new` on first creation only.
    pub fn collection(
        &self,
        name: &str,
        dim: Option<i64>,
        bit_width: i64,
        metric: Option<String>,
        embedder: Option<E>,
        lock_timeout: f64,
    ) -> Result<Arc<Mutex<Collection<E, I>>>, CoreError> {
        // Fast path
        if let Some(cached) = self.cache.lock().unwrap().get(name) {
            return Ok(cached.clone());
        }

        // Slow path: create new collection
        let coll_dir = self.inner.collection_dir(name)?;
        let coll_dir_str = coll_dir.to_string_lossy().into_owned();
        let collection =
            Collection::new(coll_dir_str, dim, bit_width, metric, embedder, lock_timeout)?;
        let arc = Arc::new(Mutex::new(collection));

        // Double-check under lock
        let mut cache = self.cache.lock().unwrap();
        if let Some(existing) = cache.get(name) {
            return Ok(existing.clone());
        }
        cache.insert(name.to_string(), arc.clone());
        Ok(arc)
    }

    /// Close all cached handles and clear the cache.
    pub fn close(&self) {
        self.cache.lock().unwrap().clear();
    }

    /// Add text documents to a collection.
    ///
    /// Embedding runs **outside** the collection lock so concurrent
    /// reads are not blocked during a slow embedder call.
    ///
    /// The embedder identity is re-checked under the write lock to
    /// guard against embedder swaps (R3). Parameters after `metadatas`
    /// are forwarded to `Collection::new` for first-creation only.
    pub fn add_text(
        &self,
        name: &str,
        ids: Vec<String>,
        documents: Vec<String>,
        metadatas: Option<Vec<String>>,
        dim: Option<i64>,
        bit_width: i64,
        metric: Option<String>,
        lock_timeout: f64,
    ) -> Result<(), CoreError> {
        // Phase 1: grab the embedder reference (brief lock, no I/O for
        // existing collections thanks to the cache).
        let emb = {
            let handle = self.collection(name, dim, bit_width, metric.clone(), None, lock_timeout)?;
            let coll = handle.lock().unwrap();
            coll.embedder().ok_or_else(|| {
                CoreError::EmbedderRequired(
                    "text was provided but this collection has no embedder; \
                     pass vectors instead, or create the collection with embedder=..."
                        .to_string(),
                )
            })?
        };

        // Phase 2: embed OUTSIDE the collection lock.
        let vectors = emb.embed(&documents)?;

        // Phase 3: re-acquire lock, re-check identity, write.
        let handle = self.collection(name, dim, bit_width, metric, None, lock_timeout)?;
        let mut coll = handle.lock().unwrap();
        // Re-check embedder identity under lock (R3): a concurrent reembed()
        // between Phase 1 and Phase 3 will have changed the stored identity,
        // making check_embedder_identity fail here.
        if let Some(emb) = coll.embedder() {
            coll.check_embedder_identity(&emb)?;
        }
        // resolve_vectors with pre-embedded vectors (fast: just normalizes).
        let vectors = coll.resolve_vectors(None, Some(vectors))?;
        coll.add(ids, None, metadatas, Some(vectors))?;
        Ok(())
    }
}

// -- Delegation methods (transparent pass-through to inner `Database`) -------

impl<E: Embedder, I: VectorIndex> std::ops::Deref for CachedDatabase<E, I> {
    type Target = Database;
    fn deref(&self) -> &Database {
        &self.inner
    }
}

impl<E: Embedder, I: VectorIndex> std::ops::DerefMut for CachedDatabase<E, I> {
    fn deref_mut(&mut self) -> &mut Database {
        &mut self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::FakeIndex;
    use ndarray::Array2;

    /// Dummy embedder that implements `Embedder` without doing anything.
    /// Used in `CachedDatabase` tests where we never call `.embed()`.
    struct NoEmbedder;

    impl Embedder for NoEmbedder {
        fn embed(&self, _docs: &[String]) -> Result<Array2<f32>, CoreError> {
            Ok(Array2::from_shape_vec((0, 0), vec![]).unwrap())
        }
        fn identity(&self) -> String {
            "NoEmbedder".to_string()
        }
    }

    fn temp_root(name: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("turbovecdb-core-database-test-{}-{}", std::process::id(), name));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn make_collection_dir(root: &Path, name: &str) {
        let dir = root.join(name);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("store.sqlite3"), b"").unwrap();
    }

    #[test]
    fn validate_name_accepts_valid_names() {
        for name in ["a", "A-b_9", &"x".repeat(128)] {
            validate_name(name).unwrap();
        }
    }

    #[test]
    fn validate_name_rejects_empty_too_long_and_bad_chars() {
        for name in ["", &"x".repeat(129), "has space", "has.dot", "has/slash", "has\\backslash", "../escape"] {
            let e = validate_name(name).unwrap_err();
            match e {
                CoreError::InvalidArgument(m) => assert!(m.contains("invalid collection name")),
                other => panic!("expected InvalidArgument, got {other:?}"),
            }
        }
    }

    #[test]
    fn collection_dir_joins_root_and_name() {
        let root = temp_root("dir");
        let db = Database::new(&root);
        let dir = db.collection_dir("foo").unwrap();
        assert_eq!(dir, root.join("foo"));
    }

    #[test]
    fn collection_dir_rejects_invalid_name_before_escape_check() {
        let root = temp_root("dir-invalid");
        let db = Database::new(&root);
        let e = db.collection_dir("../escape").unwrap_err();
        match e {
            CoreError::InvalidArgument(m) => assert!(m.contains("invalid collection name")),
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn list_collections_is_empty_when_root_missing() {
        let root = temp_root("missing");
        let db = Database::new(&root);
        assert_eq!(db.list_collections().unwrap(), Vec::<String>::new());
    }

    #[test]
    fn list_collections_sorts_and_filters_by_store_file() {
        let root = temp_root("list");
        fs::create_dir_all(&root).unwrap();
        make_collection_dir(&root, "zeta");
        make_collection_dir(&root, "alpha");
        fs::create_dir_all(root.join("not-a-collection")).unwrap(); // no store.sqlite3
        fs::write(root.join("stray-file"), b"").unwrap(); // not a dir

        let db = Database::new(&root);
        assert_eq!(db.list_collections().unwrap(), vec!["alpha".to_string(), "zeta".to_string()]);
    }

    #[test]
    fn ensure_collection_errors_when_missing() {
        let root = temp_root("ensure-missing");
        fs::create_dir_all(&root).unwrap();
        let db = Database::new(&root);
        let e = db.ensure_collection("nope").unwrap_err();
        match e {
            CoreError::CollectionNotFound(m) => assert!(m.contains("not found")),
            other => panic!("expected CollectionNotFound, got {other:?}"),
        }
    }

    #[test]
    fn ensure_collection_succeeds_when_present() {
        let root = temp_root("ensure-present");
        fs::create_dir_all(&root).unwrap();
        make_collection_dir(&root, "present");
        let db = Database::new(&root);
        assert_eq!(db.ensure_collection("present").unwrap(), root.join("present"));
    }

    #[test]
    fn remove_collection_dir_deletes_the_tree() {
        let root = temp_root("remove");
        fs::create_dir_all(&root).unwrap();
        make_collection_dir(&root, "gone");
        let db = Database::new(&root);
        db.remove_collection_dir("gone").unwrap();
        assert!(!root.join("gone").exists());
    }

    #[test]
    fn remove_collection_dir_errors_on_invalid_name() {
        let root = temp_root("remove-invalid");
        let db = Database::new(&root);
        let e = db.remove_collection_dir("../escape").unwrap_err();
        assert!(matches!(e, CoreError::InvalidArgument(_)));
    }

    #[test]
    fn delete_collection_removes_dir_under_lock() {
        let root = temp_root("delete-locked");
        fs::create_dir_all(&root).unwrap();
        make_collection_dir(&root, "gone");
        let db = Database::new(&root);
        db.delete_collection("gone", 30.0).unwrap();
        assert!(!root.join("gone").exists());
        // The sibling lock file must survive the rmtree (R1).
        assert!(root.join("gone.lock").exists());
    }

    #[test]
    fn delete_collection_missing_errors() {
        let root = temp_root("delete-missing");
        fs::create_dir_all(&root).unwrap();
        let db = Database::new(&root);
        let e = db.delete_collection("nope", 30.0).unwrap_err();
        assert!(matches!(e, CoreError::CollectionNotFound(_)));
    }

    #[test]
    fn delete_collection_times_out_with_delete_variant_message() {
        let root = temp_root("delete-contended");
        fs::create_dir_all(&root).unwrap();
        make_collection_dir(&root, "held");
        let db = Database::new(&root);
        let dir_str = root.join("held").to_string_lossy().into_owned();
        let held = FlockGuard::acquire(&write_lock_path(&dir_str), 5.0, || "x".into()).unwrap();
        let e = db.delete_collection("held", 0.3).unwrap_err();
        match e {
            CoreError::LockTimeout(m) => assert_eq!(m, delete_lock_timeout_msg(&dir_str, 0.3)),
            other => panic!("expected LockTimeout, got {other:?}"),
        }
        drop(held);
    }

    /// I4: the delete-variant message must be byte-identical to the Python
    /// wrapper's text — note the *bare integer* `30s` (not `30.0s`), since
    /// `database.py` formatted the int constant `_LOCK_TIMEOUT`.
    #[test]
    fn delete_lock_timeout_msg_is_byte_identical() {
        assert_eq!(
            delete_lock_timeout_msg("/x", 30.0),
            "could not acquire write lock on '/x' within 30s to delete collection"
        );
    }

    /// Embedder for `add_text` tests — returns dim-8 one-hot vectors.
    /// Each doc `i` gets a unit vector with 1.0 at index `i % 8`.
    struct TestEmbedder;

    impl Embedder for TestEmbedder {
        fn embed(&self, docs: &[String]) -> Result<Array2<f32>, CoreError> {
            let dim = 8usize;
            let n = docs.len();
            let mut data = Vec::with_capacity(n * dim);
            for i in 0..n {
                for j in 0..dim {
                    data.push(if i % dim == j { 1.0 } else { 0.0 });
                }
            }
            // Already unit-length, but normalize explicitly per contract.
            for row in 0..n {
                let start = row * dim;
                let sum2: f32 = data[start..start + dim].iter().map(|x| x * x).sum();
                if sum2 > 0.0 {
                    let norm = sum2.sqrt();
                    for j in 0..dim {
                        data[start + j] /= norm;
                    }
                }
            }
            Ok(Array2::from_shape_vec((n, dim), data).unwrap())
        }
        fn identity(&self) -> String {
            "TestEmbedder".into()
        }
    }

    // ------------------------------------------------------------------
    // CachedDatabase tests
    // ------------------------------------------------------------------

    #[test]
    fn cached_db_collection_returns_same_handle_on_second_call() {
        let root = temp_root("cached-same");
        let db = CachedDatabase::<NoEmbedder, FakeIndex>::new(&root);
        let h1 = db.collection("test", Some(8), 4, Some("cosine".into()), None, 5.0).unwrap();
        let h2 = db.collection("test", Some(8), 4, Some("cosine".into()), None, 5.0).unwrap();
        assert!(Arc::ptr_eq(&h1, &h2));
    }

    #[test]
    fn cached_db_collection_returns_different_handle_for_different_names() {
        let root = temp_root("cached-diff");
        let db = CachedDatabase::<NoEmbedder, FakeIndex>::new(&root);
        let ha = db.collection("a", Some(8), 4, Some("cosine".into()), None, 5.0).unwrap();
        let hb = db.collection("b", Some(8), 4, Some("cosine".into()), None, 5.0).unwrap();
        assert!(!Arc::ptr_eq(&ha, &hb));
    }

    #[test]
    fn cached_db_close_drops_all_handles() {
        let root = temp_root("cached-close");
        let db = CachedDatabase::<NoEmbedder, FakeIndex>::new(&root);
        let h1 = db.collection("a", Some(8), 4, Some("cosine".into()), None, 5.0).unwrap();
        let h2 = db.collection("b", Some(8), 4, Some("cosine".into()), None, 5.0).unwrap();
        db.close();
        let h1_again = db.collection("a", Some(8), 4, Some("cosine".into()), None, 5.0).unwrap();
        let h2_again = db.collection("b", Some(8), 4, Some("cosine".into()), None, 5.0).unwrap();
        assert!(!Arc::ptr_eq(&h1, &h1_again));
        assert!(!Arc::ptr_eq(&h2, &h2_again));
    }

    #[test]
    fn add_text_writes_embedded_documents() {
        let root = temp_root("add-text-basic");
        let db = CachedDatabase::<TestEmbedder, FakeIndex>::new(&root);
        // Pre-create the collection with an embedder (add_text doesn't
        // pass an embedder — it expects the collection to already exist).
        db.collection("coll", Some(8), 8, Some("cosine".into()), Some(TestEmbedder), 5.0).unwrap();
        db.add_text(
            "coll",
            vec!["a".into(), "b".into()],
            vec!["doc a".into(), "doc b".into()],
            None,
            Some(8),
            8,
            None,
            5.0,
        )
        .unwrap();
        // Read back via the collection handle to confirm writes.
        let handle = db.collection("coll", None, 8, None, None, 5.0).unwrap();
        let coll = handle.lock().unwrap();
        let stats = coll.health().unwrap();
        assert_eq!(stats.doc_count, 2, "add_text should insert 2 documents");
    }

    #[test]
    fn add_text_errors_when_no_embedder() {
        let root = temp_root("add-text-no-emb");
        let db = CachedDatabase::<TestEmbedder, FakeIndex>::new(&root);
        // Collection exists but was created without an embedder.
        db.collection("coll", Some(8), 8, Some("cosine".into()), None, 5.0).unwrap();
        let e = db
            .add_text(
                "coll",
                vec!["x".into()],
                vec!["text".into()],
                None,
                Some(8),
                8,
                None,
                5.0,
            )
            .unwrap_err();
        assert!(matches!(e, CoreError::EmbedderRequired(_)),
            "expected EmbedderRequired when collection has no embedder, got {e:?}");
    }

    /// Embedder that sleeps to simulate a slow network/Python embedder.
    struct SlowEmbedder;

    impl Embedder for SlowEmbedder {
        fn embed(&self, docs: &[String]) -> Result<Array2<f32>, CoreError> {
            std::thread::sleep(std::time::Duration::from_millis(500));
            let dim = 8;
            let n = docs.len();
            let mut data = Vec::with_capacity(n * dim);
            for i in 0..n {
                for j in 0..dim {
                    data.push(if i % dim == j { 1.0 } else { 0.0 });
                }
            }
            for row in 0..n {
                let start = row * dim;
                let sum2: f32 = data[start..start + dim].iter().map(|x| x * x).sum();
                if sum2 > 0.0 {
                    let norm = sum2.sqrt();
                    for j in 0..dim {
                        data[start + j] /= norm;
                    }
                }
            }
            Ok(Array2::from_shape_vec((n, dim), data).unwrap())
        }
        fn identity(&self) -> String {
            "SlowEmbedder".into()
        }
    }

    #[test]
    fn concurrent_query_not_blocked_during_slow_add_text() {
        let root = temp_root("concurrent-query-during-add");
        let db = Arc::new(CachedDatabase::<SlowEmbedder, FakeIndex>::new(&root));

        // Pre-create collection with the slow embedder.
        db.collection("coll", Some(8), 8, Some("cosine".into()), Some(SlowEmbedder), 5.0).unwrap();

        // Spawn add_text in another thread (embedding takes 500ms inside).
        let db_clone = db.clone();
        let t1 = std::thread::spawn(move || {
            db_clone.add_text(
                "coll",
                vec!["a".into()],
                vec!["hello world".into()],
                None,
                Some(8),
                8,
                None,
                5.0,
            ).unwrap();
        });

        // Give t1 enough time to get past Phase 1 (brief lock) and start embedding.
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Query during the embed phase — should NOT block.
        let start = std::time::Instant::now();
        let handle = db.collection("coll", None, 8, None, None, 5.0).unwrap();
        let coll = handle.lock().unwrap();
        coll.health().unwrap();
        drop(coll);
        let elapsed = start.elapsed();

        assert!(
            elapsed < std::time::Duration::from_millis(300),
            "query() blocked during add_text embed, took {:?} (expected < 300ms)",
            elapsed,
        );

        t1.join().unwrap();

        // Verify the document was written successfully.
        let handle = db.collection("coll", None, 8, None, None, 5.0).unwrap();
        let coll = handle.lock().unwrap();
        let stats = coll.health().unwrap();
        assert_eq!(stats.doc_count, 1);
    }

    // ------------------------------------------------------------------
    // Identity-swap concurrency test
    // ------------------------------------------------------------------

    /// Embedder with configurable identity string and sleep delay.
    /// Used to verify that `add_text` Phase 3 catches a concurrent
    /// reembed() that changed the stored embedder identity.
    #[derive(Clone)]
    struct DynEmbedder {
        name: &'static str,
        delay: std::time::Duration,
    }

    impl Embedder for DynEmbedder {
        fn embed(&self, docs: &[String]) -> Result<Array2<f32>, CoreError> {
            std::thread::sleep(self.delay);
            let dim = 8;
            let n = docs.len();
            let mut data = Vec::with_capacity(n * dim);
            for i in 0..n {
                for j in 0..dim {
                    data.push(if i % dim == j { 1.0 } else { 0.0 });
                }
            }
            for row in 0..n {
                let start = row * dim;
                let sum2: f32 = data[start..start + dim].iter().map(|x| x * x).sum();
                if sum2 > 0.0 {
                    let norm = sum2.sqrt();
                    for j in 0..dim {
                        data[start + j] /= norm;
                    }
                }
            }
            Ok(Array2::from_shape_vec((n, dim), data).unwrap())
        }
        fn identity(&self) -> String {
            self.name.to_string()
        }
    }

    #[test]
    fn reembed_during_add_text_caught_by_identity_check() {
        let root = temp_root("add-text-reembed");
        let db = Arc::new(CachedDatabase::<DynEmbedder, FakeIndex>::new(&root));

        // Pre-create collection with a slow embedder (identity "Alpha").
        db.collection(
            "coll",
            Some(8),
            8,
            Some("cosine".into()),
            Some(DynEmbedder { name: "Alpha", delay: std::time::Duration::from_millis(500) }),
            5.0,
        )
        .unwrap();

        // Add one doc so reembed has something to process.
        db.add_text(
            "coll",
            vec!["seed".into()],
            vec!["seed doc".into()],
            None,
            Some(8),
            8,
            None,
            5.0,
        )
        .unwrap();

        // Spawn add_text — Phase 1 grabs the Arc, Phase 2 embeds slowly.
        let db_clone = db.clone();
        let t1 = std::thread::spawn(move || {
            db_clone.add_text(
                "coll",
                vec!["a".into()],
                vec!["hello world".into()],
                None,
                Some(8),
                8,
                None,
                5.0,
            )
        });

        // Let t1 reach Phase 2 (embedding slowly).
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Swap the stored identity via reembed — this changes the meta table
        // but does NOT replace the Arc in the collection.
        let handle = db.collection("coll", None, 8, None, None, 5.0).unwrap();
        let mut coll = handle.lock().unwrap();
        coll.reembed(
            DynEmbedder { name: "Beta", delay: std::time::Duration::ZERO },
            None,
            None,
            10,
            None,
            "keep",   // seed doc has empty text (add_text stores text=None)
        )
        .unwrap();
        drop(coll);

        // t1 Phase 3 re-checks under the lock: check_embedder_identity
        // compares Arc identity ("Alpha") against stored meta identity
        // ("Beta") — mismatch.
        let result = t1.join().unwrap();
        assert!(
            matches!(result, Err(CoreError::EmbedderIdentityMismatch(_))),
            "expected EmbedderIdentityMismatch when embedder was swapped during add_text, got {result:?}"
        );

        // The seed doc remains (reembed re-embedded it); t1's add_text
        // must NOT have written anything.
        let handle = db.collection("coll", None, 8, None, None, 5.0).unwrap();
        let coll = handle.lock().unwrap();
        let stats = coll.health().unwrap();
        assert_eq!(stats.doc_count, 1);
    }
}
