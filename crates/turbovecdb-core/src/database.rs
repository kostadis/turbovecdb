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
        Database { root: root.into() }
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
        let cwd = std::env::current_dir().unwrap_or_default();
        let abs_root = normalize(&cwd, &self.root);
        let abs_dir = normalize(&cwd, &dir);
        if !abs_dir.starts_with(&abs_root) {
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

    // ------------------------------------------------------------------
    // CachedDatabase tests — these call todo!() stubs and will panic
    // until the builder implements the methods.
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
}
