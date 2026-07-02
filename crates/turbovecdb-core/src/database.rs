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

use crate::error::CoreError;
use std::fs;
use std::path::{Component, Path, PathBuf};

const NAME_MAX_LEN: usize = 128;

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
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
