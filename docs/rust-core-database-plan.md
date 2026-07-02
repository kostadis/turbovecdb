# Database layer in Rust (#16) — implementation plan

Plan for [#16](https://github.com/kostadis/turbovecdb/issues/16), authored
against the post-split shape (`docs/rust-core-split-design.md`, issues
#18–#26). Note #16's issue body has a stale `Files: rust/src/lib.rs`
reference — that path no longer exists; the workspace is
`crates/turbovecdb-core` + `crates/turbovecdb-py`.

## Scope correction vs. the earlier sketch

The earlier assessment ("a `Database<E, I>` in core owning the
`HashMap<String, Collection<E, I>>` cache") is **wrong**, for a reason
visible in `src/turbovecdb/collection.py`: the Python `Collection` wrapper
owns the `FileLock` + `RLock`, and repeated `Database.collection(name)`
calls must return the *same wrapper* (lock identity; first call's options
win). So the handle cache must live in Python regardless. A second cache in
Rust would (a) require converting `_core.Collection` to `Arc<Mutex<…>>`
shared ownership just so the core cache and the pyclass could both hold the
same collection, and (b) be permanently cold, because the Python cache
short-circuits every repeat lookup — dead duplicated state, a coherence
hazard with no payoff.

So the core `Database` holds **no collections and no generics**. What moves
to Rust is the pure, testable path/name/filesystem logic; what stays in
Python is exactly the lock-and-cache layer, consistent with the split's
"locking stays in the wrapper" principle.

## 1. `crates/turbovecdb-core/src/database.rs` (new)

`pub struct Database { root: PathBuf }` with:

- `validate_name(name)` — hand-rolled `[A-Za-z0-9_-]{1,128}` check (no
  regex dependency). Message parity:
  `invalid collection name '<name>': must match [A-Za-z0-9_-]{1,128}` —
  single-quote formatting matches Python's `!r` for every name without
  quotes/backslashes/control characters; the tests only match the
  `"invalid collection name"` prefix (`test_security.py`).
- `collection_dir(name) -> Result<PathBuf>` — validate + join + the
  absolute-path-prefix escape check (`… escapes database root` message),
  mirroring today's `os.path.abspath` compare. Belt-and-braces, as it is
  today (the name charset already blocks `/` and `.`).
- `list_collections() -> Vec<String>` — sorted subdirectories containing
  `store.sqlite3`; empty when the root is missing; skips non-UTF-8 entries
  (Python's `listdir` can't produce them from valid collection names).
- `ensure_collection(name) -> Result<PathBuf>` — validate + the
  isdir/`store.sqlite3` existence check; error
  `collection '<name>' not found at <dir>` via a new
  `CoreError::CollectionNotFound`.
- `remove_collection_dir(name) -> Result<()>` — validate +
  `fs::remove_dir_all`. Documented contract, same as core `Collection`:
  **the caller already holds the write lock** and has closed cached handles.

`CoreError` gains `CollectionNotFound(String)`. Cargo tests (tempdir-based):
name length/charset bounds, escape check, list ordering + `store.sqlite3`
filter, delete-missing → NotFound, delete removes the tree.

## 2. `crates/turbovecdb-py/src/database.rs` (new) + `convert.rs`

`#[pyclass] Database { inner: CoreDatabase }` exposing the five methods
above; registered in the `#[pymodule]`. `convert.rs` maps
`CollectionNotFound` → `turbovecdb.errors.CollectionNotFoundError`.

One parity detail: `remove_collection_dir`'s I/O errors convert via PyO3's
`std::io::Error → PyErr` path (which yields the right `OSError` subclass —
`FileNotFoundError`, `PermissionError`, …), preserving today's raw
`shutil.rmtree` exception behavior. `Collection`'s existing
`Io → RuntimeError` mapping is untouched.

## 3. `src/turbovecdb/database.py` — rewire, keep the lock/cache layer

- `collection()`: name validation + path resolution via `_core`; keeps the
  `threading.Lock`, the wrapper cache, the `create=False` isdir check, and
  Python `Collection` construction.
- `list_collections()`: delegates.
- `delete_collection()`: validate/exists via `_core.ensure_collection`; the
  close-cached-handle → `FileLock` → **re-check-cache-under-lock** race
  dance stays in Python verbatim (`test_delete_collection_race.py`); the
  rmtree becomes `_core.remove_collection_dir` inside the same try/finally.
- `close()`, `__enter__`/`__exit__`, `connect()`: unchanged.

## Commits (two, on `feat/rust-core`, into draft PR #27)

1. Core `database` module + `CoreError::CollectionNotFound` + cargo tests —
   the pytest suite is untouched, trivially green.
2. Adapter pyclass + convert mapping + `database.py` rewire — full suite
   green (`test_security.py`, `test_list_collections.py`,
   `test_delete_collection*.py`, `test_service.py` are the sensitive ones).

## Acceptance

- `cargo test -p turbovecdb-core` green, including the new database tests.
- Full pytest suite green.
- Error-text parity per above (no deviations expected; the `!r` quoting is
  reproduced with single quotes).

Then close #16; #17 (cutover & cleanup + completion PR) is the only slice
left.
