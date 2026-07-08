//! Cross-process advisory write lock — the Rust half of what Python
//! `filelock` did in the wrapper (`collection.py`'s `_flock`).
//!
//! On Unix, Python `filelock` acquires `fcntl.flock(LOCK_EX)` on the lock
//! file; we call the *same* syscall (`flock(2)`) on the *same* path
//! (`<root>/<name>.lock`, computed sibling-of-dir per R1) so a process
//! running the old wheel (Python filelock) and one running the new wheel
//! (this module) genuinely exclude each other (invariant I3). `flock` has no
//! native timeout, so — exactly like `filelock` — we poll with `LOCK_NB` and
//! sleep ~50ms between attempts.
//!
//! The guard is RAII: dropping it closes the fd, which releases the flock.
//! We deliberately do **not** delete the lock file on release — `filelock`
//! leaves it in place, and unlinking it would race with another process that
//! has the same path open (it would then lock a fresh, unrelated inode; the
//! R1 split-brain the sibling-path layout exists to prevent).
//!
//! flock locks are associated with the *open file description*, not the
//! process: two separate `open()` calls on the same path (even within one
//! process) get independent descriptions and therefore contend with each
//! other. Callers must account for this — a method that acquires the guard
//! must not, while holding it, call another method that acquires a second
//! guard on the same file, or it self-deadlocks until the timeout fires.

#[cfg(not(unix))]
compile_error!(
    "turbovecdb-core's flock module requires a Unix target (flock(2)). \
     Windows interop with Python filelock is already lost there (it uses \
     msvcrt.locking, a different primitive); see docs/ARCHITECTURE.md."
);

use std::fs::OpenOptions;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::time::{Duration, Instant};

use rand::Rng;

use crate::error::CoreError;

/// Poll interval between non-blocking acquire attempts — matches Python
/// `filelock`'s default `poll_interval` closely enough to be
/// indistinguishable under contention.
const POLL_INTERVAL: Duration = Duration::from_millis(50);
const MAX_BACKOFF: Duration = Duration::from_secs(1);

/// Reproduce Python's `repr()` of a `str` for the historical lock-timeout
/// messages, which format the collection directory with `{...!r}`. CPython
/// wraps in single quotes by default, switching to double quotes only when
/// the string contains a single quote but no double quote, and escapes
/// backslashes, the active quote, and control characters. Filesystem paths
/// are almost always plain printable ASCII, but we reproduce the full common
/// case so the message is byte-identical to what the Python wrapper raised
/// (invariant I4 — several tests assert on this text).
pub fn py_repr_str(s: &str) -> String {
    let has_single = s.contains('\'');
    let has_double = s.contains('"');
    let quote = if has_single && !has_double { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || (c as u32) == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// A held cross-process write lock. The flock is released when this guard is
/// dropped (the fd closes). `_file` is kept solely to own that fd.
#[derive(Debug)]
pub struct FlockGuard {
    _file: std::fs::File,
}

impl FlockGuard {
    /// Acquire an exclusive advisory lock on `path`, polling with `LOCK_NB`
    /// until `timeout_secs` elapses. On timeout, returns
    /// `CoreError::LockTimeout(timeout_msg())` — the message is built lazily
    /// by the caller so the two historical variants (the write-path message
    /// and delete's `... to delete collection` variant) live at their call
    /// sites, per invariant I4. The lock file is created if absent; its
    /// parent directory must already exist.
    pub fn acquire<F>(path: &Path, timeout_secs: f64, timeout_msg: F) -> Result<FlockGuard, CoreError>
    where
        F: FnOnce() -> String,
    {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        let fd = file.as_raw_fd();
        let start = Instant::now();
        loop {
            // SAFETY: `fd` is a valid open file descriptor owned by `file`
            // for the duration of this call. `flock` only reads it.
            let ret = unsafe { libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) };
            if ret == 0 {
                return Ok(FlockGuard { _file: file });
            }
            let err = std::io::Error::last_os_error();
            // EWOULDBLOCK (== EAGAIN on Linux) is the "someone else holds it"
            // signal; anything else is a real error (e.g. EINTR is retried by
            // the loop naturally on the next attempt, but a genuine failure
            // like EBADF must surface, not silently poll forever).
            if err.raw_os_error() != Some(libc::EWOULDBLOCK) {
                return Err(CoreError::Io(err));
            }
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed >= timeout_secs {
                return Err(CoreError::LockTimeout(timeout_msg()));
            }
            // Exponential backoff with jitter: start at POLL_INTERVAL, cap at
            // MAX_BACKOFF, add ±25% random jitter to desynchronize competing
            // waiters (thundering herd mitigation).
            let attempt = (elapsed / POLL_INTERVAL.as_secs_f64()).floor() as u32;
            let backoff_ms = 50u64 * 2u64.pow(attempt.min(5));
            let backoff = Duration::from_millis(backoff_ms.min(MAX_BACKOFF.as_millis() as u64));
            let jitter = rand::thread_rng().gen_range(0.75f64..=1.25);
            let remaining = timeout_secs - elapsed;
            let nap = Duration::from_secs_f64(
                (backoff.as_secs_f64() * jitter).min(remaining.max(0.0)),
            );
            std::thread::sleep(nap);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::thread;

    fn temp_lock_path(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("turbovecdb-core-flock-test-{}-{}", std::process::id(), name));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("x.lock")
    }

    fn no_timeout() -> String {
        "unexpected timeout".to_string()
    }

    #[test]
    fn acquire_then_release_lets_next_acquire_succeed() {
        let path = temp_lock_path("acq-rel");
        {
            let _g = FlockGuard::acquire(&path, 5.0, no_timeout).unwrap();
        } // dropped here — flock released
        // A fresh acquire must succeed promptly.
        let _g2 = FlockGuard::acquire(&path, 5.0, no_timeout).unwrap();
    }

    #[test]
    fn release_does_not_delete_the_lock_file() {
        let path = temp_lock_path("no-delete");
        {
            let _g = FlockGuard::acquire(&path, 5.0, no_timeout).unwrap();
        }
        assert!(path.exists(), "lock file must survive guard drop (filelock leaves it in place)");
    }

    #[test]
    fn second_acquire_times_out_with_the_callers_message() {
        let path = temp_lock_path("contention");
        let held = FlockGuard::acquire(&path, 5.0, no_timeout).unwrap();

        // A second, independent open of the same path (separate file
        // description) must contend with the first and time out.
        let msg = "could not acquire write lock on 'x' within 0.2s";
        let err = FlockGuard::acquire(&path, 0.2, || msg.to_string()).unwrap_err();
        match err {
            CoreError::LockTimeout(m) => assert_eq!(m, msg),
            other => panic!("expected LockTimeout, got {other:?}"),
        }
        drop(held);
    }

    #[test]
    fn two_threads_contend_across_separate_opens() {
        let path = temp_lock_path("threads");
        let (tx_locked, rx_locked) = mpsc::channel();
        let (tx_go, rx_go) = mpsc::channel();

        let p = path.clone();
        let holder = thread::spawn(move || {
            let g = FlockGuard::acquire(&p, 5.0, no_timeout).unwrap();
            tx_locked.send(()).unwrap(); // signal: lock held
            rx_go.recv().unwrap(); // wait until the contender has timed out
            drop(g);
        });

        rx_locked.recv().unwrap(); // wait until holder holds the lock
        let err = FlockGuard::acquire(&path, 0.2, || "timeout".to_string()).unwrap_err();
        assert!(matches!(err, CoreError::LockTimeout(_)));
        tx_go.send(()).unwrap();
        holder.join().unwrap();

        // Once released, acquisition succeeds.
        let _g = FlockGuard::acquire(&path, 5.0, no_timeout).unwrap();
    }

    #[test]
    fn py_repr_str_matches_cpython() {
        assert_eq!(py_repr_str("/tmp/db/c"), "'/tmp/db/c'");
        assert_eq!(py_repr_str("/tmp/o'brien/c"), "\"/tmp/o'brien/c\"");
        assert_eq!(py_repr_str("C:\\db\\c"), "'C:\\\\db\\\\c'");
        // both quotes present -> single quote, single escaped
        assert_eq!(py_repr_str("a'b\"c"), "'a\\'b\"c'");
    }
}
