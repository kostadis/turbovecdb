"""F2 regression: cross-process ``delete_collection`` vs. a stale open handle.

Finding F2 in ``docs/resiliency-review-2026-07-05.md``: if one connection holds an
open ``Collection`` handle while a *separate* connection deletes the collection
(flock + rmtree), the first handle's next ``add()`` used to still acquire the
(sibling, rmtree-surviving) write lock and commit into the now-unlinked SQLite
inode — the write was acknowledged but unrecoverable.

The fix re-``stat``s the store under the write lock and refuses the write with
``CollectionNotFoundError`` when the collection's ``store.sqlite3`` is gone or has
been replaced (inode mismatch) — see ``Collection::ensure_store_present`` /
``acquire_write_lock_checked`` in ``crates/turbovecdb-core/src/collection.rs``.

Both tests below assert the *desired* behavior: a post-delete ``add()`` on a stale
handle must be safe — it must either refuse the write (``CollectionNotFoundError``)
or leave a row that a fresh reader can still see. With the fix in place they pass.

The failure mode depends on local POSIX unlink semantics, so these tests skip on
``/mnt/*`` (drvfs/9p), which the concurrency docs already declare out of scope.
"""

import json
import os
import subprocess
import sys

import pytest

import turbovecdb
from turbovecdb import CollectionNotFoundError

DIM = 8


def _onehot(pos):
    v = [0.0] * DIM
    v[pos % DIM] = 1.0
    return v


def _skip_if_non_posix_fs(tmp_path):
    # F2 hinges on POSIX unlink semantics (writes to an unlinked-but-open inode
    # succeed and are lost on close). drvfs/9p under /mnt/* does not honor this
    # and is explicitly out of scope in docs/core/concurrency.md.
    if str(tmp_path).startswith("/mnt/"):
        pytest.skip("F2 depends on local POSIX unlink semantics; /mnt/* is out of scope")


def _after_row_visible(path):
    """True iff a *fresh* connection can still see the 'after' row (i.e. the
    stale-handle write was durable). If the collection is gone, the write is
    unrecoverable -> False."""
    db = turbovecdb.connect(path)
    try:
        col = db.collection("c", create=False)
    except CollectionNotFoundError:
        return False
    try:
        return "after" in col.get(ids=["after"]).ids
    finally:
        db.close()


def test_stale_handle_add_after_delete_is_lost_same_process(tmp_path):
    """Deterministic, single-process reproduction using two independent
    connections. ``connect()`` builds a fresh ``Database`` (own handle cache +
    own SQLite fd) each call, so handle A is never evicted when a *second*
    connection deletes the collection."""
    _skip_if_non_posix_fs(tmp_path)
    path = str(tmp_path / "db")

    # Handle A: a live, cached handle with a baseline row.
    db_a = turbovecdb.connect(path)
    col_a = db_a.collection("c", dim=DIM, create=True)
    col_a.add(ids=["before"], vectors=[_onehot(0)])
    assert col_a.count() == 1

    # A separate connection deletes the collection out from under A.
    db_b = turbovecdb.connect(path)
    db_b.delete_collection("c")
    db_b.close()
    assert not os.path.isdir(os.path.join(path, "c")), "delete should remove the collection dir"

    # A's stale handle writes again — capture what actually happens.
    try:
        col_a.add(ids=["after"], vectors=[_onehot(1)])
        add_outcome = "returned-success"
    except CollectionNotFoundError:
        add_outcome = "raised-CollectionNotFoundError"
    except Exception as exc:  # noqa: BLE001 — any error is still "not a silent accept"
        add_outcome = f"raised-{type(exc).__name__}"

    row_durable = _after_row_visible(path)
    db_a.close()

    assert add_outcome == "raised-CollectionNotFoundError" or row_durable, (
        f"F2 reproduced (same process): stale-handle add() outcome={add_outcome!r}; "
        f"'after' row visible to a fresh reader={row_durable}. The write was acknowledged "
        f"but is unrecoverable."
    )


# Child program for the cross-process case: open a handle, write a baseline row,
# announce READY, block until the parent has deleted the collection, then attempt
# the stale-handle add and report the outcome as JSON.
_CHILD = r"""
import sys, json, turbovecdb
from turbovecdb import CollectionNotFoundError
DIM = 8
path = sys.argv[1]
db = turbovecdb.connect(path)
col = db.collection("c", dim=DIM, create=True)
col.add(ids=["before"], vectors=[[1.0] + [0.0] * (DIM - 1)])
print("READY", flush=True)
sys.stdin.readline()  # block until the parent (a separate process) deletes "c"
outcome = {}
try:
    col.add(ids=["after"], vectors=[[0.0, 1.0] + [0.0] * (DIM - 2)])
    outcome["add"] = "returned-success"
    try:
        outcome["stale_count"] = col.count()
    except Exception as e:
        outcome["stale_count"] = "error:" + type(e).__name__
except CollectionNotFoundError:
    outcome["add"] = "raised-CollectionNotFoundError"
except Exception as e:
    outcome["add"] = "raised-" + type(e).__name__
print("RESULT " + json.dumps(outcome), flush=True)
"""


def test_stale_handle_add_after_delete_is_lost_cross_process(tmp_path):
    """True cross-process reproduction (F2's literal framing). A child process
    holds the open handle; the parent process deletes the collection; the child
    then writes on its stale handle. Coordinated with blocking stdin/stdout
    handshakes, so it is deterministic (no sleeps)."""
    _skip_if_non_posix_fs(tmp_path)
    path = str(tmp_path / "db")

    proc = subprocess.Popen(
        [sys.executable, "-c", _CHILD, path],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        ready = proc.stdout.readline()
        assert "READY" in ready, f"child failed to start: {ready!r}\n{proc.stderr.read()}"

        # Parent process deletes the collection out from under the child's handle.
        db_b = turbovecdb.connect(path)
        db_b.delete_collection("c")
        db_b.close()

        # Release the child to perform its stale-handle add().
        proc.stdin.write("go\n")
        proc.stdin.flush()

        result_line = proc.stdout.readline()
        assert result_line.startswith("RESULT "), (
            f"child did not report a result: {result_line!r}\n{proc.stderr.read()}"
        )
        outcome = json.loads(result_line[len("RESULT "):])
    finally:
        proc.wait(timeout=30)

    row_durable = _after_row_visible(path)

    assert outcome["add"] == "raised-CollectionNotFoundError" or row_durable, (
        f"F2 reproduced (cross-process): child stale-handle add outcome={outcome!r}; "
        f"'after' row visible to a fresh reader={row_durable}. The write was acknowledged "
        f"but is unrecoverable."
    )
