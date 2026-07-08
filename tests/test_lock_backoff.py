"""Tests for exponential backoff with jitter in cross-process lock acquisition.

Issue #58: the lock polling loop used to run every 50ms with no backoff,
creating a thundering herd under contention. The fix adds exponential backoff
(starting at 50ms, doubling each iteration, capped at 1s) with ±25% random
jitter to desynchronise competing waiters.
"""

import concurrent.futures

import pytest

import turbovecdb

DIM = 8


def test_concurrent_writers_get_lock_eventually(tmp_path):
    """Multiple writers contending for the same collection should all succeed.

    Before the backoff fix, 8 threads hammering the lock every 50ms would
    create a thundering herd, starving some writers under contention. With
    exponential backoff + jitter, all writers should acquire the lock and
    commit their documents within a reasonable timeout.
    """
    path = str(tmp_path / "db")

    def writer(name):
        try:
            db = turbovecdb.connect(path)
            col = db.collection("c", dim=DIM, create=False)
            col.add(ids=[name], documents=[f"doc {name}"], vectors=[[1.0] + [0.0] * (DIM - 1)])
            db.close()
            return name, True
        except Exception:
            return name, False

    # Create the collection first so all threads open an existing collection
    db = turbovecdb.connect(path)
    db.collection("c", dim=DIM, create=True)
    db.close()

    NUM_WRITERS = 8
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WRITERS) as pool:
        futs = [pool.submit(writer, f"w{i}") for i in range(NUM_WRITERS)]
        results = [f.result() for f in concurrent.futures.as_completed(futs)]

    assert all(ok for _, ok in results), f"Some writers failed: {results}"

    db = turbovecdb.connect(path)
    col = db.collection("c", dim=DIM, create=False)
    assert col.count() == NUM_WRITERS
    ids = col.get(limit=NUM_WRITERS).ids
    for i in range(NUM_WRITERS):
        assert f"w{i}" in ids
    db.close()


def test_lock_timeout_property(tmp_path):
    """The lock_timeout is accessible and defaults to a reasonable value."""
    path = str(tmp_path / "db")
    db = turbovecdb.connect(path)
    col = db.collection("c", dim=DIM, create=True)
    # The core exposes lock_timeout as an attribute
    timeout = col._core.lock_timeout
    assert isinstance(timeout, (int, float))
    assert timeout > 0
    db.close()


def test_lock_timeout_custom_property(tmp_path):
    """Custom lock_timeout is reflected in the core property."""
    path = str(tmp_path / "db")
    db = turbovecdb.connect(path)
    col = db.collection("c", dim=DIM, create=True, lock_timeout=5)
    assert col._core.lock_timeout == 5
    db.close()
