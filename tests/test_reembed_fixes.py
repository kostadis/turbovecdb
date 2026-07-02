"""Regression tests for two reembed() bugs:

A. skip_empty="drop" didn't drop — the empty doc (and its stale, possibly
   wrong-dimension vector) stayed in the store; only a counter moved.
B. A dimension change combined with a skipped doc corrupted the store: new_dim
   was mis-derived from updates[0] (a retained old-dim vector), the mixed-dim
   rows were committed, and the operation was not atomic — a failure left the
   collection permanently unreadable.

The fixes: drop actually deletes; new_dim comes from the real embedder output;
keep + real dim change is a clean error; and the whole rewrite is atomic.
"""
import numpy as np
import pytest

import turbovecdb
from turbovecdb.errors import DimensionMismatchError, TurboVecError


def e8(texts):
    return np.array([[float(len(t)) for _ in range(8)] for t in texts], dtype=np.float32)


def e16(texts):
    return np.array([[float(len(t)) for _ in range(16)] for t in texts], dtype=np.float32)


def _byte_lengths(coll_dir):
    """Vector blob byte-lengths straight from SQLite after a cold reopen."""
    import sqlite3
    conn = sqlite3.connect(f"{coll_dir}/store.sqlite3")
    try:
        return sorted(r[0] for r in conn.execute("SELECT length(vector) FROM docs"))
    finally:
        conn.close()


# ── Bug A ─────────────────────────────────────────────────────────────────────


def test_reembed_drop_actually_removes_empty_docs(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", dim=8, create=True)
    c.add(ids=["a", "b", "c"], documents=["hello", "", "world"], vectors=e8(["hello", "", "world"]))

    report = c.reembed(e8, skip_empty="drop", batch_size=2)

    assert report.total_docs == 3
    assert report.skipped == 1
    # The empty doc is gone from the store, not merely counted.
    assert c.count() == 2
    assert set(c.get().ids) == {"a", "c"}
    # Collection stays queryable.
    assert len(c.query(vector=e8(["q"])[0], k=5).ids) == 2
    db.close()


# ── Bug B ─────────────────────────────────────────────────────────────────────


def test_reembed_drop_with_dim_change_is_consistent(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    coll_dir = str(tmp_path / "db" / "c")
    c = db.collection("c", dim=8, create=True)
    c.add(ids=["a", "b", "c"], documents=["hello", "", "world"], vectors=e8(["hello", "", "world"]))

    report = c.reembed(e16, dim=16, skip_empty="drop", batch_size=2)

    assert report.new_dim == 16
    assert c.dim == 16
    assert c.count() == 2
    assert set(c.get().ids) == {"a", "c"}
    db.close()

    # Cold reopen: every stored vector must be 16-dim (64 bytes) — no 32-byte
    # landmines — and a query must succeed.
    assert _byte_lengths(coll_dir) == [64, 64]
    db2 = turbovecdb.connect(str(tmp_path / "db"))
    c2 = db2.collection("c")
    assert len(c2.query(vector=e16(["q"])[0], k=5).ids) == 2
    db2.close()


def test_reembed_keep_with_dim_change_raises_and_is_intact(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    coll_dir = str(tmp_path / "db" / "c")
    c = db.collection("c", dim=8, create=True)
    c.add(ids=["a", "b", "c"], documents=["hello", "", "world"], vectors=e8(["hello", "", "world"]))

    # Keeping an old 8-dim vector in a 16-dim collection is incoherent → error,
    # not corruption.
    with pytest.raises(DimensionMismatchError):
        c.reembed(e16, dim=16, skip_empty="keep", batch_size=2)

    # Atomic: collection unchanged and still usable.
    assert c.count() == 3
    assert c.dim == 8
    assert len(c.query(vector=e8(["q"])[0], k=3).ids) == 3
    db.close()
    assert _byte_lengths(coll_dir) == [32, 32, 32]


def test_reembed_rolls_back_when_index_rebuild_fails(tmp_path):
    """Force a failure AFTER the DML (during index rebuild) and prove the whole
    re-embed rolls back — this is the phase-2 rollback branch.

    The index build now happens in Rust against the native turbovec crate
    directly (no Python turbovec.IdMapIndex call left to monkeypatch), so the
    failure is triggered by an embedder returning a dimension turbovec's
    index construction rejects outright: not a positive multiple of 8. That
    surfaces exactly at reload_index()'s rebuild step, after the DML has
    already run — the same rollback branch the old monkeypatch exercised."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    coll_dir = str(tmp_path / "db" / "c")
    c = db.collection("c", dim=8, create=True)
    c.add(ids=["a", "b", "c", "d"],
          documents=["hi", "yo", "hey", "sup"],
          vectors=e8(["hi", "yo", "hey", "sup"]))
    before = c.count()

    def bad_dim_embedder(texts):
        return np.array([[1.0] * 5 for _ in texts], dtype=np.float32)

    with pytest.raises(TurboVecError):
        c.reembed(bad_dim_embedder, batch_size=2)  # embeds fine; blows up rebuilding the index

    # Nothing committed: count, dim, and vectors are exactly as before.
    assert c.count() == before
    assert c.dim == 8
    assert len(c.query(vector=e8(["q"])[0], k=4).ids) == 4
    db.close()
    assert _byte_lengths(coll_dir) == [32, 32, 32, 32]
