"""Test that WAL autocheckpoint is configured on collection init."""

import os
import sqlite3

import pytest

import turbovecdb

DIM = 8


def test_wal_autocheckpoint_is_set(tmp_path):
    """The collection should have wal_autocheckpoint set (>0) after init."""
    path = str(tmp_path / "db")
    db = turbovecdb.connect(path)
    col = db.collection("c", dim=DIM, create=True)
    col.add(ids=["a"], documents=["hello"], vectors=[[1.0] + [0.0] * (DIM - 1)])
    db.close()

    store = os.path.join(path, "c", "store.sqlite3")
    conn = sqlite3.connect(store)
    val = conn.execute("PRAGMA wal_autocheckpoint").fetchone()[0]
    assert val > 0, "wal_autocheckpoint should be explicitly set to a positive value"
    conn.close()


def test_collection_works_with_autocheckpoint(tmp_path):
    """Collection should work normally with autocheckpoint configured."""
    path = str(tmp_path / "db")
    db = turbovecdb.connect(path)
    col = db.collection("c", dim=DIM, create=True)

    for i in range(100):
        v = [0.0] * DIM
        v[i % DIM] = 1.0
        col.add(ids=[str(i)], documents=[f"doc{i}"], vectors=[v])

    assert col.count() == 100
    col.flush()
    db.close()

    db2 = turbovecdb.connect(path)
    col2 = db2.collection("c", create=False)
    assert col2.count() == 100

    store = os.path.join(path, "c", "store.sqlite3")
    conn = sqlite3.connect(store)
    val = conn.execute("PRAGMA wal_autocheckpoint").fetchone()[0]
    assert val > 0, "wal_autocheckpoint should survive reopen"
    conn.close()

    db2.close()
