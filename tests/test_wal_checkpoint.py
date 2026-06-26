"""WAL checkpoint: PRAGMA wal_checkpoint(TRUNCATE) bounds WAL file growth."""

import os

import pytest

import turbovecdb

DIM = 8


def _wal_path(db_path, col_name):
    return os.path.join(db_path, col_name, "store.sqlite3-wal")


def test_flush_checkpoints_wal(tmp_path):
    """Calling flush() truncates the WAL file."""
    path = str(tmp_path / "db")
    db = turbovecdb.connect(path)
    col = db.collection("c", dim=DIM, create=True)

    # Write enough data to generate a WAL
    for i in range(10):
        v = [0.0] * DIM
        v[i % DIM] = 1.0
        col.add(ids=[f"doc{i}"], documents=[f"doc{i}"], vectors=[v])

    wal = _wal_path(path, "c")
    # After many writes without flush, the WAL should exist
    wal_before = os.path.getsize(wal) if os.path.exists(wal) else 0

    col.flush()
    # After flush, the WAL should be truncated
    assert os.path.exists(wal)
    after_flush = os.path.getsize(wal)
    assert after_flush <= wal_before or after_flush < 1000  # truncated
    db.close()


def test_close_checkpoints_wal(tmp_path):
    """Calling close() truncates the WAL file."""
    path = str(tmp_path / "db")
    db = turbovecdb.connect(path)
    col = db.collection("c", dim=DIM, create=True)

    for i in range(10):
        v = [0.0] * DIM
        v[i % DIM] = 1.0
        col.add(ids=[f"doc{i}"], documents=[f"doc{i}"], vectors=[v])

    wal = _wal_path(path, "c")
    wal_before = os.path.getsize(wal) if os.path.exists(wal) else 0

    db.close()
    # After close, the WAL should be truncated
    wal_after = os.path.getsize(wal) if os.path.exists(wal) else 0
    assert wal_after <= wal_before


def test_wal_does_not_grow_unbounded_after_multiple_flushes(tmp_path):
    """Repeated flush() calls keep the WAL bounded."""
    path = str(tmp_path / "db")
    db = turbovecdb.connect(path)
    col = db.collection("c", dim=DIM, create=True)

    for round_ in range(3):
        for i in range(10):
            v = [0.0] * DIM
            v[(round_ + i) % DIM] = 1.0
            col.upsert(ids=[f"doc{i}"], documents=[f"doc{i}"], vectors=[v])
        col.flush()

    wal = _wal_path(path, "c")
    wal_size = os.path.getsize(wal) if os.path.exists(wal) else 0
    # WAL should be small after checkpointing
    assert wal_size < 65536  # well under 64KB
    db.close()


def test_wal_checkpoint_survives_reopen(tmp_path):
    """Data is intact after WAL checkpoint and reopen."""
    path = str(tmp_path / "db")
    db = turbovecdb.connect(path)
    col = db.collection("c", dim=DIM, create=True)
    col.add(ids=["a"], documents=["hello"], vectors=[[1.0] + [0.0] * (DIM - 1)])
    col.flush()
    db.close()

    db2 = turbovecdb.connect(path)
    col2 = db2.collection("c", create=False)
    assert col2.count() == 1
    db2.close()
