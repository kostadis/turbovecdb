"""Connection reconnection: explicit reconnect() preserves data and health."""

import pytest

import turbovecdb

DIM = 8


def test_reconnect_roundtrip(tmp_path):
    """reconnect() preserves data and health across a fresh connection."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    col = db.collection("c", dim=DIM, create=True)
    col.add(ids=["a"], documents=["hello"], vectors=[[1.0] + [0.0] * (DIM - 1)])
    assert col.count() == 1
    h = col.health()
    assert h.ok is True

    col._core.reconnect()

    assert col.count() == 1, "count must survive reconnect"
    h2 = col.health()
    assert h2.ok is True
    assert h2.doc_count == 1

    col.add(ids=["b"], documents=["world"], vectors=[[0.0] * (DIM - 1) + [1.0]])
    assert col.count() == 2
    db.close()


def test_reconnect_after_close_then_reopen(tmp_path):
    """reconnect() on a reopened collection also works."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    col = db.collection("c", dim=DIM, create=True)
    col.add(ids=["a"], documents=["hello"], vectors=[[1.0] + [0.0] * (DIM - 1)])
    db.close()

    db2 = turbovecdb.connect(str(tmp_path / "db"))
    col2 = db2.collection("c", create=False)
    assert col2.count() == 1

    col2._core.reconnect()
    assert col2.count() == 1
    assert col2.health().ok is True
    db2.close()
