"""Database.close() and delete_collection log errors instead of swallowing them."""

import logging

import pytest

import turbovecdb

DIM = 8


def test_close_normal(tmp_path):
    """Database.close() works normally with healthy collections."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    col = db.collection("c", dim=DIM, create=True)
    col.add(ids=["a"], documents=["hello"], vectors=[[1.0] + [0.0] * (DIM - 1)])
    db.close()
    # Can re-open
    db2 = turbovecdb.connect(str(tmp_path / "db"))
    col2 = db2.collection("c", create=False)
    assert col2.count() == 1
    db2.close()


def test_close_multiple_collections(tmp_path):
    """Database.close() handles multiple collections."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    for name in ("a", "b", "c"):
        col = db.collection(name, dim=DIM, create=True)
        col.add(ids=["x"], documents=["doc"], vectors=[[1.0] + [0.0] * (DIM - 1)])
    db.close()
    db2 = turbovecdb.connect(str(tmp_path / "db"))
    assert db2.list_collections() == ["a", "b", "c"]
    db2.close()


def test_close_logs_warning_on_error(tmp_path, caplog):
    """Database.close() logs a warning when a collection close fails."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    col = db.collection("c", dim=DIM, create=True)
    col.add(ids=["a"], documents=["hello"], vectors=[[1.0] + [0.0] * (DIM - 1)])

    # Corrupt the collection's connection to trigger a close error
    col._conn.close()  # close the inner connection early

    with caplog.at_level(logging.WARNING, logger="turbovecdb.database"):
        db.close()
        assert any("error closing collection" in rec.getMessage() for rec in caplog.records)


def test_delete_collection_logs_warning_on_close_error(tmp_path, caplog):
    """delete_collection logs a warning if the cached handle fails to close."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    col = db.collection("c", dim=DIM, create=True)
    col.add(ids=["a"], documents=["hello"], vectors=[[1.0] + [0.0] * (DIM - 1)])

    # Break the collection to trigger a close error
    col._conn.close()

    with caplog.at_level(logging.WARNING, logger="turbovecdb.database"):
        db.delete_collection("c")
        assert any("error closing collection" in rec.getMessage() for rec in caplog.records)


def test_close_continues_after_error(tmp_path):
    """Database.close() continues closing remaining collections after one fails."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    good = db.collection("good", dim=DIM, create=True)
    good.add(ids=["a"], documents=["hello"], vectors=[[1.0] + [0.0] * (DIM - 1)])

    bad = db.collection("bad", dim=DIM, create=True)
    bad.add(ids=["b"], documents=["world"], vectors=[[0.0, 1.0] + [0.0] * (DIM - 2)])
    bad._conn.close()  # corrupt

    # close() should not raise, and both should be evicted from cache
    db.close()
    assert db._collections == {}
