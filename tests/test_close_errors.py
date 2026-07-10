"""Database.close() and delete_collection log errors instead of swallowing them."""

import logging


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


def test_close_multiple_collections_reports_errors(tmp_path):
    """Database.close() reports errors for multiple failed collections."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    for name in ("a", "b", "c"):
        col = db.collection(name, dim=DIM, create=True)
        col.add(ids=["x"], documents=["doc"], vectors=[[1.0] + [0.0] * (DIM - 1)])
    
    # Make all collections fail to close
    def _boom():
        raise RuntimeError("boom")
    for name in ("a", "b", "c"):
        db._collections[name].close = _boom

    # close() should raise TurboVecError with all three errors
    with pytest.raises(TurboVecError, match="Failed to close 3 collections"):
        db.close()

    # All collections should be evicted from cache
    assert db._collections == {}


def test_close_reports_error(tmp_path):
    """Database.close() raises TurboVecError when a collection close fails."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    col = db.collection("c", dim=DIM, create=True)
    col.add(ids=["a"], documents=["hello"], vectors=[[1.0] + [0.0] * (DIM - 1)])

    # Make the collection's close() raise
    def _boom():
        raise RuntimeError("boom")
    col.close = _boom

    # close() should raise TurboVecError with error details
    with pytest.raises(TurboVecError, match="Failed to close 1 collection"):
        db.close()


def test_delete_collection_reports_close_error(tmp_path):
    """delete_collection propagates close errors from the cached handle."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    col = db.collection("c", dim=DIM, create=True)
    col.add(ids=["a"], documents=["hello"], vectors=[[1.0] + [0.0] * (DIM - 1)])

    # Break the collection to trigger a close error
    def _boom():
        raise RuntimeError("boom")
    col.close = _boom

    # delete_collection should propagate the close error
    with pytest.raises(RuntimeError, match="boom"):
        db.delete_collection("c")


def test_close_continues_after_error_and_reports_errors(tmp_path):
    """Database.close() reports errors and continues closing remaining collections."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    good = db.collection("good", dim=DIM, create=True)
    good.add(ids=["a"], documents=["hello"], vectors=[[1.0] + [0.0] * (DIM - 1)])
    bad = db.collection("bad", dim=DIM, create=True)
    bad.add(ids=["b"], documents=["world"], vectors=[[0.0, 1.0] + [0.0] * (DIM - 2)])
    def _boom():
        raise RuntimeError("boom")
    bad.close = _boom  # make this collection fail to close
    
    # close() should raise TurboVecError for the failed collection
    with pytest.raises(TurboVecError, match="Failed to close 1 collection"):
        db.close()
    
    # Both collections should be evicted from cache
    assert db._collections == {}
