"""Test that Database.close() reports errors instead of suppressing them."""

import pytest
import turbovecdb
from turbovecdb import TurboVecError


def test_close_reports_errors(tmp_path):
    """Database.close() raises TurboVecError when collection close fails."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    col = db.collection("c", dim=8, create=True)
    col.add(ids=["a"], documents=["hello"], vectors=[[1.0] + [0.0] * 7])

    # Make the collection's close() raise
    def _boom():
        raise RuntimeError("boom")
    col.close = _boom

    # close() should raise TurboVecError with error details
    with pytest.raises(TurboVecError, match="Failed to close 1 collection"):
        db.close()


def test_close_continues_after_error_and_reports_all(tmp_path):
    """Database.close() reports all errors and continues closing remaining collections."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    good = db.collection("good", dim=8, create=True)
    good.add(ids=["a"], documents=["hello"], vectors=[[1.0] + [0.0] * 7])
    bad = db.collection("bad", dim=8, create=True)
    bad.add(ids=["b"], documents=["world"], vectors=[[0.0, 1.0] + [0.0] * 6])

    # Make the bad collection's close() raise
    def _boom():
        raise RuntimeError("boom")
    bad.close = _boom

    # close() should raise TurboVecError with both errors (though good should succeed)
    with pytest.raises(TurboVecError, match="Failed to close 1 collection"):
        db.close()

    # Both collections should be evicted from cache
    assert db._collections == {}


def test_delete_collection_propagates_close_error(tmp_path):
    """delete_collection propagates close errors from the cached handle."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    col = db.collection("c", dim=8, create=True)
    col.add(ids=["a"], documents=["hello"], vectors=[[1.0] + [0.0] * 7])

    # Break the collection to trigger a close error
    def _boom():
        raise RuntimeError("boom")
    col.close = _boom

    # delete_collection should propagate the close error
    with pytest.raises(RuntimeError, match="boom"):
        db.delete_collection("c")
