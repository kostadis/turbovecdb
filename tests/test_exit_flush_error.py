"""Collection.__exit__ error handling: flush errors must not mask original exceptions."""

import pytest

import turbovecdb

DIM = 8


def test_exit_flush_error_preserves_original_exception(tmp_path):
    """If the with-block raises and flush() succeeds, the original exception
    propagates — flush errors must never mask the caller's exception."""
    db = turbovecdb.connect(str(tmp_path / "db"))

    class TestError(Exception):
        pass

    with pytest.raises(TestError):
        with db.collection("c", dim=DIM, create=True) as col:
            col.add(ids=["a"], documents=["hello"], vectors=[[1.0] + [0.0] * (DIM - 1)])
            raise TestError("boom from with-block")

    db.close()


def test_exit_flush_then_reopen(tmp_path):
    """On normal exit (no exception), the collection is flushed and
    re-openable — data survives across the with-block boundary."""
    db = turbovecdb.connect(str(tmp_path / "db"))

    with db.collection("c", dim=DIM, create=True) as col:
        col.add(ids=["a"], documents=["hello"], vectors=[[1.0] + [0.0] * (DIM - 1)])
        assert col.count() == 1

    db.close()

    db2 = turbovecdb.connect(str(tmp_path / "db"))
    col2 = db2.collection("c", create=False)
    assert col2.count() == 1
    db2.close()
