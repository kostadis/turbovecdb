"""Context manager protocol: Collection and Database support ``with`` statements."""

import pytest

import turbovecdb

DIM = 8


def test_database_context_manager_closes_on_exit(tmp_path):
    """Database closes all collections when the with-block exits."""
    with turbovecdb.connect(str(tmp_path / "db")) as db:
        col = db.collection("c", dim=DIM, create=True)
        col.add(ids=["a"], documents=["hello"], vectors=[[1.0] + [0.0] * (DIM - 1)])
        assert col.count() == 1
    with turbovecdb.connect(str(tmp_path / "db")) as db:
        col = db.collection("c", create=False)
        assert col.count() == 1


def test_collection_context_manager_flushes_on_exit(tmp_path):
    """Collection flushes the index when the with-block exits but the handle stays usable."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    with db.collection("c", dim=DIM, create=True) as col:
        col.add(ids=["a"], documents=["hello"], vectors=[[1.0] + [0.0] * (DIM - 1)])
        assert col.count() == 1
    # The handle is still usable after the with-block (not closed, just flushed)
    assert col.count() == 1
    # The .tvim was flushed, so a new handle can load it
    db.close()
    db2 = turbovecdb.connect(str(tmp_path / "db"))
    col2 = db2.collection("c", create=False)
    assert col2.count() == 1
    db2.close()


def test_database_context_manager_on_existing_data(tmp_path):
    """Re-opening a database with existing collections works."""
    path = str(tmp_path / "db")
    with turbovecdb.connect(path) as db:
        col = db.collection("c", dim=DIM, create=True)
        col.add(ids=["a"], documents=["hello"], vectors=[[1.0] + [0.0] * (DIM - 1)])
    with turbovecdb.connect(path) as db:
        col = db.collection("c", create=False)
        assert col.count() == 1


def test_database_context_manager_error_in_block(tmp_path):
    """An exception in the with-block still closes cleanly."""
    class TestError(Exception):
        pass
    with pytest.raises(TestError):
        with turbovecdb.connect(str(tmp_path / "db")) as db:
            db.collection("c", dim=DIM, create=True)
            raise TestError("boom")
    with turbovecdb.connect(str(tmp_path / "db")) as db:
        col = db.collection("c", create=False)
        assert col.count() == 0


def test_collection_context_manager_error_in_block(tmp_path):
    """An exception in the with-block still flushes the collection cleanly."""
    class TestError(Exception):
        pass
    db = turbovecdb.connect(str(tmp_path / "db"))
    with pytest.raises(TestError):
        with db.collection("c", dim=DIM, create=True) as col:
            col.add(ids=["a"], documents=["hello"], vectors=[[1.0] + [0.0] * (DIM - 1)])
            raise TestError("boom")
    # Collection handle is still usable after error (flushed, not closed)
    assert col.count() == 1
    db.close()


def test_database_context_manager_reuse(tmp_path):
    """Database can be reopened after context manager exit."""
    path = str(tmp_path / "db")
    with turbovecdb.connect(path) as db:
        db.collection("c", dim=DIM, create=True).add(
            ids=["a"], documents=["hello"], vectors=[[1.0] + [0.0] * (DIM - 1)]
        )
    with turbovecdb.connect(path) as db:
        col = db.collection("c", create=False)
        assert col.count() == 1
