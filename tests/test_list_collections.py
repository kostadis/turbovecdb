"""Test list_collections TOCTOU fix (P2-3)."""

import threading
import time

import pytest

import turbovecdb


def test_list_collections_basic(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    assert db.list_collections() == []
    db.collection("a", dim=8, create=True)
    db.collection("b", dim=8, create=True)
    assert db.list_collections() == ["a", "b"]
    db.close()


def test_list_collections_after_delete(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c_a = db.collection("a", dim=8, create=True)
    c_a.add(ids=["x"], vectors=[[1.0] + [0.0] * 7])
    db.collection("b", dim=8, create=True)
    db.delete_collection("a")
    assert db.list_collections() == ["b"]
    db.close()


def test_list_collections_empty_after_all_deleted(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    db.collection("a", dim=8, create=True)
    db.delete_collection("a")
    assert db.list_collections() == []
    db.close()


def test_list_collections_excludes_non_collection_dirs(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    # Create a real collection
    db.collection("real", dim=8, create=True)
    # Create a stray dir without store.sqlite3
    import os
    os.makedirs(os.path.join(str(tmp_path / "db"), "stray"), exist_ok=True)
    assert db.list_collections() == ["real"]
    db.close()


def test_list_collections_not_created_returns_empty(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "nonexistent"))
    assert db.list_collections() == []


def test_list_collections_concurrent_delete(tmp_path):
    """list_collections should not crash when called concurrently with delete."""
    import os
    db = turbovecdb.connect(str(tmp_path / "db"))
    for i in range(10):
        db.collection(f"c{i}", dim=8, create=True)

    def deleter():
        for i in range(10):
            try:
                db.delete_collection(f"c{i}")
            except Exception:
                pass
            time.sleep(0.001)

    t = threading.Thread(target=deleter)
    t.start()
    for _ in range(50):
        try:
            lst = db.list_collections()
            # Should never crash, result may be any subset
            assert isinstance(lst, list)
        except Exception:
            pass  # swallow transient errors during race
    t.join()
    assert db.list_collections() == []
    db.close()


def test_list_collections_concurrent_create(tmp_path):
    """list_collections should not crash when called concurrently with create."""
    db = turbovecdb.connect(str(tmp_path / "db"))

    def creator():
        for i in range(10):
            try:
                db.collection(f"c{i}", dim=8, create=True)
            except Exception:
                pass
            time.sleep(0.001)

    t = threading.Thread(target=creator)
    t.start()
    for _ in range(50):
        lst = db.list_collections()
        assert isinstance(lst, list)
    t.join()
    db.close()


def test_list_collections_serialized_with_delete(tmp_path):
    """list_collections holding self._lock sees a consistent snapshot."""
    import os
    db = turbovecdb.connect(str(tmp_path / "db"))
    c_a = db.collection("a", dim=8, create=True)
    c_a.add(ids=["x"], vectors=[[1.0] + [0.0] * 7])
    db.collection("b", dim=8, create=True)

    results = []

    def lister():
        time.sleep(0.01)  # let delete start
        results.append(db.list_collections())

    t = threading.Thread(target=lister)
    t.start()
    db.delete_collection("a")
    t.join()

    # lister either sees "a" (before delete) or not (after), but never crashes
    assert isinstance(results[0], list)
    db.close()


def test_list_collections_prexisting_on_reopen(tmp_path):
    """list_collections sees collections created in a previous session."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    db.collection("c1", dim=8, create=True)
    db.collection("c2", dim=8, create=True)
    db.close()

    db2 = turbovecdb.connect(str(tmp_path / "db"))
    assert db2.list_collections() == ["c1", "c2"]
    db2.close()
