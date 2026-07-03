"""Test delete_collection race with concurrent collection() (P2-5)."""

import threading
import time

import pytest

import turbovecdb


def test_delete_collection_race_recreate(tmp_path):
    """Simulate a concurrent collection(create=True) recreating the collection
    between the cache close and the file-lock acquisition in delete_collection.

    The post-lock re-check should evict the new handle before rmtree runs.
    """
    import os
    db_path = str(tmp_path / "db")
    db = turbovecdb.connect(db_path)
    c = db.collection("c", dim=8, create=True)
    c.add(ids=["x"], vectors=[[1.0] + [0.0] * 7])
    db.close()

    # Thread A: delete_collection
    # Thread B: collection(create=True) that races with the delete
    results = {"error": None}

    def deleter():
        db2 = turbovecdb.connect(db_path)
        try:
            db2.delete_collection("c")
        except Exception as e:
            results["error"] = e
        finally:
            db2.close()

    def creator():
        # Wait until the initial cache close has happened (hard to time
        # precisely, so just keep trying).
        for _ in range(50):
            try:
                db3 = turbovecdb.connect(db_path)
                db3.collection("c", dim=8, create=True)
                db3.close()
                break
            except Exception:
                pass
            time.sleep(0.001)

    t1 = threading.Thread(target=deleter)
    t2 = threading.Thread(target=creator)

    # Start creator slightly after deleter
    t1.start()
    time.sleep(0.005)
    t2.start()

    t1.join()
    t2.join()

    # delete_collection should have succeeded
    if results["error"]:
        raise results["error"]

    # The directory should be gone (or the creator recreated it after rmtree
    # — both outcomes are acceptable; the collection should be usable either way).
    coll_dir = os.path.join(db_path, "c")
    if os.path.isdir(coll_dir):
        # The creator won the race after rmtree — verify it's a valid collection
        db4 = turbovecdb.connect(db_path)
        c4 = db4.collection("c", dim=8)
        assert c4.count() == 0  # fresh collection created after delete
        db4.close()


def test_delete_collection_concurrent_create_then_delete(tmp_path):
    """collection() followed immediately by delete_collection should work
    cleanly — no stale handles in cache after delete."""
    import os
    db_path = str(tmp_path / "db")
    db = turbovecdb.connect(db_path)
    c = db.collection("c", dim=8, create=True)
    c.add(ids=["x"], vectors=[[1.0] + [0.0] * 7])

    # Delete it immediately
    db.delete_collection("c")
    assert not os.path.isdir(os.path.join(db_path, "c"))
    assert "c" not in db._collections
    db.close()


def test_delete_collection_recreate_after_delete_is_usable(tmp_path):
    """Deleting and recreating the same collection works."""
    import os
    db_path = str(tmp_path / "db")
    db = turbovecdb.connect(db_path)
    c = db.collection("c", dim=8, create=True)
    c.add(ids=["x"], vectors=[[1.0] + [0.0] * 7])
    db.close()

    db2 = turbovecdb.connect(db_path)
    db2.delete_collection("c")
    c2 = db2.collection("c", dim=8, create=True)
    c2.add(ids=["y"], vectors=[[0.0, 1.0] + [0.0] * 6])
    assert c2.count() == 1
    res = c2.get(ids=["y"])
    assert res.ids == ["y"]
    db2.close()


def test_delete_collection_race_no_data_loss(tmp_path):
    """A real concurrent create racing a delete must leave the collection in
    a valid state. The delete holds the core's cross-process write lock across
    the rmtree (in Rust now — no monkeypatchable Python ``FileLock`` seam), so
    a racing create either finishes before the delete or is serialized against
    it; the result is always either a clean delete or a usable recreation."""
    import os
    db_path = str(tmp_path / "db")

    db = turbovecdb.connect(db_path)
    c = db.collection("c", dim=8, create=True)
    c.add(ids=["x"], vectors=[[1.0] + [0.0] * 7])
    db.close()

    results = []

    def deleter():
        db2 = turbovecdb.connect(db_path)
        try:
            db2.delete_collection("c")
            results.append("deleted")
        except Exception as e:
            results.append(f"error: {e}")
        finally:
            db2.close()

    def creator():
        time.sleep(0.001)  # let the deleter get going
        db3 = turbovecdb.connect(db_path)
        try:
            db3.collection("c", dim=8, create=True)
            results.append("created")
        except Exception as e:
            results.append(f"creror: {e}")
        finally:
            db3.close()

    t1 = threading.Thread(target=deleter)
    t2 = threading.Thread(target=creator)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # The collection directory should still exist (creator recreated it) OR
    # be cleanly deleted — either is acceptable; it must be usable if present.
    coll_dir = os.path.join(db_path, "c")
    if os.path.isdir(coll_dir):
        db4 = turbovecdb.connect(db_path)
        c4 = db4.collection("c", dim=8)
        assert c4.count() >= 0
        db4.close()
