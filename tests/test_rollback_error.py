"""Issue #52 — Collection.rollback() silently swallows SQLite errors.

The Rust core's ``rollback()`` at
``crates/turbovecdb-core/src/collection.rs:371-373`` discards errors with
``let _ = conn.execute_batch("ROLLBACK")``. These tests verify that every
code path calling rollback() leaves the connection in a clean state where
subsequent operations do not fail with "cannot start a transaction within
a transaction" and the collection remains internally consistent.
"""

import pytest

import turbovecdb
from turbovecdb import DimensionMismatchError

DIM = 8


def _vec(dim=DIM):
    return [1.0] + [0.0] * (dim - 1)


def _col(tmp_path, name="c"):
    db = turbovecdb.connect(str(tmp_path / "db"))
    col = db.collection(name, dim=DIM, create=True)
    return db, col


def test_duplicate_id_rollback_then_add(tmp_path):
    """Duplicate id triggers rollback; subsequent add() must succeed."""
    db, col = _col(tmp_path)
    v = _vec()
    col.add(ids=["a"], vectors=[v])
    with pytest.raises(ValueError, match="already exists"):
        col.add(ids=["a"], vectors=[v])
    col.add(ids=["b"], vectors=[v])
    assert col.count() == 2
    db.close()


def test_dimension_mismatch_rollback_then_add(tmp_path):
    """Dimension mismatch triggers rollback; subsequent add() must succeed."""
    db, col = _col(tmp_path)
    v = _vec()
    with pytest.raises(DimensionMismatchError):
        col.add(ids=["a"], vectors=[[1.0] * 16])
    col.add(ids=["a"], vectors=[v])
    assert col.count() == 1
    db.close()


def test_consecutive_rollbacks_do_not_corrupt_state(tmp_path):
    """Multiple rollbacks on the same handle must not break the connection."""
    db, col = _col(tmp_path)
    v = _vec()
    col.add(ids=["a"], vectors=[v])
    for _ in range(5):
        with pytest.raises(ValueError, match="already exists"):
            col.add(ids=["a"], vectors=[v])
    col.add(ids=["b"], vectors=[v])
    assert col.count() == 2
    db.close()


def test_update_metadata_not_found_rollback_then_add(tmp_path):
    """update_metadata on a nonexistent id triggers rollback; add() must work."""
    db, col = _col(tmp_path)
    v = _vec()
    col.add(ids=["a"], vectors=[v])
    with pytest.raises(ValueError, match="not found"):
        col.update_metadata(ids=["nonexistent"], metadatas=[{"x": 1}])
    col.add(ids=["b"], vectors=[v])
    assert col.count() == 2
    db.close()


def test_update_documents_not_found_rollback_then_add(tmp_path):
    """update_documents on a nonexistent id triggers rollback; add() must work."""
    db, col = _col(tmp_path)
    v = _vec()
    col.add(ids=["a"], vectors=[v])
    with pytest.raises(ValueError, match="not found"):
        col.update_documents(ids=["nonexistent"], documents=["new doc"])
    col.add(ids=["b"], vectors=[v])
    assert col.count() == 2
    db.close()


def test_rollback_leaves_no_partial_write(tmp_path):
    """After a rollback no partial write is visible in the store."""
    db, col = _col(tmp_path)
    v = _vec()
    col.add(ids=["a", "b", "c"], vectors=[v, v, v])
    assert col.count() == 3
    with pytest.raises(ValueError, match="already exists"):
        col.add(ids=["a", "d", "e"], vectors=[v, v, v])
    assert col.count() == 3
    res = col.get(ids=["d", "e"])
    assert res.ids == []
    db.close()


def test_mixed_operations_after_rollback(tmp_path):
    """Reads (count/get/query) and writes all work after a rollback."""
    db, col = _col(tmp_path)
    v = _vec()
    col.add(ids=["a"], vectors=[v])
    with pytest.raises(ValueError, match="already exists"):
        col.add(ids=["a"], vectors=[v])
    assert col.count() == 1
    res = col.get(ids=["a"])
    assert res.ids == ["a"]
    qres = col.query(vector=v, k=1)
    assert qres.ids == ["a"]
    col.add(ids=["b"], vectors=[v])
    assert col.count() == 2
    db.close()


def test_clear_then_add_after_rollback(tmp_path):
    """clear() commits internally; adds after it must work."""
    db, col = _col(tmp_path)
    v = _vec()
    col.add(ids=["a", "b"], vectors=[v, v])
    col.clear()
    assert col.count() == 0
    col.add(ids=["c"], vectors=[v])
    assert col.count() == 1
    db.close()


def test_delete_then_add_after_rollback(tmp_path):
    """delete() commits internally; adds after it must work."""
    db, col = _col(tmp_path)
    v = _vec()
    col.add(ids=["a", "b", "c"], vectors=[v, v, v])
    col.delete(ids=["c"])
    assert col.count() == 2
    col.add(ids=["d"], vectors=[v])
    assert col.count() == 3
    db.close()


def test_rollback_does_not_poison_connection(tmp_path):
    """A rollback must not leave the connection in a poisoned state affecting
    future transactions across different operations."""
    db, col = _col(tmp_path)
    v = _vec()
    col.add(ids=["a"], vectors=[v])
    with pytest.raises(ValueError, match="already exists"):
        col.add(ids=["a"], vectors=[v])
    col.upsert(ids=["a"], vectors=[v])
    assert col.count() == 1
    col.delete(ids=["a"])
    assert col.count() == 0
    col.add(ids=["b"], vectors=[v])
    assert col.count() == 1
    db.close()


def test_rollback_after_metadata_update_leaves_state_consistent(tmp_path):
    """If update_metadata rolls back, the old metadata must still be intact."""
    db, col = _col(tmp_path)
    v = _vec()
    col.add(ids=["a"], vectors=[v], metadatas=[{"key": "original"}])
    with pytest.raises(ValueError, match="not found"):
        col.update_metadata(ids=["nonexistent"], metadatas=[{"key": "wrong"}])
    res = col.get(ids=["a"])
    assert res.metadatas[0]["key"] == "original"
    db.close()
