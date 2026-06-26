"""A failed commit must not leave the in-memory index diverged from SQLite.

The write paths mirror their changes into the in-memory turbovec index *before*
committing. If the commit fails, SQLite rolls back but the index keeps the
mutation. These tests assert the index is invalidated on commit failure so the
next access rebuilds from the durable store (the source of truth), rather than
silently serving stale/incomplete results for the life of the handle.
"""

import numpy as np
import pytest

import turbovecdb

DIM = 8


def _vec(seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(DIM).astype("float32").tolist()


class _FailingConn:
    """Delegates to a real sqlite connection but makes ``commit()`` raise.

    ``sqlite3.Connection.commit`` is a read-only builtin, so we can't patch it
    in place; we wrap the connection and let the collection use the proxy.
    """

    def __init__(self, real):
        self._real = real

    def commit(self):
        # A failed commit leaves the durable state at the pre-transaction
        # point; mirror that by rolling back before surfacing the error.
        self._real.rollback()
        raise RuntimeError("simulated disk-full on commit")

    def __getattr__(self, name):
        return getattr(self._real, name)


class _CommitFails:
    """Swap the collection's connection for one whose ``commit()`` fails."""

    def __init__(self, col):
        self._col = col
        self._real = col._conn

    def __enter__(self):
        self._col._conn = _FailingConn(self._real)
        return self

    def __exit__(self, *exc):
        self._col._conn = self._real
        return False


def _ids(result):
    return set(result.ids)


def test_delete_commit_failure_does_not_lose_searchability(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    col = db.collection("c", dim=DIM, create=True)
    col.add(ids=["a", "b", "c"],
            documents=["x", "y", "z"],
            vectors=[_vec(1), _vec(2), _vec(3)])

    # A delete whose commit fails: SQLite keeps all three rows, but the
    # in-memory index already removed "b".
    with _CommitFails(col):
        with pytest.raises(RuntimeError):
            col.delete(ids=["b"])

    # The row is still durable...
    assert col.count() == 3
    # ...and must still be searchable: the index self-heals from SQLite.
    found = set()
    for seed in (1, 2, 3):
        found |= _ids(col.query(vector=_vec(seed), k=3))
    assert found == {"a", "b", "c"}
    db.close()


def test_upsert_replace_commit_failure_does_not_drop_row(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    col = db.collection("c", dim=DIM, create=True)
    col.add(ids=["a"], documents=["old"], vectors=[_vec(1)])

    # upsert-replace removes the old uid from the index before committing.
    with _CommitFails(col):
        with pytest.raises(RuntimeError):
            col.upsert(ids=["a"], documents=["new"], vectors=[_vec(2)])

    # Commit was rolled back, so the durable row is still the original.
    got = col.get(ids=["a"])
    assert got.documents == ["old"]
    # And it must still be searchable rather than vanishing from the index.
    res = col.query(vector=_vec(1), k=1)
    assert res.ids == ["a"]
    db.close()


def test_add_commit_failure_keeps_existing_rows_searchable(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    col = db.collection("c", dim=DIM, create=True)
    col.add(ids=["a"], documents=["x"], vectors=[_vec(1)])

    with _CommitFails(col):
        with pytest.raises(RuntimeError):
            col.add(ids=["b"], documents=["y"], vectors=[_vec(2)])

    assert col.count() == 1
    # The phantom "b" must not have been durably written...
    assert col.get(ids=["b"]).ids == []
    # ...and the pre-existing row stays searchable.
    assert col.query(vector=_vec(1), k=5).ids == ["a"]
    db.close()
