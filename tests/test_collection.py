"""Contract, persistence, and distance tests for turbovecdb collections."""

import os

import pytest

import turbovecdb
from turbovecdb import CollectionNotFoundError, DimensionMismatchError, GetResult, QueryResult

DIM = 8  # turbovec requires dim % 8 == 0
VECS = {
    "a": [1.0, 0, 0, 0, 0, 0, 0, 0],
    "b": [0, 1.0, 0, 0, 0, 0, 0, 0],
    "c": [0, 0, 1.0, 0, 0, 0, 0, 0],
    "d": [0.9, 0.1, 0, 0, 0, 0, 0, 0],  # nearest neighbour of "a"
}


def _seed(col):
    ids = ["a", "b", "c", "d"]
    col.add(
        ids=ids,
        documents=[f"doc {i}" for i in ids],
        metadatas=[{"wing": "letters", "room": i} for i in ids],
        vectors=[VECS[i] for i in ids],
    )


@pytest.fixture
def col(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("drawers", create=True)
    _seed(c)
    yield c
    db.close()


def test_query_typed_and_ordered(col):
    res = col.query(vector=VECS["a"], k=2)
    assert isinstance(res, QueryResult)
    assert res.ids[0] == "a"        # exact match first
    assert res.ids[1] == "d"        # nearest neighbour second
    assert res.vectors is None      # not requested


def test_get_typed(col):
    res = col.get(ids=["a"])
    assert isinstance(res, GetResult)
    assert res.ids == ["a"]
    assert res.documents == ["doc a"]
    assert res.metadatas[0]["wing"] == "letters"


def test_distance_is_true_cosine(col):
    res = col.query(vector=VECS["a"], k=4)
    d = dict(zip(res.ids, res.distances))
    assert d["a"] == pytest.approx(0.0, abs=1e-5)
    assert d["b"] == pytest.approx(1.0, abs=1e-5)
    assert 0.0 < d["d"] < d["b"]


def test_vectors_returned_when_requested(col):
    res = col.query(vector=VECS["a"], k=1, include=["vectors"])
    assert res.vectors is not None
    assert len(res.vectors[0]) == DIM


def test_query_empty_collection(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("empty", create=True)
    res = c.query(vector=VECS["a"], k=3)
    assert res.ids == [] and res.distances == []
    db.close()


def test_query_requires_exactly_one_of_text_vector(col):
    with pytest.raises(ValueError):
        col.query()
    with pytest.raises(ValueError):
        col.query(text="x", vector=VECS["a"])


def test_count(col):
    assert col.count() == 4


def test_add_duplicate_raises(col):
    with pytest.raises(ValueError):
        col.add(ids=["a"], documents=["dup"], vectors=[VECS["a"]])


def test_upsert_replaces(col):
    moved = [0, 0, 0, 0, 1.0, 0, 0, 0]
    col.upsert(ids=["a"], documents=["new a"], metadatas=[{"room": "a2"}], vectors=[moved])
    assert col.count() == 4
    assert col.get(ids=["a"]).documents == ["new a"]
    q = col.query(vector=moved, k=1)
    assert q.ids[0] == "a"
    assert q.distances[0] == pytest.approx(0.0, abs=1e-5)


def test_delete_by_id(col):
    col.delete(ids=["a"])
    assert col.count() == 3
    assert "a" not in col.query(vector=VECS["a"], k=4).ids


def test_delete_by_where(col):
    col.delete(where={"room": "b"})
    assert col.count() == 3
    assert col.get(ids=["b"]).ids == []


def test_bad_dimension_raises(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("d", create=True)
    with pytest.raises(DimensionMismatchError):
        c.add(ids=["x"], documents=["x"], vectors=[[1.0, 0, 0, 0]])  # dim 4, not %8
    db.close()


def test_dimension_mismatch_after_commit(col):
    with pytest.raises(DimensionMismatchError):
        col.add(ids=["z"], documents=["z"], vectors=[[1.0] * 16])  # dim 16 != 8


def test_collection_create_false_raises(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    with pytest.raises(CollectionNotFoundError):
        db.collection("missing", create=False)


# ── persistence: .tvim load + rebuild-from-SQLite ──────────────────────────


def test_persists_and_reloads(tmp_path):
    path = str(tmp_path / "db")
    db = turbovecdb.connect(path)
    _seed(db.collection("drawers", create=True))
    db.close()  # flushes .tvim

    tvim = os.path.join(path, "drawers", "index.tvim")
    assert os.path.exists(tvim)

    db2 = turbovecdb.connect(path)
    c2 = db2.collection("drawers", create=False)
    assert c2.count() == 4
    assert c2.query(vector=VECS["a"], k=1).ids[0] == "a"
    db2.close()


def test_rebuilds_when_tvim_missing(tmp_path):
    path = str(tmp_path / "db")
    db = turbovecdb.connect(path)
    _seed(db.collection("drawers", create=True))
    db.close()

    os.remove(os.path.join(path, "drawers", "index.tvim"))  # simulate crash before flush

    db2 = turbovecdb.connect(path)
    c2 = db2.collection("drawers", create=False)
    assert c2.count() == 4
    assert c2.query(vector=VECS["a"], k=2).ids == ["a", "d"]  # rebuilt from SQLite
    db2.close()
