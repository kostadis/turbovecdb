"""Batteries-included path: an embedder turns text into vectors on add/query."""

import pytest

import turbovecdb
from turbovecdb import EmbedderRequiredError

DIM = 8


def _embedder(texts):
    """Toy embedder: 'a'-ish text -> position 0, else position 1."""
    out = []
    for t in texts:
        v = [0.0] * DIM
        v[0 if "alpha" in t else 1] = 1.0
        out.append(v)
    return out


def test_add_and_query_with_embedder(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    col = db.collection("c", embedder=_embedder, create=True)
    col.add(ids=["a", "b"], documents=["alpha document", "beta document"])
    res = col.query(text="find the alpha one", k=1)
    assert res.ids[0] == "a"
    assert res.distances[0] == pytest.approx(0.0, abs=1e-5)
    db.close()


def test_text_without_embedder_raises(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    col = db.collection("c", create=True)  # no embedder
    with pytest.raises(EmbedderRequiredError):
        col.add(ids=["a"], documents=["x"])  # no vectors, no embedder
    db.close()


def test_query_text_without_embedder_raises(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    col = db.collection("c", create=True)
    col.add(ids=["a"], documents=["x"], vectors=[[1.0] + [0.0] * (DIM - 1)])
    with pytest.raises(EmbedderRequiredError):
        col.query(text="x", k=1)
    db.close()
