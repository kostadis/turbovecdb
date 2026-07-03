"""C6: Database.collection() raises on conflicting options for a cached
handle instead of silently returning the first call's handle."""

import pytest

import turbovecdb
from turbovecdb import TurboVecError

DIM = 8


def _embedder_a(texts):
    return [[1.0] + [0.0] * (DIM - 1) for _ in texts]


def _embedder_b(texts):
    return [[0.0, 1.0] + [0.0] * (DIM - 2) for _ in texts]


def test_matching_repeat_calls_do_not_conflict(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c1 = db.collection("c", dim=DIM, bit_width=4, create=True)
    c2 = db.collection("c", dim=DIM, bit_width=4, create=True)
    assert c1 is c2
    db.close()


def test_no_opinion_repeat_calls_do_not_conflict(tmp_path):
    """A second call that doesn't specify dim/bit_width/metric/embedder at
    all must never conflict, regardless of what the cached handle has."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    c1 = db.collection("c", dim=DIM, bit_width=2, embedder=_embedder_a, create=True)
    c2 = db.collection("c")
    assert c1 is c2
    db.close()


def test_conflicting_dim_raises(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    db.collection("c", dim=8, create=True)
    with pytest.raises(TurboVecError, match="already open with different options"):
        db.collection("c", dim=16)
    db.close()


def test_conflicting_bit_width_raises(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    db.collection("c", dim=DIM, bit_width=4, create=True)
    with pytest.raises(TurboVecError, match="already open with different options"):
        db.collection("c", bit_width=2)
    db.close()


def test_conflicting_embedder_raises(tmp_path):
    """Regression: previously accepted a different embedder on a warm
    handle without complaint (compounds with C5's query-time garbage
    rankings)."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    db.collection("c", dim=DIM, embedder=_embedder_a, create=True)
    with pytest.raises(TurboVecError, match="already open with different options"):
        db.collection("c", embedder=_embedder_b)
    db.close()


def test_explicit_no_embedder_conflicts_with_cached_embedder(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    db.collection("c", dim=DIM, embedder=_embedder_a, create=True)
    with pytest.raises(TurboVecError, match="already open with different options"):
        db.collection("c", embedder=None)
    db.close()


def test_matching_embedder_by_identity_does_not_conflict(tmp_path):
    """Two distinct callables with the same derived identity (here: the
    same function object, the common real-world case) must not conflict."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    c1 = db.collection("c", dim=DIM, embedder=_embedder_a, create=True)
    c2 = db.collection("c", embedder=_embedder_a)
    assert c1 is c2
    db.close()
