"""Health check: Collection.health() reports integrity and coherence."""

import pytest

import turbovecdb
from turbovecdb import HealthResult

DIM = 8


def test_health_ok_on_empty_collection(tmp_path):
    """A new empty collection reports healthy."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    col = db.collection("c", dim=DIM, create=True)
    h = col.health()
    assert isinstance(h, HealthResult)
    assert h.ok is True
    assert h.quick_check == "ok"
    assert h.store_gen >= 0
    assert h.doc_count == 0
    db.close()


def test_health_ok_after_writes(tmp_path):
    """A collection with data reports healthy."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    col = db.collection("c", dim=DIM, create=True)
    col.add(ids=["a"], documents=["hello"], vectors=[[1.0] + [0.0] * (DIM - 1)])
    h = col.health()
    assert h.ok is True
    assert h.doc_count == 1
    db.close()


def test_health_coherence_after_flush(tmp_path):
    """After flush, the cache is coherent (store_gen == tvim_gen)."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    col = db.collection("c", dim=DIM, create=True)
    col.add(ids=["a"], documents=["hello"], vectors=[[1.0] + [0.0] * (DIM - 1)])
    col.flush()
    h = col.health()
    assert h.coherent is True
    assert h.store_gen == h.tvim_gen
    db.close()


def test_health_coherence_before_flush(tmp_path):
    """Before flush, the cache may be dirty (store_gen > tvim_gen)."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    col = db.collection("c", dim=DIM, create=True)
    col.add(ids=["a"], documents=["hello"], vectors=[[1.0] + [0.0] * (DIM - 1)])
    h = col.health()
    # store_gen was bumped but tvim_gen was not (no flush yet)
    assert h.store_gen > h.tvim_gen or h.coherent is False
    db.close()



