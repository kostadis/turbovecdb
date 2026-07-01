"""Tests for the HTTP service layer (src/turbovecdb/service.py).

These drive the op_* handlers directly (no socket) — the handlers are the
whole contract, and the earlier gap was that nothing exercised them at all,
which is how the /clear -> Collection.clear() call shipped broken.
"""
import numpy as np

import turbovecdb
from turbovecdb import service


def _vec(seed, dim=8):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32).tolist()


def test_op_clear_empties_then_stays_usable(tmp_path):
    db_path = str(tmp_path / "svc")
    items = [
        {"id": "a", "vector": _vec(1), "type": "page", "title": "Alpha"},
        {"id": "b", "vector": _vec(2), "type": "page", "title": "Beta"},
        {"id": "c", "vector": _vec(3), "type": "page", "title": "Gamma"},
    ]
    assert service.op_upsert({"db_path": db_path, "items": items})["count"] == 3
    assert service.op_count({"db_path": db_path})["count"] == 3

    # /clear must actually empty the collection (regression: used to raise
    # AttributeError because Collection.clear() was missing).
    assert service.op_clear({"db_path": db_path})["count"] == 0
    assert service.op_count({"db_path": db_path})["count"] == 0

    # And the collection must remain usable — this exercises the next_uid
    # reset + empty-index rebuild together.
    assert service.op_upsert({"db_path": db_path, "items": items[:2]})["count"] == 2
    assert service.op_count({"db_path": db_path})["count"] == 2
    pairs = service.op_candidate_pairs({"db_path": db_path, "threshold": 2.0, "k": 6})
    assert {p["a"] for p in pairs["pairs"]} | {p["b"] for p in pairs["pairs"]} <= {"a", "b"}


def test_op_clear_on_missing_collection_is_noop(tmp_path):
    # No upsert has ever created the "pages" collection here.
    assert service.op_clear({"db_path": str(tmp_path / "empty")})["count"] == 0


def test_collection_clear_directly(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("pages", dim=8, create=True)
    c.add(ids=["a", "b"], vectors=[_vec(1), _vec(2)])
    assert c.count() == 2

    c.clear()
    assert c.count() == 0
    # Index is emptied: a query returns nothing rather than stale hits.
    assert c.query(vector=_vec(1), k=5).ids == []
    assert c.health().doc_count == 0

    # Config survives the clear; the handle is still writable.
    assert c.dim == 8
    c.add(ids=["a"], vectors=[_vec(9)])
    assert c.count() == 1
    db.close()


def test_collection_clear_empty_is_noop(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("pages", dim=8, create=True)
    gen_before = c._store_gen()
    c.clear()  # nothing to clear
    assert c.count() == 0
    assert c._store_gen() == gen_before  # no needless store_gen churn
    db.close()
