"""Tests proving Phase 1 high-severity bugs are fixed."""

import os
import sqlite3

import pytest

import turbovecdb
from turbovecdb import DimensionMismatchError, TurboVecError

DIM = 8
_UNIT = [1.0, 0, 0, 0, 0, 0, 0, 0]


# ═══════════════════════════════════════════════════════════════════════
# #87 — query() doesn't validate shape/dimension before ANN search
#
# Confirmed broken: query with wrong dim PANICS + poisons mutex;
# empty vector silently returns empty; multi-row silently uses row 0.
# ═══════════════════════════════════════════════════════════════════════


def test_query_wrong_dimension_raises(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", dim=DIM, create=True)
    c.add(ids=["x"], vectors=[_UNIT])
    with pytest.raises(DimensionMismatchError):
        c.query(vector=[[1.0, 0, 0, 0]])  # dim 4 != 8
    # Handle must survive a clean rejection
    c.query(vector=[_UNIT])
    db.close()


def test_query_multi_row_raises(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", dim=DIM, create=True)
    c.add(ids=["x"], vectors=[_UNIT])
    # Single-query API must reject multi-row input
    with pytest.raises((ValueError, TurboVecError)):
        c.query(vector=[_UNIT, [0, 1.0, 0, 0, 0, 0, 0, 0]])
    db.close()


def test_query_empty_vector_raises(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", dim=DIM, create=True)
    c.add(ids=["x"], vectors=[_UNIT])
    with pytest.raises((ValueError, TurboVecError)):
        c.query(vector=[])
    db.close()


def test_query_nan_vector_raises(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", dim=DIM, create=True)
    c.add(ids=["x"], vectors=[_UNIT])
    with pytest.raises((ValueError, TurboVecError)):
        c.query(vector=[[float("nan"), 0, 0, 0, 0, 0, 0, 0]])
    # Handle must survive a clean rejection
    c.query(vector=[_UNIT])
    db.close()


def test_query_inf_vector_raises(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", dim=DIM, create=True)
    c.add(ids=["x"], vectors=[_UNIT])
    with pytest.raises((ValueError, TurboVecError)):
        c.query(vector=[[float("inf"), 0, 0, 0, 0, 0, 0, 0]])
    c.query(vector=[_UNIT])
    db.close()


# ═══════════════════════════════════════════════════════════════════════
# #85 — Corrupt / missing dim metadata silently treated as uninitialised
#
# Confirmed broken: dim=None, count=1, query returns 0 results (silent).
# ═══════════════════════════════════════════════════════════════════════


def test_corrupt_dim_meta_errors_on_reopen(tmp_path):
    path = str(tmp_path / "db")
    db = turbovecdb.connect(path)
    c = db.collection("c", dim=DIM, create=True)
    c.add(ids=["x"], documents=["hello"], vectors=[_UNIT])
    db.close()

    # Corrupt dim metadata directly in SQLite
    store = os.path.join(path, "c", "store.sqlite3")
    conn = sqlite3.connect(store)
    conn.execute("UPDATE meta SET value = 'not-a-number' WHERE key = 'dim'")
    conn.commit()
    conn.close()

    # Reopening a populated collection with corrupt dim must fail with a
    # typed error — NOT silently open as dim=None / empty query
    db2 = turbovecdb.connect(path)
    with pytest.raises((ValueError, TurboVecError)):
        db2.collection("c", create=False)
    db2.close()


def test_missing_dim_meta_errors_on_reopen(tmp_path):
    path = str(tmp_path / "db")
    db = turbovecdb.connect(path)
    c = db.collection("c", dim=DIM, create=True)
    c.add(ids=["x"], documents=["hello"], vectors=[_UNIT])
    db.close()

    # Delete dim metadata entirely
    store = os.path.join(path, "c", "store.sqlite3")
    conn = sqlite3.connect(store)
    conn.execute("DELETE FROM meta WHERE key = 'dim'")
    conn.commit()
    conn.close()

    db2 = turbovecdb.connect(path)
    with pytest.raises((ValueError, TurboVecError)):
        db2.collection("c", create=False)
    db2.close()


# ═══════════════════════════════════════════════════════════════════════
# Safety-net: these were ALREADY FIXED — verify they stay fixed.
# ═══════════════════════════════════════════════════════════════════════


def test_nan_vector_add_rejected(tmp_path):
    """Bug #88 — already fixed. NaN rejected with clean error at add boundary."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", dim=DIM, create=True)
    with pytest.raises((ValueError, TurboVecError)):
        c.add(ids=["x"], vectors=[[float("nan"), 0, 0, 0, 0, 0, 0, 0]])
    # Handle survives
    c.add(ids=["y"], vectors=[_UNIT])
    assert c.count() == 1
    db.close()


def test_inf_vector_add_rejected(tmp_path):
    """Bug #88 — already fixed. Inf rejected with clean error at add boundary."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", dim=DIM, create=True)
    with pytest.raises((ValueError, TurboVecError)):
        c.add(ids=["x"], vectors=[[float("inf"), 0, 0, 0, 0, 0, 0, 0]])
    c.add(ids=["y"], vectors=[_UNIT])
    assert c.count() == 1
    db.close()


def test_invalid_bit_width_constructor_raises(tmp_path):
    """Bug #86 — already fixed. Constructor rejects bit_width not in {2,3,4}."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    with pytest.raises((ValueError, TurboVecError)):
        db.collection("c", dim=DIM, bit_width=7, create=True)
    db.close()
