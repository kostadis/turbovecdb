"""Security: path traversal, SQL chunk limits, filter depth."""

import pytest
import turbovecdb
from turbovecdb import UnsupportedFilterError

DIM = 8


def _v(i):
    v = [0.0] * DIM
    v[i % DIM] = 1.0
    return v


# ── path traversal ────────────────────────────────────────────────────────────

def test_absolute_name_rejected(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    with pytest.raises(ValueError, match="invalid collection name"):
        db.collection("/etc/passwd")
    db.close()


def test_dotdot_name_rejected(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    with pytest.raises(ValueError, match="invalid collection name"):
        db.collection("../sibling")
    db.close()


def test_special_chars_rejected(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    for bad in ("has space", "has.dot", "has/slash", "has\\backslash", ""):
        with pytest.raises(ValueError, match="invalid collection name"):
            db.collection(bad)
    db.close()


def test_valid_names_accepted(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    for name in ("drawers", "my-collection", "snake_case", "abc123", "A" * 128):
        col = db.collection(name, create=True)
        assert col is not None
    db.close()


# ── filter recursion depth ────────────────────────────────────────────────────

def test_filter_depth_limit(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    col = db.collection("c", create=True)
    col.add(ids=["x"], documents=["d"], vectors=[_v(0)])

    # Build a deeply nested $and filter
    deep = {"field": "val"}
    for _ in range(15):
        deep = {"$and": [deep]}

    with pytest.raises(UnsupportedFilterError, match="depth"):
        col.get(where=deep)
    db.close()


# ── $in/$nin size limit ───────────────────────────────────────────────────────

def test_in_size_limit(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    col = db.collection("c", create=True)
    col.add(ids=["x"], documents=["d"], vectors=[_v(0)])
    with pytest.raises(UnsupportedFilterError, match="exceeds maximum"):
        col.get(where={"field": {"$in": ["v"] * 901}})
    db.close()


def test_nin_size_limit(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    col = db.collection("c", create=True)
    col.add(ids=["x"], documents=["d"], vectors=[_v(0)])
    with pytest.raises(UnsupportedFilterError, match="exceeds maximum"):
        col.get(where={"field": {"$nin": ["v"] * 901}})
    db.close()
