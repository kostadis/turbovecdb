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


def test_rebuilds_when_tvim_is_legacy_format(tmp_path):
    """A `.tvim` written by turbovec 0.9 is on-disk format v3, which
    turbovec 1.0 refuses to decode: v1-v4 predate the v5 rotation change
    that "altered every encoded byte". Opening such a collection must fall
    back to rebuilding from the SQLite vectors (the source of truth), not
    fail. This is the path every collection created before the 1.0 upgrade
    takes on its first open.
    """
    path = str(tmp_path / "db")
    db = turbovecdb.connect(path)
    _seed(db.collection("drawers", create=True))
    db.close()  # flushes a current-format .tvim

    tvim = os.path.join(path, "drawers", "index.tvim")
    # Stamp a turbovec 0.9 header over it: TVIM magic + version byte 3.
    # `tvim_gen` in SQLite still matches `store_gen`, so the staleness gate
    # passes and the load itself is what has to reject this.
    with open(tvim, "wb") as f:
        f.write(b"TVIM" + bytes([3]) + b"\x00" * 64)

    db2 = turbovecdb.connect(path)
    c2 = db2.collection("drawers", create=False)
    assert c2.count() == 4
    assert c2.query(vector=VECS["a"], k=2).ids == ["a", "d"]  # rebuilt from SQLite
    db2.close()

    # The rebuild marks the collection dirty, so close() rewrites the cache
    # in the current format — the legacy file does not survive. turbovec's
    # v7 container carries its own magic (b"TV7\0"), not the b"TVIM" of the
    # v3-era id-map files.
    with open(tvim, "rb") as f:
        head = f.read(4)
    assert head == b"TV7\0", f"legacy v3 cache must have been replaced, got {head!r}"


def test_rejects_dim_above_engine_maximum(tmp_path):
    """turbovec 1.0 lowered MAX_DIM from 65536 to 16384. An oversized dim
    must surface as a turbovecdb DimensionMismatchError rather than as a
    raw construct error from inside the engine.
    """
    db = turbovecdb.connect(str(tmp_path / "db"))
    col = db.collection("wide", create=True)
    with pytest.raises(DimensionMismatchError):
        col.add(ids=["x"], vectors=[[0.0] * 16392])  # multiple of 8, over the cap
    db.close()


# ── reembed tests ───────────────────────────────────────────────────────────


def test_reembed_same_dimension(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("reembed_test", dim=8, create=True)

    def embed1(texts):
        import numpy as np
        return np.array([[float(len(t)) for _ in range(8)] for t in texts])

    def embed2(texts):
        import numpy as np
        return np.array([[float(len(t) * 2) for _ in range(8)] for t in texts])

    # Add with vectors directly
    c.add(ids=["a", "b"], documents=["hello", "world"], vectors=embed1(["hello", "world"]))

    report = c.reembed(embed2, batch_size=2)
    assert report.total_docs == 2
    assert report.old_dim == 8
    assert report.new_dim == 8
    assert report.skipped == 0
    assert report.elapsed_seconds >= 0

    db.close()


def test_reembed_dimension_change(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("dim_change", dim=8, create=True)

    def embed8(texts):
        import numpy as np
        return np.array([[float(len(t)) for _ in range(8)] for t in texts])

    def embed16(texts):
        import numpy as np
        return np.array([[float(len(t)) for _ in range(16)] for t in texts])

    c.add(ids=["a"], documents=["hello"], vectors=embed8(["hello"]))

    report = c.reembed(embed16, dim=16, batch_size=2)
    assert report.total_docs == 1
    assert report.old_dim == 8
    assert report.new_dim == 16
    assert c.dim == 16

    db.close()


def test_reembed_skip_empty_keep(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("skip_empty", dim=8, create=True)

    def embed(texts):
        import numpy as np
        return np.array([[float(len(t)) for _ in range(8)] for t in texts])

    # Add with vectors directly (no embedder)
    vecs = embed(["hello", "", "world"])
    c.add(ids=["a", "b", "c"], documents=["hello", "", "world"], vectors=vecs)

    report = c.reembed(embed, skip_empty="keep", batch_size=2)
    assert report.total_docs == 3
    assert report.skipped == 1  # empty doc kept with normalized old vector

    db.close()


def test_reembed_skip_empty_drop(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("skip_empty", dim=8, create=True)

    def embed(texts):
        import numpy as np
        return np.array([[float(len(t)) for _ in range(8)] for t in texts])

    # Add with vectors directly (no embedder)
    vecs = embed(["hello", "", "world"])
    c.add(ids=["a", "b", "c"], documents=["hello", "", "world"], vectors=vecs)

    report = c.reembed(embed, skip_empty="drop", batch_size=2)
    assert report.total_docs == 3
    assert report.skipped == 1  # empty doc dropped

    db.close()


def test_reembed_skip_empty_error(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("skip_empty", dim=8, create=True)

    def embed(texts):
        import numpy as np
        return np.array([[float(len(t)) for _ in range(8)] for t in texts])

    # Add with vectors directly (no embedder)
    vecs = embed(["hello", ""])
    c.add(ids=["a", "b"], documents=["hello", ""], vectors=vecs)

    with pytest.raises(ValueError, match="Cannot re-embed empty document"):
        c.reembed(embed, skip_empty="error", batch_size=2)

    db.close()


def test_reembed_invalid_skip_empty(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("invalid", dim=8, create=True)

    def embed(texts):
        import numpy as np
        return np.array([[float(len(t)) for _ in range(8)] for t in texts])

    with pytest.raises(ValueError, match="skip_empty must be"):
        c.reembed(embed, skip_empty="invalid")

    db.close()


def test_reembed_invalid_bit_width(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("invalid", dim=8, create=True)

    def embed(texts):
        import numpy as np
        return np.array([[float(len(t)) for _ in range(8)] for t in texts])

    with pytest.raises(ValueError, match="bit_width must be"):
        c.reembed(embed, bit_width=5)

    db.close()


def test_reembed_non_callable_raises(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("invalid", dim=8, create=True)

    with pytest.raises(ValueError, match="embedder must be a callable"):
        c.reembed("not a callable")

    db.close()


def test_reembed_dimension_mismatch_raises(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("mismatch", dim=8, create=True)

    def embed8(texts):
        import numpy as np
        return np.array([[float(len(t)) for _ in range(8)] for t in texts])

    def embed16(texts):
        import numpy as np
        return np.array([[float(len(t)) for _ in range(16)] for t in texts])

    # Add with 8-dim vectors first
    c.add(ids=["a"], documents=["hello"], vectors=embed8(["hello"]))

    # Now try to reembed with 16-dim embedder but dim=8 specified
    with pytest.raises(DimensionMismatchError):
        c.reembed(embed16, dim=8, batch_size=2)

    db.close()


def test_reembed_empty_collection(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("empty", dim=8, create=True)

    def embed(texts):
        import numpy as np
        return np.array([[float(len(t)) for _ in range(8)] for t in texts])

    report = c.reembed(embed)
    assert report.total_docs == 0
    assert report.old_dim == 8
    assert report.new_dim == 8

    db.close()


# ── delete_collection tests ──────────────────────────────────────────────────


def test_delete_collection(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("to_delete", dim=8, create=True)
    import numpy as np
    c.add(ids=["a"], documents=["hello"], vectors=[np.ones(8, dtype=np.float32)])
    db.close()

    db2 = turbovecdb.connect(str(tmp_path / "db"))
    assert "to_delete" in db2.list_collections()
    db2.delete_collection("to_delete")
    assert "to_delete" not in db2.list_collections()
    db2.close()


def test_delete_collection_not_found(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))

    with pytest.raises(CollectionNotFoundError):
        db.delete_collection("nonexistent")

    db.close()


def test_delete_collection_closes_handle(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("to_close", dim=8, create=True)
    import numpy as np
    c.add(ids=["a"], documents=["hello"], vectors=[np.ones(8, dtype=np.float32)])
    db.close()

    db2 = turbovecdb.connect(str(tmp_path / "db"))
    handle = db2.collection("to_close", create=False)
    assert handle.count() == 1
    db2.delete_collection("to_close")
    # Handle should be closed
    assert "to_close" not in db2.list_collections()
    db2.close()


# ── integration: end-to-end migration ────────────────────────────────────────


def test_reembed_with_bit_width_change(tmp_path):
    """Test that bit_width can be changed during reembed."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("bitwidth_test", dim=8, bit_width=4, create=True)

    def embed(texts):
        import numpy as np
        return np.array([[float(len(t)) for _ in range(8)] for t in texts])

    c.add(ids=["a"], documents=["hello"], vectors=embed(["hello"]))

    # Reembed with different bit_width
    report = c.reembed(embed, bit_width=2, batch_size=2)
    assert report.total_docs == 1
    assert report.old_dim == 8
    assert c.dim == 8
    # bit_width should be updated
    assert c._bit_width == 2

    db.close()


def test_reembed_updates_embedder_identity(tmp_path):
    """Test that reembed updates the stored embedder identity."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("identity_test", dim=8, create=True)

    def embed1(texts):
        import numpy as np
        return np.array([[float(len(t)) for _ in range(8)] for t in texts])

    c.add(ids=["a"], documents=["hello"], vectors=embed1(["hello"]))

    # Get initial embedder identity
    initial_identity = c._meta_get("embedder_identity")
    assert initial_identity is None or "add" in initial_identity.lower()

    # Reembed with new embedder
    def embed2(texts):
        import numpy as np
        return np.array([[float(len(t) * 2) for _ in range(8)] for t in texts])

    c.reembed(embed2, batch_size=2)

    # Check embedder identity was updated
    new_identity = c._meta_get("embedder_identity")
    assert new_identity == "embed2"

    db.close()
