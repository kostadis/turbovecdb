"""Test the GAP-1 embedder identity guard on add/upsert write paths."""

import pytest

import turbovecdb
from turbovecdb import EmbedderIdentityMismatchError, EmbedderRequiredError

DIM = 8


def _embedder_a(texts):
    """'alpha'-detecting toy embedder."""
    import numpy as np
    out = []
    for t in texts:
        v = [0.0] * DIM
        v[0 if "alpha" in t else 1] = 1.0
        out.append(v)
    return out


def _embedder_b(texts):
    """Different toy embedder: places everything at position 2."""
    import numpy as np
    return [[0.0, 0.0, 1.0] + [0.0] * (DIM - 3) for _ in texts]


def test_stores_identity_on_creation(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", embedder=_embedder_a, create=True)
    identity = c._meta_get("embedder_identity")
    assert identity == "_embedder_a"
    db.close()


def test_identity_persists_across_reopen(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", embedder=_embedder_a, create=True)
    c.add(ids=["a"], documents=["alpha document"])
    db.close()

    db2 = turbovecdb.connect(str(tmp_path / "db"))
    c2 = db2.collection("c", embedder=_embedder_a)
    # Reopen with same embedder — identity already stored, add succeeds.
    c2.add(ids=["b"], documents=["beta document"])
    db2.close()


def test_mismatch_raises_on_add(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", embedder=_embedder_a, create=True)
    c.add(ids=["a"], documents=["alpha document"])
    db.close()

    db2 = turbovecdb.connect(str(tmp_path / "db"))
    c2 = db2.collection("c", embedder=_embedder_b)
    with pytest.raises(EmbedderIdentityMismatchError) as exc:
        c2.add(ids=["b"], documents=["beta document"])
    assert "_embedder_a" in str(exc.value)
    assert "_embedder_b" in str(exc.value)
    db2.close()


def test_mismatch_raises_on_upsert(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", embedder=_embedder_a, create=True)
    c.add(ids=["a"], documents=["alpha document"])
    db.close()

    db2 = turbovecdb.connect(str(tmp_path / "db"))
    c2 = db2.collection("c", embedder=_embedder_b)
    with pytest.raises(EmbedderIdentityMismatchError):
        c2.upsert(ids=["a"], documents=["updated"])
    db2.close()


def test_raw_vectors_bypasses_identity_check(tmp_path):
    """Passing vectors= directly should never check embedder identity."""
    import numpy as np
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", embedder=_embedder_a, create=True)
    c.add(ids=["a"], documents=["alpha document"])

    # Simulate a "different" embedder's vectors via raw vector path
    vec = np.array([[0.0, 0.0, 1.0] + [0.0] * (DIM - 3)], dtype=np.float32)
    c.add(ids=["b"], vectors=vec)  # should not raise
    db.close()


def test_no_embedder_no_identity_check(tmp_path):
    """Collection without embedder — identity should not be stored and no check."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", create=True)
    assert c._meta_get("embedder_identity") is None
    c.add(ids=["a"], vectors=[[1.0] + [0.0] * (DIM - 1)])
    db.close()


def test_legacy_collection_allows_first_embedder(tmp_path):
    """Populated legacy collection (no identity stored) must reject a new
    embedder's text writes — require reembed() to set a proper identity."""
    from turbovecdb.errors import EmbedderMismatchError

    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", create=True)
    c.add(ids=["a"], vectors=[[1.0] + [0.0] * (DIM - 1)])
    db.close()

    db2 = turbovecdb.connect(str(tmp_path / "db"))
    c2 = db2.collection("c", embedder=_embedder_a)
    with pytest.raises(EmbedderMismatchError, match="no embedder identity"):
        c2.add(ids=["b"], documents=["beta document"])
    # Raw vector path still works — it bypasses the embedder identity check.
    c2.add(ids=["b"], vectors=[[1.0] + [0.0] * (DIM - 1)])
    db2.close()


def test_identity_after_reembed(tmp_path):
    """reembed updates identity; subsequent adds use the new identity."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", embedder=_embedder_a, create=True)
    c.add(ids=["a"], documents=["alpha document"])
    db.close()

    db2 = turbovecdb.connect(str(tmp_path / "db"))
    c2 = db2.collection("c", embedder=_embedder_a)
    c2.reembed(_embedder_b, batch_size=2)
    # After reembed, stored identity is _embedder_b but self._embedder is _embedder_a
    # — the guard catches this mismatch.
    with pytest.raises(EmbedderIdentityMismatchError):
        c2.add(ids=["b"], documents=["beta document"])
    db2.close()

    # Reopen with the new embedder — now add should work.
    db3 = turbovecdb.connect(str(tmp_path / "db"))
    c3 = db3.collection("c", embedder=_embedder_b)
    c3.add(ids=["b"], documents=["beta document"])
    db3.close()


def test_mismatch_error_message(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", embedder=_embedder_a, create=True)
    c.add(ids=["a"], documents=["alpha document"])
    db.close()

    db2 = turbovecdb.connect(str(tmp_path / "db"))
    c2 = db2.collection("c", embedder=_embedder_b)
    with pytest.raises(EmbedderIdentityMismatchError) as exc:
        c2.add(ids=["b"], documents=["beta document"])
    msg = str(exc.value)
    assert "use reembed()" in msg
    db2.close()
