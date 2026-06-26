"""P3 tests: name validation, update_documents, embedder_identity property, vectors= warning."""

import logging

import numpy as np
import pytest

import turbovecdb

DIM = 8


# -- P3-1: delete_collection name validation ----------------------------------

def test_delete_collection_invalid_name_raises(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    with pytest.raises(ValueError, match="invalid collection name"):
        db.delete_collection("../escape")
    with pytest.raises(ValueError, match="invalid collection name"):
        db.delete_collection("")
    with pytest.raises(ValueError, match="invalid collection name"):
        db.delete_collection("a" * 200)
    db.close()


def test_delete_collection_valid_name_works(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("valid-name_123", dim=DIM, create=True)
    c.add(ids=["x"], vectors=[[1.0] + [0.0] * (DIM - 1)])
    db.delete_collection("valid-name_123")
    assert "valid-name_123" not in db.list_collections()
    db.close()


# -- P3-2: update_documents ---------------------------------------------------

def _vec():
    return np.array([[1.0] + [0.0] * (DIM - 1)], dtype=np.float32)


def test_update_documents_basic(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", dim=DIM, create=True)
    c.add(ids=["a", "b"], documents=["old1", "old2"], vectors=_vec().repeat(2, axis=0))
    c.update_documents(ids=["a", "b"], documents=["new1", "new2"])
    res = c.get(ids=["a", "b"], include=["documents"])
    assert res.documents[0] == "new1"
    assert res.documents[1] == "new2"
    db.close()


def test_update_documents_does_not_affect_metadata(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", dim=DIM, create=True)
    c.add(ids=["a"], documents=["x"], metadatas=[{"k": "v"}], vectors=_vec())
    c.update_documents(ids=["a"], documents=["y"])
    res = c.get(ids=["a"], include=["documents", "metadatas"])
    assert res.documents[0] == "y"
    assert res.metadatas[0] == {"k": "v"}
    db.close()


def test_update_documents_missing_id_raises(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", dim=DIM, create=True)
    c.add(ids=["a"], vectors=_vec())
    with pytest.raises(ValueError, match="not found"):
        c.update_documents(ids=["b"], documents=["x"])
    db.close()


def test_update_documents_length_mismatch_raises(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", dim=DIM, create=True)
    c.add(ids=["a"], vectors=_vec())
    with pytest.raises(ValueError, match="length"):
        c.update_documents(ids=["a"], documents=["x", "y"])
    db.close()


def test_update_documents_empty_ids_is_noop(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", dim=DIM, create=True)
    c.add(ids=["a"], documents=["x"], vectors=_vec())
    c.update_documents(ids=[], documents=[])
    res = c.get(ids=["a"])
    assert res.documents[0] == "x"
    db.close()


def test_update_documents_bumps_store_gen(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", dim=DIM, create=True)
    c.add(ids=["a"], vectors=_vec())
    gen1 = c._store_gen()
    c.update_documents(ids=["a"], documents=["new"])
    gen2 = c._store_gen()
    assert gen2 == gen1 + 1
    db.close()


def test_update_documents_persists_across_reopen(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", dim=DIM, create=True)
    c.add(ids=["a"], documents=["old"], vectors=_vec())
    c.update_documents(ids=["a"], documents=["new"])
    db.close()

    db2 = turbovecdb.connect(str(tmp_path / "db"))
    c2 = db2.collection("c", dim=DIM)
    res = c2.get(ids=["a"])
    assert res.documents[0] == "new"
    db2.close()


# -- P3-3: embedder_identity property -----------------------------------------

def _embedder(texts):
    return [[float(len(t)) for _ in range(DIM)] for t in texts]


def test_embedder_identity_property_on_creation(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", embedder=_embedder, create=True)
    assert c.embedder_identity == "_embedder"
    db.close()


def test_embedder_identity_property_none_without_embedder(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", dim=DIM, create=True)
    assert c.embedder_identity is None
    db.close()


def test_embedder_identity_property_after_reembed(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", embedder=_embedder, create=True)
    c.add(ids=["a"], documents=["hello"])
    c.reembed(_embedder, batch_size=2)
    assert c.embedder_identity == "_embedder"
    db.close()


# -- P3-4: vectors= bypass warning --------------------------------------------

def test_vectors_bypass_warning_logged(caplog, tmp_path):
    caplog.set_level(logging.WARNING)
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", embedder=_embedder, create=True)
    c.add(ids=["a"], documents=["hello"])
    c.add(ids=["b"], vectors=[[1.0] + [0.0] * (DIM - 1)])
    assert any("bypasses the configured embedder" in msg for msg in caplog.messages)
    db.close()


def test_vectors_bypass_no_warning_without_embedder(caplog, tmp_path):
    caplog.set_level(logging.WARNING)
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", dim=DIM, create=True)
    c.add(ids=["a"], vectors=[[1.0] + [0.0] * (DIM - 1)])
    assert not any("bypasses" in msg for msg in caplog.messages)
    db.close()


def test_vectors_bypass_warning_on_upsert(caplog, tmp_path):
    caplog.set_level(logging.WARNING)
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", embedder=_embedder, create=True)
    c.add(ids=["a"], documents=["hello"])
    c.upsert(ids=["a"], vectors=[[0.0, 1.0] + [0.0] * (DIM - 2)])
    assert any("bypasses the configured embedder" in msg for msg in caplog.messages)
    db.close()
