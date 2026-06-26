"""Test atomic metadata-only updates (GAP-3)."""

import numpy as np
import pytest

import turbovecdb

DIM = 8


def _vec():
    return np.array([[1.0] + [0.0] * (DIM - 1)], dtype=np.float32)


def test_update_metadata_basic(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", dim=DIM, create=True)
    vecs = _vec().repeat(3, axis=0)
    c.add(ids=["a", "b", "c"],
          metadatas=[{"key1": 1}, {"key2": 2}, {"key3": 3}],
          vectors=vecs)
    c.update_metadata(ids=["a", "c"], metadatas=[{"new": 10}, {"new": 30}])
    res_a = c.get(ids=["a"])
    assert res_a.metadatas[0] == {"new": 10}
    res_c = c.get(ids=["c"])
    assert res_c.metadatas[0] == {"new": 30}
    res_b = c.get(ids=["b"])
    assert res_b.metadatas[0] == {"key2": 2}
    db.close()


def test_update_metadata_does_not_affect_vectors(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", dim=DIM, create=True)
    vec = [1.0 / (DIM ** 0.5)] * DIM
    c.add(ids=["a"], metadatas=[{"old": 1}], vectors=[vec])
    c.update_metadata(ids=["a"], metadatas=[{"new": 2}])
    res = c.get(ids=["a"], include=["metadatas", "vectors"])
    assert res.metadatas[0] == {"new": 2}
    assert res.vectors[0] == pytest.approx(vec, abs=1e-5)
    db.close()


def test_update_metadata_none_entries(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", dim=DIM, create=True)
    vecs = _vec().repeat(2, axis=0)
    c.add(ids=["a", "b"],
          metadatas=[{"k": "v"}, {"k": "w"}],
          vectors=vecs)
    c.update_metadata(ids=["a"], metadatas=[None])
    res_a = c.get(ids=["a"])
    assert res_a.metadatas[0] == {}
    res_b = c.get(ids=["b"])
    assert res_b.metadatas[0] == {"k": "w"}
    db.close()


def test_update_metadata_missing_id_raises(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", dim=DIM, create=True)
    c.add(ids=["a"], vectors=_vec())
    with pytest.raises(ValueError, match="not found"):
        c.update_metadata(ids=["b"], metadatas=[{"k": "v"}])
    db.close()


def test_update_metadata_length_mismatch_raises(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", dim=DIM, create=True)
    c.add(ids=["a"], vectors=_vec())
    with pytest.raises(ValueError, match="length"):
        c.update_metadata(ids=["a"], metadatas=[{"k": "v"}, {"k2": "v2"}])
    db.close()


def test_update_metadata_empty_ids_is_noop(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", dim=DIM, create=True)
    c.add(ids=["a"], metadatas=[{"k": "v"}], vectors=_vec())
    c.update_metadata(ids=[], metadatas=[])
    res = c.get(ids=["a"])
    assert res.metadatas[0] == {"k": "v"}
    db.close()


def test_update_metadata_bumps_store_gen(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", dim=DIM, create=True)
    c.add(ids=["a"], vectors=_vec())
    gen1 = c._store_gen()
    c.update_metadata(ids=["a"], metadatas=[{"k": "v"}])
    gen2 = c._store_gen()
    assert gen2 == gen1 + 1
    db.close()


def test_update_metadata_persists_across_reopen(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", dim=DIM, create=True)
    c.add(ids=["a"], metadatas=[{"old": 1}], vectors=_vec())
    c.update_metadata(ids=["a"], metadatas=[{"new": 2}])
    db.close()

    db2 = turbovecdb.connect(str(tmp_path / "db"))
    c2 = db2.collection("c", dim=DIM)
    res = c2.get(ids=["a"])
    assert res.metadatas[0] == {"new": 2}
    db2.close()


def test_update_metadata_on_upserted_row(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", dim=DIM, create=True)
    c.add(ids=["a"], metadatas=[{"k": "v"}], vectors=_vec())
    vec1 = np.array([[0.0] * (DIM - 1) + [1.0]], dtype=np.float32)
    c.upsert(ids=["a"], metadatas=[{"k": "w"}], vectors=vec1)
    c.update_metadata(ids=["a"], metadatas=[{"updated": True}])
    res = c.get(ids=["a"], include=["metadatas"])
    assert res.metadatas[0] == {"updated": True}
    db.close()


def test_update_metadata_concurrent_safety(tmp_path):
    import threading
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", dim=DIM, create=True)
    c.add(ids=["a"], metadatas=[{"k": "v"}], vectors=_vec())

    def writer():
        db2 = turbovecdb.connect(str(tmp_path / "db"))
        c2 = db2.collection("c", dim=DIM)
        c2.update_metadata(ids=["a"], metadatas=[{"from": "writer"}])
        db2.close()

    t = threading.Thread(target=writer)
    t.start()
    t.join()

    res = c.get(ids=["a"])
    assert res.metadatas[0] == {"from": "writer"}
    db.close()
