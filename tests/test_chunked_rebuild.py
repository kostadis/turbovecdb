"""Index rebuild from the SQLite store: deleting index.tvim forces a rebuild on
the next open, and the rebuilt index must answer queries correctly."""

import os

import turbovecdb

DIM = 8


def test_rebuild_with_fewer_vectors_than_batch_size(tmp_path):
    """A small collection rebuilds in a single batch (no chunking behaviour change)."""
    path = str(tmp_path / "db")
    db = turbovecdb.connect(path)
    col = db.collection("c", dim=DIM, create=True)
    for i in range(5):
        v = [0.0] * DIM
        v[i % DIM] = 1.0
        col.add(ids=[f"doc{i}"], documents=[f"doc {i}"], vectors=[v])
    col.flush()
    db.close()
    # Delete the index so the next open triggers a rebuild
    tvim = os.path.join(path, "c", "index.tvim")
    os.remove(tvim)
    db2 = turbovecdb.connect(path)
    col2 = db2.collection("c", create=False)
    assert col2.count() == 5
    # Verify the index is correct via query
    res = col2.query(vector=[1.0, 0, 0, 0, 0, 0, 0, 0], k=1)
    assert res.ids[0] == "doc0"
    db2.close()


def test_rebuild_chunks_across_multiple_batches(tmp_path):
    """Force a small batch size so the rebuild exercises multiple chunks."""
    path = str(tmp_path / "db")
    db = turbovecdb.connect(path)
    col = db.collection("c", dim=DIM, create=True)
    for i in range(10):
        v = [0.0] * DIM
        v[i % DIM] = 1.0
        col.add(ids=[f"doc{i}"], documents=[f"doc {i}"], vectors=[v])
    col.flush()
    db.close()
    # Delete the index so the next open triggers a rebuild
    tvim = os.path.join(path, "c", "index.tvim")
    os.remove(tvim)
    db2 = turbovecdb.connect(path)
    col2 = db2.collection("c", create=False)
    assert col2.count() == 10
    # Verify the index is correct by running a query
    res = col2.query(vector=[1.0, 0, 0, 0, 0, 0, 0, 0], k=1)
    assert res.ids[0] == "doc0"
    db2.close()


def test_rebuild_chunks_restores_index_after_crash(tmp_path):
    """Simulate a crash by deleting .tvim; chunked rebuild restores it."""
    path = str(tmp_path / "db")
    db = turbovecdb.connect(path)
    col = db.collection("c", dim=DIM, create=True)
    for i in range(7):
        v = [0.0] * DIM
        v[i % DIM] = 1.0
        col.add(ids=[f"doc{i}"], documents=[f"doc {i}"], vectors=[v])
    db.close()  # flush writes the index
    os.remove(os.path.join(path, "c", "index.tvim"))  # simulate crash
    db2 = turbovecdb.connect(path)
    col2 = db2.collection("c", create=False)
    assert col2.count() == 7
    res = col2.query(vector=[0, 0, 0, 0, 0, 0, 1.0, 0], k=1)
    assert res.ids[0] == "doc6"
    db2.close()


def test_rebuild_empty_collection(tmp_path):
    """An empty collection (no vectors) rebuilds without error."""
    path = str(tmp_path / "db")
    db = turbovecdb.connect(path)
    db.collection("c", dim=DIM, create=True)
    db.close()
    db2 = turbovecdb.connect(path)
    col2 = db2.collection("c", create=False)
    assert col2.count() == 0
    db2.close()
