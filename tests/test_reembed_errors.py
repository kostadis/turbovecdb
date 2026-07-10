"""Reembed error messages include uid range for easier debugging."""

import pytest

import turbovecdb
from turbovecdb import TurboVecError

DIM = 8


def test_reembed_batch_error_includes_uid_range(tmp_path):
    """When the embedder fails mid-batch, the error includes the uid range."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    col = db.collection("c", dim=DIM, create=True)

    # Add several documents to get a batch
    for i in range(5):
        v = [0.0] * DIM
        v[i % DIM] = 1.0
        col.add(ids=[f"doc{i}"], documents=[f"document {i}"], vectors=[v])

    def failing_embedder(texts):
        if "document 3" in texts:
            raise RuntimeError("embedder crashed on doc 3")
        import numpy as np
        return np.array([[float(len(t)) for _ in range(DIM)] for t in texts])

    with pytest.raises(TurboVecError) as excinfo:
        col.reembed(failing_embedder, batch_size=2)
    assert "uids [" in str(excinfo.value)
    assert "embedder crashed" in str(excinfo.value)


def test_reembed_final_batch_error_includes_uid_range(tmp_path):
    """When the embedder fails on the final batch, the error includes the uid range."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    col = db.collection("c", dim=DIM, create=True)

    def failing_embedder(texts):
        raise RuntimeError("embedder completely dead")

    import numpy as np
    col.add(ids=["a"], documents=["hello"], vectors=[np.ones(DIM, dtype=np.float32)])

    with pytest.raises(TurboVecError) as excinfo:
        col.reembed(failing_embedder, batch_size=10)
    assert "uids [" in str(excinfo.value)
    assert "embedder completely dead" in str(excinfo.value)
