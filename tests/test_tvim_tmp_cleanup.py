"""Issue #57 — orphan .tvim.tmp files left behind on crash during flush.

``flush_impl()`` writes to ``<tvim_path>.tmp`` then renames. A crash between the
``write()`` and ``rename()`` leaves an orphan ``.tmp`` file. These tests verify
that the orphan file is cleaned up on the next open.
"""

import os

import turbovecdb

DIM = 8


def test_orphan_tmp_cleaned_on_open(tmp_path):
    """An orphan .tmp file left by a crash mid-flush must be removed on next open."""
    path = str(tmp_path / "db")
    db = turbovecdb.connect(path)
    col = db.collection("c", dim=DIM, create=True)
    col.add(ids=["a"], vectors=[[1.0] + [0.0] * (DIM - 1)])
    col.flush()
    db.close()

    # Simulate a crash during flush: orphan .tmp file remains after write()
    orphan = os.path.join(col.dir, "index.tvim.tmp")
    with open(orphan, "w") as f:
        f.write("garbage")
    assert os.path.exists(orphan)

    # Reopen — must remove the orphan .tmp
    db2 = turbovecdb.connect(path)
    col2 = db2.collection("c", dim=DIM)
    assert not os.path.exists(orphan), "orphan .tmp should be cleaned on open"
    assert col2.count() == 1
    db2.close()


def test_no_orphan_tmp_opens_normally(tmp_path):
    """A collection with no orphan .tmp files opens and works normally."""
    path = str(tmp_path / "db")
    db = turbovecdb.connect(path)
    col = db.collection("c", dim=DIM, create=True)
    col.add(ids=["a"], vectors=[[1.0] + [0.0] * (DIM - 1)])
    db.close()

    db2 = turbovecdb.connect(path)
    col2 = db2.collection("c", dim=DIM)
    assert col2.count() == 1
    db2.close()
