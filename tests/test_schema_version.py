"""Schema version: migration framework tracks on-disk format evolution."""

import gc
import os
import sqlite3


import turbovecdb

DIM = 8


def test_new_collection_has_schema_version(tmp_path):
    """A newly created collection stores the current schema version."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    col = db.collection("c", dim=DIM, create=True)
    version = int(col._meta_get("schema_version", 0))
    assert version >= 1
    db.close()


def test_schema_version_persists_across_reopen(tmp_path):
    """Schema version is still present after closing and reopening."""
    path = str(tmp_path / "db")
    db = turbovecdb.connect(path)
    col = db.collection("c", dim=DIM, create=True)
    col.add(ids=["a"], documents=["hello"], vectors=[[1.0] + [0.0] * (DIM - 1)])
    db.close()

    db2 = turbovecdb.connect(path)
    col2 = db2.collection("c", create=False)
    version = int(col2._meta_get("schema_version", 0))
    assert version >= 1
    db2.close()


def test_legacy_collection_gets_schema_version(tmp_path):
    """A collection created without schema_version (legacy) gets upgraded on open."""
    path = str(tmp_path / "db")
    db = turbovecdb.connect(path)
    col = db.collection("c", dim=DIM, create=True)
    col.add(ids=["a"], documents=["hello"], vectors=[[1.0] + [0.0] * (DIM - 1)])
    db.close()
    del col, db
    gc.collect()  # release the Rust engine's connection before touching the file

    # Simulate a legacy collection by removing schema_version directly on disk.
    conn = sqlite3.connect(os.path.join(path, "c", "store.sqlite3"))
    conn.execute("DELETE FROM meta WHERE key='schema_version'")
    conn.commit()
    conn.close()

    db2 = turbovecdb.connect(path)
    col2 = db2.collection("c", create=False)
    version = int(col2._meta_get("schema_version", 0))
    assert version >= 1
    assert col2.count() == 1  # data intact
    db2.close()


def test_schema_version_not_bumped_on_every_open(tmp_path):
    """Re-opening a collection does not unnecessarily update schema_version."""
    path = str(tmp_path / "db")
    db = turbovecdb.connect(path)
    col = db.collection("c", dim=DIM, create=True)
    db.close()

    db2 = turbovecdb.connect(path)
    col2 = db2.collection("c", create=False)
    v1 = col2._meta_get("schema_version")
    db2.close()

    db3 = turbovecdb.connect(path)
    col3 = db3.collection("c", create=False)
    v2 = col3._meta_get("schema_version")
    assert v1 == v2  # not bumped again
    db3.close()
