"""Test _seen_gen advancement on reader-only reload failure (P2-4)."""

import numpy as np
import pytest

import turbovecdb

DIM = 8


def test_seen_gen_advanced_on_corrupt_tvim(tmp_path):
    """If tvim is corrupt, rebuild succeeds, seen_gen is advanced."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", dim=DIM, create=True)
    c.add(ids=["a"], vectors=[[1.0] + [0.0] * (DIM - 1)])
    c.flush()
    gen = c._store_gen()

    # Corrupt tvim so load_index fails
    with open(c._tvim_path, "wb") as f:
        f.write(b"garbage")

    # Add more data so store_gen advances past tvim_gen
    c.add(ids=["b"], vectors=[[0.0, 1.0] + [0.0] * (DIM - 2)])
    new_gen = c._store_gen()
    assert new_gen > gen

    # Write the handle (simulate reader-only reopen)
    db.close()
    db2 = turbovecdb.connect(str(tmp_path / "db"))
    c2 = db2.collection("c", dim=DIM)

    # The tvim is corrupt, so _reload_index will fall through to rebuild,
    # which should succeed. seen_gen should be advanced.
    assert c2._seen_gen == new_gen
    res = c2.query(vector=[1.0] + [0.0] * (DIM - 1), k=10)
    assert len(res.ids) == 2
    db2.close()


def test_rebuild_failure_sets_seen_gen_and_empty_index(tmp_path):
    """If rebuild fails (e.g. corrupt vector blob), seen_gen is advanced
    and index is replaced with an empty one, preventing infinite retry."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", dim=DIM, create=True)

    # Insert a row with a valid vector
    c.add(ids=["a"], vectors=[[1.0] + [0.0] * (DIM - 1)])

    # Insert a row with a wrong-length vector blob directly into SQLite
    # to make np.stack fail during rebuild.
    c._conn.execute(
        "INSERT INTO docs(uid, str_id, document, metadata, vector) "
        "VALUES(?, ?, '', '{}', ?)",
        (999, "bad", b"too short")
    )
    c._conn.commit()
    # Bump store_gen past what the index saw
    c._meta_set("store_gen", c._store_gen() + 1)
    c._conn.commit()
    store_gen = c._store_gen()

    # Reset seen_gen to force a reload attempt
    c._seen_gen = -1

    # Now calling _ensure_current should trigger _reload_index,
    # which will fail due to the corrupt row.
    with pytest.raises(Exception):
        c._ensure_current()

    # After failure, seen_gen should be advanced and index is empty
    assert c._seen_gen == store_gen
    # Index is a valid empty index (not None), so write path won't crash
    assert c._index is not None
    db.close()


def test_query_after_rebuild_failure_returns_empty(tmp_path):
    """After a rebuild failure, subsequent queries return empty (no crash)."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", dim=DIM, create=True)
    c.add(ids=["a"], vectors=[[1.0] + [0.0] * (DIM - 1)])
    c._conn.execute(
        "INSERT INTO docs(uid, str_id, document, metadata, vector) "
        "VALUES(?, ?, '', '{}', ?)",
        (999, "bad", b"bad")
    )
    c._conn.commit()
    c._meta_set("store_gen", c._store_gen() + 1)
    c._conn.commit()
    store_gen = c._store_gen()
    c._seen_gen = -1

    # First call fails
    with pytest.raises(Exception):
        c._ensure_current()

    assert c._seen_gen == store_gen
    assert c._index is not None

    # Second call should skip reload and return empty (empty index)
    res = c.query(vector=[1.0] + [0.0] * (DIM - 1), k=10)
    assert res.ids == []
    assert res.distances == []
    db.close()


def test_write_after_rebuild_failure_works(tmp_path):
    """After a rebuild failure, a write path succeeds (empty index is valid)."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", dim=DIM, create=True)
    c.add(ids=["a"], vectors=[[1.0] + [0.0] * (DIM - 1)])
    c._conn.execute(
        "INSERT INTO docs(uid, str_id, document, metadata, vector) "
        "VALUES(?, ?, '', '{}', ?)",
        (999, "bad", b"bad")
    )
    c._conn.commit()
    c._meta_set("store_gen", c._store_gen() + 1)
    c._conn.commit()
    store_gen = c._store_gen()
    c._seen_gen = -1

    # First call fails
    with pytest.raises(Exception):
        c._ensure_current()

    assert c._seen_gen == store_gen
    assert c._index is not None

    # Remove the corrupt row so a future reload would succeed
    c._conn.execute("DELETE FROM docs WHERE uid=999")
    c._conn.commit()

    # Write path — _ensure_current skips reload (seen_gen == store_gen),
    # but the empty index is valid so the write succeeds.
    c.add(ids=["b"], vectors=[[0.0, 1.0] + [0.0] * (DIM - 2)])
    # The write added "b" to the index — query should find it.
    res = c.query(vector=[0.0, 1.0] + [0.0] * (DIM - 2), k=10)
    assert "b" in res.ids
    # "a" is gone from the index (it was never rebuilt after the crash)
    # but the write should not crash.
    db.close()


def test_normal_reload_after_seen_gen_advanced(tmp_path):
    """After a failure and state fix, normal reload works again."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", dim=DIM, create=True)
    c.add(ids=["a"], vectors=[[1.0] + [0.0] * (DIM - 1)])
    c._conn.execute(
        "INSERT INTO docs(uid, str_id, document, metadata, vector) "
        "VALUES(?, ?, '', '{}', ?)",
        (999, "bad", b"bad")
    )
    c._conn.commit()
    c._meta_set("store_gen", c._store_gen() + 1)
    c._conn.commit()
    c._seen_gen = -1

    with pytest.raises(Exception):
        c._ensure_current()

    # Fix data and advance store_gen again
    c._conn.execute("DELETE FROM docs WHERE uid=999")
    c._meta_set("store_gen", c._store_gen() + 1)
    c._conn.commit()
    new_store_gen = c._store_gen()

    # Now _ensure_current should reload successfully
    c._seen_gen = -1  # force reload
    c._ensure_current()
    assert c._seen_gen == new_store_gen
    assert c._index is not None
    res = c.query(vector=[1.0] + [0.0] * (DIM - 1), k=10)
    assert len(res.ids) == 1
    db.close()
