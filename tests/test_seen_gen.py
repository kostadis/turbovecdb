"""Cache coherence (black-box): a corrupt index.tvim rebuilds from the SQLite
store, and a second handle sees another handle's committed writes.

(Previously a white-box suite poking c._conn / c._seen_gen to force reload
failures; those internals moved into the Rust core. The rebuild-failure
resilience they exercised is now handled inside _core; here we keep the
observable guarantees.)"""

import os

import pytest
from filelock import FileLock

import turbovecdb
from turbovecdb import TurboVecError
from turbovecdb.collection import write_lock_path

DIM = 8


def _v(i):
    v = [0.0] * DIM
    v[i % DIM] = 1.0
    return v


def test_corrupt_tvim_rebuilds_from_store(tmp_path):
    """A corrupt index.tvim is ignored on reopen; the index rebuilds from SQLite."""
    path = str(tmp_path / "db")
    db = turbovecdb.connect(path)
    c = db.collection("c", dim=DIM, create=True)
    c.add(ids=["a"], vectors=[_v(0)])
    c.flush()  # writes index.tvim (tvim_gen == store_gen)

    # Corrupt the cache, then write more so store_gen advances past tvim_gen.
    with open(os.path.join(path, "c", "index.tvim"), "wb") as f:
        f.write(b"garbage")
    c.add(ids=["b"], vectors=[_v(1)])
    db.close()

    db2 = turbovecdb.connect(path)
    c2 = db2.collection("c", dim=DIM)
    assert c2.count() == 2
    assert set(c2.query(vector=_v(0), k=10).ids) == {"a", "b"}
    db2.close()


def test_second_handle_sees_committed_writes(tmp_path):
    """A separate handle to the same collection sees committed writes."""
    path = str(tmp_path / "db")
    db = turbovecdb.connect(path)
    writer = db.collection("c", dim=DIM, create=True)
    writer.add(ids=["a"], vectors=[_v(0)])

    db2 = turbovecdb.connect(path)  # independent handle to the same dir
    reader = db2.collection("c", dim=DIM)
    assert reader.count() == 1

    writer.add(ids=["b"], vectors=[_v(1)])  # advances store_gen
    assert reader.count() == 2
    assert set(reader.query(vector=_v(1), k=10).ids) == {"a", "b"}

    db.close()
    db2.close()


def test_close_takes_the_write_lock(tmp_path):
    """close() must contend for the same cross-process write lock as flush()/
    add() (regression for C1: close() used to flush index.tvim under only the
    in-process lock, letting a concurrent writer bump store_gen between the
    rename and the tvim_gen stamp — a stale .tvim silently marked coherent)."""
    path = str(tmp_path / "db")
    db1 = turbovecdb.connect(path)
    col1 = db1.collection("c", dim=DIM, create=True, lock_timeout=0.5)
    col1.add(ids=["a"], vectors=[_v(0)])

    # A second, independent holder takes the write lock externally, via a
    # Python filelock on the same sibling <name>.lock path the Rust core uses
    # (I3). col1.close() must contend for that same lock and time out.
    db2 = turbovecdb.connect(path)
    col2 = db2.collection("c", dim=DIM, lock_timeout=0.5)
    holder = FileLock(write_lock_path(col2.dir))
    holder.acquire()
    try:
        with pytest.raises(TurboVecError, match="could not acquire write lock"):
            col1.close()
    finally:
        holder.release()
