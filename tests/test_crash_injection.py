"""R7: crash-injection harness. Kills a writer mid-write (SIGKILL, no
chance to run any cleanup/rollback code) and verifies the database is
still safely recoverable afterward — the core "a crash never loses data"
promise this project makes. Complements the narrower, deterministic
regression tests elsewhere (C1's test_close_takes_the_write_lock,
R2's reload_index_rebuilds_when_loaded_dim_mismatches_meta /
test_corrupt_tvim_rebuilds_from_store) with an end-to-end check that
doesn't assume anything about *how* the crash is survived."""

import subprocess
import sys
import time

import turbovecdb

DIM = 8

_WRITER = """
import sys, turbovecdb
DIM = {DIM}
path = sys.argv[1]
db = turbovecdb.connect(path)
col = db.collection("c", dim=DIM, create=True)
N = {N}
ids = [str(i) for i in range(N)]
vectors = [[1.0] + [0.0] * (DIM - 1) for _ in range(N)]
print("STARTING", flush=True)
col.add(ids=ids, vectors=vectors)
print("DONE", flush=True)
"""


def test_sigkill_mid_write_leaves_db_recoverable(tmp_path):
    path = str(tmp_path / "db")

    # Seed a baseline the killed process's write won't touch (disjoint id
    # range), so recovery can distinguish "the big write committed" from
    # "it didn't" by count() alone.
    db0 = turbovecdb.connect(path)
    col0 = db0.collection("c", dim=DIM, create=True)
    col0.add(
        ids=["baseline-a", "baseline-b"],
        vectors=[[1.0] + [0.0] * (DIM - 1), [0.0, 1.0] + [0.0] * (DIM - 2)],
    )
    db0.close()

    n = 300000
    proc = subprocess.Popen(
        [sys.executable, "-c", _WRITER.format(DIM=DIM, N=n), path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    line = proc.stdout.readline()
    assert "STARTING" in line

    time.sleep(0.2)  # well before the ~1-1.5s this add() takes, to land mid-write
    proc.kill()  # SIGKILL — no signal handler, no chance to roll back explicitly
    proc.wait(timeout=5)

    # Reopening must not raise, and must report a consistent, valid state.
    db = turbovecdb.connect(path)
    col = db.collection("c", dim=DIM)
    health = col.health()
    assert health.ok, "SQLite integrity check must pass after a mid-write SIGKILL"

    count = col.count()
    assert count in (2, 2 + n), (
        f"count() after a crashed write must be the pre-write baseline (2) "
        f"or the fully-committed batch (2 + {n}), got {count} — a torn commit"
    )

    # get()/query() must work normally — no corruption.
    res = col.get(ids=["baseline-a", "baseline-b"])
    assert set(res.ids) == {"baseline-a", "baseline-b"}

    # The write lock must not be stuck: SIGKILL releases OS-level advisory
    # locks on process exit, so a fresh write must succeed promptly.
    col.add(ids=["after-crash"], vectors=[[0.0, 0.0, 1.0] + [0.0] * (DIM - 3)])
    assert col.count() == count + 1
    db.close()


def test_sigkill_during_flush_leaves_tvim_recoverable(tmp_path):
    """A crash mid-flush() must leave a .tvim that's either the old one, the
    new one, or absent — never a torn file the loader can't detect."""
    path = str(tmp_path / "db")

    db0 = turbovecdb.connect(path)
    col0 = db0.collection("c", dim=DIM, create=True)
    col0.add(ids=["baseline-a"], vectors=[[1.0] + [0.0] * (DIM - 1)])
    db0.close()

    writer = """
import sys, turbovecdb
path = sys.argv[1]
DIM = {DIM}
db = turbovecdb.connect(path)
col = db.collection("c", dim=DIM, create=True)
N = {N}
ids = [str(i) for i in range(N)]
vectors = [[1.0] + [0.0] * (DIM - 1) for _ in range(N)]
col.add(ids=ids, vectors=vectors)
print("STARTING_FLUSH", flush=True)
col.flush()
print("DONE", flush=True)
""".format(DIM=DIM, N=200000)

    proc = subprocess.Popen(
        [sys.executable, "-c", writer, path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    line = proc.stdout.readline()
    assert "STARTING_FLUSH" in line

    time.sleep(0.05)
    proc.kill()
    proc.wait(timeout=5)

    db = turbovecdb.connect(path)
    col = db.collection("c", dim=DIM)
    health = col.health()
    assert health.ok, "SQLite integrity check must pass after a mid-flush SIGKILL"
    res = col.get(ids=["baseline-a"])
    assert res.ids == ["baseline-a"]

    # Usable afterward: a query must succeed (rebuilding from SQLite if the
    # .tvim was torn/stale) rather than crashing.
    col.query(vector=[1.0] + [0.0] * (DIM - 1), k=1)
    db.close()
