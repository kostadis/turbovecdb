"""Multi-process concurrency: reader cache coherence + serialized writers.

Uses real subprocesses (the actual deployment shape — hooks miner + MCP server +
CLI), so the cross-process filelock and WAL visibility are exercised for real,
not simulated with threads."""

import subprocess
import sys

import pytest

import turbovecdb

DIM = 8

# A standalone writer: opens the same collection, adds `n` docs whose one-hot
# vectors sit at positions base..base+n, then closes (flushing the .tvim).
_WRITER = """
import sys, turbovecdb
path, prefix, n, base = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
db = turbovecdb.connect(path)
col = db.collection("c", dim=8, create=True)
def vec(i):
    v = [0.0] * 8
    v[(base + i) % 8] = 1.0
    return v
col.add(
    ids=[prefix + str(i) for i in range(n)],
    documents=["doc " + prefix + str(i) for i in range(n)],
    vectors=[vec(i) for i in range(n)],
)
db.close()
"""


def _onehot(pos):
    v = [0.0] * DIM
    v[pos % DIM] = 1.0
    return v


def _run_writer(path, prefix, n, base):
    return subprocess.run(
        [sys.executable, "-c", _WRITER, path, prefix, str(n), str(base)],
        capture_output=True, text=True,
    )


def test_reader_sees_other_process_writes(tmp_path):
    path = str(tmp_path / "db")
    db = turbovecdb.connect(path)
    col = db.collection("c", dim=DIM, create=True)
    col.add(ids=["p0", "p1"], documents=["dp0", "dp1"], vectors=[_onehot(0), _onehot(1)])
    assert col.count() == 2

    r = _run_writer(path, "c", 3, 4)  # child adds c0..c2 at positions 4,5,6
    assert r.returncode == 0, r.stderr

    # count() reads SQLite directly, so it's immediately current
    assert col.count() == 5
    # the index must reload to surface the child's vectors to a live reader
    res = col.query(vector=_onehot(4), k=1)
    assert res.ids[0].startswith("c")
    assert res.distances[0] == pytest.approx(0.0, abs=1e-5)
    db.close()


def test_concurrent_writers_no_loss_or_collision(tmp_path):
    path = str(tmp_path / "db")
    # pre-create so the dimension is committed before the writers race
    db = turbovecdb.connect(path)
    db.collection("c", dim=DIM, create=True)
    db.close()

    procs = [
        subprocess.Popen([sys.executable, "-c", _WRITER, path, prefix, "20", str(base)],
                         stderr=subprocess.PIPE, text=True)
        for prefix, base in (("a", 0), ("b", 1))
    ]
    for p in procs:
        _, err = p.communicate()
        assert p.returncode == 0, err

    db = turbovecdb.connect(path)
    col = db.collection("c", create=False)
    assert col.count() == 40  # no writes lost
    got = set(col.get(limit=1000).ids)
    assert got == {f"a{i}" for i in range(20)} | {f"b{i}" for i in range(20)}
    db.close()
