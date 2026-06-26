"""Lock timeout: writer crash resilience via configurable file lock timeout."""

import subprocess
import sys
import os

import pytest

import turbovecdb
from turbovecdb import TurboVecError

DIM = 8

# A subprocess that acquires the write lock and holds it indefinitely.
_HOLDER = """
import time, sys, turbovecdb
path = sys.argv[1]
db = turbovecdb.connect(path)
col = db.collection("c", dim={DIM}, create=True)
col._flock.acquire()  # hold the lock
print("LOCKED", flush=True)
time.sleep(60)  # hold until killed
"""


def test_default_lock_timeout_applies(tmp_path):
    """A normal collection should work with the default lock timeout."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    col = db.collection("c", dim=DIM, create=True)
    col.add(ids=["a"], documents=["hello"], vectors=[[1.0] + [0.0] * (DIM - 1)])
    assert col.count() == 1
    db.close()


def test_lock_timeout_custom(tmp_path):
    """lock_timeout is passed through and the lock still works normally."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    col = db.collection("c", dim=DIM, create=True, lock_timeout=10)
    col.add(ids=["a"], documents=["hello"], vectors=[[1.0] + [0.0] * (DIM - 1)])
    assert col.count() == 1
    assert col._flock.timeout == 10
    db.close()


def test_lock_timeout_default_value(tmp_path):
    """Default lock_timeout is 30 seconds."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    col = db.collection("c", dim=DIM, create=True)
    assert col._flock.timeout == 30
    db.close()


def test_lock_timeout_via_database(tmp_path):
    """Can pass lock_timeout through Database.collection()."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    col = db.collection("c", dim=DIM, create=True, lock_timeout=5)
    assert col._flock.timeout == 5
    db.close()


def test_lock_timeout_blocks_then_raises(tmp_path):
    """When another process holds the lock, a write operation times out and raises TurboVecError."""
    path = str(tmp_path / "db")

    # Start a subprocess that holds the write lock
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLDER.format(DIM=DIM), path, "c"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    # Wait for it to acquire the lock
    line = proc.stdout.readline()
    assert "LOCKED" in line

    # Now try a write operation from this process with a short timeout
    db = turbovecdb.connect(path)
    col = db.collection("c", dim=DIM, create=True, lock_timeout=0.5)
    with pytest.raises(TurboVecError, match="could not acquire write lock"):
        col.add(ids=["b"], documents=["world"], vectors=[[0.0, 1.0] + [0.0] * (DIM - 2)])

    db.close()
    proc.kill()
    proc.wait()
