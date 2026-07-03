"""R1: the write lock lives outside the collection directory it guards, so
delete_collection's rmtree can't silently orphan it."""

import os
import shutil

import pytest
from filelock import FileLock, Timeout

import turbovecdb
from turbovecdb.collection import write_lock_path

DIM = 8


def _v(i):
    v = [0.0] * DIM
    v[i % DIM] = 1.0
    return v


def test_write_lock_path_is_a_sibling_not_nested(tmp_path):
    """The lock file must not live inside the directory it guards."""
    path = str(tmp_path / "db")
    db = turbovecdb.connect(path)
    col = db.collection("c", dim=DIM, create=True)
    lock_path = write_lock_path(col.dir)
    assert not lock_path.startswith(col.dir + os.sep)
    assert os.path.dirname(lock_path) == os.path.dirname(col.dir)
    db.close()


def test_write_lock_survives_delete_collection(tmp_path):
    """delete_collection's rmtree removes the collection directory but must
    leave the sibling lock file (and its parent dir) intact, and a fresh
    create must work normally afterward."""
    path = str(tmp_path / "db")
    db = turbovecdb.connect(path)
    col = db.collection("c", dim=DIM, create=True)
    col.add(ids=["a"], vectors=[_v(0)])
    lock_path = write_lock_path(col.dir)

    db.delete_collection("c")
    assert os.path.isdir(os.path.dirname(lock_path))

    col2 = db.collection("c", dim=DIM, create=True)
    col2.add(ids=["b"], vectors=[_v(1)])
    assert col2.count() == 1
    db.close()


def test_lock_survives_rmtree_no_split_brain(tmp_path):
    """Regression for R1: before the fix, write.lock lived inside the
    collection directory. delete_collection's rmtree deleted the directory
    entry out from under a process still holding an flock on it, and a
    concurrent opener's FileLock at the same nominal path created a brand
    new inode and acquired it immediately — two processes then both
    believed they held the collection's write lock, one of them about to
    write into a directory mid-teardown.

    With the lock relocated outside the collection directory, rmtree-ing
    the directory must not touch the lock file at all: a concurrent
    acquire attempt against the same (still-held) lock must keep blocking
    rather than silently succeeding on a fresh inode."""
    path = str(tmp_path / "db")
    db = turbovecdb.connect(path)
    col = db.collection("c", dim=DIM, create=True)
    col.add(ids=["a"], vectors=[_v(0)])
    db.close()

    lock_path = write_lock_path(col.dir)

    holder = FileLock(lock_path)
    holder.acquire()
    try:
        shutil.rmtree(col.dir)

        contender = FileLock(lock_path, timeout=0)
        with pytest.raises(Timeout):
            contender.acquire()
    finally:
        holder.release()
