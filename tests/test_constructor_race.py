"""C7: first-creation of a brand-new collection must serialize meta
initialization (bit_width/dim/embedder_identity) against concurrent
creators, instead of running unguarded like the rest of the Rust
constructor. Two processes racing to create the same new collection with
different options used to each proceed believing their own value won,
while the meta table ended up with whichever wrote last."""

import os

import pytest
from filelock import FileLock

import turbovecdb
from turbovecdb import TurboVecError
from turbovecdb.collection import write_lock_path

DIM = 8


def test_first_creation_takes_the_write_lock(tmp_path):
    """Directly proves the fix's mechanism: constructing a brand-new
    collection contends for the same cross-process write lock as writes,
    instead of running the Rust constructor's meta read/write unguarded."""
    path = str(tmp_path / "db")
    coll_dir = os.path.join(path, "c")

    # Hold the (not-yet-existing) collection's write lock externally, as a
    # concurrent first-creator would while inside its own locked section.
    os.makedirs(path, exist_ok=True)
    external_lock = FileLock(write_lock_path(coll_dir))
    external_lock.acquire()
    try:
        db = turbovecdb.connect(path)
        with pytest.raises(TurboVecError, match="could not acquire write lock"):
            db.collection("c", dim=DIM, create=True, lock_timeout=0.5)
    finally:
        external_lock.release()


def test_reopening_an_existing_collection_does_not_take_the_lock(tmp_path):
    """The lock-acquire cost is paid only for first-creation — re-opening
    an already-created collection (the hot path, e.g. service.py opening
    one per request) must stay lock-free even while another process holds
    the write lock."""
    path = str(tmp_path / "db")
    db = turbovecdb.connect(path)
    db.collection("c", dim=DIM, create=True)
    db.close()

    external_lock = FileLock(write_lock_path(os.path.join(path, "c")))
    external_lock.acquire()
    try:
        db2 = turbovecdb.connect(path)
        col = db2.collection("c", dim=DIM, create=True, lock_timeout=0.5)
        assert col is not None
    finally:
        external_lock.release()
