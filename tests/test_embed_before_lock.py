"""R3: add/upsert must embed *before* trying to acquire the cross-process
write lock, not while holding it — a slow (e.g. network-backed) embedder
must not hold other processes' writers hostage for its own duration."""

import os

import pytest
from filelock import FileLock

import turbovecdb
from turbovecdb import EmbedderIdentityMismatchError, TurboVecError
from turbovecdb.collection import write_lock_path

DIM = 8


def _embedder_a(texts):
    return [[1.0] + [0.0] * (DIM - 1) for _ in texts]


def _embedder_b(texts):
    return [[0.0, 1.0] + [0.0] * (DIM - 2) for _ in texts]


def test_embed_happens_before_lock_acquisition(tmp_path):
    """Holds the write lock externally, then calls add(documents=...) with
    a short timeout. The write itself must still time out (the lock really
    is contended) — but the embedder must have run anyway, proving
    embedding isn't gated behind acquiring the lock."""
    path = str(tmp_path / "db")
    db = turbovecdb.connect(path)
    embed_calls = []

    def embedder(texts):
        embed_calls.append(list(texts))
        return [[1.0] + [0.0] * (DIM - 1) for _ in texts]

    col = db.collection("c", dim=DIM, embedder=embedder, create=True, lock_timeout=0.5)

    external_lock = FileLock(write_lock_path(col.dir))
    external_lock.acquire()
    try:
        with pytest.raises(TurboVecError, match="could not acquire write lock"):
            col.add(ids=["a"], documents=["hello"])
    finally:
        external_lock.release()

    assert embed_calls == [["hello"]], "embedder must run even though the write itself is blocked"


def test_upsert_also_embeds_before_lock(tmp_path):
    path = str(tmp_path / "db")
    db = turbovecdb.connect(path)
    embed_calls = []

    def embedder(texts):
        embed_calls.append(list(texts))
        return [[1.0] + [0.0] * (DIM - 1) for _ in texts]

    col = db.collection("c", dim=DIM, embedder=embedder, create=True, lock_timeout=0.5)

    external_lock = FileLock(write_lock_path(col.dir))
    external_lock.acquire()
    try:
        with pytest.raises(TurboVecError, match="could not acquire write lock"):
            col.upsert(ids=["a"], documents=["hello"])
    finally:
        external_lock.release()

    assert embed_calls == [["hello"]]


def test_stale_identity_caught_before_embedding_when_possible(tmp_path):
    """The fast, unlocked pre-check should reject an already-known-bad
    embedder identity without needing to embed at all."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    col = db.collection("c", dim=DIM, embedder=_embedder_a, create=True)
    col.add(ids=["a"], documents=["alpha"])
    db.close()

    embed_calls = []

    def embedder_b_spy(texts):
        embed_calls.append(list(texts))
        return _embedder_b(texts)

    db2 = turbovecdb.connect(str(tmp_path / "db"))
    col2 = db2.collection("c", embedder=embedder_b_spy)
    with pytest.raises(EmbedderIdentityMismatchError):
        col2.add(ids=["b"], documents=["beta"])
    assert embed_calls == [], "the pre-check must reject before ever calling the embedder"
    db2.close()


def test_reembed_race_caught_by_locked_recheck(tmp_path):
    """A concurrent reembed() that lands between the unlocked pre-check and
    the locked write must still be caught by the authoritative re-check
    under the lock."""
    db = turbovecdb.connect(str(tmp_path / "db"))
    col = db.collection("c", dim=DIM, embedder=_embedder_a, create=True)
    col.add(ids=["a"], documents=["alpha"])

    # Race: reembed after the pre-check would have passed, but before the
    # locked write runs. We can't inject a real mid-call race from pure
    # Python without deeper hooks, so simulate the same end state directly:
    # reembed changes the stored identity, then a subsequent add() with the
    # original (now stale) embedder must fail at the re-check.
    col.reembed(_embedder_b, batch_size=2)
    with pytest.raises(EmbedderIdentityMismatchError):
        col.add(ids=["c"], documents=["gamma"])
    db.close()
