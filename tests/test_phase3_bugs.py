"""Tests proving Phase 3 high-severity bugs exist.

Each test asserts the CORRECT behaviour (what should happen after the fix)
and is marked ``xfail`` because the bug is still present.

Bugs covered:
  #102 — reembed holds collection locks while running user code (embedder + callback)
  #101 — Opening populated legacy collection silently adopts caller's embedder identity
  #100 — Embedder identity collides across differently configured models
  #99 — Live handles do not refresh collection configuration after reembed
"""

import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import numpy as np
import pytest

import turbovecdb
from turbovecdb import TurboVecError

DIM = 8


# ═══════════════════════════════════════════════════════════════════════
# #102 — reembed holds collection locks while running user code
#
# reembed() acquires the write flock at the start and holds it across
# every embedder invocation and progress callback. A slow or network-
# backed embedder blocks every other writer across all processes.
# An embedder or callback that re-enters the same collection deadlocks
# on the non-reentrant in-process Mutex.
# ═══════════════════════════════════════════════════════════════════════


class _SlowEmbedder:
    """Embedder that takes ~0.3s per batch — simulates a network call."""
    def __init__(self, delay=0.3):
        self.delay = delay
        self.call_count = 0

    def __call__(self, texts):
        time.sleep(self.delay)
        self.call_count += 1
        return np.array([[float(len(t)) for _ in range(DIM)] for t in texts])


def test_reembed_blocks_concurrent_writer(tmp_path):
    """reembed holds the write flock across the embedder call. Another
    handle trying to write in a separate Database is blocked."""
    path = str(tmp_path / "db")

    db = turbovecdb.connect(path)
    c = db.collection("c", dim=DIM, create=True)
    for i in range(4):
        c.add(ids=[str(i)], documents=[f"doc {i}"], vectors=[[1.0] + [0.0] * (DIM - 1)])
    db.close()

    slow_emb = _SlowEmbedder(delay=0.5)

    def do_reembed():
        db1 = turbovecdb.connect(path)
        c1 = db1.collection("c", dim=DIM)
        c1.reembed(slow_emb, dim=DIM, batch_size=2)
        db1.close()

    def do_add():
        db2 = turbovecdb.connect(path)
        c2 = db2.collection("c", dim=DIM)
        c2.add(ids=["x"], vectors=[[1.0] + [0.0] * (DIM - 1)])
        db2.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        reembed_fut = pool.submit(do_reembed)
        time.sleep(0.1)  # let reembed acquire lock + start embed

        add_before = time.time()
        add_fut = pool.submit(do_add)

        try:
            add_fut.result(timeout=1.0)
        except TimeoutError:
            pytest.fail("Bug #102: add blocked by reembed's write lock")

        add_time = time.time() - add_before
        assert add_time < 0.5, (
            f"Bug #102: add blocked for {add_time:.2f}s (reembed held lock during embed)"
        )
        reembed_fut.result(timeout=10)


def test_reembed_callback_deadlocks(tmp_path):
    """An embedder that calls count() on the same collection deadlocks
    because reembed holds the in-process Mutex. The fix must release
    the Mutex during the embedder call."""
    path = str(tmp_path / "db")
    db = turbovecdb.connect(path)
    c = db.collection("c", dim=DIM, create=True)
    for i in range(3):
        c.add(ids=[str(i)], documents=[f"doc {i}"], vectors=[[1.0] + [0.0] * (DIM - 1)])

    def reentrant_embedder(texts):
        c.count()
        return np.array([[float(len(t)) for _ in range(DIM)] for t in texts])

    t = threading.Thread(
        target=lambda: c.reembed(reentrant_embedder, dim=DIM, batch_size=2),
        daemon=True,
    )
    t.start()
    t.join(timeout=3)

    if t.is_alive():
        pytest.fail("Bug #102: reembed deadlocked on re-entrant embedder (Mutex held)")
    db.close()


# ═══════════════════════════════════════════════════════════════════════
# #101 — Legacy collection silently adopts caller's embedder identity
#
# When embedder_identity is absent, the constructor stamps the identity
# of whichever embedder opens it — even if the collection already has
# vectors from an unknown source. This blesses the new embedder as
# authoritative and silently mixes mismatched vector origins.
# ═══════════════════════════════════════════════════════════════════════


def test_populated_collection_adopts_caller_embedder(tmp_path):
    """A populated collection with no embedder_identity must NOT
    automatically stamp the opener's embedder as authoritative."""
    path = str(tmp_path / "db")

    # Create collection with raw vectors (no embedder).
    db = turbovecdb.connect(path)
    c = db.collection("c", dim=DIM, create=True)
    c.add(ids=["a", "b"], vectors=[[1.0] + [0.0] * (DIM - 1), [0.0, 1.0] + [0.0] * (DIM - 2)])
    db.close()

    # Wipe the embedder_identity to simulate a legacy/restored collection.
    store = os.path.join(path, "c", "store.sqlite3")
    conn = sqlite3.connect(store)
    conn.execute("DELETE FROM meta WHERE key = 'embedder_identity'")
    conn.commit()
    conn.close()

    # Reopen with a NEW embedder.
    def my_embedder(texts):
        return np.array([[float(len(t)) for _ in range(DIM)] for t in texts])

    db2 = turbovecdb.connect(path)
    c2 = db2.collection("c", dim=DIM, embedder=my_embedder)

    # Bug: embedder_identity is now 'my_embedder' — silently adopted.
    identity = c2._meta_get("embedder_identity")
    assert identity != "my_embedder", (
        f"Bug #101: legacy collection adopted identity {identity!r} "
        f"without explicit migration"
    )
    db2.close()


# ═══════════════════════════════════════════════════════════════════════
# #100 — Embedder identity collides across differently configured models
#
# PyEmbedder::identity() uses only the callable's __name__ or
# module.class — model configuration (model name, endpoint, revision)
# is not included. Two instances of the same class with different
# configs therefore have identical identities.
# ═══════════════════════════════════════════════════════════════════════


class _ModelEmbedder:
    """Simulates two differently-configured models that produce the
    same class name but different vectors."""
    def __init__(self, model_name: str):
        self.model_name = model_name

    @property
    def _embedder_identity(self):
        return f"_ModelEmbedder(model={self.model_name})"

    def __call__(self, texts):
        # Different models produce different vectors.
        offset = hash(self.model_name) % 8
        return np.array([[float(len(t) + offset) for _ in range(DIM)] for t in texts])


def test_embedder_identity_collision(tmp_path):
    """Two instances of the same class with different model configs
    produce the same identity string. The identity guard therefore
    cannot distinguish them."""
    path = str(tmp_path / "db")

    emb_a = _ModelEmbedder("gpt-4")
    emb_b = _ModelEmbedder("gpt-4-turbo")

    # Both should have different identities.
    from turbovecdb.collection import embedder_identity
    id_a = embedder_identity(emb_a)
    id_b = embedder_identity(emb_b)

    assert id_a != id_b, (
        f"Bug #100: identity collision: {id_a!r} == {id_b!r}\n"
        f"  emb_a model=gpt-4, emb_b model=gpt-4-turbo\n"
        f"  Identity should include model config, not just class name"
    )


def test_identity_collision_allows_wrong_model_write(tmp_path):
    """Because identity only checks class name, a differently-configured
    model can write into a collection created by another instance of the
    same class — silently corrupting the index."""
    path = str(tmp_path / "db")

    emb_original = _ModelEmbedder("model-v1")
    emb_wrong = _ModelEmbedder("model-v2")  # same class, different model

    # Create collection with model-v1.
    db = turbovecdb.connect(path)
    c = db.collection("c", dim=DIM, embedder=emb_original, create=True)
    c.add(ids=["a"], documents=["hello"], vectors=emb_original(["hello"]))
    db.close()

    # Reopen with model-v2 — should be rejected by identity guard.
    db2 = turbovecdb.connect(path)
    c2 = db2.collection("c", dim=DIM, embedder=emb_wrong)

    # This add should raise EmbedderIdentityMismatchError, but with the
    # bug it succeeds because both embedders have the same class name.
    with pytest.raises(TurboVecError, match="identity"):
        # No vectors= — this goes through the text path which checks
        # identity. emb_wrong's identity doesn't match stored model-v1.
        c2.add(ids=["b"], documents=["world"])
    db2.close()


# ═══════════════════════════════════════════════════════════════════════
# #99 — Live handles do not refresh collection configuration after
#       reembed
#
# ensure_current() only checks store_gen — it does not refresh dim,
# bit_width, or embedder_identity before rebuilding the index. A live
# handle continues to use stale config and its original embedder
# object after another handle reembeds with different settings.
# ═══════════════════════════════════════════════════════════════════════


def test_live_handle_stale_dim_after_reembed(tmp_path):
    """A handle that was opened before reembed uses its original dim
    even after another handle changes it."""
    path = str(tmp_path / "db")

    # Handle A: create collection with dim=8.
    db_a = turbovecdb.connect(path)
    a = db_a.collection("c", dim=DIM, create=True)
    a.add(ids=["a"], documents=["hello"], vectors=[[1.0] + [0.0] * (DIM - 1)])

    # Handle B: reembed to dim=16.
    db_b = turbovecdb.connect(path)
    b = db_b.collection("c", dim=DIM)

    def embed16(texts):
        return np.array([[float(len(t)) for _ in range(16)] for t in texts])

    b.reembed(embed16, dim=16, batch_size=2)
    db_b.close()

    # Handle A's dim should now be 16 (it was changed by B).
    assert a.dim == 16, (
        f"Bug #99: live handle has stale dim={a.dim} after reembed to dim=16"
    )
    db_a.close()


def test_live_handle_stale_bit_width_after_reembed(tmp_path):
    """Same as above but for bit_width."""
    path = str(tmp_path / "db")

    db_a = turbovecdb.connect(path)
    a = db_a.collection("c", dim=DIM, bit_width=4, create=True)
    a.add(ids=["a"], documents=["hello"], vectors=[[1.0] + [0.0] * (DIM - 1)])

    db_b = turbovecdb.connect(path)
    b = db_b.collection("c", dim=DIM)

    def embed(texts):
        return np.array([[float(len(t)) for _ in range(DIM)] for t in texts])

    b.reembed(embed, bit_width=2, batch_size=2)
    db_b.close()

    assert a._bit_width == 2, (
        f"Bug #99: live handle has stale bit_width={a._bit_width} after reembed to 2"
    )
    db_a.close()


def test_live_handle_stale_embedder_after_reembed(tmp_path):
    """After reembed by another handle, a live handle's embedder object is
    the original one. Text operations correctly fail the identity check;
    raw vector operations still work (they bypass the embedder)."""
    path = str(tmp_path / "db")

    def embedder_v1(texts):
        return np.array([[1.0] * DIM for _ in texts])

    def embedder_v2(texts):
        return np.array([[2.0] * DIM for _ in texts])

    # Handle A: create with embedder_v1.
    db_a = turbovecdb.connect(path)
    a = db_a.collection("c", dim=DIM, embedder=embedder_v1, create=True)
    a.add(ids=["a"], documents=["hello"])  # identity = embedder_v1

    # Handle B: reembed with embedder_v2.
    db_b = turbovecdb.connect(path)
    b = db_b.collection("c", dim=DIM, embedder=embedder_v2)
    b.reembed(embedder_v2, dim=DIM, batch_size=2)
    db_b.close()

    # The stored identity is now embedder_v2 (set by B's reembed).
    stored = a._meta_get("embedder_identity")
    assert stored == "embedder_v2", (
        f"Bug #99: reembed did not update embedder_identity in meta"
    )

    # Handle A still has embedder_v1. Text operations correctly fail
    # because check_embedder_identity reads fresh from meta.
    with pytest.raises(TurboVecError, match="identity"):
        a.add(ids=["b"], documents=["world"])

    # Raw vector operations bypass the embedder entirely — they still work.
    a.add(ids=["b"], vectors=[[1.0] + [0.0] * (DIM - 1)])

    db_a.close()
