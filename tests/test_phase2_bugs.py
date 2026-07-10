"""Tests proving Phase 2 high-severity bugs exist.

Each test asserts the CORRECT behaviour (what should happen after the fix)
and is marked ``xfail`` because the bug is still present.

Bugs covered:
  #95 — Fallible ops after BEGIN leave SQLite transactions open
  #96 — Post-COMMIT bookkeeping failure returns error for committed mutation
  #91 — Concurrent flushes stamp wrong gen on wrong .tvim file
  #90 — .tvim validation does not bind cached UIDs/vectors to SQLite
"""

import os
import sqlite3
import threading

import numpy as np
import pytest

import turbovecdb
from turbovecdb import TurboVecError

DIM = 8
_UNIT = [1.0, 0, 0, 0, 0, 0, 0, 0]
_V2 = [0, 1.0, 0, 0, 0, 0, 0, 0]


# ═══════════════════════════════════════════════════════════════════════
# #95 — Fallible ops after BEGIN can leave SQLite transactions open
#
# Several mutation paths (write_locked, clear, delete, update_metadata,
# update_documents) use `?` after `BEGIN` without a rollback guard.
# If `store_gen_val()` fails mid-transaction, the `?` returns an error
# immediately and the SQLite transaction stays open. The next write
# then fails with "cannot start a transaction within a transaction".
# ═══════════════════════════════════════════════════════════════════════


def test_clear_leaked_transaction_on_store_gen_failure(tmp_path):
    """clear() has store_gen_val()? after BEGIN without rollback guard.
    Corrupt store_gen, call clear — the leak makes the next add fail."""
    path = str(tmp_path / "db")
    db = turbovecdb.connect(path)
    c = db.collection("c", dim=DIM, create=True)
    c.add(ids=["a", "b"], vectors=[_UNIT, _V2])
    c.flush()

    store = os.path.join(path, "c", "store.sqlite3")
    conn = sqlite3.connect(store)
    conn.execute("UPDATE meta SET value = 'corrupt' WHERE key = 'store_gen'")
    conn.commit()
    conn.close()

    with pytest.raises((ValueError, TurboVecError)):
        c.clear()

    # After a proper fix the next write should work. With the bug the
    # leaked transaction blocks it.
    try:
        c.add(ids=["x"], vectors=[_UNIT])
    except TurboVecError as e:
        if "transaction" in str(e).lower():
            pytest.fail(f"Leaked transaction blocked write: {e}")
    db.close()


def test_delete_leaked_transaction_on_store_gen_failure(tmp_path):
    path = str(tmp_path / "db")
    db = turbovecdb.connect(path)
    c = db.collection("c", dim=DIM, create=True)
    c.add(ids=["a", "b"], vectors=[_UNIT, _V2])
    c.flush()

    store = os.path.join(path, "c", "store.sqlite3")
    conn = sqlite3.connect(store)
    conn.execute("UPDATE meta SET value = 'corrupt' WHERE key = 'store_gen'")
    conn.commit()
    conn.close()

    with pytest.raises((ValueError, TurboVecError)):
        c.delete(ids=["a"])

    try:
        c.add(ids=["x"], vectors=[_UNIT])
    except TurboVecError as e:
        if "transaction" in str(e).lower():
            pytest.fail(f"Leaked transaction blocked write: {e}")
    db.close()


# ═══════════════════════════════════════════════════════════════════════
# #96 — Mutations can return an error after COMMIT has succeeded
#
# Some paths perform fallible bookkeeping after COMMIT succeeds, so
# the API returns an error for a mutation that is already durable.
# Affected: clear() creates a new index after COMMIT; add/upsert and
# delete re-read store_gen after COMMIT.
#
# NOTE: These tests verify the fix works through observable behavior.
# The specific fault-injection scenario (make_index failing after COMMIT)
# cannot be tested from Python because metadata is cached in memory;
# see the Rust test commit_failure_in_clear_rolls_back_and_recovers for
# the deterministic error-path test.
# ═══════════════════════════════════════════════════════════════════════


def test_clear_after_add_does_not_leak(tmp_path):
    """Verify clear() works and collection is reusable afterwards (#96 fix)."""
    path = str(tmp_path / "db")
    db = turbovecdb.connect(path)
    c = db.collection("c", dim=DIM, create=True)
    c.add(ids=["a", "b"], vectors=[_UNIT, _V2])
    c.flush()
    assert c.count() == 2

    c.clear()
    assert c.count() == 0

    # Re-usable after clear
    c.add(ids=["x"], vectors=[_UNIT])
    assert c.count() == 1
    db.close()


def test_add_after_reopen_preserves_data(tmp_path):
    """Verify add commits correctly and data survives reopen (#96 fix)."""
    path = str(tmp_path / "db")
    db = turbovecdb.connect(path)
    c = db.collection("c", dim=DIM, create=True)
    c.add(ids=["a"], vectors=[_UNIT])
    db.close()

    db2 = turbovecdb.connect(path)
    c2 = db2.collection("c", create=False)
    assert c2.count() == 1
    r = c2.get(ids=["a"])
    assert "a" in r.ids
    db2.close()


# ═══════════════════════════════════════════════════════════════════════
# #91 — Concurrent flushes stamp the wrong .tvim as current
#
# flush_file_io() writes to index.tvim.tmp WITHOUT the cross-process
# write lock; flush() then acquires the lock and stamps tvim_gen.
# Two handles flushing different snapshots can interleave so that
# the later metadata stamp describes the other handle's file. This is
# a timing-dependent race that is best tested with Rust-level fault
# injection.
#
# The Rust test `flush_cross_stamp_simulated_health_hides_it` in
# collection.rs proves the race RESULT deterministically by directly
# simulating the cross-stamp end state (stale .tvim content + matched
# tvim_gen == store_gen) and showing that health() reports coherent
# == true when it should report false.
#
# The Python test below runs the concurrent flush pattern as a
# best-effort check; if it XPASSes the race simply didn't trigger
# (the code is not safe — just lucky this run).
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.xfail(strict=False,
                   reason="Bug #91: concurrent flushes can cross-stamp .tvim (timing-dependent)")
def test_concurrent_flush_cross_stamp(tmp_path):
    """Two handles flushing different generations can interleave so
    one handle's tvim_gen stamp describes the other handle's file."""
    path = str(tmp_path / "db")

    db1 = turbovecdb.connect(path)
    h1 = db1.collection("c", dim=DIM, create=True)
    h1.add(ids=["a"], vectors=[_UNIT])
    h1.flush()

    db2 = turbovecdb.connect(path)
    h2 = db2.collection("c", dim=DIM)
    assert h2.count() == 1

    # Both handles add independently (different store_gen increments).
    h1.add(ids=["b"], vectors=[_V2])
    h2.add(ids=["c"], vectors=[_V2])

    results = []

    def flush_h1():
        try:
            h1.flush()
            results.append("h1_ok")
        except Exception as e:
            results.append(f"h1_err:{e}")

    def flush_h2():
        try:
            h2.flush()
            results.append("h2_ok")
        except Exception as e:
            results.append(f"h2_err:{e}")

    t1 = threading.Thread(target=flush_h1)
    t2 = threading.Thread(target=flush_h2)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    db1.close()
    db2.close()

    # Verify all three docs are queryable.
    db3 = turbovecdb.connect(path)
    c3 = db3.collection("c", dim=DIM)
    assert c3.count() == 3, "Bug #91: concurrent flush lost a document"
    hits = c3.query(vector=_UNIT, k=3)
    assert set(hits.ids) == {"a", "b", "c"}, (
        f"Bug #91: wrong cache gave wrong results: {hits.ids}"
    )
    db3.close()


# ═══════════════════════════════════════════════════════════════════════
# #90 — .tvim validation does not bind cached UIDs/vectors to SQLite
#
# A .tvim is trusted when generation, dim, bit_width, and row count
# match. None of these checks verify that the cached UID set or vector
# content corresponds to the SQLite rows. A cache from another
# collection with the same shape/properties but different vector layout
# is accepted as coherent — and because the ANN candidate pool is
# wrong, the exact re-rank (which works on SQLite data) is restricted
# to the wrong candidates and returns wrong results.
#
# Use >50 docs to exceed RERANK_FLOOR so the wrong candidate pool
# can't be masked by a full-table candidate set.
# ═══════════════════════════════════════════════════════════════════════


def test_tvim_from_other_collection_produces_wrong_results(tmp_path):
    """A .tvim from collection A (same dim/bit_width/count but
    different vector layout) is trusted by collection B, and the
    resulting ANN candidate pool is wrong. Because the pool does not
    contain B's true nearest neighbours, the exact re-rank returns
    wrong results."""
    path = str(tmp_path / "db")
    N = 100  # > RERANK_FLOOR (50)

    # Collection A: vectors at dim-0, dim-1, dim-2, ...
    db = turbovecdb.connect(path)
    a = db.collection("A", dim=DIM, create=True)
    for i in range(N):
        v = [0.0] * DIM
        v[i % DIM] = 1.0 - (i // DIM) * 0.01
        a.add(ids=[str(i)], vectors=[v])
    a.flush()
    db.close()

    # Collection B: vectors at dim-2, dim-3, dim-4, ... (shifted by 2)
    db2 = turbovecdb.connect(path)
    b = db2.collection("B", dim=DIM, create=True)
    for i in range(N):
        v = [0.0] * DIM
        v[(i + 2) % DIM] = 1.0 - (i // DIM) * 0.01
        b.add(ids=[str(i)], vectors=[v])
    b.flush()
    db2.close()

    # Replace B's .tvim with A's.
    import shutil
    shutil.copy2(
        os.path.join(path, "A", "index.tvim"),
        os.path.join(path, "B", "index.tvim"),
    )

    # Reopen B with the wrong cache.
    db3 = turbovecdb.connect(path)
    b2 = db3.collection("B", dim=DIM)

    # Query B for its dim-2 vectors (first true nearest is index 0).
    query_v = [0.0] * DIM
    query_v[2] = 1.0
    hits = b2.query(vector=query_v, k=5)
    # Correct B's top-5 for dim-2: indices where (i+2)%8 == 2 = i%8 == 0
    expected = [str(i) for i in range(0, N, 8)][:5]

    # After rejecting the swapped .tvim, the index rebuilds from SQLite.
    # Verify all results are from the correct collection (B's dim-2 vectors).
    correct_ids = {str(i) for i in range(0, N, 8)}
    assert all(rid in correct_ids for rid in hits.ids), (
        f"Bug #90: wrong .tvim produced wrong results.\n"
        f"  Correct IDs: {correct_ids}\n"
        f"  Got:         {hits.ids}"
    )
    db3.close()
