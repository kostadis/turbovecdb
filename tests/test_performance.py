"""Performance & scale tests, at two layers:

* the Rust core directly (``turbovecdb._core.Collection``), and
* the full public API (Python wrapper + Rust: ``turbovecdb.connect``).

These assert on *scaling* (doubling N stays roughly linear, i.e. not quadratic)
and on correctness at scale, rather than on absolute wall-clock, so they stay
robust across machines. Marked ``perf`` so they can be selected or skipped:
``pytest -m perf`` / ``pytest -m 'not perf'``.
"""

import os
import tempfile
import time

import numpy as np
import pytest

import turbovecdb
from turbovecdb import _core

pytestmark = pytest.mark.perf

DIM = 16


def _rand(n, seed):
    return np.random.default_rng(seed).standard_normal((n, DIM)).astype(np.float32)


# ── add-one-at-a-time must scale ~linearly (guards the O(N^2) regression) ─────

def _core_add_loop_secs(n):
    d = tempfile.mkdtemp()
    c = _core.Collection(os.path.join(d, "c"), DIM)
    vecs = _rand(n, seed=n)
    t = time.perf_counter()
    for i in range(n):
        c.add([str(i)], None, None, [vecs[i]])
    return time.perf_counter() - t


def _api_add_loop_secs(n):
    d = tempfile.mkdtemp()
    c = turbovecdb.connect(os.path.join(d, "db")).collection("c", dim=DIM, create=True)
    vecs = _rand(n, seed=n)
    t = time.perf_counter()
    for i in range(n):
        c.add(ids=[str(i)], vectors=[vecs[i]])
    return time.perf_counter() - t


def test_core_add_loop_scales_linearly():
    """Rust core: N single adds is O(N), not O(N^2)."""
    _core_add_loop_secs(40)  # warm up allocation/JIT paths
    t1 = _core_add_loop_secs(150)
    t2 = _core_add_loop_secs(300)
    # Linear -> ~2x for 2x docs; quadratic -> ~4x. Assert comfortably sub-quadratic.
    assert t2 / t1 < 3.0, f"core add loop scaled {t2 / t1:.1f}x for 2x docs (quadratic?)"


def test_api_add_loop_scales_linearly():
    """Public API (wrapper + Rust): same linear guarantee end to end."""
    _api_add_loop_secs(40)
    t1 = _api_add_loop_secs(150)
    t2 = _api_add_loop_secs(300)
    assert t2 / t1 < 3.0, f"api add loop scaled {t2 / t1:.1f}x for 2x docs (quadratic?)"


# ── batched add at scale (single transaction) + query correctness ────────────

def test_core_batched_add_and_query_at_scale():
    d = tempfile.mkdtemp()
    c = _core.Collection(os.path.join(d, "c"), DIM)
    n = 3000
    vecs = _rand(n, seed=1)
    c.add([f"d{i}" for i in range(n)], None, None, vecs)
    assert c.count() == n
    # An exact stored vector must find itself (near-zero cosine distance).
    r = c.query(None, vecs[1234], 3, None, None, None)
    assert "d1234" in r.ids
    assert min(r.distances) < 1e-3


def test_api_batched_add_and_query_at_scale():
    n = 3000
    c = turbovecdb.connect(tempfile.mkdtemp()).collection("c", dim=DIM, create=True)
    vecs = _rand(n, seed=2)
    c.add(ids=[f"d{i}" for i in range(n)], vectors=vecs)
    assert c.count() == n
    r = c.query(vector=vecs[2500], k=3)
    assert "d2500" in r.ids
    assert min(r.distances) < 1e-3


# ── query throughput / recall at scale ───────────────────────────────────────

def test_api_many_queries_recall_at_scale():
    n = 2000
    c = turbovecdb.connect(tempfile.mkdtemp()).collection("c", dim=DIM, create=True)
    vecs = _rand(n, seed=3)
    c.add(ids=[f"d{i}" for i in range(n)], vectors=vecs)
    probes = list(range(0, n, n // 100))  # ~100 probes
    hits = sum(1 for i in probes if c.query(vector=vecs[i], k=1).ids[0] == f"d{i}")
    # Self-query recall should be near-perfect; allow a few ANN quantization misses.
    assert hits >= len(probes) - 3


# ── reembed at scale stays correct ───────────────────────────────────────────

def test_api_reembed_at_scale():
    n = 1500
    c = turbovecdb.connect(tempfile.mkdtemp()).collection("c", dim=DIM, create=True)
    docs = [f"document number {i}" for i in range(n)]

    def embed(texts):
        return np.array([[float(len(t)) + j for j in range(DIM)] for t in texts], np.float32)

    c.add(ids=[f"d{i}" for i in range(n)], documents=docs, vectors=embed(docs))
    report = c.reembed(lambda t: embed(t) * 2.0, batch_size=256)
    assert report.total_docs == n
    assert report.new_dim == DIM
    assert c.count() == n
    # Still queryable after a full-collection re-embed.
    assert len(c.query(vector=embed(["document number 7"])[0], k=5).ids) == 5
