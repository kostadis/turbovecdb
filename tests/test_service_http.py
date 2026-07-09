"""HTTP-level integration tests for turbovecdb-service (/v1/ endpoints).

Spins up the real ThreadingHTTPServer, hits it via urllib.request.
"""

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
import numpy as np

from turbovecdb import service


def _vec(seed, dim=8):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32).tolist()


@pytest.fixture
def server():
    service._databases.clear()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), service.Handler)
    port = srv.socket.getsockname()[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield port
    srv.shutdown()
    service._databases.clear()


def _post(port, path, body):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read())


def _get(port, path):
    resp = urllib.request.urlopen(f"http://127.0.0.1:{port}{path}")
    return json.loads(resp.read())


def test_health(server):
    port = server
    resp = _get(port, "/v1/health")
    assert resp == {"ok": True}


def test_unversioned_404(server):
    port = server
    with pytest.raises(urllib.error.HTTPError) as exc:
        _post(port, "/upsert", {"db_path": "/tmp/x", "collection": "x", "items": []})
    assert exc.value.code == 404


def test_upsert_then_count(server, tmp_path):
    port = server
    db = str(tmp_path / "db")
    items = [
        {"id": "a", "vector": _vec(1), "type": "page", "title": "Alpha"},
        {"id": "b", "vector": _vec(2), "type": "page", "title": "Beta"},
        {"id": "c", "vector": _vec(3), "type": "note", "title": "Gamma"},
    ]
    r = _post(port, "/v1/upsert", {"db_path": db, "collection": "pages", "items": items})
    assert r["count"] == 3

    r = _post(port, "/v1/count", {"db_path": db, "collection": "pages"})
    assert r["count"] == 3


def test_upsert_then_clear(server, tmp_path):
    port = server
    db = str(tmp_path / "db")
    items = [
        {"id": "a", "vector": _vec(1), "type": "page", "title": "Alpha"},
        {"id": "b", "vector": _vec(2), "type": "page", "title": "Beta"},
    ]
    r = _post(port, "/v1/upsert", {"db_path": db, "collection": "pages", "items": items})
    assert r["count"] == 2

    r = _post(port, "/v1/clear", {"db_path": db, "collection": "pages"})
    assert r["count"] == 0

    r = _post(port, "/v1/count", {"db_path": db, "collection": "pages"})
    assert r["count"] == 0


def test_candidate_pairs(server, tmp_path):
    port = server
    db = str(tmp_path / "db")
    items = [
        {"id": "a", "vector": _vec(1), "type": "page", "title": "Alpha"},
        {"id": "b", "vector": _vec(2), "type": "page", "title": "Beta"},
        {"id": "c", "vector": _vec(3), "type": "note", "title": "Gamma"},
        {"id": "d", "vector": _vec(1), "type": "page", "title": "Alpha2"},
        {"id": "e", "vector": _vec(2), "type": "page", "title": "Beta2"},
    ]
    _post(port, "/v1/upsert", {"db_path": db, "collection": "pages", "items": items})

    r = _post(port, "/v1/candidate_pairs", {
        "db_path": db, "collection": "pages", "threshold": 1.0, "k": 6
    })
    assert "pairs" in r
    for p in r["pairs"]:
        assert {"a", "b", "distance", "a_title", "b_title", "a_type", "b_type"} <= p.keys()
    for i in range(len(r["pairs"]) - 1):
        assert r["pairs"][i]["distance"] <= r["pairs"][i + 1]["distance"]
    for p in r["pairs"]:
        assert p["a"] != p["b"]


def test_invalid_json_400(server):
    port = server
    data = b"this is not json"
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/count",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req)
    assert exc.value.code == 400


def test_missing_collection_count_zero(server, tmp_path):
    port = server
    db = str(tmp_path / "db")
    r = _post(port, "/v1/count", {"db_path": db, "collection": "nonexistent"})
    assert r == {"count": 0}


# ── S6 typed error codes ──────────────────────────────────────────────────


def _post_expect_error(port, path, body):
    """POST and return (status_code, parsed_body) even on non-2xx."""
    if isinstance(body, (bytes, bytearray)):
        data = body
    else:
        data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req)
        return (resp.getcode(), json.loads(resp.read()))
    except urllib.error.HTTPError as exc:
        return (exc.code, json.loads(exc.read()))


def test_error_collection_not_found(server, tmp_path):
    port = server
    db = str(tmp_path / "db")
    # Missing collection returns empty result, not an error
    code, body = _post_expect_error(
        port, "/v1/count", {"db_path": db, "collection": "nonexistent"}
    )
    assert code == 200
    assert body == {"count": 0}


def test_error_missing_field(server):
    port = server
    code, body = _post_expect_error(port, "/v1/count", {})
    assert code == 400
    err = body.get("error", {})
    assert err.get("code") == "MISSING_FIELD"


def test_error_invalid_json(server):
    port = server
    code, body = _post_expect_error(port, "/v1/count", b"not json")
    assert code == 400
    err = body.get("error", {})
    assert err.get("code") == "INVALID_JSON"


def test_error_not_found_route(server):
    port = server
    code, body = _post_expect_error(port, "/v1/nonexistent", {})
    assert code == 404
    err = body.get("error", {})
    assert err.get("code") == "NOT_FOUND"


def test_error_dimension_mismatch(server, tmp_path):
    port = server
    db = str(tmp_path / "db")
    items = [{"id": "a", "vector": _vec(1, dim=8), "type": "page", "title": "A"}]
    r = _post(port, "/v1/upsert", {"db_path": db, "collection": "test", "items": items})
    assert r["count"] == 1

    items2 = [{"id": "b", "vector": _vec(2, dim=16), "type": "page", "title": "B"}]
    code, body = _post_expect_error(
        port, "/v1/upsert", {"db_path": db, "collection": "test", "items": items2}
    )
    assert code == 400, f"expected 400, got {code}: {body}"
    err = body.get("error", {})
    assert err.get("code") == "CONFLICT"


def test_success_no_error_field(server, tmp_path):
    port = server
    db = str(tmp_path / "db")
    r = _post(port, "/v1/count", {"db_path": db, "collection": "pages"})
    assert "error" not in r
