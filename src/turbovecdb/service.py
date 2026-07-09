#!/usr/bin/env python3
"""
turbovecdb-service — a thin HTTP layer over the EXISTING (unchanged) turbovecdb,
shaped for llm_wiki's dedup candidate-generation. Stdlib only (http.server + json);
the only third-party import is turbovecdb itself.

This is the "plug the existing turbovecdb in and see if it works" layer — it adds
NO NEW FEATURES to turbovecdb; it only exposes the existing turbovecdb API through a
RESTful HTTP interface.

Every request carries an absolute `db_path` and a `collection` name.

Endpoints (all POST, JSON in/out; GET /v1/health):
  POST /v1/upsert          {db_path, collection, items:[{id, vector:[float], type?, title?}]}  -> {count}
  POST /v1/candidate_pairs {db_path, collection, threshold, k=6}  -> {pairs:[{a,b,distance,a_title,b_title,a_type,b_type}]}
  POST /v1/count           {db_path, collection}  -> {count}
  POST /v1/clear           {db_path, collection}  -> {count:0}

Run:  python3 -m turbovecdb.service [--host 127.0.0.1] [--port 8077]
"""
from __future__ import annotations
import argparse, json, threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import turbovecdb
from .errors import (
    CollectionNotFoundError,
    DimensionMismatchError,
    EmbedderRequiredError,
    EmbedderIdentityMismatchError,
    TurboVecError,
)

# One lock per db_path: turbovecdb is multi-process safe, but we serialize
# same-DB access within this process to avoid concurrent-writer surprises.
#
# Follow-up (lock migration): this is now HTTP-request serialization, not a
# correctness requirement. The Rust core already serializes in-process (its
# Mutex) and cross-process (its flock), so concurrent handlers can no longer
# corrupt state without it. It may still be worth keeping to smooth latency
# under same-DB write bursts (avoids piling handlers on the core Mutex /
# flock poll loop), but it is no longer load-bearing for correctness and
# could be removed if request-level parallelism is preferred.
_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
_locks_guard = threading.Lock()
_databases: dict[str, turbovecdb.Database] = {}

def _lock_for(db_path: str) -> threading.Lock:
    with _locks_guard:
        return _locks[db_path]

def _get_db(db_path: str) -> turbovecdb.Database:
    """Return a cached Database handle for the given path."""
    db = _databases.get(db_path)
    if db is None:
        db = turbovecdb.connect(db_path)
        _databases[db_path] = db
    return db


_ERROR_CODES: dict[type, tuple[int, str]] = {
    CollectionNotFoundError: (404, "COLLECTION_NOT_FOUND"),
    DimensionMismatchError: (400, "DIMENSION_MISMATCH"),
    EmbedderRequiredError: (400, "EMBEDDER_REQUIRED"),
    EmbedderIdentityMismatchError: (400, "EMBEDDER_IDENTITY_MISMATCH"),
}

def _error_payload(e: Exception) -> tuple[int, dict]:
    for exc_type, (status, code) in _ERROR_CODES.items():
        if isinstance(e, exc_type):
            return status, {"error": {"code": code, "message": str(e), "status": status}}
    msg = str(e).lower()
    if "lock timeout" in msg:
        return 429, {"error": {"code": "LOCK_TIMEOUT", "message": str(e), "status": 429, "retryable": True}}
    if "connection closed" in msg:
        return 500, {"error": {"code": "CONNECTION_CLOSED", "message": str(e), "status": 500, "retryable": True}}
    if "already open with different options" in msg:
        return 400, {"error": {"code": "CONFLICT", "message": str(e), "status": 400}}
    return 500, {"error": {"code": "INTERNAL_ERROR", "message": str(e), "status": 500}}


def op_upsert(req: dict) -> dict:
    db_path = req["db_path"]
    collection = req["collection"]
    items = req.get("items", [])
    if not items:
        return {"count": 0}
    # Work around turbovecdb gap G2: it errors on duplicate ids within one
    # batch instead of deduping. Pre-dedupe by id (last wins).
    by_id: dict[str, dict] = {}
    for it in items:
        by_id[it["id"]] = it
    items = list(by_id.values())
    dim = len(items[0]["vector"])
    with _lock_for(db_path):
        db = _get_db(db_path)
        col = db.collection(collection, dim=dim, create=True)
        col.upsert(
            ids=[it["id"] for it in items],
            vectors=[it["vector"] for it in items],
            metadatas=[
                {"pid": it["id"], "type": it.get("type", ""), "title": it.get("title", "")}
                for it in items
            ],
        )
        return {"count": col.count()}


def op_candidate_pairs(req: dict) -> dict:
    db_path = req["db_path"]
    collection = req["collection"]
    tau = float(req.get("threshold", 0.15))
    k = int(req.get("k", 6))
    with _lock_for(db_path):
        db = _get_db(db_path)
        try:
            col = db.collection(collection)
        except CollectionNotFoundError:
            return {"pairs": []}
        if col.count() == 0:
            return {"pairs": []}
        allrows = col.get(include=["metadatas", "vectors"])
        ids = allrows.ids
        vecs = allrows.vectors
        metas = allrows.metadatas
        meta_by_id = {pid: m for pid, m in zip(ids, metas)}
        seen: set[tuple[str, str]] = set()
        pairs = []
        batch = col.query_batch(vectors=vecs, k=k)
        for pid, qr in zip(ids, batch):
            for nid, dist in zip(qr.ids, qr.distances):
                if nid == pid or dist > tau:
                    continue
                key = tuple(sorted((pid, nid)))
                if key in seen:
                    continue
                seen.add(key)
                ma, mb = meta_by_id.get(key[0], {}), meta_by_id.get(key[1], {})
                pairs.append({
                    "a": key[0], "b": key[1], "distance": round(float(dist), 5),
                    "a_title": ma.get("title", ""), "b_title": mb.get("title", ""),
                    "a_type": ma.get("type", ""), "b_type": mb.get("type", ""),
                })
        pairs.sort(key=lambda p: p["distance"])
        return {"pairs": pairs}


def op_count(req: dict) -> dict:
    with _lock_for(req["db_path"]):
        db = _get_db(req["db_path"])
        try:
            col = db.collection(req["collection"])
            return {"count": col.count()}
        except CollectionNotFoundError:
            return {"count": 0}


def op_clear(req: dict) -> dict:
    db_path = req["db_path"]
    collection = req["collection"]
    with _lock_for(db_path):
        db = _get_db(db_path)
        try:
            col = db.collection(collection)
        except CollectionNotFoundError:
            return {"count": 0}
        col.clear()
        return {"count": col.count()}


ROUTES = {
    "/v1/upsert": op_upsert,
    "/v1/candidate_pairs": op_candidate_pairs,
    "/v1/count": op_count,
    "/v1/clear": op_clear,
}


class BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer with a bounded thread pool."""

    def __init__(self, *args, max_workers=10, **kwargs):
        super().__init__(*args, **kwargs)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def process_request(self, request, client_address):
        self._executor.submit(self.process_request_thread, request, client_address)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/health":
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": {"code": "NOT_FOUND", "message": "not found", "status": 404}})

    def do_POST(self):
        fn = ROUTES.get(self.path)
        if fn is None:
            self._send(404, {"error": {"code": "NOT_FOUND", "message": f"unknown route {self.path}", "status": 404}})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            self._send(200, fn(req))
        except json.JSONDecodeError as e:
            self._send(400, {"error": {"code": "INVALID_JSON", "message": str(e), "status": 400}})
        except KeyError as e:
            self._send(400, {"error": {"code": "MISSING_FIELD", "message": f"missing required field: {e}", "status": 400}})
        except Exception as e:
            status, payload = _error_payload(e)
            self._send(status, payload)

    def log_message(self, *a):  # quieter logs
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--max-workers", type=int, default=10)
    args = ap.parse_args()
    srv = BoundedThreadingHTTPServer(
        (args.host, args.port), Handler, max_workers=args.max_workers
    )
    print(f"turbovecdb-service on http://{args.host}:{args.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()