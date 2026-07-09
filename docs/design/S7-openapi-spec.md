# S7: OpenAPI spec and versioned routes

**Issue:** [#68](https://github.com/kostadis/turbovecdb/issues/68)
**Status:** Implemented (2026-07-09)
**Priority:** Low

## Problem

All routes are at the root (`/upsert`, `/candidate_pairs`, `/count`,
`/clear`, `/health`) with no version prefix. No OpenAPI spec exists.
Breaking changes are invisible and untrackable.

Out of scope: formal version negotiation (Accept headers, etc.) —
a single-user service doesn't need it. A simple path prefix is enough.

## Design

### 1. Version all routes as `/v1/<name>`

The route table becomes:

| Before          | After               |
|-----------------|---------------------|
| `/upsert`       | `/v1/upsert`        |
| `/candidate_pairs` | `/v1/candidate_pairs` |
| `/count`        | `/v1/count`         |
| `/clear`        | `/v1/clear`         |
| `/health`       | `/v1/health`        |

The unversioned paths return 404. No redirects, no compat shims.

### 2. OpenAPI spec at `docs/service-openapi.yaml`

Minimal single-file spec covering the 5 endpoints. Key elements:

- **Info:** title `turbovecdb-service`, version `0.1.0`
- **Server:** `http://localhost:8077`
- **Schemas:**
  - `Error` — `{error: string}`
  - `Health` — `{ok: boolean}`
  - `UpsertRequest` — `{db_path, collection, items: [{id, vector, type?, title?}]}`
  - `UpsertResponse` — `{count: integer}`
  - `CandidatePairsRequest` — `{db_path, collection, threshold, k?}`
  - `CandidatePairsResponse` — `{pairs: [{a, b, distance, a_title, b_title, a_type, b_type}]}`
  - `CountRequest` — `{db_path, collection}`
  - `CountResponse` — `{count: integer}`
  - `ClearRequest` — `{db_path, collection}`
  - `ClearResponse` — `{count: integer}`
- **Endpoints:**
  - `GET /v1/health` → `200 Health`
  - `POST /v1/upsert` → `200 UpsertResponse`, `500 Error`
  - `POST /v1/candidate_pairs` → `200 CandidatePairsResponse`, `500 Error`
  - `POST /v1/count` → `200 CountResponse`, `500 Error`
  - `POST /v1/clear` → `200 ClearResponse`, `500 Error`

### 3. Code changes

In `service.py`, update `ROUTES`:

```python
ROUTES = {
    "/v1/upsert": op_upsert,
    "/v1/candidate_pairs": op_candidate_pairs,
    "/v1/count": op_count,
    "/v1/clear": op_clear,
}
```

Update `do_GET`:

```python
def do_GET(self):
    if self.path == "/v1/health":
        self._send(200, {"ok": True})
    else:
        self._send(404, {"error": "not found"})
```

Update docstring to reflect versioned paths.

### 4. Test plan

- Existing tests drive `op_*` directly (not via HTTP), so they're unaffected
- Add one integration test that starts the server, sends a request to
  `/v1/health`, and checks the response
- Add test hitting an unversioned path (`/upsert`) confirms 404

## Future

When a breaking change is needed (e.g. v2 request format), add
`/v2/<name>` routes alongside v1. The old v1 routes stay until
they're no longer needed.
