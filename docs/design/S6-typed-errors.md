# S6: Flat error string with no typed error codes

**Issue:** [#69](https://github.com/kostadis/turbovecdb/issues/69)
**Status:** Implemented (2026-07-09)
**Priority:** Low

## Problem

Every error response is:

```json
{"error": "ExceptionType: message"}
```

Callers must string-parse to distinguish "collection not found" from
"lock timeout" from "invalid argument".

## Design: Structured error responses

### Error schema

```json
{
  "error": {
    "code": "COLLECTION_NOT_FOUND",
    "message": "collection 'foo' not found",
    "status": 404
  }
}
```

### Error code table

| Python exception | HTTP status | Error code | Retryable |
|---|---|---|---|
| `CollectionNotFoundError` | 404 | `COLLECTION_NOT_FOUND` | No |
| `DimensionMismatchError` | 400 | `DIMENSION_MISMATCH` | No |
| `EmbedderRequiredError` | 400 | `EMBEDDER_REQUIRED` | No |
| `EmbedderIdentityMismatchError` | 400 | `EMBEDDER_IDENTITY_MISMATCH` | No |
| `json.JSONDecodeError` | 400 | `INVALID_JSON` | No |
| `KeyError` | 400 | `MISSING_FIELD` | No |
| `TurboVecError` (message contains "lock timeout") | 429 | `LOCK_TIMEOUT` | Yes |
| `TurboVecError` (message contains "connection closed") | 500 | `CONNECTION_CLOSED` | Yes |
| Any other `TurboVecError` | 500 | `INTERNAL_ERROR` | No |
| Any other `Exception` | 500 | `INTERNAL_ERROR` | No |

### Implementation

Add a helper that maps exceptions to `(status, code, message)`:

```python
_ERROR_CODES: dict[type, tuple[int, str, bool]] = {
    CollectionNotFoundError: (404, "COLLECTION_NOT_FOUND", False),
    DimensionMismatchError: (400, "DIMENSION_MISMATCH", False),
    EmbedderRequiredError: (400, "EMBEDDER_REQUIRED", False),
    EmbedderIdentityMismatchError: (400, "EMBEDDER_IDENTITY_MISMATCH", False),
    json.JSONDecodeError: (400, "INVALID_JSON", False),
    KeyError: (400, "MISSING_FIELD", False),
}

def _error_payload(e: Exception) -> tuple[int, dict]:
    for exc_type, (status, code, retryable) in _ERROR_CODES.items():
        if isinstance(e, exc_type):
            payload = {"code": code, "message": str(e), "status": status}
            if retryable:
                payload["retryable"] = True
            return status, {"error": payload}
    # Fallback: check message for knowable patterns
    msg = str(e).lower()
    if "lock timeout" in msg:
        return 429, {"error": {"code": "LOCK_TIMEOUT", "message": str(e), "status": 429, "retryable": True}}
    if "connection closed" in msg:
        return 500, {"error": {"code": "CONNECTION_CLOSED", "message": str(e), "status": 500, "retryable": True}}
    return 500, {"error": {"code": "INTERNAL_ERROR", "message": str(e), "status": 500}}
```

Update `do_POST`:

```python
def do_POST(self):
    fn = ROUTES.get(self.path)
    if fn is None:
        self._send(404, {"error": {"code": "NOT_FOUND", "message": f"unknown route {self.path}", "status": 404}})
        return
    try:
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        self._send(200, fn(req))
    except Exception as e:
        status, payload = _error_payload(e)
        self._send(status, payload)
```

Update `do_GET` 404:

```python
self._send(404, {"error": {"code": "NOT_FOUND", "message": "not found", "status": 404}})
```

Remove the 404 route check from `do_GET` — the `NOT_FOUND` code is already handled above.

### OpenAPI update

Add error schema to `docs/service-openapi.yaml`:

```yaml
Error:
  type: object
  properties:
    error:
      type: object
      properties:
        code:
          type: string
        message:
          type: string
        status:
          type: integer
        retryable:
          type: boolean
      required: [code, message, status]
```

Update each endpoint's `"500"` response to use this schema.

### Test plan

- Add tests for each error code path:
  - `POST /v1/count` on absent collection → 404 with `COLLECTION_NOT_FOUND`
  - `POST /v1/count` with no `db_path` → 400 with `MISSING_FIELD`
  - `POST /v1/count` with garbage body → 400 with `INVALID_JSON`
  - Unversioned route → 404 with `NOT_FOUND`
- Existing tests unaffected (they check strings, update assertions)
