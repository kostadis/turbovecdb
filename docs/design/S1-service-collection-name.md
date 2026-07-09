# S1: Configurable collection name in service.py

**Issue:** [#75](https://github.com/kostadis/turbovecdb/issues/75)
**Status:** Design
**Priority:** Low

## Problem

`service.py` hardcodes `COLLECTION = "pages"` at line 30. Every endpoint
operates on this single collection regardless of caller intent.

## Design

Remove the global `COLLECTION` constant. Every request **must** include
a `collection` field in its JSON body. No defaults, no env var, no CLI
flag — the caller always specifies which collection to operate on.

### What changes

#### `_open()` — accept collection as required parameter

```python
def _open(db_path: str, collection: str, dim: int | None = None, create: bool = False):
```

#### Each `op_*` function — extract `collection` from body (required)

```python
def op_upsert(req: dict) -> dict:
    collection = req["collection"]
    ...
```

Missing `collection` raises `KeyError` → 500 response (fine for a
single-user service).

#### Remove `COLLECTION` constant, CLI flag, env var

All gone. The `--collection` startup arg is not added. The startup
message drops the `(collection='...')` suffix.

#### Result

```
POST /upsert  {"db_path":"...", "collection":"articles", "items":[...]}
POST /upsert  {"db_path":"...", "collection":"images",   "items":[...]}
```

### Test plan

- Verify `collection` field is required (missing → error)
- Verify multiple collections within one server session
- Update existing tests to pass `collection` in every request body
