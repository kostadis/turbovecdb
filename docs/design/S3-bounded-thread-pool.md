# S3: Synchronous http.server blocks thread per request

**Issue:** [#70](https://github.com/kostadis/turbovecdb/issues/70)
**Status:** Implemented (2026-07-09)
**Priority:** Low

## Problem

`ThreadingHTTPServer` spawns an unbounded thread per request. Under load,
threads pile up on the GIL, memory grows without bound, and the kernel
eventually refuses to create more threads.

Root cause: `socketserver.ThreadingMixIn.process_request` does
`threading.Thread(target=...).start()` with no cap.

## Design: Bounded thread pool

Replace `ThreadingHTTPServer` with a minimal subclass that uses a
`concurrent.futures.ThreadPoolExecutor` instead of raw threads.

A bounded pool:
- Caps concurrent requests (default 10)
- Queues excess requests (they wait, not crash)
- Reuses threads (no create/teardown per request)
- Stays stdlib-only (no new dependencies)

### `BoundedThreadingHTTPServer`

```python
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer

class BoundedThreadingHTTPServer(ThreadingHTTPServer):
    """Like ThreadingHTTPServer but with a bounded thread pool."""

    def __init__(self, *args, max_workers=10, **kwargs):
        super().__init__(*args, **kwargs)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def process_request(self, request, client_address):
        self._executor.submit(self.process_request_thread, request, client_address)
```

Only `process_request` is overridden — everything else (request
parsing, routing, response) is identical.

### CLI flag

Add `--max-workers` to control the pool size:

```python
ap.add_argument("--max-workers", type=int, default=10)
srv = BoundedThreadingHTTPServer((args.host, args.port), Handler, max_workers=args.max_workers)
```

### Why not async (aiohttp/uvicorn)?

The `op_*` handlers call `turbovecdb` — a synchronous C extension that
releases the GIL during Rust calls. Wrapping them in async would still
need a thread pool executor for every I/O call, adding complexity
with no throughput benefit. The bottleneck is the flock, not the
event loop.

### Why not waitress?

Waitress is a production WSGI server with a bounded thread pool and
better HTTP compliance, but it's a third-party dependency. The stdlib
`ThreadPoolExecutor` achieves the same resource-bounding with zero new
dependencies.

### Test plan

- Add `--max-workers` test: start server with `max_workers=2`, send 3
  concurrent requests, verify all complete (pool queues the 3rd)
- Existing op-level tests unaffected
- HTTP integration tests (test_service_http.py) still pass
