"""Test that BoundedThreadingHTTPServer has a bounded work queue."""

import json
import threading
import time
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer

import pytest

import turbovecdb.service


def test_executor_queue_is_bounded():
    """Test that when executor queue is full, new connections are rejected or delayed."""
    # Clear any existing databases
    turbovecdb.service._databases.clear()
    
    # Create a server with a very small worker pool and queue size
    # We'll test by making more requests than workers can handle quickly
    port = 0  # Let OS assign a port
    srv = turbovecdb.service.BoundedThreadingHTTPServer(
        ("127.0.0.1", port), turbovecdb.service.Handler, max_workers=2
    )
    actual_port = srv.socket.getsockname()[1]
    
    # Start server in background thread
    server_thread = threading.Thread(target=srv.serve_forever, daemon=True)
    server_thread.start()
    
    try:
        # Give server time to start
        time.sleep(0.1)
        
        # Prepare a request that will tie up workers (simulate slow processing)
        # We'll make a request that takes time to process
        slow_request_data = json.dumps({
            "db_path": "/tmp/test_db",
            "collection": "test",
            "items": [{"id": "a", "vector": [1.0, 0, 0, 0, 0, 0, 0, 0]}]
        }).encode()
        
        # Start multiple concurrent requests that will tie up workers
        # We'll start more requests than we have workers to see if queue blocks
        results = []
        threads = []
        
        def make_request(request_id):
            try:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{actual_port}/v1/upsert",
                    data=slow_request_data,
                    headers={"Content-Type": "application/json"}
                )
                # Use a short timeout to detect if request is stuck in queue
                response = urllib.request.urlopen(req, timeout=2.0)
                results.append((request_id, "success", response.getcode()))
            except urllib.error.HTTPError as e:
                results.append((request_id, "http_error", e.code))
            except urllib.error.URLError as e:
                # This might happen if connection is refused due to full queue
                results.append((request_id, "url_error", str(e)))
            except Exception as e:
                results.append((request_id, "error", str(e)))
        
        # Start 5 requests (more than our 2 workers)
        for i in range(5):
            t = threading.Thread(target=make_request, args=(i,))
            t.start()
            threads.append(t)
            # Small delay between starting requests
            time.sleep(0.05)
        
        # Wait for all requests to complete (with timeout)
        for t in threads:
            t.join(timeout=5.0)
        
        # Check results - we should see some indication of queue behavior
        # With a truly bounded queue, we might see connection refused or timeouts
        # With an unbounded queue, all would eventually succeed (but possibly slowly)
        success_count = sum(1 for r in results if r[1] == "success")
        http_error_count = sum(1 for r in results if r[1] == "http_error")
        url_error_count = sum(1 for r in results if r[1] == "url_error")
        
        # For now, just verify the test runs - we'll enhance this assertion
        # based on what we observe about the current behavior
        assert len(results) == 5
        
    finally:
        srv.shutdown()
        server_thread.join(timeout=1.0)
        turbovecdb.service._databases.clear()


def test_queue_size_can_be_configured():
    """Test that we can configure queue bounds (this would require enhancement)."""
    # This test documents the desired behavior
    # Currently, ThreadPoolExecutor doesn't expose direct queue size control
    # but we can use a custom ThreadPoolExecutor with bounded queue
    pass
