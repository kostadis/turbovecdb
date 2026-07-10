"""Tests proving Phase 4 high-severity bugs exist.

Each test asserts the CORRECT behaviour (what should happen after the fix)
and is marked ``xfail`` because the bug is still present.

Bugs covered:
  #83  — Relative database paths retarget live handles after os.chdir()
  #82  — Live handles can write successfully after their collection is deleted
  #93  — Databases with a future schema_version are opened as compatible
  #107 — HTTP request bodies are unbounded and have no read timeout
"""

import json
import os
import sqlite3
import time
from pathlib import Path

import numpy as np
import pytest

import turbovecdb
from turbovecdb import TurboVecError

DIM = 8


# ═══════════════════════════════════════════════════════════════════════
# #83 — Relative database paths retarget live handles after os.chdir()
#
# Database stores relative path strings. After cwd changes, lock files,
# .tvim, and WAL paths re-resolve against the new cwd, but the SQLite
# connection still points to the original inode — splitting the
# coordination domain.
# ═══════════════════════════════════════════════════════════════════════


def test_relative_path_retargets_after_chdir(tmp_path):
    """Flushing after chdir() writes .tvim to new cwd instead of
    original database directory. The coordination domain splits:
    SQLite metadata goes to original dir, .tvim goes to new cwd."""
    original_cwd = os.getcwd()
    db_rel = "tvdb_rel_test"
    db_abs = str(tmp_path / db_rel)

    try:
        # Open with a RELATIVE path.
        os.chdir(tmp_path)
        db = turbovecdb.connect(db_rel)
        c = db.collection("c", dim=DIM, create=True)
        c.add(ids=["a"], vectors=[[1.0] + [0.0] * (DIM - 1)])
        # Flush writes the .tvim file.
        c.flush()
        db.close()

        # .tvim should be inside the database directory.
        # With the bug, a normal cwd operation works — we need to chdir
        # THEN write through the existing handle.

        # Re-open and set up a handle with a relative-path db.
        os.chdir(tmp_path)
        db2 = turbovecdb.connect(db_rel)
        c2 = db2.collection("c", dim=DIM)

        # Add another doc.
        c2.add(ids=["b"], vectors=[[0.0, 1.0] + [0.0] * (DIM - 2)])

        # NOW change cwd to a subdirectory.
        subdir = tmp_path / "elsewhere"
        subdir.mkdir(exist_ok=True)
        os.chdir(subdir)

        # Flush — this writes .tvim. With the bug, .tvim lands in subdir
        # (the new cwd) instead of the original db_rel directory.
        c2.flush()

        # The .tvim file should be in the original database directory.
        tvim_original = Path(db_abs) / "c" / "index.tvim"
        tvim_wrong = subdir / db_rel / "c" / "index.tvim"

        assert tvim_original.exists(), (
            f"Bug #83: .tvim not in original dir {tvim_original} "
            f"(instead found in {tvim_wrong if tvim_wrong.exists() else 'nowhere'})"
        )

        # Also check: reopening at the original absolute path should
        # see both documents.
        db3 = turbovecdb.connect(db_abs)
        c3 = db3.collection("c", dim=DIM)
        assert c3.count() == 2, (
            f"Bug #83: after chdir+flush, count is {c3.count()} (expected 2)"
        )
        db3.close()
        db2.close()

    finally:
        os.chdir(original_cwd)


# ═══════════════════════════════════════════════════════════════════════
# #82 — Live handles can write after collection is deleted
#
# delete_collection() removes the directory, but a live handle connected
# to the unlinked inode reports successful writes even though the data
# is unreachable from the collection path.
# ═══════════════════════════════════════════════════════════════════════


def test_stale_handle_write_after_delete(tmp_path):
    """A handle opened before delete_collection should fail on
    subsequent operations but currently succeeds."""
    db_path = str(tmp_path / "db")

    db1 = turbovecdb.connect(db_path)
    stale = db1.collection("c", dim=DIM, create=True)
    stale.add(ids=["before"], vectors=[[1.0] + [0.0] * (DIM - 1)])

    # Delete via a separate Database handle.
    db2 = turbovecdb.connect(db_path)
    db2.delete_collection("c")
    db2.close()

    # The collection directory should be gone.
    assert not Path(db_path, "c").exists()

    # Writing through the stale handle SHOULD fail.
    try:
        stale.add(ids=["lost"], vectors=[[0.0, 1.0] + [0.0] * (DIM - 2)])
        # If we reach here, the write "succeeded" (even though it went
        # to an unlinked inode). This is the bug.
        assert False, (
            "Bug #82: stale handle wrote successfully after "
            "collection was deleted (data lost)"
        )
    except TurboVecError as e:
        if "stale" in str(e).lower() or "deleted" in str(e).lower() or "closed" in str(e).lower():
            return  # Correctly rejected
        raise  # Unexpected error
    finally:
        db1.close()


def test_stale_handle_query_after_delete(tmp_path):
    """Querying after delete_collection should fail but currently
    returns stale cached data."""
    db_path = str(tmp_path / "db")

    db1 = turbovecdb.connect(db_path)
    stale = db1.collection("c", dim=DIM, create=True)
    stale.add(ids=["a"], vectors=[[1.0] + [0.0] * (DIM - 1)])

    db2 = turbovecdb.connect(db_path)
    db2.delete_collection("c")
    db2.close()

    try:
        stale.count()
        stale.get(ids=["a"])
        # Bug: stale handle returns data from unlinked inode.
        # After fix: should raise stale-handle error.
        pytest.fail("Bug #82: stale handle query succeeded after collection was deleted")
    except TurboVecError as e:
        if "stale" in str(e).lower() or "deleted" in str(e).lower() or "closed" in str(e).lower():
            return
        raise
    finally:
        db1.close()


# ═══════════════════════════════════════════════════════════════════════
# #93 — Databases with a future schema_version are opened as compatible
#
# migrate_schema() accepts schema_version >= SCHEMA_VERSION (line 313),
# including 999, 2147483647, or any future value. An older binary
# should refuse to open a database from a newer version.
# ═══════════════════════════════════════════════════════════════════════


def test_future_schema_version_rejected(tmp_path):
    """Opening a collection with schema_version > SCHEMA_VERSION must
    fail with an error but currently succeeds."""
    path = str(tmp_path / "db")

    # Create and populate a collection.
    db = turbovecdb.connect(path)
    c = db.collection("c", dim=DIM, create=True)
    c.add(ids=["a"], vectors=[[1.0] + [0.0] * (DIM - 1)])
    db.close()

    # Mutate schema_version to a future value.
    store = os.path.join(path, "c", "store.sqlite3")
    conn = sqlite3.connect(store)
    conn.execute("UPDATE meta SET value = '999' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()

    # Reopen — should raise an error because schema is from the future.
    try:
        db2 = turbovecdb.connect(path)
        c2 = db2.collection("c", dim=DIM)
        c2.count()  # triggers schema check
        # Bug: open succeeded despite future schema_version
        pytest.fail(
            "Bug #93: opened collection with schema_version=999 "
            "(future version should be rejected)"
        )
    except TurboVecError as e:
        if "schema" in str(e).lower() or "version" in str(e).lower() or "future" in str(e).lower():
            return
        raise
    finally:
        try:
            db2.close()
        except Exception:
            pass


def test_future_schema_version_allows_write(tmp_path):
    """Not only does open succeed, but writes also work — making
    silent corruption possible."""
    path = str(tmp_path / "db")

    db = turbovecdb.connect(path)
    c = db.collection("c", dim=DIM, create=True)
    c.add(ids=["a"], vectors=[[1.0] + [0.0] * (DIM - 1)])
    db.close()

    store = os.path.join(path, "c", "store.sqlite3")
    conn = sqlite3.connect(store)
    conn.execute("UPDATE meta SET value = '999' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()

    db2 = turbovecdb.connect(path)
    try:
        c2 = db2.collection("c", dim=DIM)
        pytest.fail(
            "Bug #93: opened collection with schema_version=999 "
            "(future version should be rejected at open)"
        )
    except TurboVecError as e:
        if "schema" in str(e).lower() or "version" in str(e).lower():
            db2.close()
            return
        raise


# ═══════════════════════════════════════════════════════════════════════
# #107 — HTTP request bodies are unbounded with no read timeout
#
# The service trusts Content-Length and calls self.rfile.read(n) with no
# maximum, negative-value validation, or read deadline. A client can
# force allocation of an arbitrarily large body or hold a worker
# indefinitely.
# ═══════════════════════════════════════════════════════════════════════


def _start_service():
    """Start the HTTP service on a random port.

    Returns (port, server) where server.shutdown() stops it.
    """
    import socket
    import threading

    import turbovecdb.service

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    server = turbovecdb.service.BoundedThreadingHTTPServer(
        ("127.0.0.1", port), turbovecdb.service.Handler, max_workers=2
    )
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return port, server


def test_http_oversized_content_length(tmp_path):
    """POST with an oversized Content-Length must be rejected with
    413 Payload Too Large. With the bug, the server blocks trying
    to read the oversized body."""
    import socket

    port, server = _start_service()

    body = b'{"db_path": "%s", "collection": "c", "items": [{"id": "a", "vector": [1,0,0,0,0,0,0,0]}]}' % str(tmp_path / "db").encode()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    sock.connect(("127.0.0.1", port))

    # Advertise a huge Content-Length but only send the real body.
    raw = (
        "POST /v1/upsert HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\n"
        "Content-Length: 100000000\r\n"
        "Content-Type: application/json\r\n"
        "Connection: close\r\n"
        "\r\n"
    ).encode() + body

    sock.sendall(raw)

    try:
        response = sock.recv(4096)
    except socket.timeout:
        pytest.fail(
            "Bug #107: server blocked trying to read oversized body "
            "(no Content-Length validation)"
        )

    status_line = response.split(b"\r\n")[0].decode()
    status_code = int(status_line.split(" ")[1])

    assert status_code == 413, (
        f"Bug #107: oversized Content-Length returned {status_code} "
        f"instead of 413 Payload Too Large"
    )
    sock.close()
    server.shutdown()


def test_http_negative_content_length(tmp_path):
    """POST with a negative Content-Length must be rejected with
    400 Bad Request. With the bug, rfile.read(-1) reads until EOF
    and the request succeeds."""
    import socket

    port, server = _start_service()

    body = b'{"db_path": "%s", "collection": "c", "items": [{"id": "a", "vector": [1,0,0,0,0,0,0,0]}]}' % str(tmp_path / "db").encode()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    sock.connect(("127.0.0.1", port))

    raw = (
        "POST /v1/upsert HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\n"
        "Content-Length: -1\r\n"
        "Content-Type: application/json\r\n"
        "Connection: close\r\n"
        "\r\n"
    ).encode() + body

    sock.sendall(raw)

    try:
        response = sock.recv(4096)
    except socket.timeout:
        pytest.fail(
            "Bug #107: server blocked on negative Content-Length "
            "(rfile.read(-1) reads until EOF)"
        )

    status_line = response.split(b"\r\n")[0].decode()
    status_code = int(status_line.split(" ")[1])

    assert status_code == 400, (
        f"Bug #107: negative Content-Length returned {status_code} "
        f"instead of 400 Bad Request"
    )
    sock.close()
    server.shutdown()
