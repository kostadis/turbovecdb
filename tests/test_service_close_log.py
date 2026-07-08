"""Tests that service._close_db() logs warnings on close errors instead of
silently swallowing them (regression guard for Issue #54)."""

import logging

import turbovecdb.service as svc

DIM = 8


class _Mock:
    def __init__(self, close_raises=False):
        if close_raises:
            self.close = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        else:
            self.close = lambda: None


def test_close_db_logs_warning_on_col_error(caplog):
    col = _Mock(close_raises=True)
    db = _Mock(close_raises=False)
    with caplog.at_level(logging.WARNING, logger="turbovecdb.service"):
        svc._close_db(db, col)
        assert any("error closing collection" in rec.getMessage() for rec in caplog.records)


def test_close_db_logs_warning_on_db_error(caplog):
    col = _Mock(close_raises=False)
    db = _Mock(close_raises=True)
    with caplog.at_level(logging.WARNING, logger="turbovecdb.service"):
        svc._close_db(db, col)
        assert any("error closing database" in rec.getMessage() for rec in caplog.records)


def test_close_db_normal_no_warnings(caplog):
    col = _Mock(close_raises=False)
    db = _Mock(close_raises=False)
    with caplog.at_level(logging.WARNING, logger="turbovecdb.service"):
        svc._close_db(db, col)
        assert len(caplog.records) == 0


def test_close_db_none_handling(caplog):
    with caplog.at_level(logging.WARNING, logger="turbovecdb.service"):
        svc._close_db(None, None)
        assert len(caplog.records) == 0
