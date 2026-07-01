"""Translate where / where_document filter dicts into SQL over the sidecar.

The compiler itself now lives in the Rust core (:mod:`turbovecdb._core`, a PyO3
extension). This module is a thin adapter that preserves the historical Python
signatures and re-raises the core's ``FilterError`` as the public
:class:`UnsupportedFilterError`, so callers and the existing test-suite are
unchanged.

Metadata is stored as a JSON column, so field predicates compile to
``json_extract(metadata, '$.field')`` comparisons. The JSON *path* is bound as a
SQL parameter (not interpolated) and every operand is a bound parameter too, so
arbitrary field names and values can't inject SQL.

Supported operator set (compatible with the Chroma / Mongo subset MemPalace
uses):

* field: bare scalar equality, ``$eq``, ``$ne``, ``$gt``, ``$gte``, ``$lt``,
  ``$lte``, ``$in``, ``$nin``
* logical: ``$and``, ``$or`` (recursive, max depth 10)
* where_document: ``$contains``

Anything else raises :class:`UnsupportedFilterError`.
"""

from ._core import FilterError as _FilterError
from ._core import combined_sql as _rs_combined_sql
from ._core import where_document_to_sql as _rs_where_document_to_sql
from ._core import where_to_sql as _rs_where_to_sql
from .errors import UnsupportedFilterError


def where_to_sql(where, _depth=0):
    """Return ``(sql, params)`` for a ``where`` dict; ``("", [])`` when empty."""
    try:
        return _rs_where_to_sql(where, _depth)
    except _FilterError as e:
        raise UnsupportedFilterError(str(e)) from None


def where_document_to_sql(where_document):
    """Return ``(sql, params)`` for a ``where_document`` dict; ``("", [])`` when empty."""
    try:
        return _rs_where_document_to_sql(where_document)
    except _FilterError as e:
        raise UnsupportedFilterError(str(e)) from None


def combined_sql(where, where_document):
    """AND-combine where + where_document into a single ``(sql, params)``."""
    try:
        return _rs_combined_sql(where, where_document)
    except _FilterError as e:
        raise UnsupportedFilterError(str(e)) from None
