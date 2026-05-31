"""Translate where / where_document filter dicts into SQL over the sidecar.

Metadata is stored as a JSON column, so field predicates compile to
``json_extract(metadata, '$.field')`` comparisons. The JSON *path* is bound as a
SQL parameter (not interpolated) and every operand is a bound parameter too, so
arbitrary field names and values can't inject SQL. The resulting ``(sql,
params)`` selects the ``uid`` set that a query then passes to turbovec as an
``allowlist``.

Supported operator set (compatible with the Chroma / Mongo subset MemPalace
uses):

* field: bare scalar equality, ``$eq``, ``$ne``, ``$gt``, ``$gte``, ``$lt``,
  ``$lte``, ``$in``, ``$nin``
* logical: ``$and``, ``$or`` (recursive)
* where_document: ``$contains``

Anything else raises :class:`UnsupportedFilterError`.
"""

from .errors import UnsupportedFilterError

_CMP = {"$eq": "=", "$ne": "!=", "$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}


def _field_clause(field, value):
    path = f"$.{field}"
    if isinstance(value, dict):
        if len(value) != 1:
            raise UnsupportedFilterError(
                f"field {field!r} predicate must have exactly one operator, got {sorted(value)}"
            )
        op, operand = next(iter(value.items()))
        if op in ("$in", "$nin"):
            if not isinstance(operand, (list, tuple)) or not operand:
                raise UnsupportedFilterError(f"{op} on {field!r} requires a non-empty list")
            placeholders = ",".join("?" for _ in operand)
            negate = "NOT " if op == "$nin" else ""
            return (f"json_extract(metadata, ?) {negate}IN ({placeholders})", [path, *operand])
        if op in _CMP:
            return (f"json_extract(metadata, ?) {_CMP[op]} ?", [path, operand])
        raise UnsupportedFilterError(f"unsupported operator {op!r} on field {field!r}")
    # bare scalar → equality
    return ("json_extract(metadata, ?) = ?", [path, value])


def where_to_sql(where):
    """Return ``(sql, params)`` for a ``where`` dict; ``("", [])`` when empty."""
    if not where:
        return ("", [])
    if "$and" in where or "$or" in where:
        if len(where) != 1:
            raise UnsupportedFilterError("a logical operator ($and/$or) cannot have sibling keys")
        op, subs = next(iter(where.items()))
        if not isinstance(subs, (list, tuple)) or not subs:
            raise UnsupportedFilterError(f"{op} requires a non-empty list of clauses")
        joiner = " AND " if op == "$and" else " OR "
        frags, params = [], []
        for sub in subs:
            f, p = where_to_sql(sub)
            if f:
                frags.append(f"({f})")
                params.extend(p)
        return (joiner.join(frags), params)
    # reject any other top-level operator ($not, $nor, ...)
    for key in where:
        if isinstance(key, str) and key.startswith("$"):
            raise UnsupportedFilterError(f"unsupported top-level operator {key!r}")
    frags, params = [], []
    for field, value in where.items():
        f, p = _field_clause(field, value)
        frags.append(f)
        params.extend(p)
    return (" AND ".join(frags), params)


def _like_escape(s):
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def where_document_to_sql(where_document):
    """Return ``(sql, params)`` for a ``where_document`` dict; ``("", [])`` when empty."""
    if not where_document:
        return ("", [])
    if set(where_document.keys()) != {"$contains"}:
        raise UnsupportedFilterError(
            f"where_document supports only $contains, got {sorted(where_document)}"
        )
    needle = where_document["$contains"]
    return (r"document LIKE ? ESCAPE '\'", [f"%{_like_escape(needle)}%"])


def combined_sql(where, where_document):
    """AND-combine where + where_document into a single ``(sql, params)``."""
    frags, params = [], []
    for f, p in (where_to_sql(where), where_document_to_sql(where_document)):
        if f:
            frags.append(f"({f})")
            params.extend(p)
    return (" AND ".join(frags), params)
