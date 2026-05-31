"""Full filter-operator coverage. Uses get(where=...) so assertions are on the
matched id set, independent of vector ranking."""

import pytest

import turbovecdb
from turbovecdb import UnsupportedFilterError

DIM = 8


def _v(i):
    v = [0.0] * DIM
    v[i % DIM] = 1.0
    return v


@pytest.fixture
def col(tmp_path):
    db = turbovecdb.connect(str(tmp_path / "db"))
    c = db.collection("c", create=True)
    rows = [
        ("d1", {"lang": "en", "year": 2019}),
        ("d2", {"lang": "en", "year": 2021}),
        ("d3", {"lang": "fr", "year": 2021}),
        ("d4", {"lang": "la", "year": 2023}),
    ]
    c.add(
        ids=[r[0] for r in rows],
        documents=["the quick brown fox", "lorem ipsum dolor", "bonjour le monde", "carpe diem"],
        metadatas=[r[1] for r in rows],
        vectors=[_v(i) for i in range(len(rows))],
    )
    yield c
    db.close()


def ids(col, **kw):
    return set(col.get(**kw).ids)


def test_bare_equality(col):
    assert ids(col, where={"lang": "en"}) == {"d1", "d2"}


def test_eq(col):
    assert ids(col, where={"lang": {"$eq": "fr"}}) == {"d3"}


def test_ne(col):
    assert ids(col, where={"lang": {"$ne": "en"}}) == {"d3", "d4"}


def test_in(col):
    assert ids(col, where={"lang": {"$in": ["fr", "la"]}}) == {"d3", "d4"}


def test_nin(col):
    assert ids(col, where={"lang": {"$nin": ["fr", "la"]}}) == {"d1", "d2"}


def test_gt_gte_lt_lte(col):
    assert ids(col, where={"year": {"$gt": 2021}}) == {"d4"}
    assert ids(col, where={"year": {"$gte": 2021}}) == {"d2", "d3", "d4"}
    assert ids(col, where={"year": {"$lt": 2021}}) == {"d1"}
    assert ids(col, where={"year": {"$lte": 2021}}) == {"d1", "d2", "d3"}


def test_and(col):
    assert ids(col, where={"$and": [{"lang": "en"}, {"year": {"$gte": 2020}}]}) == {"d2"}


def test_or(col):
    assert ids(col, where={"$or": [{"lang": "fr"}, {"year": 2023}]}) == {"d3", "d4"}


def test_nested_and_or(col):
    where = {"$and": [{"year": {"$gte": 2021}}, {"$or": [{"lang": "en"}, {"lang": "la"}]}]}
    assert ids(col, where=where) == {"d2", "d4"}


def test_where_document_contains(col):
    assert ids(col, where_document={"$contains": "fox"}) == {"d1"}


def test_where_plus_where_document(col):
    got = ids(col, where={"lang": "en"}, where_document={"$contains": "lorem"})
    assert got == {"d2"}


def test_query_with_filter(col):
    # filter applied as an allowlist before ANN search
    res = col.query(vector=_v(0), k=10, where={"lang": "fr"})
    assert res.ids == ["d3"]


def test_filter_no_matches_returns_empty(col):
    assert col.query(vector=_v(0), k=10, where={"lang": "zz"}).ids == []


@pytest.mark.parametrize("bad", [
    {"$regex": "x"},
    {"$not": [{"lang": "en"}]},
    {"lang": {"$contains": "e"}},     # $contains is for where_document, not fields
    {"year": {"$between": [1, 2]}},
])
def test_unsupported_where_raises(col, bad):
    with pytest.raises(UnsupportedFilterError):
        col.get(where=bad)


def test_unsupported_where_document_raises(col):
    with pytest.raises(UnsupportedFilterError):
        col.get(where_document={"$not_contains": "x"})
