"""turbovecdb — an embedded vector database.

turbovec 4-bit ANN for fast approximate search, paired with a durable SQLite
sidecar that is the source of truth for documents, metadata, the id map, and the
exact float32 vectors. Metadata filters, persistence, exact-cosine re-rank, and
multi-process safety are built in. Bring your own embeddings, or pass an
``embedder`` callable and hand it text.

    import turbovecdb
    db = turbovecdb.connect("/path/to/db")
    col = db.collection("docs", embedder=my_embed_fn, create=True)
    col.add(ids=["a"], documents=["hello world"])
    hits = col.query(text="greeting", k=5)
    print(hits.ids, hits.distances)
"""

from .collection import Collection, GetResult, HealthResult, QueryResult, ReembedReport
from .database import Database, connect
from .errors import (
    CollectionNotFoundError,
    DimensionMismatchError,
    EmbedderIdentityMismatchError,
    EmbedderRequiredError,
    TurboVecError,
    UnsupportedFilterError,
)

__version__ = "0.5.0"

__all__ = [
    "connect",
    "Database",
    "Collection",
    "QueryResult",
    "GetResult",
    "HealthResult",
    "ReembedReport",
    "TurboVecError",
    "CollectionNotFoundError",
    "UnsupportedFilterError",
    "DimensionMismatchError",
    "EmbedderRequiredError",
    "EmbedderIdentityMismatchError",
]
