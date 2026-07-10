"""Exception hierarchy for turbovecdb."""


class TurboVecError(Exception):
    """Base class for every turbovecdb error."""


class CollectionNotFoundError(TurboVecError, FileNotFoundError):
    """Raised by ``Database.collection(create=False)`` when the collection is absent.

    Subclasses ``FileNotFoundError`` so callers that catch the latter keep working.
    """


class UnsupportedFilterError(TurboVecError):
    """Raised when a where / where_document clause uses an unsupported operator.

    Silent dropping of unknown operators is forbidden — a filter that can't be
    honored must fail loudly rather than return wrong results.
    """


class DimensionMismatchError(TurboVecError):
    """Raised when a vector's dimensionality is wrong for the collection.

    Also raised when the dimension is not a positive multiple of 8 — a hard
    requirement of the underlying turbovec index.
    """


class EmbedderRequiredError(TurboVecError):
    """Raised when text is given (to add/query) but the collection has no embedder."""


class EmbedderIdentityMismatchError(TurboVecError):
    """Raised when re-embed is called with an embedder that doesn't match stored identity.

    This is the GAP-1 guard in action — it prevents accidental embedder swaps that
    would corrupt the index without the user's explicit awareness.
    """


class EmbedderMismatchError(TurboVecError):
    """Raised when a collection has documents but no embedder identity.

    A legacy or restored collection with data but no stored embedder identity
    cannot silently adopt a new embedder. Use ``reembed()`` to set a proper
    identity before adding text documents.
    """
