"""turbovec index lifecycle helpers — implemented in the Rust core.

The turbovec ``IdMapIndex`` maps ``uint64`` ids → quantized vectors. It is a
*derived cache*: the durable copy of every vector is a float32 BLOB in SQLite,
so the index can always be rebuilt. The lifecycle helpers (create / build /
load / atomic write) plus vector L2 normalization now live in
:mod:`turbovecdb._core`, which drives the turbovec Python API via PyO3
(turbovec has no Rust crate). This module just re-exports them so callers keep
importing ``turbovecdb.index``.
"""

from ._core import (  # noqa: F401
    build_index,
    l2_normalize,
    load_index,
    new_index,
    write_index_atomic,
)

# turbovec's bit menu is {2, 3, 4}; 4 is the recall ceiling.
DEFAULT_BIT_WIDTH = 4
