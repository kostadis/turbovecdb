"""Vector L2 normalization — implemented in the Rust core.

The turbovec ``IdMapIndex`` maps ``uint64`` ids → quantized vectors. It is a
*derived cache*: the durable copy of every vector is a float32 BLOB in SQLite,
so the index can always be rebuilt. `Collection`'s Rust core owns the index
lifecycle directly against the native ``turbovec`` crate (see
``docs/rust-core-split-design.md``); this module now only re-exports vector
L2 normalization, which remains a standalone public helper.
"""

from ._core import l2_normalize  # noqa: F401

# turbovec's bit menu is {2, 3, 4}; 4 is the recall ceiling.
DEFAULT_BIT_WIDTH = 4
