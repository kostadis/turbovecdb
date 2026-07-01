"""turbovec index lifecycle helpers.

The turbovec ``IdMapIndex`` maps ``uint64`` ids → quantized vectors. It is a
*derived cache*: the durable copy of every vector is a float32 BLOB in SQLite,
so the index can always be rebuilt. These helpers keep that lifecycle in one
place (build / load / atomic write) plus vector normalization.
"""

import os

import numpy as np
import turbovec

# Row-wise L2 normalization now lives in the Rust core (turbovecdb._core).
from ._core import l2_normalize  # noqa: F401

# turbovec's bit menu is {2, 3, 4}; 4 is the recall ceiling.
DEFAULT_BIT_WIDTH = 4


def new_index(dim, bit_width):
    return turbovec.IdMapIndex(dim=dim, bit_width=bit_width)


def build_index(dim, bit_width, uids, vecs):
    """Build an index from parallel ``uids`` (1-D) and ``vecs`` (list of 1-D)."""
    idx = turbovec.IdMapIndex(dim=dim, bit_width=bit_width)
    if len(uids):
        idx.add_with_ids(
            np.ascontiguousarray(np.stack(vecs), dtype=np.float32),
            np.asarray(uids, dtype=np.uint64),
        )
    return idx


def load_index(path):
    return turbovec.IdMapIndex.load(path)


def write_index_atomic(index, path):
    """Serialize ``index`` to ``path`` atomically (temp file + os.replace)."""
    tmp = path + ".tmp"
    index.write(tmp)
    os.replace(tmp, path)
