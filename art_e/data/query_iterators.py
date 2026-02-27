from art_e.data.types_enron import SyntheticQuery
from typing import List, Optional
import os
import random
import pyarrow.ipc as ipc

# Path to local Arrow dataset (relative to this file)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DATASET_DIR = os.path.join(BASE_DIR, "..", "..", "data", "art_e_vince_kaminski")

# A few spot-checked synthetic queries that we found to be ambiguous or
# contradicted by other source emails. We'll exclude them for a more accurate
# model.
bad_queries = [49, 101, 129, 171, 208, 266, 327]


def _load_arrow_split(split: str) -> list[dict]:
    """Load an Arrow split directly using PyArrow (bypasses datasets metadata)."""
    arrow_path = os.path.join(LOCAL_DATASET_DIR, split, "data-00000-of-00001.arrow")
    with open(arrow_path, "rb") as f:
        reader = ipc.open_stream(f)
        table = reader.read_all()
    return table.to_pylist()


def load_synthetic_queries(
    split: str = "train",
    limit: Optional[int] = None,
    max_messages: Optional[int] = 1,
    shuffle: bool = False,
    exclude_known_bad_queries: bool = True,
) -> List[SyntheticQuery]:
    rows = _load_arrow_split(split)

    queries = [SyntheticQuery(**row) for row in rows]

    if max_messages is not None:
        queries = [q for q in queries if len(q.message_ids) <= max_messages]

    if exclude_known_bad_queries:
        queries = [q for q in queries if q.id not in bad_queries]

    if shuffle:
        random.shuffle(queries)

    if limit is not None:
        return queries[:limit]
    else:
        return queries
