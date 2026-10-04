"""Property 32: Index save/load round trip.

**Validates: Requirements 8.11**
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from nab_sentry.store.vector_index import DEFAULT_DIM, VectorIndex

DIM = DEFAULT_DIM


def _unit_vectors(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, DIM)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


# An op is ("add", ids, seed) or ("remove", ids). IDs come from a small pool so removes hit.
_id_pool = st.integers(min_value=0, max_value=200)
_add_op = st.tuples(
    st.just("add"),
    st.lists(_id_pool, min_size=0, max_size=15, unique=True),
    st.integers(min_value=0, max_value=2**32 - 1),
)
_remove_op = st.tuples(st.just("remove"), st.lists(_id_pool, min_size=0, max_size=15))
ops_strategy = st.lists(st.one_of(_add_op, _remove_op), min_size=0, max_size=8)

# Path segments, including non-ASCII names (save/load uses byte-level serialization).
dir_names = st.sampled_from(["idx", "índice", "カメラ", "with space"])


def _apply(index: VectorIndex, ops) -> None:
    for op in ops:
        if op[0] == "add":
            present = set(index.ids().tolist())
            ids = [i for i in op[1] if i not in present]
            index.add(np.array(ids, dtype=np.int64), _unit_vectors(op[2], len(ids)))
        else:
            index.remove(np.array(op[1], dtype=np.int64))


@settings(max_examples=100, deadline=None)
@given(
    ops=ops_strategy,
    query_seed=st.integers(min_value=0, max_value=2**32 - 1),
    k=st.integers(min_value=1, max_value=50),
    dir_name=dir_names,
)
def test_save_load_round_trip(ops, query_seed, k, dir_name):
    """Feature: nab-sentry, Property 32: Index save/load round trip."""
    index = VectorIndex(DIM)
    _apply(index, ops)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / dir_name / "vectors.faiss"
        index.save(path)
        loaded = VectorIndex.load(path, dim=DIM)

        # No leftover temp files from the atomic write.
        assert sorted(p.name for p in path.parent.iterdir()) == ["vectors.faiss"]

    assert loaded.ntotal == index.ntotal
    assert set(loaded.ids().tolist()) == set(index.ids().tolist())

    q = _unit_vectors(query_seed, 1)[0]
    before = index.search(q, k)
    after = loaded.search(q, k)
    assert [h.vector_id for h in after] == [h.vector_id for h in before]
    assert [h.similarity for h in after] == [h.similarity for h in before]
