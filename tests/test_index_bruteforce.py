"""Property 31: Index search equals brute-force ranking.

**Validates: Requirements 8.2, 8.7**

For any index of up to 2,000 random unit vectors with arbitrary distinct IDs and any query vector,
``search(q, k)`` with no filter returns ``min(k, ntotal)`` Hits equal to a NumPy brute-force
inner-product ranking in IDs and order, with similarities within 1e-5, disregarding order among
near-ties (< 1e-6).
"""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from nab_sentry.store.vector_index import VectorIndex

DIM = 512
SIM_TOL = 1e-5
TIE_TOL = 1e-6


def _unit(rng: np.random.Generator, n: int) -> np.ndarray:
    v = rng.standard_normal((n, DIM)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v


@st.composite
def index_cases(draw):
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    n = draw(st.integers(min_value=1, max_value=2000))
    k = draw(st.integers(min_value=1, max_value=2100))
    # Fraction of rows that duplicate an earlier row, to force exact similarity ties.
    dup_frac = draw(st.sampled_from([0.0, 0.0, 0.1, 0.5]))
    # Query either a fresh random unit vector or one of the stored vectors.
    query_stored = draw(st.booleans())
    rng = np.random.default_rng(seed)

    vecs = _unit(rng, n)
    n_dup = int(n * dup_frac)
    if n_dup and n > 1:
        dst = rng.choice(np.arange(1, n), size=min(n_dup, n - 1), replace=False)
        src = rng.integers(0, dst)  # copy from an earlier row
        vecs[dst] = vecs[src]

    # Arbitrary distinct non-negative int64 IDs, in random (non-sorted) order.
    ids = np.unique(rng.integers(0, 2**62, size=n * 2, dtype=np.int64))
    while ids.size < n:  # practically unreachable with a 2**62 range
        ids = np.unique(np.concatenate([ids, rng.integers(0, 2**62, size=n, dtype=np.int64)]))
    ids = rng.permutation(ids)[:n]

    q = vecs[rng.integers(0, n)].copy() if query_stored else _unit(rng, 1)[0]
    return ids, vecs, q, k


@given(index_cases())
def test_search_equals_bruteforce(case):
    ids, vecs, q, k = case
    index = VectorIndex(dim=DIM)
    index.add(ids, vecs)

    hits = index.search(q, k)

    n_expected = min(k, len(ids))
    assert len(hits) == n_expected

    # Brute force in float64 over the same float32 data.
    bf = vecs.astype(np.float64) @ q.astype(np.float64)
    sim_by_id = dict(zip(ids.tolist(), bf.tolist()))
    order = np.lexsort((ids, -bf))  # (-similarity, vector_id)
    bf_ids = ids[order][:n_expected]
    bf_sims = bf[order][:n_expected]

    hit_ids = [h.vector_id for h in hits]
    assert len(set(hit_ids)) == len(hit_ids), "duplicate IDs in result"
    assert all(i in sim_by_id for i in hit_ids), "result contains unknown ID"

    # Each reported similarity matches the brute-force inner product for that ID.
    for h in hits:
        assert abs(h.similarity - sim_by_id[h.vector_id]) <= SIM_TOL

    # Position-wise similarities match the brute-force ranking.
    for h, s in zip(hits, bf_sims):
        assert abs(h.similarity - s) <= SIM_TOL

    # IDs and order match, except where the differing entries are near-ties.
    for pos, (got, want) in enumerate(zip(hit_ids, bf_ids.tolist())):
        if got != want:
            assert abs(sim_by_id[got] - sim_by_id[want]) < TIE_TOL, (
                f"position {pos}: got id {got} ({sim_by_id[got]}), "
                f"expected id {want} ({sim_by_id[want]})"
            )

    # Any ID outside the brute-force top set must tie with the boundary similarity.
    boundary = bf_sims[-1]
    for got in set(hit_ids) - set(bf_ids.tolist()):
        assert abs(sim_by_id[got] - boundary) < TIE_TOL

    # Returned order is (-similarity, vector_id) on the reported similarities.
    keys = [(-h.similarity, h.vector_id) for h in hits]
    assert keys == sorted(keys)


def test_exact_duplicates_order_by_id():
    """Identical vectors tie exactly and are ordered by ascending vector_id."""
    rng = np.random.default_rng(7)
    v = _unit(rng, 1)
    vecs = np.repeat(v, 5, axis=0)
    ids = np.array([40, 3, 17, 99, 8], dtype=np.int64)
    index = VectorIndex(dim=DIM)
    index.add(ids, vecs)

    hits = index.search(v[0], 10)
    assert [h.vector_id for h in hits] == [3, 8, 17, 40, 99]
    assert all(abs(h.similarity - 1.0) <= SIM_TOL for h in hits)
