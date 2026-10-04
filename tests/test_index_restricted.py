"""Property 30: Restricted search and post-filter fallback agree.

**Validates: Requirements 8.4, 8.6**
"""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from nab_sentry.store.vector_index import Hit, VectorIndex

SIM_TOL = 1e-5  # similarities must agree within this
TIE_TOL = 1e-5  # order among hits closer than this is disregarded


@st.composite
def index_cases(draw):
    """A random unit-vector index, an allowed subset, a query, and k."""
    dim = draw(st.sampled_from([4, 16, 64, 512]))
    n = draw(st.integers(min_value=1, max_value=2000))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(seed)

    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    # Occasionally duplicate rows so exact similarity ties occur.
    if draw(st.booleans()) and n > 1:
        dup = rng.integers(0, n, size=max(1, n // 10))
        vecs[rng.integers(0, n, size=dup.size)] = vecs[dup]
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    # Distinct non-negative, non-contiguous IDs.
    ids = rng.choice(10 * n + 10, size=n, replace=False).astype(np.int64)

    density = draw(st.sampled_from([0.0, 0.01, 0.1, 0.5, 1.0]))
    mask = rng.random(n) < density
    allowed = ids[mask]
    if draw(st.booleans()):
        # IDs not present in the index must simply never appear.
        allowed = np.concatenate([allowed, np.arange(10 * n + 10, 10 * n + 15, dtype=np.int64)])
    if allowed.size == 0:
        allowed = ids[: 1 + int(rng.integers(0, n))]

    q = rng.standard_normal(dim).astype(np.float32)
    q /= np.linalg.norm(q)
    k = draw(st.integers(min_value=1, max_value=max(1, min(n + 5, 200))))
    return dim, ids, vecs, allowed, q, k


def _build(dim: int, ids: np.ndarray, vecs: np.ndarray, **kw) -> VectorIndex:
    idx = VectorIndex(dim, **kw)
    idx.add(ids, vecs)
    return idx


def _assert_agree(a: list[Hit], b: list[Hit], allowed: np.ndarray) -> None:
    allowed_set = set(allowed.tolist())
    assert len(a) == len(b)
    for h in a + b:
        assert h.vector_id in allowed_set
    for lst in (a, b):
        assert len({h.vector_id for h in lst}) == len(lst)
        assert lst == sorted(lst, key=lambda h: (-h.similarity, h.vector_id))
    # Rank-wise similarities agree.
    for ha, hb in zip(a, b):
        assert abs(ha.similarity - hb.similarity) <= SIM_TOL
    if not a:
        return
    sim_a = {h.vector_id: h.similarity for h in a}
    sim_b = {h.vector_id: h.similarity for h in b}
    # Common IDs carry the same similarity.
    for vid in sim_a.keys() & sim_b.keys():
        assert abs(sim_a[vid] - sim_b[vid]) <= SIM_TOL
    # Outside a tie with the cut-off similarity, both paths select the same IDs.
    cutoff = min(a[-1].similarity, b[-1].similarity)
    strict_a = {v for v, s in sim_a.items() if s > cutoff + TIE_TOL}
    strict_b = {v for v, s in sim_b.items() if s > cutoff + TIE_TOL}
    assert strict_a == strict_b
    # Positions where neither neighbour is tied must hold the same ID.
    sims = [h.similarity for h in a]
    for i, (ha, hb) in enumerate(zip(a, b)):
        tied = (i > 0 and sims[i - 1] - sims[i] < TIE_TOL) or (
            i + 1 < len(sims) and sims[i] - sims[i + 1] < TIE_TOL
        )
        if not tied and ha.similarity > cutoff + TIE_TOL:
            assert ha.vector_id == hb.vector_id


def _brute_allowed(ids, vecs, allowed, q, k) -> list[Hit]:
    """Reference: exact top-k over the allowed rows."""
    allowed_set = set(allowed.tolist())
    sims = vecs @ q
    hits = [Hit(int(i), float(s)) for i, s in zip(ids, sims) if int(i) in allowed_set]
    hits.sort(key=lambda h: (-h.similarity, h.vector_id))
    return hits[:k]


@given(index_cases())
def test_restricted_and_postfilter_agree(case):
    dim, ids, vecs, allowed, q, k = case
    idx = _build(dim, ids, vecs)
    fallback_k = idx.ntotal + int(np.random.default_rng(k).integers(0, 3))  # >= ntotal

    restricted = idx.search_restricted(q, k, allowed)
    post = idx.search_postfilter(q, k, allowed, fallback_k)

    _assert_agree(restricted, post, allowed)
    # Both equal the exact restricted ranking.
    _assert_agree(restricted, _brute_allowed(ids, vecs, allowed, q, k), allowed)
    # And the public entry point matches as well.
    _assert_agree(idx.search(q, k, allowed), post, allowed)


@given(index_cases())
def test_search_falls_back_when_restricted_raises(case):
    dim, ids, vecs, allowed, q, k = case
    idx = _build(dim, ids, vecs)

    def boom(*_a, **_kw):
        raise RuntimeError("IDSelectorBatch unsupported")

    idx.search_restricted = boom  # instance-level override; class untouched
    got = idx.search(q, k, allowed)  # must not raise
    expected = idx.search_postfilter(q, k, allowed, min(idx.ntotal, 10 * k))
    assert got == expected

    # The force_postfilter hook takes the same path.
    forced = _build(dim, ids, vecs, force_postfilter=True)
    assert forced.search(q, k, allowed) == expected
