"""Sanity unit tests for nab_sentry.store.vector_index (properties live in tasks 8.6-8.9)."""

from __future__ import annotations

import faiss
import numpy as np
import pytest

from nab_sentry.store.vector_index import Hit, IndexUnavailable, VectorIndex


def _unit(rng: np.random.Generator, n: int, d: int = 512) -> np.ndarray:
    v = rng.standard_normal((n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture
def populated() -> tuple[VectorIndex, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    vecs = _unit(rng, 50)
    ids = np.arange(100, 150, dtype=np.int64)
    idx = VectorIndex()
    idx.add(ids, vecs)
    return idx, ids, vecs


def test_index_type_and_ids(populated):
    idx, ids, _ = populated
    assert isinstance(idx.faiss_index, faiss.IndexIDMap2)
    assert isinstance(faiss.downcast_index(idx.faiss_index.index), faiss.IndexFlatIP)
    assert idx.ntotal == 50
    assert sorted(idx.ids().tolist()) == ids.tolist()


def test_add_rejects_duplicates_and_bad_shapes(populated):
    idx, _, vecs = populated
    with pytest.raises(ValueError):
        idx.add(np.array([100]), vecs[:1])  # already present
    with pytest.raises(ValueError):
        idx.add(np.array([1, 1]), vecs[:2])  # duplicate within batch
    with pytest.raises(ValueError):
        idx.add(np.array([1]), vecs[:1, :10])  # wrong dim
    with pytest.raises(ValueError):
        idx.add(np.array([1, 2]), vecs[:1])  # count mismatch
    assert idx.ntotal == 50


def test_search_matches_brute_force_and_is_sorted(populated):
    idx, ids, vecs = populated
    q = vecs[3]
    hits = idx.search(q, 10)
    sims = vecs @ q
    expected = [int(ids[i]) for i in np.argsort(-sims)[:10]]
    assert [h.vector_id for h in hits] == expected
    assert hits[0] == Hit(103, pytest.approx(1.0, abs=1e-5))
    assert all(a.similarity >= b.similarity for a, b in zip(hits, hits[1:]))
    # k larger than ntotal -> no -1 padding
    assert len(idx.search(q, 1000)) == 50


def test_restricted_and_postfilter_agree(populated):
    idx, _, vecs = populated
    allowed = np.array([101, 120, 133, 149, 999], dtype=np.int64)
    r = idx.search(vecs[0], 3, allowed)
    assert len(r) == 3 and {h.vector_id for h in r} <= set(allowed.tolist())
    idx.force_postfilter = True
    p = idx.search(vecs[0], 3, allowed)
    assert [h.vector_id for h in p] == [h.vector_id for h in r]
    assert idx.search(vecs[0], 3, np.array([], dtype=np.int64)) == []


def test_restricted_failure_falls_back(populated, monkeypatch):
    idx, _, vecs = populated

    def boom(*a, **k):
        raise RuntimeError("selector unsupported")

    monkeypatch.setattr(idx, "search_restricted", boom)
    allowed = np.array([100, 110])
    hits = idx.search(vecs[0], 2, allowed)
    # fallback K = min(50, 10 * 2) = 20, so only allowed IDs inside the top 20 survive
    expected = [h for h in idx.search(vecs[0], 20) if h.vector_id in {100, 110}]
    assert hits == expected
    assert hits[0].vector_id == 100


def test_remove(populated):
    idx, _, vecs = populated
    assert idx.remove(np.array([100, 101, 5000])) == 2
    assert idx.ntotal == 48
    assert 100 not in {h.vector_id for h in idx.search(vecs[0], 50)}


def test_save_load_round_trip(populated, tmp_path):
    idx, _, vecs = populated
    path = tmp_path / "sub" / "index.faiss"
    idx.save(path)
    assert [p.name for p in path.parent.iterdir()] == ["index.faiss"]
    loaded = VectorIndex.load(path)
    assert loaded.ntotal == idx.ntotal
    assert sorted(loaded.ids().tolist()) == sorted(idx.ids().tolist())
    assert loaded.search(vecs[7], 20) == idx.search(vecs[7], 20)


def test_load_rejects_missing_wrong_type_and_dim(tmp_path):
    with pytest.raises(IndexUnavailable):
        VectorIndex.load(tmp_path / "missing.faiss")
    garbage = tmp_path / "garbage.faiss"
    garbage.write_bytes(b"not an index")
    with pytest.raises(IndexUnavailable):
        VectorIndex.load(garbage)
    flat = tmp_path / "flat.faiss"
    flat.write_bytes(faiss.serialize_index(faiss.IndexFlatIP(512)).tobytes())
    with pytest.raises(IndexUnavailable):
        VectorIndex.load(flat)
    l2 = tmp_path / "l2.faiss"
    l2.write_bytes(faiss.serialize_index(faiss.IndexIDMap2(faiss.IndexFlatL2(512))).tobytes())
    with pytest.raises(IndexUnavailable):
        VectorIndex.load(l2)
    small = tmp_path / "small.faiss"
    VectorIndex(dim=8).save(small)
    with pytest.raises(IndexUnavailable):
        VectorIndex.load(small)
