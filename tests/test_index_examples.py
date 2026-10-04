"""Example-based unit tests for the Vector_Index (task 8.9).

Covers: index type is ``IndexIDMap2(IndexFlatIP(512))`` (8.1); a failing restricted search falls
back to the post-filter path without raising (8.4); missing or wrong-type index files raise
``IndexUnavailable`` (8.15).
"""

from __future__ import annotations

import logging

import faiss
import numpy as np
import pytest

from nab_sentry.store import vector_index as vi
from nab_sentry.store.vector_index import IndexUnavailable, VectorIndex


def _unit(rng: np.random.Generator, n: int, d: int = 512) -> np.ndarray:
    v = rng.standard_normal((n, d)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture
def index() -> tuple[VectorIndex, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    vecs = _unit(rng, 40)
    ids = np.arange(1, 41, dtype=np.int64)  # mirrors vectors-table row IDs starting at 1
    idx = VectorIndex()
    idx.add(ids, vecs)
    return idx, ids, vecs


# --------------------------------------------------------------------------- 8.1 index type
def test_index_is_idmap2_over_flat_ip_512(index):
    idx, ids, _ = index
    fi = idx.faiss_index
    assert type(fi) is faiss.IndexIDMap2
    inner = faiss.downcast_index(fi.index)
    assert type(inner) is faiss.IndexFlatIP
    assert inner.metric_type == faiss.METRIC_INNER_PRODUCT
    assert fi.d == 512 and idx.dim == 512
    # Vector IDs are exactly the IDs supplied (the vectors-table row IDs).
    assert sorted(idx.ids().tolist()) == ids.tolist()


def test_default_constructor_is_empty_512():
    idx = VectorIndex()
    assert idx.ntotal == 0
    assert type(idx.faiss_index) is faiss.IndexIDMap2
    assert idx.faiss_index.d == 512


# --------------------------------------------------------------------------- 8.4 fallback
def _expected_postfilter(idx, q, k, allowed):
    fallback_k = min(idx.ntotal, 10 * k)
    allowed_set = set(int(a) for a in allowed)
    return [h for h in idx.search(q, fallback_k) if h.vector_id in allowed_set][:k]


def test_monkeypatched_search_restricted_raising_falls_back(index, monkeypatch, caplog):
    idx, _, vecs = index
    calls = []

    def boom(*args, **kwargs):
        calls.append(args)
        raise RuntimeError("IDSelectorBatch unsupported")

    monkeypatch.setattr(idx, "search_restricted", boom)
    allowed = np.array([2, 5, 9, 17, 33], dtype=np.int64)
    q = vecs[4]  # ID 5 is the query itself
    with caplog.at_level(logging.WARNING, logger=vi.__name__):
        hits = idx.search(q, 3, allowed)

    assert calls, "restricted path should have been attempted"
    assert hits == _expected_postfilter(idx, q, 3, allowed)
    assert hits[0].vector_id == 5
    assert all(h.vector_id in set(allowed.tolist()) for h in hits)
    assert all(a.similarity >= b.similarity for a, b in zip(hits, hits[1:]))
    assert any("falling back" in r.getMessage() for r in caplog.records)


def test_faiss_search_parameters_raising_falls_back(index, monkeypatch):
    """Failure inside the real restricted path (FAISS selector API) also falls back."""
    idx, _, vecs = index

    def broken_params(*args, **kwargs):
        raise AttributeError("SearchParameters not available in this FAISS build")

    monkeypatch.setattr(vi.faiss, "SearchParameters", broken_params)
    allowed = np.array([1, 10, 20, 30, 40], dtype=np.int64)
    q = vecs[9]  # ID 10
    hits = idx.search(q, 2, allowed)
    assert hits == _expected_postfilter(idx, q, 2, allowed)
    assert hits[0].vector_id == 10


def test_fallback_uses_min_ntotal_10k(index, monkeypatch):
    idx, _, vecs = index
    seen = {}
    real_postfilter = idx.search_postfilter

    def spy(q, k, allowed, fallback_k):
        seen["fallback_k"] = fallback_k
        return real_postfilter(q, k, allowed, fallback_k)

    monkeypatch.setattr(idx, "search_restricted", lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))
    monkeypatch.setattr(idx, "search_postfilter", spy)

    idx.search(vecs[0], 2, np.array([1, 2]))
    assert seen["fallback_k"] == 20  # 10 * 2 < 40
    idx.search(vecs[0], 7, np.array([1, 2]))
    assert seen["fallback_k"] == 40  # min(40, 70)


# --------------------------------------------------------------------------- 8.15 unavailable
def test_load_missing_file_raises(tmp_path):
    with pytest.raises(IndexUnavailable):
        VectorIndex.load(tmp_path / "does_not_exist.faiss")


def test_load_directory_path_raises(tmp_path):
    with pytest.raises(IndexUnavailable):
        VectorIndex.load(tmp_path)


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda: faiss.IndexFlatIP(512), id="bare-flat-ip"),
        pytest.param(lambda: faiss.IndexIDMap(faiss.IndexFlatIP(512)), id="idmap-not-idmap2"),
        pytest.param(lambda: faiss.IndexIDMap2(faiss.IndexFlatL2(512)), id="idmap2-flat-l2"),
        pytest.param(lambda: faiss.IndexIDMap2(faiss.IndexFlatIP(256)), id="idmap2-wrong-dim"),
    ],
)
def test_load_wrong_type_raises(tmp_path, make):
    path = tmp_path / "wrong.faiss"
    path.write_bytes(faiss.serialize_index(make()).tobytes())
    with pytest.raises(IndexUnavailable):
        VectorIndex.load(path)


def test_load_corrupt_bytes_raises(tmp_path):
    path = tmp_path / "corrupt.faiss"
    path.write_bytes(b"\x00\x01garbage" * 10)
    with pytest.raises(IndexUnavailable):
        VectorIndex.load(path)


def test_load_valid_file_succeeds(index, tmp_path):
    idx, ids, _ = index
    path = tmp_path / "index.faiss"
    idx.save(path)
    loaded = VectorIndex.load(path)
    assert type(loaded.faiss_index) is faiss.IndexIDMap2
    assert sorted(loaded.ids().tolist()) == ids.tolist()
