"""FAISS vector index wrapper: ``IndexIDMap2(IndexFlatIP(dim))`` keyed by ``vectors`` row IDs.

Requirements 8.1-8.4, 8.7, 8.10, 8.11. Vector IDs equal the ``vectors`` table row IDs (8.1).
Filtered search uses ``SearchParameters(sel=IDSelectorBatch)``; any failure there falls back to an
unrestricted search with ``fallback_k = min(ntotal, 10 * top_k)`` followed by a post-filter (8.4).
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

log = logging.getLogger(__name__)

DEFAULT_DIM = 512


class IndexUnavailable(RuntimeError):
    """The index file is missing, unreadable, or not an ``IndexIDMap2(IndexFlatIP(dim))``."""


@dataclass(frozen=True)
class Hit:
    vector_id: int
    similarity: float


def _as_ids(ids: np.ndarray | list[int]) -> np.ndarray:
    arr = np.asarray(ids)
    if arr.ndim != 1:
        raise ValueError(f"ids must be 1-D, got shape {arr.shape}")
    if arr.size and not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"ids must be integers, got dtype {arr.dtype}")
    return np.ascontiguousarray(arr, dtype=np.int64)


def _sorted_hits(dist: np.ndarray, labels: np.ndarray) -> list[Hit]:
    """Drop ``-1`` padding and order by ``(-similarity, vector_id)``."""
    hits = [
        Hit(int(i), float(d))
        for d, i in zip(dist.tolist(), labels.tolist())
        if i != -1
    ]
    hits.sort(key=lambda h: (-h.similarity, h.vector_id))
    return hits


class VectorIndex:
    """Exact inner-product index with explicit int64 IDs."""

    def __init__(self, dim: int = DEFAULT_DIM, *, force_postfilter: bool = False) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = dim
        # Test/diagnostic hook (Config.force_postfilter): skip the IDSelectorBatch path.
        self.force_postfilter = force_postfilter
        self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    # ------------------------------------------------------------------ persistence
    @classmethod
    def load(cls, path: Path, dim: int = DEFAULT_DIM, *, force_postfilter: bool = False) -> VectorIndex:
        path = Path(path)
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise IndexUnavailable(f"vector index file cannot be read: {path} ({exc})") from exc
        try:
            idx = faiss.deserialize_index(np.frombuffer(data, dtype=np.uint8))
        except Exception as exc:  # faiss raises RuntimeError on corrupt input
            raise IndexUnavailable(f"vector index file cannot be loaded: {path} ({exc})") from exc
        if not isinstance(idx, faiss.IndexIDMap2):
            raise IndexUnavailable(
                f"vector index {path} has type {type(idx).__name__}, expected IndexIDMap2"
            )
        inner = faiss.downcast_index(idx.index)
        if not isinstance(inner, faiss.IndexFlatIP) or inner.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise IndexUnavailable(
                f"vector index {path} wraps {type(inner).__name__}, expected IndexFlatIP"
            )
        if idx.d != dim:
            raise IndexUnavailable(f"vector index {path} has dim {idx.d}, expected {dim}")
        obj = cls(dim, force_postfilter=force_postfilter)
        obj._index = idx
        return obj

    def save(self, path: Path) -> None:
        """Atomically write the index: temp file in the same dir, fsync, ``os.replace``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # serialize_index avoids FAISS's narrow-string file APIs (non-ASCII paths on Windows).
        data = faiss.serialize_index(self._index).tobytes()
        fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------ contents
    @property
    def ntotal(self) -> int:
        return int(self._index.ntotal)

    @property
    def faiss_index(self) -> faiss.Index:
        return self._index

    def ids(self) -> np.ndarray:
        return faiss.vector_to_array(self._index.id_map).astype(np.int64, copy=False)

    def add(self, ids: np.ndarray, vecs: np.ndarray) -> None:
        id_arr = _as_ids(ids)
        v = np.asarray(vecs)
        if v.ndim != 2 or v.shape[1] != self.dim:
            raise ValueError(f"vecs must have shape (n, {self.dim}), got {v.shape}")
        if v.shape[0] != id_arr.shape[0]:
            raise ValueError(f"{id_arr.shape[0]} ids but {v.shape[0]} vectors")
        if id_arr.size == 0:
            return
        v = np.ascontiguousarray(v, dtype=np.float32)
        if not np.all(np.isfinite(v)):
            raise ValueError("vecs contain non-finite values")
        if np.any(id_arr < 0):
            raise ValueError("ids must be non-negative")
        if np.unique(id_arr).size != id_arr.size:
            raise ValueError("duplicate ids within batch")
        clash = np.intersect1d(id_arr, self.ids())
        if clash.size:
            raise ValueError(f"ids already present in index: {clash[:10].tolist()}")
        self._index.add_with_ids(v, id_arr)

    def remove(self, ids: np.ndarray) -> int:
        id_arr = _as_ids(ids)
        if id_arr.size == 0:
            return 0
        return int(self._index.remove_ids(faiss.IDSelectorBatch(id_arr)))

    # ------------------------------------------------------------------ search
    def _query(self, q: np.ndarray) -> np.ndarray:
        arr = np.asarray(q, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape != (1, self.dim):
            raise ValueError(f"query must have shape ({self.dim},) or (1, {self.dim}), got {np.shape(q)}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("query contains non-finite values")
        return np.ascontiguousarray(arr)

    def search(self, q: np.ndarray, k: int, allowed: np.ndarray | None = None) -> list[Hit]:
        """Top-``k`` hits; restricted to ``allowed`` when given, with post-filter fallback (8.4)."""
        if k <= 0 or self.ntotal == 0:
            return []
        if allowed is None:
            return self._plain(self._query(q), min(k, self.ntotal))
        allowed_arr = _as_ids(allowed)
        if allowed_arr.size == 0:
            return []
        fallback_k = min(self.ntotal, 10 * k)
        if not self.force_postfilter:
            try:
                return self.search_restricted(q, min(k, allowed_arr.size), allowed_arr)
            except Exception:
                log.warning("restricted search failed; falling back to post-filter", exc_info=True)
        return self.search_postfilter(q, k, allowed_arr, fallback_k)

    def _plain(self, xq: np.ndarray, k: int) -> list[Hit]:
        if k <= 0:
            return []
        dist, labels = self._index.search(xq, k)
        return _sorted_hits(dist[0], labels[0])

    def search_restricted(self, q: np.ndarray, k: int, allowed: np.ndarray) -> list[Hit]:
        xq = self._query(q)
        allowed_arr = _as_ids(allowed)
        k = min(k, self.ntotal)
        if k <= 0 or allowed_arr.size == 0:
            return []
        sel = faiss.IDSelectorBatch(allowed_arr)
        params = faiss.SearchParameters(sel=sel)
        dist, labels = self._index.search(xq, k, params=params)
        return _sorted_hits(dist[0], labels[0])[:k]

    def search_postfilter(
        self, q: np.ndarray, k: int, allowed: np.ndarray, fallback_k: int
    ) -> list[Hit]:
        xq = self._query(q)
        allowed_set = set(_as_ids(allowed).tolist())
        if k <= 0 or not allowed_set:
            return []
        hits = self._plain(xq, min(fallback_k, self.ntotal))
        return [h for h in hits if h.vector_id in allowed_set][:k]
