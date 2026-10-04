"""Unit tests for SearchEngine readiness and the unknown-camera case.

Missing encoder and inconsistent FAISS/DB ID sets raise ``SearchUnavailable`` (8.15); an unknown
camera returns ``[]`` without touching the index (8.13).

_Requirements: 8.13, 8.15_
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from nab_sentry.config import Config
from nab_sentry.search.engine import (
    IndexInconsistent,
    SearchEngine,
    SearchFilter,
    SearchUnavailable,
)
from nab_sentry.store.db import MetadataStore, NewFrame, NewVector, NewVideo
from nab_sentry.store.vector_index import VectorIndex
from tests.fakes import RED_DIRECTION, FakeEncoder

T0 = datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc)


class SpyIndex:
    """Wraps a real ``VectorIndex`` and records ``search`` calls."""

    def __init__(self, inner: VectorIndex) -> None:
        self.inner = inner
        self.search_calls = 0

    def ids(self) -> np.ndarray:
        return self.inner.ids()

    def search(self, q, k, allowed=None):
        self.search_calls += 1
        return self.inner.search(q, k, allowed)


def _build_store(n: int = 3) -> tuple[MetadataStore, list[int]]:
    """Camera 'cam1' with ``n`` red frame vectors; returns the store and the vector IDs."""
    db = MetadataStore(":memory:")
    db.upsert_camera("cam1", "Loading Dock")
    vid = db.insert_video(NewVideo("cam1", "a.mp4", "h1", T0, 10.0, 16, 12, 60.0))
    ids = []
    for i in range(n):
        off = float(i)
        fid = db.insert_frame(NewFrame(vid, i * 10, off, T0 + timedelta(seconds=off),
                                       "motion", 0.5, f"t{i}.jpg"))
        ids.append(db.insert_vector(NewVector(fid, "frame")))
    db.finalize_video(vid, sampled=60, passed=n, duration_s=60.0, playback_path="a_play.mp4")
    return db, ids


def _index_with(ids: list[int]) -> VectorIndex:
    index = VectorIndex()
    if ids:
        vecs = np.stack([RED_DIRECTION] * len(ids)).astype(np.float32)
        index.add(np.array(ids, dtype=np.int64), vecs)
    return index


def _engine(db, index, encoder="default") -> SearchEngine:
    enc = FakeEncoder(colour_mode=True) if encoder == "default" else encoder
    return SearchEngine(Config(root=Path(".")), db, index, enc)


def test_missing_encoder_raises_search_unavailable():
    db, ids = _build_store()
    try:
        eng = _engine(db, _index_with(ids), encoder=None)
        with pytest.raises(SearchUnavailable):
            eng.check_ready()
        with pytest.raises(SearchUnavailable):
            eng.search("red", SearchFilter(), 10)
    finally:
        db.close()


def test_orphan_index_id_raises_index_inconsistent():
    db, ids = _build_store()
    try:
        eng = _engine(db, _index_with(ids + [max(ids) + 1000]))
        with pytest.raises(IndexInconsistent) as exc:
            eng.check_ready()
        assert isinstance(exc.value, SearchUnavailable)
        assert "--repair" in str(exc.value)
        with pytest.raises(IndexInconsistent):
            eng.search("red", SearchFilter(), 10)
    finally:
        db.close()


def test_db_vector_missing_from_index_raises_index_inconsistent():
    db, ids = _build_store()
    try:
        eng = _engine(db, _index_with(ids[:-1]))
        with pytest.raises(IndexInconsistent) as exc:
            eng.check_ready()
        assert isinstance(exc.value, SearchUnavailable)
        assert "--repair" in str(exc.value)
    finally:
        db.close()


def test_empty_store_and_index_is_ready_and_returns_empty():
    db = MetadataStore(":memory:")
    try:
        eng = _engine(db, _index_with([]))
        eng.check_ready()  # no exception
        assert eng.search("red", SearchFilter(), 10) == []
    finally:
        db.close()


def test_consistent_store_is_ready():
    db, ids = _build_store()
    try:
        eng = _engine(db, _index_with(ids))
        eng.check_ready()
        assert eng.search("red", SearchFilter(), 10) != []
    finally:
        db.close()


def test_unknown_camera_returns_empty_without_index_search():
    db, ids = _build_store()
    try:
        spy = SpyIndex(_index_with(ids))
        eng = _engine(db, spy)
        assert eng.search("red", SearchFilter(camera_id="no-such-camera"), 10) == []
        assert spy.search_calls == 0
    finally:
        db.close()
