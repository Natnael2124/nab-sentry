"""Fast sanity checks for SearchEngine (spec tests live in tasks 11.9-11.11)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from nab_sentry.config import Config
from nab_sentry.search.engine import (
    FilterError,
    IndexInconsistent,
    SearchEngine,
    SearchFilter,
    SearchUnavailable,
    validate_filter,
)
from nab_sentry.store.db import MetadataStore, NewFrame, NewVector, NewVideo
from nab_sentry.store.vector_index import VectorIndex
from tests.fakes import BLUE_DIRECTION, RED_DIRECTION, FakeEncoder

T0 = datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc)


class SpyIndex:
    def __init__(self, inner: VectorIndex) -> None:
        self.inner = inner
        self.search_calls = 0

    def ids(self) -> np.ndarray:
        return self.inner.ids()

    def search(self, q, k, allowed=None):
        self.search_calls += 1
        return self.inner.search(q, k, allowed)


def _build(db: MetadataStore, index: VectorIndex) -> None:
    """Camera 'cam1' ('Loading Dock'): frame vectors at 0,1,2 s (red) and 30 s (blue), plus a
    red 'person' crop at 1 s."""
    db.upsert_camera("cam1", "Loading Dock")
    vid = db.insert_video(NewVideo("cam1", "a.mp4", "h1", T0, 10.0, 16, 12, 60.0))
    ids, vecs = [], []
    for off, vec in [(0.0, RED_DIRECTION), (1.0, RED_DIRECTION), (2.0, RED_DIRECTION),
                     (30.0, BLUE_DIRECTION)]:
        fid = db.insert_frame(NewFrame(vid, int(off * 10), off, T0 + timedelta(seconds=off),
                                       "motion", 0.5, f"t{int(off)}.jpg"))
        ids.append(db.insert_vector(NewVector(fid, "frame")))
        vecs.append(vec)
        if off == 1.0:
            ids.append(db.insert_vector(NewVector(fid, "crop", "person", 0.9, (0, 0, 4, 4))))
            vecs.append(RED_DIRECTION)
    db.finalize_video(vid, sampled=60, passed=4, duration_s=60.0, playback_path="a_play.mp4")
    index.add(np.array(ids, dtype=np.int64), np.stack(vecs).astype(np.float32))


@pytest.fixture
def setup(tmp_path):
    db = MetadataStore(":memory:")
    index = VectorIndex()
    _build(db, index)
    spy = SpyIndex(index)
    eng = SearchEngine(Config(root=tmp_path), db, spy, FakeEncoder(colour_mode=True))
    yield eng, spy, db
    db.close()


def test_validate_filter_parts():
    with pytest.raises(FilterError) as e:
        validate_filter(SearchFilter(start=T0, end=T0 - timedelta(seconds=1)))
    assert e.value.part == "time_range"
    with pytest.raises(FilterError) as e:
        validate_filter(SearchFilter(cls="dragon"))
    assert e.value.part == "cls"
    validate_filter(SearchFilter(start=T0, end=T0, cls="person"))


def test_unfiltered_search_clusters_and_boosts(setup):
    eng, _, _ = setup
    events = eng.search("red dock", SearchFilter(), 20)
    assert events[0].camera_id == "cam1"
    # Red hits at 0,1,1,2 s merge into one event; the label boost applies ("dock").
    top = events[0]
    assert sorted(h.frame_offset_s for h in top.hits) == [0.0, 1.0, 1.0, 2.0]
    assert top.start_s == 0.0 and top.end_s == pytest.approx(5.0)
    assert top.score == pytest.approx(1.0 + eng.cfg.label_boost)
    assert len(eng.search("red dock", SearchFilter(), 1)) == 1


def test_class_filter_restricts_to_crops(setup):
    eng, _, db = setup
    events = eng.search("red", SearchFilter(cls="person"), 20)
    allowed = set(db.allowed_vector_ids(SearchFilter(cls="person")).tolist())
    assert events and all(h.vector_id in allowed for e in events for h in e.hits)


def test_invalid_and_empty_filters_skip_index(setup):
    eng, spy, _ = setup
    with pytest.raises(FilterError):
        eng.search("red", SearchFilter(cls="dragon"), 20)
    assert eng.search("red", SearchFilter(camera_id="nope"), 20) == []
    assert spy.search_calls == 0


def test_check_ready(setup, tmp_path):
    eng, _, db = setup
    eng.check_ready()
    with pytest.raises(SearchUnavailable):
        SearchEngine(Config(root=tmp_path), db, VectorIndex(), None).check_ready()
    with pytest.raises(IndexInconsistent):
        SearchEngine(Config(root=tmp_path), db, VectorIndex(), FakeEncoder()).check_ready()
