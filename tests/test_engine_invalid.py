"""Property 33: Invalid filters are rejected before searching.

*For any* time range with ``start > end`` and *for any* class string not in the Target_Classes,
``SearchEngine.search`` raises ``FilterError`` naming ``time_range`` or ``cls`` respectively, and a
spy index records zero search calls; *for any* filter whose Allowed_ID_Set is empty, ``search``
returns ``[]`` with zero index calls.

When both parts are invalid, ``validate_filter`` checks the time range first, so the reported
part is ``time_range``.

**Validates: Requirements 8.12, 8.14**
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st

from nab_sentry.config import Config
from nab_sentry.search.engine import VALID_CLASSES, FilterError, SearchEngine, SearchFilter
from nab_sentry.store.db import MetadataStore, NewFrame, NewVector, NewVideo
from nab_sentry.store.vector_index import VectorIndex
from tests.fakes import BLUE_DIRECTION, RED_DIRECTION, FakeEncoder

T0 = datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc)
DATA_END = T0 + timedelta(seconds=30)


# ---------------------------------------------------------------------------- spies


class SpyIndex:
    """Wraps a real ``VectorIndex`` and records ``search``/``ids`` calls."""

    def __init__(self, inner: VectorIndex) -> None:
        self.inner = inner
        self.search_calls: list[int] = []
        self.ids_calls = 0

    def ids(self) -> np.ndarray:
        self.ids_calls += 1
        return self.inner.ids()

    def search(self, q, k, allowed=None):
        self.search_calls.append(int(k))
        return self.inner.search(q, k, allowed)


class SpyStore:
    """Wraps a real ``MetadataStore`` and records ``allowed_vector_ids``/``hit_rows`` calls."""

    def __init__(self, inner: MetadataStore) -> None:
        self.inner = inner
        self.allowed_calls: list[SearchFilter] = []
        self.hit_rows_calls = 0

    def allowed_vector_ids(self, f):
        self.allowed_calls.append(f)
        return self.inner.allowed_vector_ids(f)

    def hit_rows(self, ids):
        self.hit_rows_calls += 1
        return self.inner.hit_rows(ids)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _build_store() -> tuple[MetadataStore, VectorIndex]:
    """Camera 'cam1': frame vectors at 0, 1, 2 s (red) and 30 s (blue), a 'person' crop at 1 s."""
    db = MetadataStore(":memory:")
    index = VectorIndex()
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
    return db, index


def _engine():
    db, index = _build_store()
    spy_db, spy_index, encoder = SpyStore(db), SpyIndex(index), FakeEncoder(colour_mode=True)
    eng = SearchEngine(Config(root=Path(".")), spy_db, spy_index, encoder)
    return eng, spy_db, spy_index, encoder, db


# ---------------------------------------------------------------------------- strategies

ZONES = [
    None,  # naive -> local time zone
    timezone.utc,
    timezone(timedelta(hours=5, minutes=30)),
    timezone(timedelta(hours=-8)),
    timezone(timedelta(hours=14)),
    timezone(timedelta(hours=-12)),
    timezone(timedelta(hours=9, minutes=45)),
]

# 2001..2037 keeps naive ``timestamp()`` (local mktime) well-defined on Windows.
any_dt = st.datetimes(
    min_value=datetime(2001, 1, 1), max_value=datetime(2037, 12, 31),
    timezones=st.sampled_from(ZONES),
)


def _ts(dt: datetime) -> float:
    """Independent oracle: naive ``timestamp()`` interprets the value in local time."""
    return dt.timestamp()


@st.composite
def inverted_range(draw):
    start, end = draw(any_dt), draw(any_dt)
    assume(_ts(start) > _ts(end))
    return start, end


@st.composite
def non_inverted_range(draw):
    kind = draw(st.sampled_from(["none", "start", "end", "both"]))
    if kind == "none":
        return None, None
    if kind == "start":
        return draw(any_dt), None
    if kind == "end":
        return None, draw(any_dt)
    a, b = draw(any_dt), draw(any_dt)
    return (a, b) if _ts(a) <= _ts(b) else (b, a)


_names = sorted(VALID_CLASSES)
case_and_space_variants = st.sampled_from(_names).flatmap(
    lambda n: st.sampled_from([
        n.upper(), n.title(), n.capitalize(), " " + n, n + " ", "\t" + n, n + "\n",
        " " + n + " ", n + "s",
    ])
)
invalid_cls = st.one_of(case_and_space_variants, st.text(max_size=20)).filter(
    lambda s: s not in VALID_CLASSES
)
camera_ids = st.one_of(st.none(), st.just("cam1"), st.text(max_size=10))
queries = st.sampled_from(["red", "blue square", "person at the dock", "x"])


@st.composite
def invalid_filter(draw):
    """(filter, expected FilterError part)."""
    kind = draw(st.sampled_from(["time_range", "cls", "both"]))
    camera = draw(camera_ids)
    if kind == "time_range":
        start, end = draw(inverted_range())
        cls = draw(st.one_of(st.none(), st.sampled_from(_names)))
        return SearchFilter(camera_id=camera, start=start, end=end, cls=cls), "time_range"
    if kind == "cls":
        start, end = draw(non_inverted_range())
        return SearchFilter(camera_id=camera, start=start, end=end, cls=draw(invalid_cls)), "cls"
    start, end = draw(inverted_range())
    # Both parts invalid: the time range is checked first.
    return SearchFilter(camera_id=camera, start=start, end=end, cls=draw(invalid_cls)), "time_range"


# Valid filters whose Allowed_ID_Set is empty for the fixture data.
_before = T0 - timedelta(days=1)
_after = DATA_END + timedelta(days=1)
empty_set_filter = st.one_of(
    # Unknown camera (8.13).
    st.builds(SearchFilter, camera_id=st.text(max_size=10).filter(lambda c: c != "cam1")),
    # Time window entirely before / after the data.
    st.builds(SearchFilter, start=st.none(), end=st.just(T0 - timedelta(seconds=1))),
    st.builds(SearchFilter, start=st.just(DATA_END + timedelta(seconds=1)), end=st.none()),
    st.builds(SearchFilter, start=st.just(_before), end=st.just(_before + timedelta(hours=1))),
    st.builds(SearchFilter, start=st.just(_after), end=st.just(_after + timedelta(hours=1))),
    # A valid class with no crops in the data.
    st.builds(SearchFilter, camera_id=st.sampled_from([None, "cam1"]),
              cls=st.sampled_from([n for n in _names if n != "person"])),
)


# ---------------------------------------------------------------------------- properties


@given(fp=invalid_filter(), query=queries, limit=st.integers(min_value=1, max_value=50))
def test_invalid_filter_rejected_before_any_work(fp, query, limit):
    """Property 33 (8.14): FilterError with the right part; no encode, no allowed set, no index."""
    f, part = fp
    eng, spy_db, spy_index, encoder, db = _engine()
    try:
        try:
            eng.search(query, f, limit)
        except FilterError as e:
            assert e.part == part
        else:
            raise AssertionError(f"search accepted invalid filter {f!r}")

        # search_vector validates too, independently of search().
        try:
            eng.search_vector(RED_DIRECTION.copy(), f, query)
        except FilterError as e:
            assert e.part == part
        else:
            raise AssertionError(f"search_vector accepted invalid filter {f!r}")

        assert spy_index.search_calls == []
        assert spy_index.ids_calls == 0  # rejected even before the readiness check
        assert encoder.text_calls == []
        assert spy_db.allowed_calls == []
        assert spy_db.hit_rows_calls == 0
    finally:
        db.close()


@given(f=empty_set_filter, query=queries, limit=st.integers(min_value=1, max_value=50))
def test_empty_allowed_set_returns_empty_without_index_call(f, query, limit):
    """Property 33 (8.12): an empty Allowed_ID_Set gives ``[]`` and zero index searches."""
    eng, spy_db, spy_index, encoder, db = _engine()
    try:
        assert db.allowed_vector_ids(f).size == 0  # precondition on the real store
        assert eng.search(query, f, limit) == []
        assert spy_index.search_calls == []
        assert spy_db.allowed_calls == [f]
        assert spy_db.hit_rows_calls == 0
        assert encoder.text_calls == [query.strip()]
    finally:
        db.close()


def test_both_parts_invalid_reports_time_range():
    eng, _, spy_index, encoder, db = _engine()
    try:
        f = SearchFilter(start=T0, end=T0 - timedelta(microseconds=1), cls="Person")
        try:
            eng.search("red", f, 10)
        except FilterError as e:
            assert e.part == "time_range"
        else:
            raise AssertionError("expected FilterError")
        assert spy_index.search_calls == [] and encoder.text_calls == []
    finally:
        db.close()


def test_mixed_zone_inversion_detected_by_instant_not_wall_clock():
    """20:00+14:00 is 06:00Z, earlier than 13:00Z although its wall clock is later."""
    eng, _, spy_index, _, db = _engine()
    try:
        start = datetime(2024, 5, 1, 20, 0, tzinfo=timezone(timedelta(hours=14)))
        end = datetime(2024, 5, 1, 13, 0, tzinfo=timezone.utc)  # window contains the data
        eng.search("red", SearchFilter(start=start, end=end), 10)  # valid: start < end
        try:
            eng.search("red", SearchFilter(start=end, end=start), 10)
        except FilterError as e:
            assert e.part == "time_range"
        else:
            raise AssertionError("expected FilterError")
        assert len(spy_index.search_calls) == 1  # only the valid search reached the index
    finally:
        db.close()
