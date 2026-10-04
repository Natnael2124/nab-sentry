"""Property 29: Filtered hits are always allowed.

**Validates: Requirements 8.3, 8.5**
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from nab_sentry.config import Config
from nab_sentry.search.engine import VALID_CLASSES, SearchEngine, SearchFilter
from nab_sentry.store.db import MetadataStore, NewFrame, NewVector, NewVideo
from nab_sentry.store.vector_index import VectorIndex

DIM = 8
CLASSES = tuple(sorted(VALID_CLASSES))
CAMERAS = ("cam-a", "cam-b", "cam-c")
UNKNOWN_CAMERA = "cam-unknown"

BASE_MS = 1_700_000_000_000
WINDOW_MS = 3_600_000
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

offsets = st.sampled_from([-5 * 60, 0, 5 * 60 + 30, 14 * 60]).map(lambda m: timezone(timedelta(minutes=m)))


def at(ms: int, tz: timezone) -> datetime:
    return (EPOCH + timedelta(milliseconds=ms)).astimezone(tz)


@dataclass(frozen=True)
class GenVector:
    kind: str
    det_class: str | None


@dataclass(frozen=True)
class GenFrame:
    offset_ms: int
    vectors: tuple[GenVector, ...]


@dataclass(frozen=True)
class GenVideo:
    camera_id: str
    tz: timezone
    start_ms: int
    complete: bool
    frames: tuple[GenFrame, ...]


vectors_st = st.one_of(
    st.just(GenVector("frame", None)),
    st.sampled_from(CLASSES).map(lambda c: GenVector("crop", c)),
)

frames_st = st.builds(
    GenFrame,
    offset_ms=st.integers(0, 600_000),
    vectors=st.lists(vectors_st, min_size=0, max_size=4).map(tuple),
)


@st.composite
def videos_st(draw: st.DrawFn) -> GenVideo:
    raw = draw(st.lists(frames_st, min_size=0, max_size=5))
    seen: dict[int, GenFrame] = {}
    for fr in raw:
        seen.setdefault(fr.offset_ms, fr)
    return GenVideo(
        camera_id=draw(st.sampled_from(CAMERAS)),
        tz=draw(offsets),
        start_ms=BASE_MS + draw(st.integers(0, WINDOW_MS)),
        complete=draw(st.booleans()),
        frames=tuple(sorted(seen.values(), key=lambda fr: fr.offset_ms)),
    )


bound_ms = st.integers(BASE_MS - 60_000, BASE_MS + WINDOW_MS + 660_000)


@st.composite
def filters_st(draw: st.DrawFn) -> SearchFilter:
    """Valid filters only: start <= end whenever both are given."""
    camera = draw(st.one_of(st.none(), st.sampled_from(CAMERAS + (UNKNOWN_CAMERA,))))
    start_ms = draw(st.one_of(st.none(), bound_ms))
    end_ms = draw(st.one_of(st.none(), bound_ms))
    if start_ms is not None and end_ms is not None and start_ms > end_ms:
        start_ms, end_ms = end_ms, start_ms
    start = None if start_ms is None else at(start_ms, draw(offsets))
    end = None if end_ms is None else at(end_ms, draw(offsets))
    cls = draw(st.one_of(st.none(), st.sampled_from(CLASSES)))
    return SearchFilter(camera_id=camera, start=start, end=end, cls=cls)


def build_store(videos: list[GenVideo]) -> MetadataStore:
    store = MetadataStore(":memory:")
    with store.transaction():
        for cam in CAMERAS:
            store.upsert_camera(cam, f"Label {cam}")
        for i, v in enumerate(videos):
            vid = store.insert_video(
                NewVideo(
                    camera_id=v.camera_id, src_path=f"v{i}.mp4", src_hash=f"{i:064x}",
                    start_ts=at(v.start_ms, v.tz), fps=25.0, width=640, height=480,
                    est_duration_s=700.0,
                )
            )
            for j, fr in enumerate(v.frames):
                fid = store.insert_frame(
                    NewFrame(
                        video_id=vid, frame_idx=j, offset_s=fr.offset_ms / 1000,
                        abs_time=at(v.start_ms + fr.offset_ms, v.tz), gate_reason="motion",
                        motion_frac=0.5, thumb_path=f"t{i}_{j}.jpg",
                    )
                )
                for gv in fr.vectors:
                    if gv.kind == "frame":
                        nv = NewVector(frame_id=fid, kind="frame")
                    else:
                        nv = NewVector(frame_id=fid, kind="crop", det_class=gv.det_class,
                                       det_conf=0.9, box=(0, 0, 10, 10))
                    store.insert_vector(nv)
            if v.complete:
                store.finalize_video(vid, sampled=len(v.frames), passed=len(v.frames),
                                     duration_s=700.0, playback_path=f"p{i}.mp4")
    return store


def unit_rows(rng: np.random.Generator, n: int) -> np.ndarray:
    m = rng.standard_normal((n, DIM)).astype(np.float32)
    m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-6)
    return m


@settings(max_examples=150, deadline=None)
@given(
    videos=st.lists(videos_st(), min_size=0, max_size=6),
    f=filters_st(),
    seed=st.integers(0, 2**32 - 1),
    top_k=st.integers(1, 30),
)
def test_filtered_hits_are_allowed(videos, f, seed, top_k):
    """Feature: nab-sentry, Property 29: Filtered hits are always allowed."""
    rng = np.random.default_rng(seed)
    store = build_store(videos)
    try:
        ids = np.asarray(store.vector_ids(), dtype=np.int64)
        index = VectorIndex(DIM)
        index.add(ids, unit_rows(rng, ids.size))
        qvec = unit_rows(rng, 1)[0]
        cfg = Config(top_k=top_k)
        allowed = set(store.allowed_vector_ids(f).tolist())
        db_ids = set(ids.tolist())

        for force in (False, True):
            index.force_postfilter = force
            engine = SearchEngine(cfg, store, index, encoder=None)
            events = engine.search_vector(qvec, f, "person")
            hit_ids = [h.vector_id for e in events for h in e.hits]

            assert len(hit_ids) == len(set(hit_ids)), "a hit appears in more than one Event"
            assert len(hit_ids) <= top_k
            if f.is_empty():
                assert set(hit_ids) <= db_ids
            else:
                assert set(hit_ids) <= allowed, (force, sorted(set(hit_ids) - allowed))
                if not allowed:
                    assert events == []
    finally:
        store.close()
