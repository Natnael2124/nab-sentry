"""Property 28: Allowed ID set equals the reference filter.

**Validates: Requirements 8.3, 8.8, 8.9, 8.13**
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from hypothesis import given, settings
from hypothesis import strategies as st

from nab_sentry.search.engine import SearchFilter
from nab_sentry.store.db import MetadataStore, NewFrame, NewVector, NewVideo

CLASSES = ("person", "car", "truck", "bus", "motorcycle", "bicycle")
CAMERAS = ("cam-a", "cam-b", "cam-c")
UNKNOWN_CAMERA = "cam-unknown"

# All generated instants are whole milliseconds inside a ~1 hour window so filter bounds
# frequently land on, just before, or just after frame timestamps.
BASE_MS = 1_700_000_000_000
WINDOW_MS = 3_600_000
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)

offsets = st.sampled_from([-12 * 60, -5 * 60, -3 * 60 - 30, 0, 5 * 60 + 30, 9 * 60 + 45, 14 * 60]).map(
    lambda m: timezone(timedelta(minutes=m))
)


def at(ms: int, tz: timezone) -> datetime:
    """Aware datetime for an integer epoch-ms instant, expressed in ``tz``."""
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
    # Frames of a video have distinct offsets in practice; keep them unique and sorted.
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
def filters_st(draw: st.DrawFn) -> tuple[SearchFilter, int | None, int | None]:
    camera = draw(st.one_of(st.none(), st.sampled_from(CAMERAS + (UNKNOWN_CAMERA,))))
    start_ms = draw(st.one_of(st.none(), bound_ms))
    end_ms = draw(st.one_of(st.none(), bound_ms))
    start = None if start_ms is None else at(start_ms, draw(offsets))
    end = None if end_ms is None else at(end_ms, draw(offsets))
    cls = draw(st.one_of(st.none(), st.sampled_from(CLASSES)))
    return SearchFilter(camera_id=camera, start=start, end=end, cls=cls), start_ms, end_ms


def build_store(videos: list[GenVideo]) -> tuple[MetadataStore, list[tuple[int, str, int, bool, GenVector]]]:
    """Populate a fresh store; return rows (vector_id, camera, abs_ms, complete, vector)."""
    store = MetadataStore(":memory:")
    rows: list[tuple[int, str, int, bool, GenVector]] = []
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
                abs_ms = v.start_ms + fr.offset_ms
                fid = store.insert_frame(
                    NewFrame(
                        video_id=vid, frame_idx=j, offset_s=fr.offset_ms / 1000,
                        abs_time=at(abs_ms, v.tz), gate_reason="motion", motion_frac=0.5,
                        thumb_path=f"t{i}_{j}.jpg",
                    )
                )
                for gv in fr.vectors:
                    if gv.kind == "frame":
                        nv = NewVector(frame_id=fid, kind="frame")
                    else:
                        nv = NewVector(frame_id=fid, kind="crop", det_class=gv.det_class,
                                       det_conf=0.9, box=(0, 0, 10, 10))
                    rows.append((store.insert_vector(nv), v.camera_id, abs_ms, v.complete, gv))
            if v.complete:
                store.finalize_video(vid, sampled=len(v.frames), passed=len(v.frames),
                                     duration_s=700.0, playback_path=f"p{i}.mp4")
    return store, rows


def reference(rows, f: SearchFilter, start_ms: int | None, end_ms: int | None) -> list[int]:
    out = []
    for vector_id, camera, abs_ms, complete, gv in rows:
        if not complete:
            continue
        if f.camera_id is not None and camera != f.camera_id:
            continue
        if start_ms is not None and abs_ms < start_ms:
            continue
        if end_ms is not None and abs_ms > end_ms:
            continue
        if f.cls is not None and not (gv.kind == "crop" and gv.det_class == f.cls):
            continue
        out.append(vector_id)
    return sorted(out)


@settings(max_examples=200, deadline=None)
@given(videos=st.lists(videos_st(), min_size=0, max_size=6), filt=filters_st())
def test_allowed_ids_equal_reference(videos, filt):
    """Feature: nab-sentry, Property 28: Allowed ID set equals the reference filter."""
    f, start_ms, end_ms = filt
    store, rows = build_store(videos)
    try:
        got = store.allowed_vector_ids(f).tolist()
        assert got == reference(rows, f, start_ms, end_ms)
        if f.camera_id == UNKNOWN_CAMERA:  # 8.13
            assert got == []
    finally:
        store.close()
