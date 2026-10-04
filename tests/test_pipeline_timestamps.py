"""Property 7: Absolute timestamp equals start plus offset (task 10.4).

Year ranges: naive start times are converted to the local zone with ``astimezone()``, which on
Windows goes through ``localtime``/``mktime`` and rejects instants outside roughly 1970-3000, so
naive years are restricted to 1971-2100. Aware start times never touch ``localtime`` and use
1971-3000 (the range in the design). Naive local times inside a DST gap (wall-clock times that do
not exist) are excluded with ``assume()``.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from nab_sentry.config import Config
from nab_sentry.ingest.metadata import to_aware_local
from nab_sentry.ingest.motion import GateDecision
from nab_sentry.ingest.pipeline import IngestPipeline, build_new_frame
from nab_sentry.store.db import MetadataStore, NewVideo
from nab_sentry.store.vector_index import VectorIndex
from tests.fakes import FakeEncoder, FakePlayback, FakeSource

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
US = timedelta(microseconds=1)
TOL_US = 1000  # 1 ms


def _exact_us(aware: datetime) -> int:
    """Exact epoch microseconds via integer timedelta arithmetic (no float rounding)."""
    return (aware - EPOCH) // US


# --- strategies ---------------------------------------------------------------------------------

fixed_zones = st.one_of(
    st.just(timezone.utc),
    st.integers(min_value=-(23 * 60 + 59), max_value=23 * 60 + 59).map(
        lambda m: timezone(timedelta(minutes=m))
    ),
)

aware_starts = st.datetimes(
    min_value=datetime(1971, 1, 1), max_value=datetime(3000, 12, 31), timezones=fixed_zones
)

naive_starts = st.datetimes(min_value=datetime(1971, 1, 2), max_value=datetime(2100, 12, 31))

start_times = st.one_of(aware_starts, naive_starts)

# Offsets in [0, 86400] s at millisecond precision.
offsets_ms = st.integers(min_value=0, max_value=86_400_000)


def _not_in_dst_gap(naive: datetime) -> bool:
    """A naive local time exists iff it survives a local round trip."""
    return datetime.fromtimestamp(naive.timestamp()) == naive


def _store_with_video(start: datetime) -> tuple[MetadataStore, int]:
    db = MetadataStore(":memory:")
    db.upsert_camera("CAM-A", "Camera A")
    vid = db.insert_video(
        NewVideo(
            camera_id="CAM-A", src_path="x.mp4", src_hash="0" * 64, start_ts=start, fps=25.0,
            width=64, height=48, est_duration_s=86_400.0,
        )
    )
    return db, vid


# --- Property 7 -----------------------------------------------------------------------------------


@settings(max_examples=300, deadline=None)
@given(start=start_times, off_ms=offsets_ms, frame_idx=st.integers(0, 10_000_000))
def test_property7_abs_timestamp_equals_start_plus_offset(
    start: datetime, off_ms: int, frame_idx: int
) -> None:
    """**Validates: Requirements 1.12, 2.3, 6.2**"""
    if start.tzinfo is None:
        assume(_not_in_dst_gap(start))
        expected_start = start.astimezone()  # 1.12: naive -> local zone
    else:
        expected_start = start
    expected_us = _exact_us(expected_start) + off_ms * 1000
    offset_s = off_ms / 1000

    nf = build_new_frame(
        video_id=1, start_time=start, frame_idx=frame_idx, offset_s=offset_s,
        decision=GateDecision(True, "motion", 0.5), thumb_path="t.jpg",
    )
    assert nf.abs_time.tzinfo is not None and nf.abs_time.utcoffset() is not None
    assert abs(_exact_us(nf.abs_time) - expected_us) <= TOL_US
    # Aware start: result keeps the start's offset (fixed-offset zones only here).
    if start.tzinfo is not None:
        assert nf.abs_time.utcoffset() == start.utcoffset()

    db, vid = _store_with_video(start)
    try:
        nf = build_new_frame(
            video_id=vid, start_time=start, frame_idx=frame_idx, offset_s=offset_s,
            decision=GateDecision(True, "motion", 0.5), thumb_path="t.jpg",
        )
        db.insert_frame(nf)
        abs_ts, abs_epoch_ms = db._conn.execute(
            "SELECT abs_ts, abs_epoch_ms FROM frames WHERE video_id = ?", (vid,)
        ).fetchone()
    finally:
        db.close()

    assert abs(abs_epoch_ms * 1000 - expected_us) <= TOL_US
    parsed = datetime.fromisoformat(abs_ts)
    assert parsed.utcoffset() is not None
    assert abs(_exact_us(parsed) - expected_us) <= TOL_US


# --- end to end through IngestPipeline -------------------------------------------------------------

W, H = 64, 48


def _frames(n: int) -> list[np.ndarray]:
    out = []
    for i in range(n):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        x = (i * 9) % (W - 12)
        img[10:22, x:x + 12] = 255
        out.append(img)
    return out


@settings(
    max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
@given(
    start=st.datetimes(min_value=datetime(1971, 1, 2), max_value=datetime(2100, 12, 31)).map(
        lambda d: d.replace(microsecond=0)
    ),
    fps=st.sampled_from([2.0, 4.0, 7.5, 12.0, 29.97]),
    n=st.integers(min_value=4, max_value=40),
)
def test_property7_end_to_end_frames_rows(start: datetime, fps: float, n: int) -> None:
    """**Validates: Requirements 1.12, 2.3, 6.2**

    Start time comes from a filename (naive -> local zone); every stored frame row satisfies
    ``abs_epoch_ms == start_epoch_ms + round(offset_s * 1000)`` within 1 ms.
    """
    assume(_not_in_dst_gap(start))
    name = f"CAM-A_{start:%Y%m%dT%H%M%S}.mp4"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "ws"
        cfg = Config(
            root=root, sample_rate=2.0, keyframe_interval_s=1.0, enable_detector=False,
            batch_size=4,
        )  # type: ignore[arg-type]
        videos = root / "videos"
        videos.mkdir(parents=True)
        (videos / name).write_bytes(f"video-{start.isoformat()}-{fps}-{n}".encode())
        db = MetadataStore(cfg.db_path)
        try:
            pipe = IngestPipeline(
                cfg, db, VectorIndex(), FakeEncoder(batch_size=cfg.batch_size), None,
                open_source=lambda p: FakeSource(_frames(n), fps=fps, width=W, height=H, path=p),
                start_playback=FakePlayback(),
            )
            report = pipe.ingest_paths([videos])
            assert [r.status for r in report.results] == ["ingested"]

            (start_epoch_ms,) = db._conn.execute("SELECT start_epoch_ms FROM videos").fetchone()
            assert abs(start_epoch_ms * 1000 - _exact_us(start.astimezone())) <= TOL_US
            rows = db._conn.execute("SELECT offset_s, abs_ts, abs_epoch_ms FROM frames").fetchall()
            assert rows, "at least the first frame passes the gate"
            for offset_s, abs_ts, abs_epoch_ms in rows:
                expected_ms = start_epoch_ms + round(offset_s * 1000)
                assert abs(abs_epoch_ms - expected_ms) <= 1
                parsed_ms = _exact_us(datetime.fromisoformat(abs_ts)) / 1000
                assert abs(parsed_ms - expected_ms) <= 1
        finally:
            db.close()
