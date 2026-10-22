"""Per-camera live tracking across Segments (V2 Requirement 10.4-10.6, task 10.3)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nab_sentry.config import Config
from nab_sentry.ingest.detector import Detection
from nab_sentry.ingest.pipeline import IngestPipeline
from nab_sentry.ingest.sources import DecodedFrame
from nab_sentry.store.db import MetadataStore
from nab_sentry.store.vector_index import VectorIndex
from nab_sentry.track import TrackSink
from nab_sentry.track.bytetrack import ByteTracker
from nab_sentry.track.live import LiveTracks
from tests.fakes import FailureInjector, FakeDetector, FakeEncoder, FakePlayback, FakeSource

W, H = 64, 48
GREY = np.full((H, W, 3), 100, np.uint8)
DT = 1 / 3


def live(**kw) -> LiveTracks:
    return LiveTracks(lambda: ByteTracker(0.3, 2, 2.0), gate_width=64, **kw)


def person(x: int = 10) -> list[Detection]:
    return [Detection("person", 0.9, (x, 5, x + 20, 40))]


def feed(lt: LiveTracks, cam: str, start: float, n: int, img=GREY, x0: int = 10) -> set[int]:
    """One video of ``n`` tracked frames at 3 fps; returns the track keys assigned."""
    lt.begin_video(cam, start)
    for i in range(n):
        lt.update(DecodedFrame(i, i * DT, img), person(x0 + i))
    keys = {a.track_key for a in lt.tracker(cam).last_assignments}
    lt.end_video(start + n * DT)
    return keys


def test_live_tracks_is_a_track_sink():
    assert isinstance(live(), TrackSink)


def test_track_continues_into_next_segment_within_gap():
    lt = live(gap_reset_s=3.0)
    k1 = feed(lt, "CAM01", 1000.0, 6)
    k2 = feed(lt, "CAM01", 1000.0 + 6 * DT + 0.5, 4, x0=16)  # 0.5 s gap
    assert k1 == k2 and None not in k1
    assert lt.stats("CAM01")["gap_resets"] == 0


def test_long_gap_ends_tracks():
    lt = live(gap_reset_s=3.0)
    k1 = feed(lt, "CAM01", 1000.0, 6)
    k2 = feed(lt, "CAM01", 1000.0 + 2 + 10.0, 4, x0=16)
    assert k1.isdisjoint(k2)
    assert lt.stats("CAM01")["gap_resets"] == 1


def test_small_overlap_from_name_resolution_is_tolerated():
    """The next Segment's name is truncated to the second, so it can start ~0.8 s early."""
    lt = live()
    k1 = feed(lt, "CAM01", 1000.0, 6)
    k2 = feed(lt, "CAM01", 1000.0 + 6 * DT - 0.8, 4, x0=16)  # no exception, same track
    assert k1 == k2


def test_large_overlap_resets():
    lt = live(overlap_tolerance_s=2.0)
    k1 = feed(lt, "CAM01", 1000.0, 6)
    k2 = feed(lt, "CAM01", 1000.0 - 5.0, 4, x0=16)
    assert k1.isdisjoint(k2) and lt.stats("CAM01")["gap_resets"] == 1


def test_cameras_are_independent():
    lt = live()
    a = feed(lt, "CAM01", 1000.0, 5)
    b = feed(lt, "CAM02", 1000.0, 5)
    assert lt.tracker("CAM01") is not lt.tracker("CAM02")


def test_scene_cut_ends_tracks_also_across_segments():
    lt = live()
    k1 = feed(lt, "CAM01", 1000.0, 5)
    other = np.full((H, W, 3), 230, np.uint8)  # a completely different view
    k2 = feed(lt, "CAM01", 1000.0 + 5 * DT, 4, img=other, x0=15)  # camera moved between Segments
    assert k1.isdisjoint(k2)
    assert lt.stats("CAM01")["scene_cuts"] == 1
    k3 = feed(lt, "CAM01", 1000.0 + 9 * DT, 3, img=other, x0=19)  # same view: continues
    assert k3 == k2


def test_update_without_begin_is_an_error():
    with pytest.raises(RuntimeError):
        live().update(DecodedFrame(0, 0.0, GREY), [])


# -- rollback ----------------------------------------------------------------------------------


@settings(max_examples=40, deadline=None)
@given(st.lists(st.tuples(st.integers(1, 6), st.booleans(), st.floats(0.0, 6.0)), min_size=1, max_size=8))
def test_rolled_back_videos_leave_no_trace(videos):
    """Property: feeding videos where some roll back == feeding only the committed ones."""
    a, b = live(), live()
    t = 1000.0
    for n, ok, gap in videos:
        a.begin_video("CAM01", t)
        for i in range(n):
            a.update(DecodedFrame(i, i * DT, GREY), person(10 + i))
        if ok:
            ra = a.end_video(t + n * DT)
            b.begin_video("CAM01", t)
            for i in range(n):
                b.update(DecodedFrame(i, i * DT, GREY), person(10 + i))
            rb = b.end_video(t + n * DT)
            assert ra == rb
            t += n * DT + gap
        else:
            a.rollback_video()
    ta, tb = a.tracker("CAM01"), b.tracker("CAM01")
    if tb is None:
        assert ta is None
    else:
        assert [(x.key, x.box, x.hits) for x in ta.tracks] == [(x.key, x.box, x.hits) for x in tb.tracks]
        assert a.stats("CAM01") == b.stats("CAM01")


# -- pipeline ----------------------------------------------------------------------------------


def _tracker_state(lt: LiveTracks, cam: str):
    tr = lt.tracker(cam)
    return (tr._next_key, tr._now, tr._base_t,
            [(x.key, x.cls, x.box, x.velocity, x.hits, x.first_t, x.last_t, x.confirmed) for x in tr.tracks],
            lt.stats(cam))


def _ingest(root: Path, names: list[str], checkpoint=None, retry: str | None = None) -> LiveTracks:
    rng = np.random.default_rng(1)
    images = [rng.integers(0, 256, (H, W, 3), dtype=np.uint8) for _ in range(20)]  # 4 s, always moving
    cfg = Config(root=root, enable_detector=True, sample_rate=1.0, track_rate=3.0)
    cfg.videos_dir.mkdir(parents=True)
    for n in names:
        (cfg.videos_dir / n).write_bytes(n.encode())
    lt = live()
    # Position from the frame's pixels, not the call count: a failed attempt must not
    # shift what the retry sees.
    det = FakeDetector(lambda frame, i: person(10 + int(frame[0, 0, 0]) % 5))
    db = MetadataStore(cfg.db_path)
    try:
        pipe = IngestPipeline(cfg, db, VectorIndex(), FakeEncoder(), det,
                              open_source=lambda p: FakeSource(images, fps=5.0, width=W, height=H, path=p),
                              start_playback=FakePlayback(), tracker=lt, checkpoint=checkpoint)
        res = pipe.ingest_paths([cfg.videos_dir]).results
        if retry is not None:
            assert [r.status for r in res][-1] in ("failed", "unreadable")
            assert pipe.ingest_video(cfg.videos_dir / retry).status == "ingested"
        else:
            assert {r.status for r in res} == {"ingested"}
    finally:
        db.close()
    return lt


@settings(max_examples=18, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.sampled_from(["frame", "thumbnail", "embed", "playback_wait", "finalize", "index_add",
                        "index_save", "after_index_save", "commit"]),
       st.integers(1, 3))
def test_pipeline_failure_rolls_tracks_back_and_retry_matches_clean_run(step, nth):
    """Segment 2 fails at ``step`` and is retried (as watch mode does): the tracks end up
    exactly as after a clean run of both Segments (V2 Requirement 10.6)."""
    names = ["CAM-A_20250101T080000.mp4", "CAM-A_20250101T080004.mp4"]  # contiguous 4 s Segments
    with tempfile.TemporaryDirectory() as tmp:
        clean = _ingest(Path(tmp) / "clean", names)
        failed = _ingest(Path(tmp) / "failed", names,
                         checkpoint=FailureInjector(step, nth=nth if step in ("frame", "thumbnail") else 1,
                                                    video=names[1]),
                         retry=names[1])
    assert _tracker_state(failed, "CAM-A") == _tracker_state(clean, "CAM-A")
    assert _tracker_state(clean, "CAM-A")[3], "the person should be tracked"
