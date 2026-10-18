"""Tracking-rate path (V2 Requirement 9.2-9.3, tasks 9.3-9.4).

* ``DualSampler``: the ``on_sample`` frames are exactly the V1 sample-rate selection, the
  tracking-rate frames exactly the selection at the tracking rate, nothing else is yielded.
* Property 23 (V1 search unchanged): ingesting with a ``TrackSink`` writes the same videos,
  frames and vectors rows (and the same index vectors) as ingesting without one.
* The tracker sees frames in time order, only from the two cadences, and the detector runs on
  a tracking-only frame only while the scene is active.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nab_sentry.config import Config
from nab_sentry.ingest.detector import Detection
from nab_sentry.ingest.pipeline import IngestPipeline
from nab_sentry.ingest.sampler import DualSampler, sample_indices
from nab_sentry.ingest.sources import DecodedFrame
from nab_sentry.store.db import MetadataStore
from nab_sentry.store.vector_index import VectorIndex
from nab_sentry.track import TrackSink
from tests.fakes import FakeEncoder, FakePlayback, FakeSource

W, H = 32, 24
NAME = "CAM-A_20250101T080000.mp4"


# -- DualSampler -----------------------------------------------------------------------------


@given(st.integers(0, 120), st.sampled_from([5.0, 10.0, 12.5, 29.97, 30.0]),
       st.sampled_from([0.5, 1.0, 2.0]), st.sampled_from([1.0, 2.0, 3.0, 5.0, 7.0]),
       st.sets(st.integers(0, 119), max_size=20))
def test_dual_sampler_matches_both_reference_selections(n, fps, rate, track, undecodable):
    track = max(track, rate)
    frames = [DecodedFrame(i, i / fps, np.zeros((1, 1, 3), np.uint8))
              for i in range(n) if i not in undecodable]
    dual = DualSampler(rate, track)
    out = list(dual.select(frames))
    dec = lambda i: i not in undecodable  # noqa: E731
    want_sample = sample_indices(n, fps, rate, dec)
    want_track = sample_indices(n, fps, track, dec)
    assert [f.index for f, on in out if on] == want_sample
    assert [f.index for f, _ in out] == sorted(set(want_sample) | set(want_track))
    assert dual.selected_count == len(want_sample)
    assert dual.track_count == len(want_track)


# -- pipeline --------------------------------------------------------------------------------


def detect_from_pixels(frame: np.ndarray) -> list[Detection]:
    """Deterministic detections from the frame content (not from the call count), so a frame
    gets the same detections however often the detector runs."""
    h = hashlib.sha256(frame.tobytes()).digest()
    out = []
    for k in range(h[0] % 4):
        x1, y1 = h[1 + 4 * k] % (W - 4), h[2 + 4 * k] % (H - 4)
        x2, y2 = x1 + 2 + h[3 + 4 * k] % 8, y1 + 2 + h[4 + 4 * k] % 8
        conf = (h[20 + k] % 100) / 100
        out.append(Detection("person" if k % 2 == 0 else "car", conf, (x1, y1, x2, y2),
                             high=conf >= 0.35))
    return out


class PixelDetector:
    def __init__(self) -> None:
        self.calls: list[float] = []  # mean pixel value per call, just to count

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.calls.append(float(frame.mean()))
        return detect_from_pixels(frame)


class FakeTrackSink:
    """Records updates; ``active`` follows a scripted pattern over update counts."""

    def __init__(self, pattern: list[bool]) -> None:
        self.pattern = pattern
        self.updates: list[tuple[int, float, int]] = []  # (frame index, offset, n detections)

    @property
    def active(self) -> bool:
        if not self.pattern:
            return False
        return self.pattern[len(self.updates) % len(self.pattern)]

    def update(self, frame, detections) -> None:
        self.updates.append((frame.index, frame.offset_s, len(detections)))


@st.composite
def videos(draw):
    n = draw(st.integers(1, 30))
    imgs: list[np.ndarray] = []
    for i in range(n):
        if i and draw(st.integers(0, 2)) > 0:  # mostly static, some change
            imgs.append(imgs[-1].copy())
            continue
        rng = np.random.default_rng(draw(st.integers(0, 2**31 - 1)))
        imgs.append(rng.integers(0, 256, (H, W, 3), dtype=np.uint8))
    return imgs


def _ingest(root: Path, images, fps, cfg_kw, tracker):
    cfg = Config(root=root, keyframe_interval_s=2.0, enable_detector=True, **cfg_kw)
    vdir = root / "videos"
    vdir.mkdir(parents=True)
    (vdir / NAME).write_bytes(b"v")
    det = PixelDetector()
    db = MetadataStore(cfg.db_path)
    index = VectorIndex()
    try:
        pipe = IngestPipeline(
            cfg, db, index, FakeEncoder(), det,
            open_source=lambda p: FakeSource(images, fps=fps, width=W, height=H, path=p),
            start_playback=FakePlayback(), tracker=tracker)
        [res] = pipe.ingest_paths([vdir]).results
        c = db._conn
        rows = {
            "status": res.status,
            "video": c.execute("SELECT sampled_count, passed_count FROM videos").fetchall(),
            "frames": c.execute("SELECT frame_idx, offset_s, gate_reason, motion_frac, thumb_path "
                                "FROM frames ORDER BY frame_id").fetchall(),
            "vectors": c.execute("SELECT f.frame_idx, v.kind, v.det_class, v.det_conf, v.x1, v.y1, "
                                 "v.x2, v.y2 FROM vectors v JOIN frames f USING (frame_id) "
                                 "ORDER BY v.vector_id").fetchall(),
        }
        if res.status == "ingested":
            ids = np.asarray(sorted(index.ids().tolist()), dtype=np.int64)
            rows["index"] = [index.faiss_index.reconstruct(int(i)).round(6).tolist() for i in ids]
        return rows, det, cfg
    finally:
        db.close()


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(videos(), st.sampled_from([5.0, 10.0]), st.sampled_from([1.0, 2.0]),
       st.sampled_from([1.0, 3.0, 5.0]), st.lists(st.booleans(), max_size=6))
def test_property_23_v1_rows_unchanged_with_tracker(images, fps, rate, track, pattern):
    kw = {"sample_rate": rate, "track_rate": track}
    with tempfile.TemporaryDirectory() as tmp:
        v1, det1, _ = _ingest(Path(tmp) / "a", images, fps, kw, None)
        sink = FakeTrackSink(pattern)
        v2, det2, cfg = _ingest(Path(tmp) / "b", images, fps, kw, sink)
    assert v2 == v1

    # Without a tracker the detector runs exactly once per passed sample frame (V1).
    assert len(det1.calls) == len(v1["frames"])
    # The tracker sees time-ordered frames from the two cadences only.
    offsets = [o for _, o, _ in sink.updates]
    assert offsets == sorted(offsets) and len(set(offsets)) == len(offsets)
    eff_track = max(track, rate)
    allowed = set(sample_indices(len(images), fps, rate)) | set(sample_indices(len(images), fps, eff_track))
    assert {i for i, _, _ in sink.updates} <= allowed
    # Every passed sample frame reaches the tracker with all its detections (high and low).
    passed = {r[0] for r in v1["frames"]}
    got = {i: n for i, _, n in sink.updates}
    for i in passed:
        assert got[i] == len(detect_from_pixels(images[i]))
    assert len(det2.calls) == len(sink.updates)


def test_inactive_tracker_on_static_scene_adds_no_detector_runs():
    """After the first frame a static scene fails the gate; with no live track the detector
    must not run on tracking-only frames (no CPU spent on an empty scene)."""
    still = [np.full((H, W, 3), 90, np.uint8)] * 50  # 10 s at 5 fps
    with tempfile.TemporaryDirectory() as tmp:
        sink = FakeTrackSink([False])
        rows, det, _ = _ingest(Path(tmp) / "a", still, 5.0,
                               {"sample_rate": 1.0, "track_rate": 5.0}, sink)
    # A still scene passes the gate only at the first frame and at keyframes (every 2 s);
    # those are no motion, so tracking-rate frames are never detected: the detector runs on
    # the passed sample frames alone (5 runs instead of up to 50).
    passed = [r[0] for r in rows["frames"]]
    assert passed == [0, 10, 20, 30, 40]
    assert [i for i, _, _ in sink.updates] == passed
    assert len(det.calls) == len(passed)


def test_motion_starts_tracking_rate_detection():
    """A real motion pass makes the following tracking-only frames detected."""
    rng = np.random.default_rng(0)
    frames = [np.full((H, W, 3), 90, np.uint8)] * 10 + [
        rng.integers(0, 256, (H, W, 3), dtype=np.uint8) for _ in range(10)]  # motion from 2 s
    with tempfile.TemporaryDirectory() as tmp:
        sink = FakeTrackSink([False])
        rows, det, _ = _ingest(Path(tmp) / "a", frames, 5.0,
                               {"sample_rate": 1.0, "track_rate": 5.0}, sink)
    reasons = {r[0]: r[2] for r in rows["frames"]}
    # Frame 10 (2.0 s) is labelled "keyframe" (checked before motion) but carries motion.
    assert reasons[10] == "keyframe" and reasons[15] == "motion"
    updated = [i for i, _, _ in sink.updates]
    assert updated[:1] == [0]  # first frame: no tracking-rate follow-up
    assert updated[1:] == list(range(10, 20))  # every frame from the first motion pass on


def test_fake_track_sink_satisfies_protocol():
    assert isinstance(FakeTrackSink([]), TrackSink)
