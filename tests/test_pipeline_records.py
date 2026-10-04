"""Property test for the per-video record shape written by the ingest pipeline (task 10.5).

Property 17: Pipeline record shape per video, plus the pipeline half of Property 16 (every
stored ``crop`` box lies inside its video's width and height).

**Validates: Requirements 3.6, 3.7, 4.5, 4.7, 4.8, 5.1, 5.2, 6.2, 6.3**
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nab_sentry.config import Config
from nab_sentry.ingest.detector import TARGET_CLASSES, Detection, clamp_box
from nab_sentry.ingest.motion import MotionGate
from nab_sentry.ingest.pipeline import IngestPipeline
from nab_sentry.ingest.sampler import Sampler
from nab_sentry.store.db import MetadataStore
from nab_sentry.store.vector_index import VectorIndex
from tests.fakes import FakeDetector, FakeEncoder, FakePlayback, FakeSource

W, H = 32, 24
CLASSES = sorted(TARGET_CLASSES.values())
NAMES = ["CAM-A_20250101T080000.mp4", "CAM-B_20250102T093000.mp4"]

# -- strategies ------------------------------------------------------------------------------

# Coordinates range well outside the frame so some boxes clamp to None (fully outside,
# zero-width, inverted) and others are partly clipped.
_coord_x = st.floats(min_value=-20, max_value=W + 20, allow_nan=False, allow_infinity=False)
_coord_y = st.floats(min_value=-20, max_value=H + 20, allow_nan=False, allow_infinity=False)

detections = st.builds(
    lambda cls, conf, x1, y1, x2, y2: Detection(cls, conf, (x1, y1, x2, y2)),
    st.sampled_from(CLASSES),
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    _coord_x, _coord_y, _coord_x, _coord_y,
)


@st.composite
def frame_images(draw) -> list[np.ndarray]:
    """1..10 small frames; each either repeats the previous one (static) or is new."""
    n = draw(st.integers(min_value=1, max_value=10))
    out: list[np.ndarray] = []
    for i in range(n):
        if i > 0 and draw(st.booleans()):
            out.append(out[-1].copy())
            continue
        seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
        rng = np.random.default_rng(seed)
        img = np.full((H, W, 3), int(rng.integers(0, 256)), dtype=np.uint8)
        if draw(st.booleans()):  # a random bright block on top
            x, y = int(rng.integers(0, W - 8)), int(rng.integers(0, H - 8))
            img[y:y + 8, x:x + 8] = 255 - img[0, 0, 0]
        out.append(img)
    return out


@st.composite
def video_specs(draw) -> dict:
    return {
        "images": draw(frame_images()),
        "fps": draw(st.sampled_from([2.0, 4.0, 5.0])),
        # detections per detector call (call i == i-th passed frame of this video)
        "script": draw(st.lists(st.lists(detections, max_size=4), max_size=10)),
        "raise_on": draw(st.sets(st.integers(min_value=0, max_value=9), max_size=4)),
    }


# -- helpers ---------------------------------------------------------------------------------


def _expected_passed(images: list[np.ndarray], fps: float, cfg: Config) -> tuple[int, list[int]]:
    """Replay the sampler and motion gate: (sampled count, passed frame indices)."""
    src = FakeSource(images, fps=fps, width=W, height=H)
    sampler = Sampler(cfg.sample_rate)
    gate = MotionGate(cfg.motion_threshold, cfg.keyframe_interval_s, cfg.gate_width,
                      cfg.motion_pixel_delta, cfg.motion_method)  # type: ignore[arg-type]
    passed = [f.index for f in sampler.select(src.frames()) if gate.evaluate(f).passed]
    return sampler.selected_count, passed


def _expected_crops(spec: dict, call: int) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    if call in spec["raise_on"]:
        return []
    dets = spec["script"][call] if call < len(spec["script"]) else []
    out = []
    for d in dets:
        box = clamp_box(*d.box, W, H)
        if box is not None:
            out.append((d.cls, d.conf, box))
    return out


# -- property --------------------------------------------------------------------------------


@settings(max_examples=50, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
@given(
    specs=st.lists(video_specs(), min_size=1, max_size=2),
    sample_rate=st.sampled_from([1.0, 2.0, 5.0]),
    batch_size=st.integers(min_value=1, max_value=5),
)
def test_pipeline_record_shape_per_video(specs: list[dict], sample_rate: float,
                                         batch_size: int) -> None:
    """Property 17 (+ Property 16 stored-row bounds).

    **Validates: Requirements 3.6, 3.7, 4.5, 4.7, 4.8, 5.1, 5.2, 6.2, 6.3**
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "ws"
        cfg = Config(root=root, sample_rate=sample_rate, keyframe_interval_s=1.0,
                     enable_detector=True, batch_size=batch_size)
        videos_dir = root / "videos"
        videos_dir.mkdir(parents=True)
        by_name = dict(zip(NAMES, specs))
        for i, name in enumerate(by_name):
            (videos_dir / name).write_bytes(f"video-{i}-{name}".encode())

        detectors = {name: FakeDetector(s["script"], raise_on=s["raise_on"])
                     for name, s in by_name.items()}

        class Router:
            """Routes detect() calls to the current video's scripted detector."""
            current: str = ""

            def detect(self, frame):
                return detectors[self.current].detect(frame)

        router = Router()

        def open_source(path: Path) -> FakeSource:
            router.current = Path(path).name
            s = by_name[router.current]
            return FakeSource(s["images"], fps=s["fps"], width=W, height=H, path=path)

        db = MetadataStore(cfg.db_path)
        index = VectorIndex()
        try:
            pipe = IngestPipeline(cfg, db, index, FakeEncoder(batch_size=batch_size), router,
                                  open_source=open_source, start_playback=FakePlayback())
            report = pipe.ingest_paths([videos_dir])
            assert [r.status for r in report.results] == ["ingested"] * len(by_name)

            conn = db._conn
            all_thumbs = {p.name for p in cfg.thumbs_dir.glob("*")}
            seen_thumbs: set[str] = set()
            for r in report.results:
                spec = by_name[r.path.name]
                exp_sampled, exp_passed = _expected_passed(spec["images"], spec["fps"], cfg)

                # videos row: sampled == sampler selections, 1 <= passed <= sampled (3.6)
                sampled, passed, width, height, status = conn.execute(
                    "SELECT sampled_count, passed_count, width, height, status "
                    "FROM videos WHERE video_id = ?", (r.video_id,)).fetchone()
                assert status == "complete"
                assert (width, height) == (W, H)
                assert sampled == r.sampled == exp_sampled
                assert passed == r.passed == len(exp_passed)
                assert 1 <= passed <= sampled

                # one frames row (+ thumbnail) per passed frame, none for rejected ones (3.7, 6.2)
                frames = conn.execute(
                    "SELECT frame_id, frame_idx, thumb_path FROM frames "
                    "WHERE video_id = ? ORDER BY frame_id", (r.video_id,)).fetchall()
                assert [f[1] for f in frames] == exp_passed
                thumbs = {f[2] for f in frames}
                assert len(thumbs) == len(frames) and thumbs <= all_thumbs
                seen_thumbs |= thumbs

                # detector called once per passed frame
                assert detectors[r.path.name].call_count == len(exp_passed)

                n_vectors = 0
                for call, (frame_id, _idx, _thumb) in enumerate(frames):
                    rows = conn.execute(
                        "SELECT kind, det_class, det_conf, x1, y1, x2, y2 FROM vectors "
                        "WHERE frame_id = ? ORDER BY vector_id", (frame_id,)).fetchall()
                    n_vectors += len(rows)
                    kinds = [row[0] for row in rows]
                    assert kinds.count("frame") == 1  # exactly one frame vector (5.1)
                    assert rows[0] == ("frame", None, None, None, None, None, None)
                    crops = [row for row in rows if row[0] == "crop"]
                    expected = _expected_crops(spec, call)  # 0 when detector raised (4.7, 4.8)
                    assert len(crops) == len(expected)
                    for row, (cls, conf, box) in zip(crops, expected):  # (4.5, 5.2, 6.3)
                        _k, c, cf, x1, y1, x2, y2 = row
                        assert c == cls
                        assert abs(cf - conf) < 1e-9
                        assert (x1, y1, x2, y2) == box
                        # Property 16, stored-row half: inside the video's frame
                        assert 0 <= x1 < x2 <= width
                        assert 0 <= y1 < y2 <= height
                assert n_vectors == r.vectors

            # no thumbnails beyond the passed frames
            assert seen_thumbs == all_thumbs
            # store vector IDs == index IDs (memory and disk)
            db_ids = set(db.vector_ids().tolist())
            assert set(index.ids().tolist()) == db_ids
            assert set(VectorIndex.load(cfg.index_path).ids().tolist()) == db_ids
        finally:
            db.close()
