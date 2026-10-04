"""Example tests for synthetic output and motion gating on the default synthetic video.

Validates: Requirements 15.1, 15.2, 15.3, 15.5, 15.6, 15.9
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from nab_sentry.config import Config
from nab_sentry.ingest.metadata import parse_filename, parse_sidecar_text
from nab_sentry.ingest.motion import MotionGate
from nab_sentry.ingest.sampler import Sampler
from nab_sentry.ingest.sources import DecodedFrame
from nab_sentry.synthetic import (
    GROUND_TRUTH_NAME,
    SyntheticError,
    SyntheticSpec,
    background,
    ground_truth_document,
    render_frame,
    write_synthetic,
)

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_synthetic.py"
RED_BGR = (0, 0, 255)
BLUE_BGR = (255, 0, 0)
DEFAULT_INTERVALS = [(10.0, 25.0), (50.0, 65.0)]


def _script():
    name = "nab_sentry_make_synthetic_examples"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _in_interval(t: float) -> bool:
    return any(s <= t < e for s, e in DEFAULT_INTERVALS)


# --------------------------------------------------------------------------- 15.1


def test_default_spec_renders_objects_only_inside_intervals():
    spec = SyntheticSpec()
    assert (spec.frame_count, spec.fps, spec.width, spec.height) == (900, 10, 640, 480)
    bg = background(spec)
    prev_mask = None
    for i in range(spec.frame_count):
        frame = render_frame(i, spec)
        assert frame.shape == (480, 640, 3) and frame.dtype == np.uint8
        t = i / spec.fps
        changed = np.any(frame != bg, axis=2)
        if 10 <= t < 25:
            red = np.all(frame == RED_BGR, axis=2)
            assert red.sum() >= 60 * 60, i
            assert np.array_equal(changed, red), i  # only the red square differs
        elif 50 <= t < 65:
            blue = np.all(frame == BLUE_BGR, axis=2)
            assert blue.sum() >= int(np.pi * 30 * 30), i
            assert np.array_equal(changed, blue), i
        else:
            assert not changed.any(), f"object visible outside intervals at frame {i}"
            prev_mask = None
            continue
        # The object changes position in every consecutive frame inside an interval.
        if prev_mask is not None:
            assert not np.array_equal(changed, prev_mask), i
        prev_mask = changed


# --------------------------------------------------------------------------- 15.2, 15.3


def test_write_default_sidecar_and_ground_truth_match(tmp_path: Path):
    spec = SyntheticSpec()
    video, sidecar, gt = write_synthetic(spec, tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [video.name, sidecar.name, GROUND_TRUTH_NAME]
    )
    assert video.name == "CAM-SYN01_20250101T080000.mp4"
    assert sidecar.name == "CAM-SYN01_20250101T080000.json"
    assert video.stat().st_size > 0

    # Filename and sidecar resolve to identical camera ID and start time.
    cam, start = parse_filename(video.name)
    parsed = parse_sidecar_text(sidecar.read_text(encoding="utf-8")).sidecar
    assert parsed is not None
    assert (parsed.camera_id, parsed.label, parsed.start_time) == (
        "CAM-SYN01", "Synthetic Yard", spec.start_time,
    )
    assert (parsed.camera_id, parsed.start_time) == (cam, start)

    # ground_truth.json lists each object plus the sidecar's camera ID and start time.
    doc = json.loads(gt.read_text(encoding="utf-8"))
    assert doc == ground_truth_document(spec)
    (entry,) = doc["videos"]
    assert entry["file"] == video.name
    assert entry["camera_id"] == parsed.camera_id
    assert entry["start_time"] == parsed.start_time.isoformat()
    assert entry["objects"] == [
        {"label": "red square", "start_s": 10.0, "end_s": 25.0},
        {"label": "blue circle", "start_s": 50.0, "end_s": 65.0},
    ]


# --------------------------------------------------------------------------- 15.9


def test_unwritable_output_dir_leaves_no_files(tmp_path: Path):
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    spec = SyntheticSpec(duration_s=1.0, fps=2, width=160, height=120, objects=())
    with pytest.raises(SyntheticError, match="output directory is not writable"):
        write_synthetic(spec, blocker / "out")
    assert [p.name for p in tmp_path.iterdir()] == ["blocker"]
    assert blocker.read_text() == "x"


def test_cli_unwritable_output_dir_exits_nonzero(tmp_path: Path, capsys):
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    rc = _script().main(["--out", str(blocker / "out"), "--duration", "1", "--fps", "2",
                         "--width", "160", "--height", "120",
                         "--red", "0", "0.5", "--blue", "0.5", "1"])
    assert rc != 0
    assert "output directory is not writable" in capsys.readouterr().err
    assert [p.name for p in tmp_path.iterdir()] == ["blocker"]


# --------------------------------------------------------------------------- 15.5, 15.6


def _frames(spec: SyntheticSpec) -> Iterator[DecodedFrame]:
    for i in range(spec.frame_count):  # rendered lazily
        yield DecodedFrame(index=i, offset_s=i / spec.fps, image=render_frame(i, spec))


def _gate_default_video() -> list[tuple[float, str]]:
    cfg = Config()
    spec = SyntheticSpec()
    sampler = Sampler(cfg.sample_rate)
    gate = MotionGate(cfg.motion_threshold, cfg.keyframe_interval_s, cfg.gate_width,
                      cfg.motion_pixel_delta, cfg.motion_method)
    gate.reset()
    passes = []
    for frame in sampler.select(_frames(spec)):
        d = gate.evaluate(frame)
        if d.passed:
            passes.append((frame.offset_s, d.reason))
    return passes


def test_gate_passes_frames_inside_each_interval_and_few_outside():
    cfg = Config()
    passes = _gate_default_video()
    assert passes, "gate passed nothing"

    # 15.5: at least one passed sampled frame inside each ground-truth interval.
    for s, e in DEFAULT_INTERVALS:
        assert any(s <= t < e for t, _ in passes), (s, e, passes)

    # 15.6: outside the intervals only first frame, keyframes, and <= 1 disappearance frame
    # within 1 s after each interval end.
    assert passes[0] == (0.0, "first")
    after_end_counts = {e: 0 for _, e in DEFAULT_INTERVALS}
    last_pass = None
    for idx, (t, reason) in enumerate(passes):
        if idx > 0 and not _in_interval(t):
            end = next((e for _, e in DEFAULT_INTERVALS if e <= t <= e + 1.0), None)
            if end is not None and reason != "keyframe":
                after_end_counts[end] += 1
            else:
                assert reason == "keyframe", (t, reason, passes)
                assert t - last_pass >= cfg.keyframe_interval_s - 1e-9, (t, last_pass)
        last_pass = t
    assert all(c <= 1 for c in after_end_counts.values()), after_end_counts
