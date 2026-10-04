"""Quick sanity checks for the Synthetic_Generator (small, fast videos).

Validates: Requirements 15.1, 15.2, 15.3, 15.8, 15.9
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from nab_sentry.ingest.metadata import parse_filename, parse_sidecar_text
from nab_sentry.synthetic import (
    SyntheticError,
    SyntheticObject,
    SyntheticSpec,
    background,
    ground_truth_document,
    render_frame,
    validate_spec,
    write_synthetic,
)

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_synthetic.py"


def _script():
    name = "nab_sentry_make_synthetic_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _small_spec(seed: int = 3) -> SyntheticSpec:
    return SyntheticSpec(
        seed=seed, duration_s=2.0, fps=5, width=160, height=120,
        objects=(
            SyntheticObject("red square", "square", "red", 0.4, 1.0),
            SyntheticObject("blue circle", "circle", "blue", 1.2, 1.8),
        ),
    )


def test_default_spec_matches_requirements():
    spec = SyntheticSpec()
    validate_spec(spec)
    assert spec.frame_count == 900
    assert (spec.width, spec.height, spec.fps) == (640, 480, 10)
    assert spec.video_name == "CAM-SYN01_20250101T080000.mp4"
    doc = ground_truth_document(spec)
    assert doc["videos"][0]["objects"] == [
        {"label": "red square", "start_s": 10.0, "end_s": 25.0},
        {"label": "blue circle", "start_s": 50.0, "end_s": 65.0},
    ]
    assert doc["videos"][0]["start_time"] == "2025-01-01T08:00:00"


def test_render_frame_objects_only_in_intervals():
    spec = SyntheticSpec()
    bg = background(spec)
    assert BG_RANGE_OK(bg)
    for i in (0, 99, 100, 101, 249, 250, 499, 500, 649, 650, 899):
        f = render_frame(i, spec)
        t = i / spec.fps
        changed = np.any(f != bg, axis=2)
        red = np.all(f == (0, 0, 255), axis=2)
        blue = np.all(f == (255, 0, 0), axis=2)
        if 10 <= t < 25:
            assert red.sum() == 100 * 100 and changed.sum() == red.sum()
        elif 50 <= t < 65:
            assert blue.sum() > 0.7 * 100 * 100 and changed.sum() == blue.sum()
        else:
            assert not changed.any()
    # Object moves every frame inside the interval.
    assert not np.array_equal(render_frame(100, spec), render_frame(101, spec))


def BG_RANGE_OK(bg: np.ndarray) -> bool:
    return bg.dtype == np.uint8 and bg.min() >= 150 and bg.max() <= 210


def test_write_synthetic_small(tmp_path: Path):
    spec = _small_spec()
    video, sidecar, gt = write_synthetic(spec, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [video.name, sidecar.name, "ground_truth.json"]
    )
    cam, start = parse_filename(video.name)
    parsed = parse_sidecar_text(sidecar.read_text(encoding="utf-8")).sidecar
    assert parsed is not None and (parsed.camera_id, parsed.start_time) == (cam, start)
    assert json.loads(gt.read_text(encoding="utf-8")) == ground_truth_document(spec)

    cap = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    assert len(frames) == spec.frame_count
    # Background (outside objects) is identical in frames without objects.
    assert np.array_equal(frames[0], frames[-1])


@pytest.mark.parametrize(
    "kwargs, name",
    [
        ({"duration_s": 0}, "duration"),
        ({"fps": -1}, "fps"),
        ({"objects": (SyntheticObject("red square", "square", "red", -1, 1),)}, "red square"),
        ({"objects": (SyntheticObject("red square", "square", "red", 1, 3),)}, "red square"),
        ({"objects": (SyntheticObject("blue circle", "circle", "blue", 1, 1),)}, "blue circle"),
    ],
)
def test_invalid_settings_named_and_no_output(tmp_path: Path, kwargs, name):
    base = dict(duration_s=2.0, fps=5, width=160, height=120)
    base.update(kwargs)
    spec = SyntheticSpec(**base)
    with pytest.raises(SyntheticError, match=name):
        write_synthetic(spec, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_unwritable_output_dir(tmp_path: Path, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    rc = _script().main(["--out", str(blocker / "sub"), "--duration", "1", "--fps", "2",
                         "--width", "160", "--height", "120",
                         "--red", "0", "0.5", "--blue", "0.5", "1"])
    assert rc == 1
    assert "output directory is not writable" in capsys.readouterr().err
    assert [p.name for p in tmp_path.iterdir()] == ["file"]


def test_cli_invalid_setting_exit_code(tmp_path: Path, capsys):
    rc = _script().main(["--out", str(tmp_path), "--fps", "0"])
    assert rc == 1
    assert "fps" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_cli_start_time_naive_default():
    args = _script().build_parser().parse_args([])
    spec = _script().spec_from_args(args)
    assert spec == SyntheticSpec()
    assert spec.start_time == datetime(2025, 1, 1, 8, 0, 0)
