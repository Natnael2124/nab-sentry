"""Example and slow tests for ``OnnxYoloDetector`` (Requirements 4.1, 4.6, 4.9).

- 4.6: a missing ONNX file raises ``ModelMissingError`` naming the path and the
  Model_Fetcher command.
- 4.9: invalid ``conf`` / ``max_det`` raise ``ConfigError`` at construction, naming
  the parameter and its allowed range, before the model file is even looked at.
- 4.1 (slow): the real YOLO11n model returns Target_Class boxes in source-frame
  pixel coordinates (not 640x640 model-input coordinates). Skipped when
  ``models/yolo11n.onnx`` is not present.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from nab_sentry.config import Config, ConfigError
from nab_sentry.errors import ModelMissingError
from nab_sentry.ingest.detector import TARGET_CLASSES, Detection, OnnxYoloDetector
from nab_sentry.startup import FETCH_HINT

DEFAULTS = {"conf": 0.35, "iou": 0.45, "max_det": 10}


# --------------------------------------------------------------------- Req 4.6


def test_missing_onnx_names_path_and_fetch_command(tmp_path: Path) -> None:
    path = tmp_path / "models" / "yolo11n.onnx"
    with pytest.raises(ModelMissingError) as ei:
        OnnxYoloDetector(path, **DEFAULTS)
    msg = str(ei.value)
    assert str(path) in msg
    assert FETCH_HINT in msg
    assert "fetch_models.py" in msg
    assert not path.exists()  # nothing was downloaded or created


# --------------------------------------------------------------------- Req 4.9


@pytest.mark.parametrize(
    "override, name, allowed",
    [
        ({"conf": 1.01}, "det_conf", "0.0..1.0"),
        ({"conf": -0.5}, "det_conf", "0.0..1.0"),
        ({"conf": float("nan")}, "det_conf", "0.0..1.0"),
        ({"conf": "high"}, "det_conf", "0.0..1.0"),
        ({"max_det": 0}, "det_max_per_frame", "1..100"),
        ({"max_det": 101}, "det_max_per_frame", "1..100"),
        ({"max_det": 2.5}, "det_max_per_frame", "1..100"),
    ],
)
def test_invalid_param_raises_config_error_at_init(
    tmp_path: Path, override: dict, name: str, allowed: str
) -> None:
    # The model path does not exist: validation must win over the missing-file check.
    with pytest.raises(ConfigError) as ei:
        OnnxYoloDetector(tmp_path / "yolo11n.onnx", **(DEFAULTS | override))
    issues = ei.value.issues
    assert [i.name for i in issues] == [name]
    assert issues[0].allowed == allowed
    assert name in str(ei.value) and allowed in str(ei.value)


def test_multiple_invalid_params_all_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as ei:
        OnnxYoloDetector(tmp_path / "yolo11n.onnx", conf=2.0, iou=0.45, max_det=0)
    assert {i.name for i in ei.value.issues} == {"det_conf", "det_max_per_frame"}


def test_boundary_values_are_valid(tmp_path: Path) -> None:
    # Range ends are inclusive, so these get past validation to the missing-file check.
    for conf, max_det in ((0.0, 1), (1.0, 100)):
        with pytest.raises(ModelMissingError):
            OnnxYoloDetector(tmp_path / "yolo11n.onnx", conf=conf, iou=0.45, max_det=max_det)


# --------------------------------------------------------------------- Req 4.1 (slow)


def _sample_scene(w: int, h: int) -> np.ndarray:
    """Deterministic street-like BGR scene: sky, road, a car-like and a person-like shape."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[: h // 2] = (200, 170, 120)  # sky
    img[h // 2 :] = (80, 80, 80)  # road
    # car: body, cabin, wheels
    cv2.rectangle(img, (w // 8, h * 9 // 16), (w * 3 // 8, h * 3 // 4), (30, 30, 180), -1)
    cv2.rectangle(img, (w * 3 // 16, h // 2), (w * 5 // 16, h * 9 // 16), (60, 60, 200), -1)
    for cx in (w * 3 // 16, w * 5 // 16):
        cv2.circle(img, (cx, h * 3 // 4), h // 20, (10, 10, 10), -1)
    # person: head, torso, legs
    px = w * 3 // 4
    cv2.circle(img, (px, h * 3 // 8), h // 24, (150, 180, 220), -1)
    cv2.rectangle(img, (px - w // 60, h * 5 // 12), (px + w // 60, h * 5 // 8), (120, 40, 40), -1)
    cv2.line(img, (px, h * 5 // 8), (px - w // 60, h * 13 // 16), (40, 40, 40), max(1, w // 160))
    cv2.line(img, (px, h * 5 // 8), (px + w // 60, h * 13 // 16), (40, 40, 40), max(1, w // 160))
    rng = np.random.default_rng(0)
    noise = rng.integers(-8, 9, size=img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _assert_valid(dets: list[Detection], w: int, h: int, conf: float, max_det: int) -> None:
    assert len(dets) <= max_det
    for d in dets:
        assert d.cls in TARGET_CLASSES.values()
        assert conf <= d.conf <= 1.0
        x1, y1, x2, y2 = d.box
        assert all(isinstance(v, int) for v in d.box)
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h, (d, w, h)


@pytest.mark.slow
def test_real_detector_returns_frame_coordinate_boxes(network_guard) -> None:
    pytest.importorskip("onnxruntime")
    model = Config().models_dir / "yolo11n.onnx"
    if not model.is_file():
        pytest.skip(f"real model not present: {model} (run {FETCH_HINT})")

    conf, max_det = 0.10, 20
    det = OnnxYoloDetector(model, conf=conf, iou=0.45, max_det=max_det)

    # 640x360 scene and its exact 2x nearest-neighbour upscale (1280x720). Letterboxing
    # the 1280x720 frame to 640 reproduces the 640x360 frame pixel-for-pixel, so the
    # model sees the same input; the returned boxes must differ only by the 2x scale.
    small = _sample_scene(640, 360)
    big = cv2.resize(small, (1280, 720), interpolation=cv2.INTER_NEAREST)

    dets_small = det.detect(small)
    dets_big = det.detect(big)
    _assert_valid(dets_small, 640, 360, conf, max_det)
    _assert_valid(dets_big, 1280, 720, conf, max_det)

    assert [d.cls for d in dets_big] == [d.cls for d in dets_small]
    for b, s in zip(dets_big, dets_small):
        assert b.conf == pytest.approx(s.conf, abs=1e-4)
        for vb, vs in zip(b.box, s.box):
            assert abs(vb - 2 * vs) <= 2, (b, s)

    # A non-square, non-16:9 frame also stays inside its own bounds.
    tall = _sample_scene(360, 800)
    _assert_valid(det.detect(tall), 360, 800, conf, max_det)

    network_guard.assert_no_attempts()
