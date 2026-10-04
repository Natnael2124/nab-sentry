"""Sanity tests for letterbox / postprocess / OnnxYoloDetector (no model required)."""

from __future__ import annotations

import numpy as np
import pytest

from nab_sentry.config import ConfigError
from nab_sentry.errors import ModelMissingError
from nab_sentry.ingest.detector import (
    Detection,
    Detector,
    DetectorInferenceError,
    LetterboxMeta,
    OnnxYoloDetector,
    letterbox,
    postprocess,
)
from nab_sentry.startup import FETCH_HINT

N = 8400


def _raw(*anchors: tuple[float, float, float, float, int, float]) -> np.ndarray:
    """Build a (1, 84, N) tensor; each anchor is (cx, cy, w, h, class_id, score)."""
    raw = np.zeros((1, 84, N), dtype=np.float32)
    for i, (cx, cy, w, h, cid, score) in enumerate(anchors):
        raw[0, :4, i] = (cx, cy, w, h)
        raw[0, 4 + cid, i] = score
    return raw


IDENT = LetterboxMeta(1.0, 0.0, 0.0)


def test_letterbox_shape_padding_and_meta():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[..., 2] = 255  # pure red in BGR
    t, meta = letterbox(frame, 64)
    assert t.shape == (1, 3, 64, 64) and t.dtype == np.float32
    assert meta == LetterboxMeta(0.32, 0.0, 16.0)
    # padding rows are 114/255 in every channel
    assert np.allclose(t[0, :, 0, :], 114 / 255)
    # content is RGB: red channel first
    assert np.allclose(t[0, 0, 32, 32], 1.0) and np.allclose(t[0, 2, 32, 32], 0.0)


def test_postprocess_undoes_letterbox_and_filters_classes():
    meta = LetterboxMeta(0.5, 10.0, 20.0)
    raw = _raw(
        (60.0, 70.0, 20.0, 40.0, 0, 0.9),   # person -> frame (80,60)-(120,140)
        (300.0, 300.0, 20.0, 20.0, 15, 0.99),  # cat: not a Target_Class
        (200.0, 200.0, 20.0, 20.0, 2, 0.2),   # below threshold
    )
    dets = postprocess(raw, meta, 1000, 1000, conf=0.35, iou=0.45, max_det=10)
    assert len(dets) == 1
    d = dets[0]
    assert d.cls == "person" and d.box == (80, 60, 120, 140)
    assert d.conf == pytest.approx(0.9)


def test_postprocess_classwise_nms_and_sort_and_max_det():
    raw = _raw(
        (50, 50, 20, 20, 2, 0.8),   # car
        (51, 51, 20, 20, 2, 0.7),   # overlapping car -> suppressed
        (50, 50, 20, 20, 0, 0.6),   # same box, other class -> kept
        (200, 200, 20, 20, 7, 0.8),  # tie in conf with first car; larger x1
    )
    dets = postprocess(raw, IDENT, 640, 640, conf=0.35, iou=0.45, max_det=10)
    assert [(d.cls, d.box[0]) for d in dets] == [("car", 40), ("truck", 190), ("person", 40)]
    top2 = postprocess(raw, IDENT, 640, 640, conf=0.35, iou=0.45, max_det=2)
    assert top2 == dets[:2]


def test_postprocess_clamps_and_drops_outside_boxes():
    raw = _raw(
        (5, 5, 20, 20, 0, 0.9),        # partially outside -> clamped
        (900, 900, 20, 20, 0, 0.8),    # fully outside the 100x100 frame -> dropped
    )
    dets = postprocess(raw, IDENT, 100, 100, conf=0.35, iou=0.45, max_det=10)
    assert dets == [Detection("person", pytest.approx(0.9), (0, 0, 15, 15))]


def test_postprocess_empty():
    assert postprocess(_raw(), IDENT, 640, 640, 0.35, 0.45, 10) == []


@pytest.mark.parametrize(
    "kwargs, name",
    [
        ({"conf": 1.5}, "det_conf"),
        ({"conf": -0.1}, "det_conf"),
        ({"iou": 2.0}, "det_iou"),
        ({"max_det": 0}, "det_max_per_frame"),
        ({"max_det": 101}, "det_max_per_frame"),
    ],
)
def test_detector_validates_params(tmp_path, kwargs, name):
    args = {"conf": 0.35, "iou": 0.45, "max_det": 10} | kwargs
    with pytest.raises(ConfigError) as ei:
        OnnxYoloDetector(tmp_path / "yolo11n.onnx", **args)
    assert [i.name for i in ei.value.issues] == [name]
    assert name in str(ei.value) and ".." in str(ei.value)


def test_detector_missing_model(tmp_path):
    path = tmp_path / "models" / "yolo11n.onnx"
    with pytest.raises(ModelMissingError) as ei:
        OnnxYoloDetector(path, 0.35, 0.45, 10)
    assert str(path) in str(ei.value) and FETCH_HINT in str(ei.value)


def test_detect_wraps_inference_errors():
    class Boom:
        def run(self, *_a, **_k):
            raise RuntimeError("bad")

    det = object.__new__(OnnxYoloDetector)
    det.conf, det.iou, det.max_det, det.input_size = 0.35, 0.45, 10, 64
    det._session, det._input_name = Boom(), "images"
    assert isinstance(det, Detector)
    with pytest.raises(DetectorInferenceError, match="bad"):
        det.detect(np.zeros((32, 32, 3), dtype=np.uint8))
