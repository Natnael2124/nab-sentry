"""Rectangular detector input (V2 task 9.0): letterbox to (h, w), the box round trip through
``postprocess``, the ONNX shape check, and model-file selection from Config."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from nab_sentry.config import Config
from nab_sentry.errors import ModelMissingError
from nab_sentry.ingest.detector import LETTERBOX_PAD, OnnxYoloDetector, input_hw, letterbox
from nab_sentry.model_files import DETECTOR_REL, detector_input, detector_path, detector_rel

dims = st.integers(1, 40).map(lambda k: 32 * k)  # 32..1280


def test_input_hw():
    assert input_hw(640) == (640, 640)
    assert input_hw((384, 640)) == (384, 640)


@given(st.integers(8, 400), st.integers(8, 400), dims, dims)
def test_letterbox_rect_shape_and_inverse(fh, fw, th, tw):
    frame = np.zeros((fh, fw, 3), np.uint8)
    frame[fh // 2, fw // 2] = 255
    tensor, meta = letterbox(frame, (th, tw))
    assert tensor.shape == (1, 3, th, tw)
    assert meta.scale == pytest.approx(min(tw / fw, th / fh))
    # The frame's corners map inside the input and back again.
    for x, y in ((0, 0), (fw, fh)):
        ix, iy = x * meta.scale + meta.pad_x, y * meta.scale + meta.pad_y
        assert -1 <= ix <= tw + 1 and -1 <= iy <= th + 1
        assert (ix - meta.pad_x) / meta.scale == pytest.approx(x)


def test_letterbox_16_9_into_384x640_has_little_padding():
    tensor, meta = letterbox(np.full((776, 1376, 3), 200, np.uint8), (384, 640))
    assert meta.pad_x == 0 and meta.pad_y == (384 - round(776 * 640 / 1376)) // 2
    pad = np.isclose(tensor[0, 0], LETTERBOX_PAD / 255.0).mean()
    assert pad < 0.07  # 6% here vs about 44% padding in a 640 x 640 square


def test_letterbox_int_is_square_as_before():
    a, ma = letterbox(np.zeros((50, 100, 3), np.uint8), 64)
    b, mb = letterbox(np.zeros((50, 100, 3), np.uint8), (64, 64))
    assert np.array_equal(a, b) and ma == mb


def _tiny_onnx(path: Path, h: int, w: int) -> Path:
    import onnx
    from onnx import TensorProto, helper

    inp = helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, h, w])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 3, h, w])
    graph = helper.make_graph([helper.make_node("Identity", ["images"], ["out"])], "g", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def test_detector_rejects_model_exported_for_another_shape(tmp_path: Path):
    model = _tiny_onnx(tmp_path / "yolo11n.onnx", 64, 64)
    with pytest.raises(ModelMissingError, match="exported for a 64x64 input"):
        OnnxYoloDetector(model, 0.35, 0.45, 10, input_size=(32, 64))
    det = OnnxYoloDetector(model, 0.35, 0.45, 10, input_size=64)
    assert det.input_size == 64
    rect = OnnxYoloDetector(_tiny_onnx(tmp_path / "r.onnx", 32, 64), 0.35, 0.45, 10, input_size=(32, 64))
    assert rect.input_size == (32, 64)


def test_model_file_follows_config(tmp_path: Path):
    cfg = Config(root=tmp_path)
    assert detector_input(cfg) == (384, 640)
    assert detector_path(cfg, {}).name == "yolo11n_384x640.onnx"
    square = Config(root=tmp_path, det_input_h=640)
    assert detector_path(square, {}).name == DETECTOR_REL
    assert detector_rel(480, 480) == "yolo11n_480x480.onnx"
    assert Config(det_input_h=10).validate()[0].name == "det_input_h"
    assert Config(det_threads=-1).validate()[0].name == "det_threads"
