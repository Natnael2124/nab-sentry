"""Sanity unit tests for nab_sentry.ingest.motion (properties live in 6.6-6.8)."""

import numpy as np

from nab_sentry.ingest.motion import MotionGate, changed_fraction, downscale_gray
from nab_sentry.ingest.sources import DecodedFrame


def _frame(i: int, offset: float, image: np.ndarray) -> DecodedFrame:
    return DecodedFrame(index=i, offset_s=offset, image=image)


def _gate(**kw) -> MotionGate:
    args = dict(threshold=0.02, keyframe_interval_s=10.0, gate_width=320, pixel_delta=25)
    args.update(kw)
    return MotionGate(**args)


def test_downscale_keeps_aspect_and_never_upscales():
    big = np.zeros((720, 1280, 3), np.uint8)
    assert downscale_gray(big, 320).shape == (180, 320)
    small = np.zeros((48, 64), np.uint8)
    assert downscale_gray(small, 320).shape == (48, 64)


def test_changed_fraction_basics():
    a = np.zeros((10, 10), np.uint8)
    b = a.copy()
    b[:5] = 200
    assert changed_fraction(a, a, 25) == 0.0
    assert changed_fraction(a, b, 25) == 0.5
    assert changed_fraction(a, np.zeros((5, 5), np.uint8), 25) == 1.0


def test_decision_order_and_state():
    gate = _gate()
    black = np.zeros((240, 320, 3), np.uint8)
    white = np.full((240, 320, 3), 255, np.uint8)
    empty = np.zeros((0, 0, 3), np.uint8)

    assert _gate().evaluate(_frame(0, 0.0, empty)).reason == "empty"

    d = gate.evaluate(_frame(0, 0.0, black))
    assert (d.passed, d.reason, d.fraction) == (True, "first", 0.0)
    assert gate.evaluate(_frame(1, 1.0, black)).reason == "static"
    assert gate.evaluate(_frame(2, 2.0, empty)).reason == "empty"
    d = gate.evaluate(_frame(3, 3.0, white))
    assert (d.passed, d.reason, d.fraction) == (True, "motion", 1.0)
    # empty frame did not replace the reference; white vs white is static
    assert gate.evaluate(_frame(4, 4.0, white)).reason == "static"
    # keyframe is due 10 s after the last pass (3.0)
    assert gate.evaluate(_frame(12, 12.9, white)).reason == "static"
    assert gate.evaluate(_frame(13, 13.0, white)).reason == "keyframe"

    gate.reset()
    assert gate.evaluate(_frame(0, 0.0, white)).reason == "first"


def test_grayscale_input_and_resolution_change():
    gate = _gate()
    assert gate.evaluate(_frame(0, 0.0, np.zeros((100, 100), np.uint8))).reason == "first"
    d = gate.evaluate(_frame(1, 1.0, np.zeros((50, 100), np.uint8)))
    assert (d.reason, d.fraction) == ("motion", 1.0)


def test_identical_frames_pass_once_per_interval():
    gate = _gate(keyframe_interval_s=2.0)
    img = np.full((60, 80, 3), 90, np.uint8)
    passed = [gate.evaluate(_frame(i, i * 0.5, img)).passed for i in range(13)]  # D = 6 s
    assert sum(passed) == 1 + 6 // 2


def test_mog2_method_detects_change():
    gate = _gate(method="mog2")
    black = np.zeros((120, 160, 3), np.uint8)
    for i in range(5):
        gate.evaluate(_frame(i, i * 0.1, black))
    d = gate.evaluate(_frame(5, 0.5, np.full_like(black, 255)))
    assert d.reason == "motion" and d.fraction > 0.5
