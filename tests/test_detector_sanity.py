"""Sanity tests for nab_sentry.ingest.detector basics and FakeDetector."""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest

from nab_sentry.ingest.detector import TARGET_CLASSES, Detection, Detector, clamp_box
from tests.fakes import FakeDetector, FakeDetectorError


def test_target_classes():
    assert TARGET_CLASSES == {0: "person", 1: "bicycle", 2: "car",
                              3: "motorcycle", 5: "bus", 7: "truck"}


def test_module_does_not_import_onnxruntime():
    # Importing detector.py must not pull in onnxruntime (lazy import in OnnxYoloDetector).
    import subprocess
    code = ("import sys, nab_sentry.ingest.detector; "
            "sys.exit(1 if 'onnxruntime' in sys.modules else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


@pytest.mark.parametrize(
    "box, w, h, expected",
    [
        ((1.2, 2.7, 5.1, 6.0), 10, 10, (1, 2, 6, 6)),        # floor / ceil
        ((-3.0, -1.5, 12.2, 20.0), 10, 8, (0, 0, 10, 8)),    # clamped to frame
        ((3.2, 3.2, 3.4, 3.4), 10, 10, (3, 3, 4, 4)),        # sub-pixel -> 1 px
        ((5.1, 6.0, 1.2, 2.7), 10, 10, (1, 2, 6, 6)),        # swapped corners
        ((3.0, 3.0, 3.0, 5.0), 10, 10, None),                # zero width
        ((12.0, 1.0, 15.0, 4.0), 10, 10, None),              # fully outside
        ((-5.0, 1.0, -1.0, 4.0), 10, 10, None),              # fully outside (left)
        ((math.nan, 1.0, 4.0, 4.0), 10, 10, None),
        ((0.0, 0.0, math.inf, 4.0), 10, 10, None),
        ((0.0, 0.0, 4.0, 4.0), 0, 10, None),                 # empty frame
    ],
)
def test_clamp_box_examples(box, w, h, expected):
    assert clamp_box(*box, w, h) == expected


def test_clamp_box_result_types_are_int():
    out = clamp_box(np.float32(1.5), np.float64(0.1), 4.9, 3.0, 10, 10)
    assert out == (1, 0, 5, 3)
    assert all(type(v) is int for v in out)


def test_fake_detector_scripted_sequence_and_raise():
    d1 = Detection("person", 0.9, (0, 0, 4, 4))
    d2 = Detection("car", 0.5, (1, 1, 3, 3))
    det = FakeDetector([[d1], [], [d1, d2]], raise_on={1})
    assert isinstance(det, Detector)
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    assert det.detect(frame) == [d1]
    with pytest.raises(FakeDetectorError):
        det.detect(frame)
    assert det.detect(frame) == [d1, d2]
    assert det.detect(frame) == []  # past end of script
    assert det.call_count == 4
    assert det.calls[0] == (0, (8, 8, 3))


def test_fake_detector_callable_and_mapping():
    d = Detection("bus", 0.7, (0, 0, 2, 2))
    by_call = FakeDetector(lambda frame, i: [d] if i % 2 == 0 else [])
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    assert [len(by_call.detect(frame)) for _ in range(4)] == [1, 0, 1, 0]

    mapped = FakeDetector({2: [d]}, raise_on=[0], error=ValueError("boom"))
    with pytest.raises(ValueError, match="boom"):
        mapped.detect(frame)
    assert mapped.detect(frame) == []
    assert mapped.detect(frame) == [d]
