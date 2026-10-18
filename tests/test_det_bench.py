"""Fast tests for scripts/det_bench.py pure helpers (no models)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from nab_sentry.ingest.detector import Detection

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "det_bench.py"


def _load():
    spec = importlib.util.spec_from_file_location("det_bench_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look up the defining module
    spec.loader.exec_module(mod)
    return mod


db = _load()


def det(cls, x1, y1, x2, y2):
    return Detection(cls=cls, conf=0.9, box=(x1, y1, x2, y2))


def test_parse_size():
    assert db.parse_size("416") == (416, 416)
    assert db.parse_size("384x640") == (384, 640)
    for bad in ("400", "0", "384x640x2", "16"):
        with pytest.raises(ValueError):
            db.parse_size(bad)
    assert db.size_label((384, 640)) == "384x640" and db.size_label((416, 416)) == "416"


def test_match_requires_same_class_and_iou():
    base = [det("person", 0, 0, 100, 200), det("car", 300, 300, 500, 400)]
    other = [det("person", 5, 5, 100, 200), det("truck", 300, 300, 500, 400), det("car", 0, 0, 10, 10)]
    flags, extra = db.match(base, other)
    assert flags == [True, False]
    assert extra == 2


def test_agreement_buckets_and_recall():
    base = [[det("person", 0, 0, 10, 50), det("person", 0, 0, 10, 150), det("person", 0, 0, 10, 300)]]
    other = [[det("person", 0, 0, 10, 150), det("person", 0, 0, 10, 300)]]
    a = db.agreement(base, other)
    assert a.total == {"small": 1, "medium": 1, "large": 1}
    assert a.recall("small") == 0.0 and a.recall("large") == 1.0
    assert a.recall() == pytest.approx(2 / 3) and a.extra == 0
    assert np.isnan(db.agreement([[]], [[]]).recall())


def test_busy_fraction():
    assert db.busy_fraction(100.0, 5) == pytest.approx(0.5)


def test_letterbox_rect_shape_and_meta():
    frame = np.zeros((776, 1376, 3), np.uint8)
    tensor, meta = db.letterbox_rect(frame, (384, 640))
    assert tensor.shape == (1, 3, 384, 640)
    assert meta.scale == pytest.approx(640 / 1376)
    assert meta.pad_x == 0 and meta.pad_y == pytest.approx((384 - round(776 * 640 / 1376)) // 2)


class _Fake:
    def __init__(self, tag):
        self.tag, self.calls = tag, 0

    def detect(self, frame):
        self.calls += 1
        return [self.tag]


def test_time_interleaved_runs_every_detector_on_every_frame():
    dets = {"a": _Fake("a"), "b": _Fake("b")}
    frames = [np.zeros((4, 4, 3), np.uint8)] * 7
    out = db.time_interleaved(dets, frames, iters=2)
    for k in dets:
        times, per_frame = out[k]
        assert len(times) == 14 and per_frame == [[k]] * 7
        assert dets[k].calls == 5 + 14  # warm-up + timed
