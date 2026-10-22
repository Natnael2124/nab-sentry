"""Low-confidence detections for the tracker (V2 Requirement 9.1, task 9.2).

The central property: with any ``low_conf <= conf``, the high detections are exactly the V1
result for ``conf`` (same boxes, scores, order), so crop embeddings and search are unchanged.
New fields never make a V1 config invalid: ``det_low_conf`` above ``det_conf`` acts as
``det_conf`` and the tracking rate never drops below ``sample_rate``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from nab_sentry.config import Config, ConfigError, effective_low_conf, effective_track_rate
from nab_sentry.ingest.detector import Detection, LetterboxMeta, OnnxYoloDetector, postprocess
from tests.fakes import FakeDetector
from tests.test_detector_post import cases

META = LetterboxMeta(1.0, 0.0, 0.0)


def _two_people() -> np.ndarray:
    raw = np.zeros((1, 84, 2), np.float32)
    raw[0, :4, 0], raw[0, 4, 0] = (100, 100, 40, 80), 0.9  # person, high
    raw[0, :4, 1], raw[0, 4, 1] = (300, 300, 40, 80), 0.2  # person, low
    return raw


@given(cases(), st.floats(0.0, 1.0))
def test_high_detections_equal_v1(case, frac):
    raw, meta, fw, fh, conf, iou, max_det = case
    v1 = postprocess(raw, meta, fw, fh, conf, iou, max_det)
    both = postprocess(raw, meta, fw, fh, conf, iou, max_det, conf * frac)
    assert [d for d in both if d.high] == v1
    assert all(d.high for d in v1)


@given(cases(), st.floats(0.0, 1.0))
def test_low_detections_are_bounded_and_below_conf(case, frac):
    raw, meta, fw, fh, conf, iou, max_det = case
    low_conf = conf * frac
    both = postprocess(raw, meta, fw, fh, conf, iou, max_det, low_conf)
    low = [d for d in both if not d.high]
    assert len(low) <= max_det
    for d in low:
        assert low_conf - 1e-6 <= d.conf < conf + 1e-6
    flags = [d.high for d in both]  # highs first, each group by confidence desc
    assert flags == sorted(flags, reverse=True)
    confs = [d.conf for d in low]
    assert confs == sorted(confs, reverse=True)


def test_low_conf_none_equal_or_above_conf_is_v1():
    raw = _two_people()
    v1 = postprocess(raw, META, 640, 640, 0.35, 0.45, 10)
    assert [d.conf for d in v1] == [pytest.approx(0.9)]
    assert postprocess(raw, META, 640, 640, 0.35, 0.45, 10, 0.35) == v1
    assert postprocess(raw, META, 640, 640, 0.35, 0.45, 10, 0.8) == v1
    both = postprocess(raw, META, 640, 640, 0.35, 0.45, 10, 0.1)
    assert [(round(d.conf, 2), d.high) for d in both] == [(0.9, True), (0.2, False)]


def test_detector_validates_low_conf_range(tmp_path: Path):
    with pytest.raises(ConfigError) as exc:
        OnnxYoloDetector(tmp_path / "x.onnx", conf=0.35, iou=0.45, max_det=10, low_conf=-0.1)
    assert [i.name for i in exc.value.issues] == ["det_low_conf"]


def test_config_effective_values_keep_v1_configs_valid():
    assert Config().validate() == []
    # A V1 user lowering det_conf or raising sample_rate gets no error about the new fields.
    assert Config(det_conf=0.05).validate() == []
    assert effective_low_conf(Config(det_conf=0.05)) == 0.05
    assert effective_low_conf(Config()) == 0.1
    assert Config(sample_rate=5).validate() == []
    assert effective_track_rate(Config(sample_rate=5)) == 5.0
    assert effective_track_rate(Config()) == 3.0
    assert [i.name for i in Config(track_rate=0.5).validate()] == ["track_rate"]
    assert [i.name for i in Config(det_low_conf=1.5).validate()] == ["det_low_conf"]


def test_pipeline_embeds_only_high_detections():
    """A low detection reaches the pipeline but gets no crop (task 9.1)."""
    from nab_sentry.ingest.pipeline import IngestPipeline

    det = FakeDetector(lambda frame, i: [
        Detection("person", 0.9, (2, 3, 30, 40)),
        Detection("car", 0.2, (10, 5, 60, 45), high=False),
    ])
    pipe = IngestPipeline.__new__(IngestPipeline)
    kept = pipe._detect(det, Path("v.mp4"), 0.0, np.zeros((64, 64, 3), np.uint8))
    assert [(d.cls, d.high) for d, _ in kept] == [("person", True)]
