"""Motion gate ROI (V2 Requirement 3.1, task 3.2): Properties 3 and 4."""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from nab_sentry.ingest.motion import MotionGate
from nab_sentry.ingest.sources import DecodedFrame

H, W = 48, 64


@st.composite
def clips(draw):
    """Short frame sequences mixing static stretches and random changes."""
    n = draw(st.integers(1, 12))
    frames, cur = [], np.full((H, W, 3), draw(st.integers(0, 255)), np.uint8)
    for i in range(n):
        if draw(st.booleans()):
            rng = np.random.default_rng(draw(st.integers(0, 2**31 - 1)))
            y, x = int(rng.integers(0, H - 8)), int(rng.integers(0, W - 8))
            cur = cur.copy()
            cur[y:y + int(rng.integers(2, 20)), x:x + int(rng.integers(2, 30))] = int(rng.integers(0, 256))
        frames.append(DecodedFrame(i, float(i), cur))
    return frames


params = st.fixed_dictionaries({
    "threshold": st.sampled_from([0.0, 0.01, 0.05, 0.2]),
    "keyframe_interval_s": st.sampled_from([1.0, 3.0, 600.0]),
    "gate_width": st.sampled_from([32, 64]),
    "pixel_delta": st.sampled_from([10, 25]),
    "method": st.sampled_from(["diff", "mog2"]),
})


@given(params, clips())
def test_property_3_full_frame_roi_equals_no_roi(p, frames):
    a, b = MotionGate(**p), MotionGate(**p, roi=(0.0, 0.0, 1.0, 1.0))
    for f in frames:
        da, db = a.evaluate(f), b.evaluate(f)
        assert (da.passed, da.reason) == (db.passed, db.reason)
        assert abs(da.fraction - db.fraction) < 1e-12


@given(params.filter(lambda p: p["threshold"] > 0), st.integers(1, 12), st.data())
def test_property_4_changes_outside_the_roi_never_pass(p, n, data):
    roi = (0.5, 0.5, 1.0, 1.0)  # bottom-right quarter
    gate = MotionGate(**{**p, "keyframe_interval_s": 600.0}, roi=roi)
    base = np.full((H, W, 3), 100, np.uint8)
    for i in range(n):
        img = base.copy()
        if i:
            v = data.draw(st.integers(0, 255))
            img[: H // 2 - 4, :] = v  # top half changes (outside, with a blur margin)
            img[:, : W // 2 - 4] = v  # left half changes
        d = gate.evaluate(DecodedFrame(i, float(i), img))
        assert d.passed == (i == 0)  # only the first frame (no keyframe within 600 s)


def test_change_inside_the_roi_passes():
    gate = MotionGate(0.02, 600.0, 64, 25, roi=(0.5, 0.5, 1.0, 1.0))
    base = np.full((H, W, 3), 100, np.uint8)
    gate.evaluate(DecodedFrame(0, 0.0, base))
    moved = base.copy()
    moved[30:44, 40:60] = 250
    assert gate.evaluate(DecodedFrame(1, 1.0, moved)).reason == "motion"
