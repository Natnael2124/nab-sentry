"""Property 14: Identical frames pass once per keyframe interval.

**Validates: Requirements 3.5**

For any sample rate ``r``, integer ``m >= 1`` with ``K = m / r`` in [1, 600], and a
run of ``n`` identical sampled frames at offsets ``k / r`` (``k = 0..n-1``, so the
run spans ``D = (n - 1) / r`` seconds from 0), the Motion_Gate passes exactly
``1 + floor(D / K)`` frames.

Notes and assumptions:

- Expected count uses exact integers: ``floor(D / K) = floor((n - 1) / m)``, so no
  float division is involved on the expected side. The gate compares float offsets
  with ``EPS = 1e-9``; with offsets <= a few thousand seconds the accumulated float
  error is ~1e-12, well inside EPS, while a one-sample shortfall is >= 1/30 s.
- Motion_Threshold is drawn from (0, 1]. Identical frames have changed fraction
  exactly 0.0, so threshold 0.0 would pass every frame as "motion"
  (0.0 >= 0.0, Req 3.2) and the 1 + floor(D / K) count cannot hold. Req 3.5 /
  Property 14 implicitly assume threshold > 0.
- ``method="diff"`` (the default, design key decision 4). MOG2's output on a
  static scene is model-dependent and is not part of this property.
- To keep runs fast, ``m`` is capped at 300 samples per interval (K still reaches
  600 s at low rates) and the run covers at most 5 intervals.
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, strategies as st

from nab_sentry.ingest.motion import MotionGate
from nab_sentry.ingest.sampler import Sampler
from nab_sentry.ingest.sources import DecodedFrame
from tests.fakes import FakeSource

M_CAP = 300
MAX_INTERVALS = 4


@st.composite
def keyframe_cases(draw):
    rate = draw(st.floats(min_value=0.1, max_value=30.0,
                          allow_nan=False, allow_infinity=False))
    m_lo = max(1, math.ceil(rate))  # K = m / rate >= 1
    m_hi = min(M_CAP, math.floor(600.0 * rate))  # K <= 600
    m = draw(st.integers(min_value=m_lo, max_value=max(m_lo, m_hi)))
    k_interval = m / rate
    # Guard against rounding at the [1, 600] edges.
    if not (1.0 <= k_interval <= 600.0):
        m = m_lo
        k_interval = m / rate
    q = draw(st.integers(min_value=0, max_value=MAX_INTERVALS))
    rem = draw(st.integers(min_value=0, max_value=m - 1))
    n = q * m + rem + 1  # number of identical sampled frames
    return rate, m, k_interval, n


gate_params = st.fixed_dictionaries({
    "threshold": st.floats(min_value=0.0, max_value=1.0, exclude_min=True,
                           allow_nan=False),
    "gate_width": st.integers(min_value=64, max_value=1920),
    "pixel_delta": st.integers(min_value=1, max_value=255),
})


@st.composite
def static_images(draw):
    h = draw(st.integers(min_value=4, max_value=24))
    w = draw(st.integers(min_value=4, max_value=32))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    return np.random.default_rng(seed).integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _count_passes(gate: MotionGate, frames) -> tuple[int, list[str]]:
    reasons = [gate.evaluate(f) for f in frames]
    return sum(d.passed for d in reasons), [d.reason for d in reasons]


@given(case=keyframe_cases(), params=gate_params, image=static_images())
def test_identical_frames_pass_once_per_keyframe_interval(case, params, image):
    rate, m, k_interval, n = case
    gate = MotionGate(params["threshold"], k_interval, params["gate_width"],
                      params["pixel_delta"], method="diff")
    frames = [DecodedFrame(index=k, offset_s=k / rate, image=image) for k in range(n)]

    passed, reasons = _count_passes(gate, frames)

    expected = 1 + (n - 1) // m  # 1 + floor(D / K) with D = (n-1)/r, K = m/r
    assert passed == expected, (rate, m, k_interval, n, reasons)
    # Passes are the first frame and a keyframe at every multiple of m samples.
    assert reasons[0] == "first"
    passed_idx = [i for i, r in enumerate(reasons) if r != "static"]
    assert passed_idx == list(range(0, n, m))
    assert all(r == "keyframe" for r in reasons[m::m])


@given(case=keyframe_cases(), params=gate_params, image=static_images(),
       fps_mult=st.integers(min_value=1, max_value=3))
def test_identical_frames_via_sampler(case, params, image, fps_mult):
    """Same property with offsets produced by the Sampler over a source whose fps
    is an integer multiple of the sample rate."""
    rate, m, k_interval, n = case
    fps = rate * fps_mult
    n_source = (n - 1) * fps_mult + 1
    source = FakeSource(images=[image] * n_source, fps=fps)
    sampler = Sampler(rate)
    sampled = list(sampler.select(source.frames()))
    assert len(sampled) == n

    gate = MotionGate(params["threshold"], k_interval, params["gate_width"],
                      params["pixel_delta"], method="diff")
    passed, reasons = _count_passes(gate, sampled)

    assert passed == 1 + (n - 1) // m, (rate, m, fps_mult, n, reasons)


def test_example_default_interval_at_one_fps():
    """60 s of identical frames at 1 fps with K = 10 s -> frames 0,10,...,60 pass."""
    image = np.full((12, 16, 3), 80, dtype=np.uint8)
    gate = MotionGate(0.02, 10.0, 320, 25)
    frames = [DecodedFrame(index=k, offset_s=float(k), image=image) for k in range(61)]
    passed, reasons = _count_passes(gate, frames)
    assert passed == 7
    assert [i for i, r in enumerate(reasons) if r != "static"] == list(range(0, 61, 10))
