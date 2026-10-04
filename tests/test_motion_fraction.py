"""Property 12: Changed fraction is bounded and downscaling never upscales.

**Validates: Requirements 3.1**
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from nab_sentry.ingest.motion import changed_fraction, downscale_gray

gate_widths = st.integers(min_value=64, max_value=1920)
pixel_deltas = st.integers(min_value=1, max_value=255)
seeds = st.integers(min_value=0, max_value=2**32 - 1)


def _image(seed: int, h: int, w: int, channels: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = (h, w) if channels == 0 else (h, w, channels)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


@settings(max_examples=150, deadline=None)
@given(
    seed=seeds,
    h=st.integers(min_value=1, max_value=240),
    w=st.integers(min_value=1, max_value=2600),
    channels=st.sampled_from([0, 1, 3, 4]),
    gate_width=gate_widths,
)
def test_downscale_never_upscales_and_keeps_aspect(seed, h, w, channels, gate_width):
    """**Validates: Requirements 3.1**"""
    img = _image(seed, h, w, channels)
    out = downscale_gray(img, gate_width)

    assert out.ndim == 2
    assert out.dtype == np.uint8
    out_h, out_w = out.shape
    assert out_w == min(w, gate_width)
    assert abs(out_h - h * out_w / w) <= 1.0
    if w <= gate_width:
        assert out.shape == (h, w)


@settings(max_examples=150, deadline=None)
@given(
    seed_a=seeds,
    seed_b=seeds,
    h=st.integers(min_value=1, max_value=120),
    w=st.integers(min_value=1, max_value=160),
    delta=pixel_deltas,
    mode=st.sampled_from(["random", "identical", "partial"]),
)
def test_changed_fraction_bounded_zero_on_identical_symmetric(seed_a, seed_b, h, w, delta, mode):
    """**Validates: Requirements 3.1**"""
    a = _image(seed_a, h, w, 0)
    if mode == "identical":
        b = a.copy()
    elif mode == "random":
        b = _image(seed_b, h, w, 0)
    else:
        b = a.copy()
        rng = np.random.default_rng(seed_b)
        mask = rng.random((h, w)) < 0.5
        b[mask] = 255 - b[mask]

    f = changed_fraction(a, b, delta)
    assert isinstance(f, float)
    assert 0.0 <= f <= 1.0
    assert changed_fraction(b, a, delta) == f
    assert changed_fraction(a, a.copy(), delta) == 0.0
    if mode == "identical":
        assert f == 0.0

    # Same properties on the gate's actual inputs (downscaled, blurred).
    sa, sb = downscale_gray(a, 64), downscale_gray(b, 64)
    g = changed_fraction(sa, sb, delta)
    assert 0.0 <= g <= 1.0
    assert changed_fraction(sa, sa.copy(), delta) == 0.0


@given(
    seed=seeds,
    h=st.integers(min_value=1, max_value=60),
    w=st.integers(min_value=1, max_value=60),
    delta=pixel_deltas,
)
def test_changed_fraction_differing_shapes_is_full_change(seed, h, w, delta):
    """**Validates: Requirements 3.1**"""
    a = _image(seed, h, w, 0)
    b = _image(seed, h + 1, w, 0)
    assert changed_fraction(a, b, delta) == 1.0
    assert changed_fraction(b, a, delta) == 1.0
