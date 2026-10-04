"""Property 16 (clamp_box part): bounding boxes are clamped integer boxes inside the frame.

**Validates: Requirements 4.3, 5.10, 6.3**
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from nab_sentry.ingest.detector import clamp_box

frame_dims = st.integers(min_value=1, max_value=4000)

# Mix of coordinates near the frame (common case, sub-pixel and boundary values)
# and arbitrary finite floats far outside it (negative, huge).
coords = st.one_of(
    st.floats(min_value=-50.0, max_value=4100.0, allow_nan=False, allow_infinity=False),
    st.integers(min_value=-10, max_value=4010).map(float),
    st.floats(allow_nan=False, allow_infinity=False),
)


def _oracle_axis(a: float, b: float, size: int) -> tuple[int, int] | None:
    """Independent oracle: round the float interval outward, intersect with [0, size]."""
    lo = math.floor(min(a, b))
    hi = math.ceil(max(a, b))
    lo_c = lo if lo > 0 else 0
    lo_c = size if lo_c > size else lo_c
    hi_c = hi if hi < size else size
    hi_c = 0 if hi_c < 0 else hi_c
    if hi_c - lo_c < 1:
        return None
    return lo_c, hi_c


@settings(max_examples=500)
@given(x1=coords, y1=coords, x2=coords, y2=coords, w=frame_dims, h=frame_dims)
def test_clamp_box_bounds_containment_and_none_oracle(x1, y1, x2, y2, w, h):
    result = clamp_box(x1, y1, x2, y2, w, h)

    ox = _oracle_axis(x1, x2, w)
    oy = _oracle_axis(y1, y2, h)

    # None exactly when the outward-rounded intersection is < 1 px on either axis.
    assert (result is None) == (ox is None or oy is None)
    if result is None:
        return

    assert len(result) == 4
    assert all(type(v) is int for v in result)
    rx1, ry1, rx2, ry2 = result
    assert 0 <= rx1 < rx2 <= w
    assert 0 <= ry1 < ry2 <= h
    assert (rx1, rx2) == ox
    assert (ry1, ry2) == oy

    # Containment: the result covers the float box's intersection with the frame.
    fx_lo, fx_hi = max(min(x1, x2), 0.0), min(max(x1, x2), float(w))
    fy_lo, fy_hi = max(min(y1, y2), 0.0), min(max(y1, y2), float(h))
    if fx_lo <= fx_hi:
        assert rx1 <= fx_lo and fx_hi <= rx2
    if fy_lo <= fy_hi:
        assert ry1 <= fy_lo and fy_hi <= ry2

    # The crop is non-empty.
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    crop = frame[ry1:ry2, rx1:rx2]
    assert crop.shape[0] >= 1 and crop.shape[1] >= 1


@given(
    x1=coords, y1=coords, x2=coords, y2=coords, w=frame_dims, h=frame_dims,
    bad=st.sampled_from([math.nan, math.inf, -math.inf]),
    idx=st.integers(min_value=0, max_value=3),
)
def test_clamp_box_non_finite_returns_none(x1, y1, x2, y2, w, h, bad, idx):
    args = [x1, y1, x2, y2]
    args[idx] = bad
    assert clamp_box(*args, w, h) is None


@given(x1=coords, y1=coords, x2=coords, y2=coords,
       w=st.integers(min_value=-5, max_value=4000), h=st.integers(min_value=-5, max_value=0))
def test_clamp_box_empty_frame_returns_none(x1, y1, x2, y2, w, h):
    assert clamp_box(x1, y1, x2, y2, w, h) is None
    assert clamp_box(x1, y1, x2, y2, h, w) is None


def test_clamp_box_examples():
    assert clamp_box(10.2, 20.7, 50.1, 60.0, 100, 100) == (10, 20, 51, 60)
    assert clamp_box(50.1, 60.0, 10.2, 20.7, 100, 100) == (10, 20, 51, 60)  # swapped
    assert clamp_box(-30.0, -5.5, 150.0, 120.0, 100, 80) == (0, 0, 100, 80)
    assert clamp_box(200.0, 10.0, 300.0, 20.0, 100, 100) is None  # fully outside
    assert clamp_box(5.0, 5.0, 5.0, 9.0, 100, 100) is None  # zero width
    assert clamp_box(5.2, 5.0, 5.3, 9.0, 100, 100) == (5, 5, 6, 9)  # sub-pixel -> 1 px
