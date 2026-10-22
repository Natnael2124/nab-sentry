"""Zone and line geometry (V2 Requirement 12.1-12.2, task 12.2): Properties 13 and 14."""

from __future__ import annotations

import math

import cv2
import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st

from nab_sentry.rules.geometry import anchor, crossing, point_in_polygon, side

unit = st.floats(0.0, 1.0, allow_nan=False)
points = st.tuples(unit, unit)


def _dist_to_segment(p, a, b) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
    return math.hypot(p[0] - (ax + t * dx), p[1] - (ay + t * dy))


@st.composite
def simple_polygons(draw):
    """Star-shaped polygons (vertices sorted by angle around a centre): always simple, often
    non-convex."""
    cx, cy = draw(st.floats(0.3, 0.7)), draw(st.floats(0.3, 0.7))
    n = draw(st.integers(3, 9))
    angles = sorted(draw(st.lists(st.floats(0, 2 * math.pi, exclude_max=True), min_size=n,
                                  max_size=n, unique=True)))
    assume(all(b - a > 0.05 for a, b in zip(angles, angles[1:] + [angles[0] + 2 * math.pi])))
    radii = draw(st.lists(st.floats(0.05, 0.3), min_size=n, max_size=n))
    return [(cx + r * math.cos(t), cy + r * math.sin(t)) for t, r in zip(angles, radii)]


# -- Property 13 -------------------------------------------------------------------------------


@given(simple_polygons(), points)
def test_property_13_point_in_polygon_matches_opencv(poly, p):
    n = len(poly)
    near_edge = min(_dist_to_segment(p, poly[i], poly[(i + 1) % n]) for i in range(n))
    assume(near_edge > 1e-6)  # OpenCV works in float32 here; keep clear of the boundary
    ref = cv2.pointPolygonTest(np.array(poly, np.float32).reshape(-1, 1, 2), p, False) > 0
    assert point_in_polygon(p, poly) == ref


@given(simple_polygons(), st.floats(0, 1), st.integers(0, 20))
def test_property_13_boundary_points_are_inside(poly, t, edge):
    a, b = poly[edge % len(poly)], poly[(edge + 1) % len(poly)]
    on_edge = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
    assert point_in_polygon(on_edge, poly)
    assert all(point_in_polygon(v, poly) for v in poly)


def test_point_in_polygon_examples():
    square = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
    assert point_in_polygon((0.5, 0.5), square)
    assert not point_in_polygon((0.9, 0.5), square)
    assert point_in_polygon((0.8, 0.5), square)  # on the edge
    u_shape = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.6, 0.9), (0.6, 0.4), (0.4, 0.4), (0.4, 0.9), (0.1, 0.9)]
    assert not point_in_polygon((0.5, 0.7), u_shape)  # inside the notch = outside
    assert point_in_polygon((0.2, 0.7), u_shape)
    assert not point_in_polygon((0.5, 0.5), [(0, 0), (1, 1)])


# -- Property 14 -------------------------------------------------------------------------------


@given(points, points, points, points)
def test_property_14_reversing_the_movement_negates_the_crossing(p, q, a, b):
    assert crossing(q, p, a, b) == -crossing(p, q, a, b)


@given(points, points, points, points)
def test_property_14_a_crossing_really_passes_through_the_segment(p, q, a, b):
    c = crossing(p, q, a, b)
    if c != 0:
        assert side(a, b, p) == -c and side(a, b, q) == c
        # The intersection point of the two lines lies within both segments.
        d = (q[0] - p[0]) * (b[1] - a[1]) - (q[1] - p[1]) * (b[0] - a[0])
        assume(abs(d) > 1e-9)
        t = ((a[0] - p[0]) * (b[1] - a[1]) - (a[1] - p[1]) * (b[0] - a[0])) / d
        u = ((a[0] - p[0]) * (q[1] - p[1]) - (a[1] - p[1]) * (q[0] - p[0])) / d
        assert -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9


@given(points, points, points, points, st.floats(1.5, 5.0))
def test_property_14_movement_beside_the_segment_never_crosses(p, q, a, b, k):
    """Moving the line far along its own direction past the movement removes any crossing."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    assume(math.hypot(dx, dy) > 1e-3)
    far_a = (a[0] + k * 10 * dx, a[1] + k * 10 * dy)
    far_b = (b[0] + k * 10 * dx, b[1] + k * 10 * dy)
    # p, q in 0..1 and the shifted segment starts >= 15 line-lengths away
    assume(min(math.hypot(far_a[0] - x, far_a[1] - y) for x, y in (p, q)) > 2)
    assert crossing(p, q, far_a, far_b) == 0


def test_crossing_direction_as_seen_on_screen():
    a, b = (0.2, 0.5), (0.8, 0.5)  # arrow pointing right across the middle of the frame
    up, down = (0.5, 0.3), (0.5, 0.7)  # above / below on screen
    assert side(a, b, up) == 1  # left of a right-pointing arrow is up on screen
    assert crossing(down, up, a, b) == 1  # walking up the screen = "in"
    assert crossing(up, down, a, b) == -1
    assert crossing((0.9, 0.7), (0.9, 0.3), a, b) == 0  # beside the segment
    assert crossing(down, (0.5, 0.5), a, b) == 0  # onto the line: not yet a crossing


def test_anchor_is_bottom_centre_normalized():
    assert anchor((100, 50, 140, 250), 400, 500) == (0.3, 0.5)
