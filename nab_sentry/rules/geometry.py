"""Pure geometry for zones and lines (V2 Requirement 12.1-12.2).

Coordinates are normalized to the frame: ``x`` and ``y`` in 0..1 from the top-left corner, as
in ``config/cameras.yaml``.

* :func:`anchor` - where an object touches the ground: the bottom-centre of its box.
* :func:`point_in_polygon` - points on the boundary count as inside.
* :func:`side` / :func:`crossing` - which side of a directed line a point is on, and whether a
  movement crossed the line segment and in which direction. A line ``[a, b]`` is drawn as an
  arrow from ``a`` to ``b`` (``scripts/snapshot.py``); crossing to the arrow's **left** is
  ``+1`` ("in"), to its right ``-1`` ("out"). In image coordinates (``y`` grows downwards)
  "left of the arrow" is the side you see on the left when looking along the arrow on screen.
"""

from __future__ import annotations

from collections.abc import Sequence

Point = tuple[float, float]

EPS = 1e-12


def anchor(box: Sequence[float], width: int, height: int) -> Point:
    """Normalized bottom-centre of a pixel box ``(x1, y1, x2, y2)``."""
    x1, _y1, x2, y2 = box
    return ((x1 + x2) / 2 / width, y2 / height)


def _cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _on_segment(p: Point, a: Point, b: Point) -> bool:
    if abs(_cross(a, b, p)) > EPS * max(1.0, abs(b[0] - a[0]) + abs(b[1] - a[1])):
        return False
    return (min(a[0], b[0]) - EPS <= p[0] <= max(a[0], b[0]) + EPS
            and min(a[1], b[1]) - EPS <= p[1] <= max(a[1], b[1]) + EPS)


def point_in_polygon(p: Point, poly: Sequence[Point]) -> bool:
    """True if ``p`` is inside the simple polygon ``poly`` or on its boundary."""
    n = len(poly)
    if n < 3:
        return False
    for i in range(n):
        if _on_segment(p, poly[i], poly[(i + 1) % n]):
            return True
    inside = False
    x, y = p
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):  # edge straddles the horizontal ray
            x_at = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_at:
                inside = not inside
    return inside


def side(a: Point, b: Point, p: Point) -> int:
    """Which side of the directed line ``a -> b`` ``p`` is on, as seen on screen (``y``
    downwards): ``+1`` left, ``-1`` right, ``0`` on the line."""
    c = _cross(a, b, p)
    if abs(c) <= EPS:
        return 0
    # With y pointing down the screen, a positive cross product is a clockwise turn, i.e. the
    # point is to the right of the arrow as seen on screen.
    return -1 if c > 0 else 1


def crossing(p: Point, q: Point, a: Point, b: Point) -> int:
    """Direction in which the movement ``p -> q`` crossed the segment ``a``-``b``.

    ``+1``: from the arrow's right to its left ("in"); ``-1``: left to right ("out"); ``0``: no
    crossing - the endpoints are not strictly on opposite sides of the line, or the movement
    passes beside the segment rather than through it. Callers that let an object pause exactly
    on the line pass its last position that was off the line as ``p``.
    """
    sp, sq = side(a, b, p), side(a, b, q)
    if sp == 0 or sq == 0 or sp == sq:
        return 0
    # The movement crosses the infinite line; check it crosses within the segment a-b: a and b
    # must not lie strictly on the same side of the movement's line.
    da, db = _cross(p, q, a), _cross(p, q, b)
    if (da > EPS and db > EPS) or (da < -EPS and db < -EPS):
        return 0
    return sq
