"""Activity log entries from a track's history (Step 3, Requirement 19) - pure functions.

:func:`zone_seconds` and :func:`line_crossings` turn a track's ground points (normalized
anchors with times) into time spent per zone and crossings per line; :func:`summary` writes the
one-line log entry, e.g. ``20:02 · Back yard · person in a red top · on the lawn 14 s, crossed
the path (in)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from nab_sentry.rules.geometry import Point, crossing, point_in_polygon, side

MAX_GAP_S = 2.0  # a gap between observations counts at most this long (missed detections)


def zone_seconds(points: Sequence[tuple[float, Point]], zones: dict[str, Sequence[Point]]) -> dict[str, float]:
    """Seconds spent in each zone: consecutive observations both inside count their time
    difference (capped at ``MAX_GAP_S``). Zones never entered are left out."""
    out: dict[str, float] = {}
    for name, poly in zones.items():
        total, prev = 0.0, None
        for t, p in points:
            inside = point_in_polygon(p, poly)
            if inside and prev is not None and prev[1]:
                total += min(MAX_GAP_S, max(0.0, t - prev[0]))
            if inside or prev is not None:
                prev = (t, inside)
        entered = any(point_in_polygon(p, poly) for _, p in points)
        if entered:
            out[name] = round(total, 1)
    return out


def line_crossings(points: Sequence[tuple[float, Point]],
                   lines: dict[str, tuple[Point, Point]]) -> list[dict]:
    """Crossings in time order: ``{"line", "direction": "in"|"out", "t"}``, measured from the
    last position that was off the line (a pause on the line still counts)."""
    out = []
    for name, (a, b) in lines.items():
        last_off = None
        for t, p in points:
            if side(a, b, p) == 0:
                continue
            if last_off is not None:
                c = crossing(last_off, p, a, b)
                if c:
                    out.append({"line": name, "direction": "in" if c > 0 else "out", "t": t})
            last_off = p
    return sorted(out, key=lambda c: c["t"])


def _secs(s: float) -> str:
    s = int(round(s))
    return f"{s} s" if s < 90 else f"{s // 60} min {s % 60} s" if s < 3600 else f"{s // 3600} h {s % 3600 // 60} min"


def summary(first_t: float, last_t: float, camera_label: str, what: str, zones: dict[str, float],
            crossings: Sequence[dict]) -> str:
    """One readable line; zones with under a second are mentioned without a time."""
    parts = []
    for name, secs in zones.items():
        parts.append(f"on the {name.replace('_', ' ')} {_secs(secs)}" if secs >= 1 else f"entered the {name.replace('_', ' ')}")
    for c in crossings:
        parts.append(f"crossed the {c['line'].replace('_', ' ')} ({c['direction']})")
    if not parts:
        parts.append(f"seen for {_secs(last_t - first_t)}")
    when = datetime.fromtimestamp(first_t).strftime("%H:%M")
    return f"{when} · {camera_label} · {what} · {', '.join(parts)}"
