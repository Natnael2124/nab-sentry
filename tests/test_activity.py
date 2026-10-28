"""Activity log entries (Step 3, task 18.2): zone time, crossings, summaries."""

from __future__ import annotations

from datetime import datetime

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nab_sentry.describe.activity import MAX_GAP_S, line_crossings, summary, zone_seconds

SQUARE = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
LINE = ((0.5, 0.0), (0.5, 1.0))  # pointing down: moving right = "in"


def walk(xs, y=0.5, dt=1 / 3):
    return [(i * dt, (x, y)) for i, x in enumerate(xs)]


def test_zone_seconds_counts_consecutive_inside_time():
    pts = walk([0.1, 0.3, 0.4, 0.5, 0.9])  # inside for 3 observations = 2 intervals
    assert zone_seconds(pts, {"yard": SQUARE}) == {"yard": pytest.approx(0.7, abs=0.05)}
    assert zone_seconds(walk([0.1, 0.9]), {"yard": SQUARE}) == {}


def test_zone_seconds_caps_gaps():
    pts = [(0.0, (0.5, 0.5)), (60.0, (0.5, 0.5))]  # one minute without observations
    assert zone_seconds(pts, {"yard": SQUARE})["yard"] == MAX_GAP_S


@given(st.lists(st.tuples(st.floats(0.01, 3.0), st.floats(0, 1), st.floats(0, 1)), max_size=40))
def test_zone_seconds_bounded_by_duration(steps):
    t, pts = 0.0, []
    for dt, x, y in steps:
        t += dt
        pts.append((t, (x, y)))
    secs = zone_seconds(pts, {"yard": SQUARE}).get("yard", 0.0)
    span = pts[-1][0] - pts[0][0] if pts else 0.0
    assert 0 <= secs <= span + 0.1


def test_line_crossings_directions_and_pause():
    there = walk([0.3, 0.5, 0.5, 0.7])  # pauses on the line
    back = walk([0.7, 0.3])
    assert [c["direction"] for c in line_crossings(there, {"path": LINE})] == ["in"]
    assert [c["direction"] for c in line_crossings(back, {"path": LINE})] == ["out"]
    assert line_crossings(walk([0.1, 0.4]), {"path": LINE}) == []


def test_summary_text():
    t = datetime(2026, 10, 23, 20, 2).timestamp()
    s = summary(t, t + 30, "Back yard", "person in a red top", {"lawn": 14.2, "drive_way": 0.3},
                [{"line": "path", "direction": "in", "t": t + 5}])
    assert s == "20:02 · Back yard · person in a red top · on the lawn 14 s, entered the drive way, crossed the path (in)"
    assert summary(t, t + 125, "Gate", "car", {}, []).endswith("· car · seen for 2 min 5 s")
