"""Pure parts of scripts/acceptance.py and scripts/make_scene.py (V2 task 17)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_script", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


acc = _load("acceptance")
scene = _load("make_scene")


def test_match_alerts_counts_and_delays():
    exp = [{"rule": "a", "t": 10.0}, {"rule": "a", "t": 100.0}, {"rule": "b", "t": 50.0}]
    act = [{"rule": "a", "t": 11.2}, {"rule": "a", "t": 300.0}, {"rule": "c", "t": 5.0}]
    s = acc.match_alerts(exp, act, tol=5.0)
    assert s["rules"]["a"] == {"expected": 2, "actual": 2, "tp": 1, "fn": 1, "fp": 1,
                               "missed_at": [100.0], "extra_at": [300.0], "delay_s": [1.2]}
    assert s["rules"]["b"]["fn"] == 1 and s["rules"]["c"]["fp"] == 1
    assert s["recall"] == pytest.approx(1 / 3) and s["precision"] == pytest.approx(1 / 3)
    assert acc.match_alerts([], [], 5.0)["recall"] == 1.0


@given(st.lists(st.tuples(st.sampled_from("ab"), st.floats(0, 500)), max_size=12), st.floats(-4, 4))
def test_shifted_copy_is_a_perfect_match(items, shift):
    exp = [{"rule": r, "t": t} for r, t in items]
    act = [{"rule": r, "t": t + shift} for r, t in items]
    s = acc.match_alerts(exp, act, tol=5.0)
    assert s["recall"] == 1.0 and s["precision"] == 1.0


def test_idle():
    objs = [{"visible": [100.0, 120.0]}, {"visible": None}]
    assert acc.idle(50.0, objs) and acc.idle(124.0, objs)
    assert not acc.idle(98.0, objs) and not acc.idle(110.0, objs)


def test_scene_walks_and_expected_alert_coverage():
    walks = scene.scenario(30)
    w = walks[0]
    a, b = w.interval
    assert w.at(a - 1) is None and w.at(b + 1) is None
    x0, _ = w.at(a)
    x1, _ = w.at(b)
    assert x0 < 0 < 1 < x1  # enters and leaves outside the frame
    stop = walks[2]  # E3 stands still for 40 s
    k = stop.schedule()
    assert any(k[i][1:] == k[i + 1][1:] and k[i + 1][0] - k[i][0] == pytest.approx(40) for i in range(len(k) - 1))
    starts = sorted(w.interval[0] for w in walks)
    assert starts[-1] < 30 * 60  # every event inside the footage


gs = _load("gate_sweep")


def test_missed_incidents_and_idle_fraction():
    inc = [(10.0, 20.0), (50.0, 55.0)]
    assert gs.missed_incidents([12.0], inc) == [(50.0, 55.0)]
    assert gs.missed_incidents([12.0, 56.0], inc) == []  # 1 s slack after the end
    sampled = [float(t) for t in range(0, 100)]
    passed = [0.0, 12.0, 30.0, 52.0, 90.0]
    # idle seconds: outside [8, 22] and [48, 57]
    idle_n = sum(1 for t in sampled if not (8 <= t <= 22 or 48 <= t <= 57))
    assert gs.idle_pass_fraction(sampled, passed, inc) == pytest.approx(3 / idle_n)
    assert gs.idle_pass_fraction([12.0], [12.0], inc) is None


@given(st.lists(st.floats(0, 100), max_size=30), st.lists(st.tuples(st.floats(0, 90), st.floats(0, 10)), max_size=5))
def test_more_passes_never_miss_more(passed, raw):
    inc = [(a, a + d) for a, d in raw]
    fewer = gs.missed_incidents(passed[: len(passed) // 2], inc)
    assert len(gs.missed_incidents(passed, inc)) <= len(fewer)
