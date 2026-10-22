"""Watch rules (V2 Requirement 12.3-12.6, task 13.3): loader, Properties 15-17, trigger examples."""

from __future__ import annotations

from datetime import datetime, time, timedelta

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nab_sentry.live.camera_config import CameraSpec
from nab_sentry.rules.config import DAYS, Rule, RulesConfigError, Schedule, parse_rules
from nab_sentry.rules.engine import Obs, RuleEngine

SQUARE = ((0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8))
LINE = ((0.0, 0.5), (1.0, 0.5))  # arrow to the right; "in" = moving up the screen
CAM = CameraSpec("CAM01", "Gate", "rtsp://h/x", zones={"yard": SQUARE}, lines={"gate": LINE})
MONDAY = datetime(2026, 10, 19, 12, 0)  # a Monday, noon
DT = 1 / 3


def rule(trigger="enter", **kw) -> Rule:
    base = dict(id="r1", name="R", camera="CAM01", classes=frozenset({"person"}), trigger=trigger,
                zone="yard", zone_poly=SQUARE, debounce=2, cooldown_s=0.0)
    if trigger == "cross":
        base.update(zone=None, zone_poly=None, line="gate", line_pts=LINE)
    base.update(kw)
    return Rule(**base)


def run(engine: RuleEngine, frames, t0: float = 0.0, local: datetime = MONDAY):
    """frames: list of lists of Obs, one list per frame at 3 fps."""
    out = []
    for i, obs in enumerate(frames):
        out += engine.step(t0 + i * DT, local + timedelta(seconds=i * DT), obs)
    return out


# -- loader --------------------------------------------------------------------------------------


def _raw(**kw):
    r = {"id": "yard-night", "name": "Yard at night", "camera": "CAM01", "zone": "yard",
         "classes": ["person"], "trigger": {"type": "enter"},
         "schedule": {"days": ["mon"], "start": "22:00", "end": "06:00"}}
    r.update(kw)
    return r


def test_loader_accepts_valid_rules_and_resolves_geometry():
    rules = parse_rules({"rules": [
        _raw(),
        {"id": "gate-in", "camera": "CAM01", "line": "gate", "direction": "in",
         "classes": ["person", "car"], "trigger": {"type": "cross"}},
        _raw(id="loiter", trigger={"type": "dwell", "seconds": 20}, schedule=None),
        _raw(id="crowd", trigger={"type": "count", "min": 3}, schedule=None),
    ]}, [CAM])
    assert [r.trigger for r in rules] == ["enter", "cross", "dwell", "count"]
    assert rules[0].zone_poly == SQUARE and rules[0].debounce == 2 and rules[0].cooldown_s == 300
    assert rules[1].line_pts == LINE and rules[1].direction == 1 and rules[1].name == "gate-in"
    assert rules[2].seconds == 20 and rules[3].min_count == 3
    assert parse_rules(None, [CAM]) == [] and parse_rules({"rules": []}, [CAM]) == []


BAD = {
    "id": {"id": "bad id!"},
    "camera": {"camera": "CAM09"},
    "zone": {"zone": "nowhere"},
    "classes": {"classes": ["dog"]},
    "trigger.type": {"trigger": {"type": "teleport"}},
    "trigger.seconds": {"trigger": {"type": "dwell"}},
    "trigger.min": {"trigger": {"type": "count", "min": 0}},
    "schedule.start": {"schedule": {"days": ["mon"], "start": "25:00", "end": "06:00"}},
    "schedule.days": {"schedule": {"days": ["funday"], "start": "22:00", "end": "06:00"}},
    "debounce": {"debounce": 0},
    "cooldown_s": {"cooldown_s": -1},
    "severity": {"severity": "panic"},
    "line": {"line": "gate"},  # a zone trigger with a line
}


@pytest.mark.parametrize("field", sorted(BAD))
def test_loader_reports_each_bad_field_once(field):
    with pytest.raises(RulesConfigError) as exc:
        parse_rules({"rules": [_raw(), {**_raw(id="second"), **BAD[field]}]}, [CAM])
    assert len(exc.value.issues) == 1
    assert exc.value.issues[0].startswith(f"rules[1].{field}")


def test_loader_cross_rule_errors_and_duplicates():
    with pytest.raises(RulesConfigError) as exc:
        parse_rules({"rules": [_raw(), _raw()]}, [CAM])
    assert exc.value.issues == ["rules[1].id: duplicate 'yard-night'"]
    with pytest.raises(RulesConfigError) as exc:
        parse_rules({"rules": [{"id": "x", "camera": "CAM01", "line": "nope", "direction": "sideways",
                                "classes": ["person"], "trigger": {"type": "cross"}}]}, [CAM])
    assert [i.split(":")[0] for i in exc.value.issues] == ["rules[0].line", "rules[0].direction"]


# -- Property 15: schedule -----------------------------------------------------------------------


def reference_active(sched: Schedule, local: datetime) -> bool:
    """Each listed day d contributes the interval [d start, d start + duration) where duration is
    (end - start) mod 24 h, or 24 h when start == end."""
    start_m = sched.start.hour * 60 + sched.start.minute
    end_m = sched.end.hour * 60 + sched.end.minute
    dur = (end_m - start_m) % (24 * 60) or 24 * 60
    for back in (0, 1):
        day = (local - timedelta(days=back)).date()
        if day.weekday() in sched.days:
            begin = datetime.combine(day, sched.start)
            if begin <= local < begin + timedelta(minutes=dur):
                return True
    return False


times = st.builds(time, st.integers(0, 23), st.sampled_from([0, 15, 30, 59]))


@given(st.sets(st.integers(0, 6), min_size=1), times, times,
       st.datetimes(min_value=datetime(2026, 1, 1), max_value=datetime(2027, 12, 31)))
def test_property_15_schedule_matches_reference(days, start, end, local):
    sched = Schedule(frozenset(days), start, end)
    assert sched.active(local) == reference_active(sched, local)


def test_schedule_examples_overnight_and_dst():
    s = Schedule(frozenset({0}), time(22), time(6))  # Monday night
    assert s.active(datetime(2026, 10, 19, 23, 30))  # Monday 23:30
    assert s.active(datetime(2026, 10, 20, 5, 59))  # Tuesday 05:59 (still Monday's window)
    assert not s.active(datetime(2026, 10, 20, 6, 0))
    assert not s.active(datetime(2026, 10, 18, 23, 0))  # Sunday night: not listed
    # Wall-clock based: a DST change day (Europe, 2026-10-25) needs no special case.
    night = Schedule(frozenset(range(7)), time(22), time(6))
    assert night.active(datetime(2026, 10, 25, 2, 30)) and not night.active(datetime(2026, 10, 25, 12, 0))
    assert DAYS[MONDAY.weekday()] == "mon"


# -- Properties 16, 17 on random streams ---------------------------------------------------------


@st.composite
def streams(draw):
    n_tracks = draw(st.integers(1, 4))
    n_frames = draw(st.integers(1, 40))
    frames = []
    for _ in range(n_frames):
        obs = []
        for k in range(1, n_tracks + 1):
            if draw(st.booleans()):  # observed in this frame?
                p = draw(st.tuples(st.sampled_from([0.1, 0.3, 0.5, 0.7, 0.9]),
                                   st.sampled_from([0.1, 0.3, 0.45, 0.55, 0.7, 0.9])))
                obs.append(Obs(k, draw(st.sampled_from(["person", "car"])), p, draw(st.booleans())))
        frames.append(obs)
    return frames


rules_any = st.builds(
    lambda trig, deb, cool, secs, mc, direction: rule(trig, debounce=deb, cooldown_s=cool, seconds=secs,
                                                      min_count=mc, direction=direction),
    st.sampled_from(["enter", "dwell", "cross", "count"]), st.integers(1, 4),
    st.sampled_from([0.0, 1.0, 5.0, 60.0]), st.sampled_from([0.5, 2.0]), st.integers(1, 3),
    st.sampled_from([-1, 0, 1]))


@given(rules_any, streams())
def test_property_16_once_per_track_and_cooldown(r, frames):
    firings = run(RuleEngine([r]), frames)
    if r.trigger != "count":
        keys = [f.track_key for f in firings]
        assert len(keys) == len(set(keys))
    ts = [f.t for f in firings]
    assert all(b - a >= r.cooldown_s - 1e-9 for a, b in zip(ts, ts[1:]))


@given(rules_any.filter(lambda r: r.trigger in ("enter", "dwell")), streams())
def test_property_17_debounce_needs_consecutive_qualifying_observations(r, frames):
    firings = run(RuleEngine([r]), frames)
    for f in firings:
        i = round(f.t / DT)
        seen = [next(o for o in fr if o.track_key == f.track_key) for fr in frames[: i + 1]
                if any(o.track_key == f.track_key for o in fr)]
        last = seen[-r.debounce:]
        assert len(last) == r.debounce
        from nab_sentry.rules.geometry import point_in_polygon
        assert all(o.confirmed and o.cls in r.classes and point_in_polygon(o.point, SQUARE) for o in last)


# -- examples ------------------------------------------------------------------------------------


def person(k, x, y, confirmed=True):
    return Obs(k, "person", (x, y), confirmed)


def test_enter_debounce_and_unconfirmed():
    eng = RuleEngine([rule("enter", debounce=2)])
    frames = [[person(1, 0.5, 0.5, confirmed=False)], [person(1, 0.5, 0.5)], [person(1, 0.5, 0.5)]]
    [f] = run(eng, frames)
    assert f.track_key == 1 and f.t == pytest.approx(2 * DT)


def test_missed_frames_do_not_reset_but_leaving_does():
    eng = RuleEngine([rule("enter", debounce=3)])
    frames = [[person(1, 0.5, 0.5)], [], [person(1, 0.5, 0.5)], [person(1, 0.9, 0.9)],
              [person(1, 0.5, 0.5)], [person(1, 0.5, 0.5)], [], [person(1, 0.5, 0.5)]]
    [f] = run(eng, frames)
    assert f.t == pytest.approx(7 * DT)  # the outside observation at frame 3 reset the count


def test_dwell_needs_time_in_zone():
    eng = RuleEngine([rule("dwell", seconds=2.0, debounce=1)])
    frames = [[person(1, 0.5, 0.5)] for _ in range(10)]
    [f] = run(eng, frames)
    assert f.t == pytest.approx(6 * DT)  # first frame with t - entry >= 2 s


def test_cross_direction_debounce_and_pause_on_line():
    walk_in = [[person(1, 0.5, 0.7)], [person(1, 0.5, 0.5)], [person(1, 0.5, 0.4)], [person(1, 0.5, 0.3)]]
    [f] = run(RuleEngine([rule("cross", direction=1, debounce=2)]), walk_in)  # paused on the line
    assert f.t == pytest.approx(3 * DT)
    assert run(RuleEngine([rule("cross", direction=-1, debounce=1)]), walk_in) == []  # wrong way
    beside = [[person(1, 1.5, 0.7)], [person(1, 1.5, 0.3)]]  # x beyond the segment's end
    assert run(RuleEngine([rule("cross", direction=0, debounce=1)]), beside) == []


def test_count_fires_once_per_episode():
    eng = RuleEngine([rule("count", min_count=2, debounce=1)])
    two = [person(1, 0.5, 0.5), person(2, 0.4, 0.4)]
    frames = [[person(1, 0.5, 0.5)], two, two, [person(1, 0.5, 0.5)], two]
    fs = run(eng, frames)
    assert [round(f.t / DT) for f in fs] == [1, 4] and fs[0].n_tracks == 2 and fs[0].track_key == 2


def test_cooldown_holds_back_but_does_not_consume_other_tracks():
    eng = RuleEngine([rule("enter", debounce=1, cooldown_s=2.0)])
    frames = [[person(1, 0.5, 0.5)], [person(1, 0.5, 0.5), person(2, 0.4, 0.4)]] + \
             [[person(2, 0.4, 0.4)] for _ in range(8)]
    fs = run(eng, frames)
    assert [(f.track_key, round(f.t / DT)) for f in fs] == [(1, 0), (2, 6)]  # 2 waits for 2 s


def test_schedule_gates_firing():
    r = rule("enter", debounce=1, schedule=Schedule(frozenset({0}), time(22), time(6)))
    eng = RuleEngine([r])
    assert run(eng, [[person(1, 0.5, 0.5)]], local=datetime(2026, 10, 19, 12)) == []
    assert len(run(eng, [[person(2, 0.5, 0.5)]], t0=10, local=datetime(2026, 10, 19, 23))) == 1


def test_snapshot_restore_replays_identically():
    eng = RuleEngine([rule("enter", debounce=2, cooldown_s=5.0)])
    run(eng, [[person(1, 0.5, 0.5)]])
    snap = eng.snapshot()
    later = [[person(1, 0.5, 0.5)], [person(2, 0.4, 0.4)], [person(2, 0.4, 0.4)]]
    a = run(eng, later, t0=1.0)
    eng.restore(snap)
    assert run(eng, later, t0=1.0) == a
