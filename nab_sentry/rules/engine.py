"""Deterministic rule evaluation (V2 Requirement 12.3-12.4, task 13.2).

:class:`RuleEngine` holds the rules of **one camera** and is fed, frame by frame in time order,
the observations of the tracked objects in that frame (:class:`Obs`: track key, class,
confirmed flag, normalized ground point). It returns the rules that fire on that frame.

Semantics, per rule:

* **qualifying** observation: confirmed track, class in the rule's classes, local time inside
  the schedule, and the spatial condition of the trigger;
* ``enter`` fires after ``debounce`` consecutive qualifying observations of a track in the zone;
* ``dwell`` fires once a track has been in the zone for at least ``seconds`` (and ``debounce``
  consecutive observations);
* ``cross`` fires when a track crossed the line in the rule's direction and then stayed on the new
  side for ``debounce`` observations (the crossing one included); the crossing is measured from
  the track's last position that was off the line, so pausing on the line still counts;
* ``count`` fires when at least ``min_count`` tracks qualify in the same frame, ``debounce``
  frames in a row; it re-arms only after the count drops below ``min_count``.

An observation that does not qualify resets the track's progress; a frame in which a track is
not observed (missed detection) changes nothing. A rule fires at most once per track (``count``:
once per episode) and never twice within ``cooldown_s`` on its camera. A track held back by the
cooldown is not marked: it can still fire once the cooldown is over, as it may be a different
intruder than the one already reported.

Muting is not the engine's business: the pipeline stores a muted firing as ``suppressed``, and it
still starts the cooldown (so a mute never causes a burst of alerts when it ends).
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from nab_sentry.rules.config import Rule
from nab_sentry.rules.geometry import Point, crossing, point_in_polygon, side

FORGET_S = 120.0  # per-track state of tracks not observed for this long is dropped


@dataclass(frozen=True)
class Obs:
    track_key: int
    cls: str
    point: Point  # normalized ground point (geometry.anchor)
    confirmed: bool = True


@dataclass(frozen=True)
class Firing:
    rule_id: str
    t: float  # epoch seconds
    track_key: int | None  # count rules: the track whose arrival reached the count
    point: Point | None
    n_tracks: int = 1


@dataclass
class _TrackState:
    consec: int = 0
    entry_t: float | None = None  # dwell: when the track entered the zone
    off_line: Point | None = None  # cross: last position strictly off the line
    cross_dir: int = 0  # cross: direction of a pending crossing
    fired: bool = False
    last_t: float = 0.0


@dataclass
class _RuleState:
    tracks: dict[int, _TrackState] = field(default_factory=dict)
    count_consec: int = 0
    count_armed: bool = True
    last_fire: float | None = None


class RuleEngine:
    def __init__(self, rules: Sequence[Rule]) -> None:
        self.rules = list(rules)
        self._state: dict[str, _RuleState] = {r.id: _RuleState() for r in self.rules}

    # -- cooldown seed (from stored alerts after a restart) ------------------------------------

    def set_last_fire(self, rule_id: str, t: float) -> None:
        if rule_id in self._state:
            self._state[rule_id].last_fire = t

    def last_fire(self, rule_id: str) -> float | None:
        return self._state[rule_id].last_fire

    # -- evaluation ----------------------------------------------------------------------------

    def step(self, t: float, local: datetime, observations: Sequence[Obs]) -> list[Firing]:
        out: list[Firing] = []
        for rule in self.rules:
            st = self._state[rule.id]
            active = rule.in_schedule(local)
            if rule.trigger == "count":
                f = self._count(rule, st, t, active, observations)
            else:
                f = self._per_track(rule, st, t, active, observations)
            if f is not None:
                out.append(f)
            for key in [k for k, ts in st.tracks.items() if t - ts.last_t > FORGET_S]:
                del st.tracks[key]
        return out

    def _cooled(self, rule: Rule, st: _RuleState, t: float) -> bool:
        return st.last_fire is None or t - st.last_fire >= rule.cooldown_s

    def _per_track(self, rule: Rule, st: _RuleState, t: float, active: bool,
                   observations: Sequence[Obs]) -> Firing | None:
        ready: list[tuple[int, Obs]] = []
        for ob in sorted(observations, key=lambda o: o.track_key):
            ts = st.tracks.setdefault(ob.track_key, _TrackState())
            ts.last_t = t
            eligible = active and ob.confirmed and ob.cls in rule.classes
            if rule.trigger == "cross":
                ok = self._cross_progress(rule, ts, ob, eligible)
            else:
                inside = point_in_polygon(ob.point, rule.zone_poly or ())
                ok = eligible and inside
                if ok:
                    if ts.entry_t is None:
                        ts.entry_t = t
                    ts.consec += 1
                else:
                    ts.consec, ts.entry_t = 0, None
                if rule.trigger == "dwell" and ok and t - ts.entry_t < rule.seconds:  # type: ignore[operator]
                    ok = False
            if ok and not ts.fired and ts.consec >= rule.debounce:
                ready.append((ob.track_key, ob))
        if not ready or not self._cooled(rule, st, t):
            return None
        key, ob = ready[0]  # one alert per rule per frame: the lowest track key
        st.tracks[key].fired = True
        st.last_fire = t
        return Firing(rule.id, t, key, ob.point)

    def _cross_progress(self, rule: Rule, ts: _TrackState, ob: Obs, eligible: bool) -> bool:
        a, b = rule.line_pts  # type: ignore[misc]
        s = side(a, b, ob.point)
        if s == 0:
            return False  # exactly on the line: no progress, no reset; wait for a side
        if ts.off_line is not None and side(a, b, ts.off_line) != s:
            c = crossing(ts.off_line, ob.point, a, b)
            if c != 0 and (rule.direction == 0 or c == rule.direction):
                ts.cross_dir, ts.consec = c, 0
            else:
                ts.cross_dir, ts.consec = 0, 0  # crossed the wrong way or beside the segment
        ts.off_line = ob.point
        if ts.cross_dir != 0 and s == ts.cross_dir and eligible:
            ts.consec += 1
            return True
        if ts.cross_dir != 0 and not eligible:
            ts.consec = 0
        return False

    def _count(self, rule: Rule, st: _RuleState, t: float, active: bool,
               observations: Sequence[Obs]) -> Firing | None:
        inside = sorted((ob for ob in observations
                         if active and ob.confirmed and ob.cls in rule.classes
                         and point_in_polygon(ob.point, rule.zone_poly or ())),
                        key=lambda o: o.track_key)
        if len(inside) < rule.min_count:
            st.count_consec, st.count_armed = 0, True
            return None
        st.count_consec += 1
        if not st.count_armed or st.count_consec < rule.debounce or not self._cooled(rule, st, t):
            return None
        st.count_armed = False
        st.last_fire = t
        newest = inside[-1]
        return Firing(rule.id, t, newest.track_key, newest.point, len(inside))

    # -- rollback ------------------------------------------------------------------------------

    def snapshot(self) -> object:
        return copy.deepcopy(self._state)

    def restore(self, snap: object) -> None:
        self._state = copy.deepcopy(snap)  # type: ignore[assignment]
