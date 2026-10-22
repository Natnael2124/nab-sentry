"""Watch rules: ``config/rules.yaml`` (V2 Requirement 12.3-12.6).

::

    rules:
      - id: yard-night                     # 1-64 letters, digits, '_' or '-'; unique
        name: Person in the yard at night  # shown in alerts
        camera: CAM01
        zone: yard                         # a zone of that camera (enter / dwell / count)
        classes: [person]
        schedule: {days: [mon, tue, wed, thu, fri, sat, sun], start: "22:00", end: "06:00"}
        trigger: {type: enter}             # or {type: dwell, seconds: 20} / {type: count, min: 3}
        debounce: 2                        # consecutive qualifying observations (default 2)
        cooldown_s: 300                    # min seconds between alerts of this rule (default 300)
        severity: immediate                # immediate | verify (ask the VLM first, see verify)
      - id: gate-in
        name: Someone came in through the gate
        camera: CAM01
        line: gate-line                    # a line of that camera (cross)
        direction: in                      # in | out | any
        classes: [person, car]
        trigger: {type: cross}

``verify`` (with ``severity: verify``) is a yes/no question the on-demand VLM answers about the
alert snapshot before the alert is sent, e.g. ``"Is a person climbing over the fence?"``; a "no"
keeps the alert from being sent, a VLM failure sends it marked "not verified".

``schedule`` is optional (always active). A window whose ``end`` is not after its ``start``
wraps past midnight and belongs to the day it starts on: Monday 22:00-06:00 runs from Monday
22:00 to Tuesday 06:00; ``start == end`` is 24 hours from ``start`` (``"00:00"`` to ``"00:00"``
is the whole day). :func:`load_rules` reports every bad
field at once in a :class:`RulesConfigError`, mirroring the Camera_Config loader.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any

import yaml

from nab_sentry.ingest.detector import TARGET_CLASSES
from nab_sentry.live.camera_config import CameraSpec, Point

ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
TRIGGERS = ("enter", "dwell", "count", "cross")
DIRECTIONS = {"in": 1, "out": -1, "any": 0}
SEVERITIES = ("immediate", "verify")
CLASSES = frozenset(TARGET_CLASSES.values())
TIME_RE = re.compile(r"([01]\d|2[0-3]):([0-5]\d)")
DEFAULT_DEBOUNCE = 2
DEFAULT_COOLDOWN_S = 300.0


class RulesConfigError(ValueError):
    def __init__(self, path: Path | str, issues: list[str]) -> None:
        self.issues = list(issues)
        lines = "\n".join(f"  - {i}" for i in self.issues)
        super().__init__(f"invalid rules {path} ({len(self.issues)} issue(s)):\n{lines}")


@dataclass(frozen=True)
class Schedule:
    days: frozenset[int]  # 0 = Monday
    start: time
    end: time

    def active(self, local: datetime) -> bool:
        """Is the wall-clock time ``local`` inside the window? (Pure; DST-safe because it only
        looks at the local date and time of day.)"""
        t = local.time()
        today = local.weekday()
        if self.start < self.end:
            return today in self.days and self.start <= t < self.end
        # wraps past midnight (start == end: 24 h from start), belongs to the day it starts on
        yesterday = (local - timedelta(days=1)).weekday()
        return (today in self.days and t >= self.start) or (yesterday in self.days and t < self.end)


@dataclass(frozen=True)
class Rule:
    id: str
    name: str
    camera: str
    classes: frozenset[str]
    trigger: str  # enter | dwell | count | cross
    zone: str | None = None
    zone_poly: tuple[Point, ...] | None = None
    line: str | None = None
    line_pts: tuple[Point, Point] | None = None
    direction: int = 0  # +1 in, -1 out, 0 any (cross only)
    seconds: float = 0.0  # dwell
    min_count: int = 1  # count
    schedule: Schedule | None = None
    debounce: int = DEFAULT_DEBOUNCE
    cooldown_s: float = DEFAULT_COOLDOWN_S
    severity: str = "immediate"
    verify: str | None = None  # yes/no question for the on-demand VLM (severity: verify)

    def in_schedule(self, local: datetime) -> bool:
        return self.schedule is None or self.schedule.active(local)


def _num(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _parse_schedule(raw: Any, name: str, issues: list[str]) -> Schedule | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        issues.append(f"{name}.schedule: must be a mapping with days, start, end")
        return None
    days = raw.get("days", list(DAYS))
    if not isinstance(days, list) or not days or any(d not in DAYS for d in days):
        issues.append(f"{name}.schedule.days: must be a non-empty list of {', '.join(DAYS)}")
        return None
    parsed = []
    for key in ("start", "end"):
        v = raw.get(key)
        m = TIME_RE.fullmatch(v) if isinstance(v, str) else None
        if m is None:
            issues.append(f"{name}.schedule.{key}: must be \"HH:MM\" (24 h), got {v!r}")
            return None
        parsed.append(time(int(m.group(1)), int(m.group(2))))
    return Schedule(frozenset(DAYS.index(d) for d in days), parsed[0], parsed[1])


def _parse_rule(i: int, raw: Any, cams: dict[str, CameraSpec], issues: list[str]) -> Rule | None:
    name = f"rules[{i}]"
    if not isinstance(raw, dict):
        issues.append(f"{name}: must be a mapping")
        return None
    before = len(issues)
    rid = raw.get("id")
    if not isinstance(rid, str) or not ID_RE.fullmatch(rid):
        issues.append(f"{name}.id: must be 1-64 letters, digits, '_' or '-', got {rid!r}")
    title = raw.get("name", rid)
    if not isinstance(title, str) or not title.strip() or len(title) > 128:
        issues.append(f"{name}.name: must be a non-empty string of at most 128 characters")
    cam = raw.get("camera")
    spec = cams.get(cam) if isinstance(cam, str) else None
    if spec is None:
        issues.append(f"{name}.camera: unknown camera {cam!r} (not in the Camera_Config)")
    classes = raw.get("classes")
    if not isinstance(classes, list) or not classes or any(c not in CLASSES for c in classes):
        issues.append(f"{name}.classes: must be a non-empty list of {', '.join(sorted(CLASSES))}")
        classes = []

    trig = raw.get("trigger")
    ttype = trig.get("type") if isinstance(trig, dict) else None
    seconds, min_count = 0.0, 1
    if ttype not in TRIGGERS:
        issues.append(f"{name}.trigger.type: must be one of {', '.join(TRIGGERS)}, got {ttype!r}")
    elif ttype == "dwell":
        seconds = _num(trig.get("seconds"))  # type: ignore[assignment]
        if seconds is None or seconds <= 0:
            issues.append(f"{name}.trigger.seconds: dwell needs seconds > 0")
            seconds = 0.0
    elif ttype == "count":
        mc = trig.get("min")
        if isinstance(mc, bool) or not isinstance(mc, int) or mc < 1:
            issues.append(f"{name}.trigger.min: count needs an integer min >= 1")
        else:
            min_count = mc

    zone = raw.get("zone")
    line = raw.get("line")
    zone_poly = line_pts = None
    direction = 0
    if ttype == "cross":
        if zone is not None:
            issues.append(f"{name}.zone: a cross trigger uses a line, not a zone")
        if spec is not None:
            if not isinstance(line, str) or line not in spec.lines:
                issues.append(f"{name}.line: unknown line {line!r} on camera {cam!r}")
            else:
                line_pts = spec.lines[line]
        d = raw.get("direction", "any")
        if d not in DIRECTIONS:
            issues.append(f"{name}.direction: must be in, out or any, got {d!r}")
        else:
            direction = DIRECTIONS[d]
    elif ttype in TRIGGERS:
        if line is not None:
            issues.append(f"{name}.line: only a cross trigger uses a line")
        if spec is not None:
            if not isinstance(zone, str) or zone not in spec.zones:
                issues.append(f"{name}.zone: unknown zone {zone!r} on camera {cam!r}")
            else:
                zone_poly = spec.zones[zone]

    schedule = _parse_schedule(raw.get("schedule"), name, issues)
    deb = raw.get("debounce", DEFAULT_DEBOUNCE)
    if isinstance(deb, bool) or not isinstance(deb, int) or not 1 <= deb <= 100:
        issues.append(f"{name}.debounce: must be an integer 1 to 100, got {deb!r}")
    cool = _num(raw.get("cooldown_s", DEFAULT_COOLDOWN_S))
    if cool is None or cool < 0:
        issues.append(f"{name}.cooldown_s: must be a number >= 0")
    sev = raw.get("severity", "immediate")
    if sev not in SEVERITIES:
        issues.append(f"{name}.severity: must be immediate or verify, got {sev!r}")
    question = raw.get("verify")
    if question is not None and (not isinstance(question, str) or not question.strip() or len(question) > 300):
        issues.append(f"{name}.verify: must be a question of 1-300 characters")
    elif question is not None and sev != "verify":
        issues.append(f"{name}.verify: needs severity: verify")

    if len(issues) != before:
        return None
    return Rule(rid, title.strip(), cam, frozenset(classes), ttype, zone if ttype != "cross" else None,
                zone_poly, line if ttype == "cross" else None, line_pts, direction, float(seconds),
                min_count, schedule, deb, float(cool), sev,
                question.strip() if isinstance(question, str) else None)


def parse_rules(data: Any, cameras: list[CameraSpec], path: Path | str = "<data>") -> list[Rule]:
    if data is None or (isinstance(data, dict) and not data.get("rules")):
        return []  # an empty file means no rules
    if not isinstance(data, dict) or not isinstance(data.get("rules"), list):
        raise RulesConfigError(path, ["rules: must be a list"])
    cams = {c.camera_id: c for c in cameras}
    issues: list[str] = []
    rules = [_parse_rule(i, r, cams, issues) for i, r in enumerate(data["rules"])]
    seen: set[str] = set()
    for i, r in enumerate(data["rules"]):
        rid = r.get("id") if isinstance(r, dict) else None
        if isinstance(rid, str):
            if rid in seen:
                issues.append(f"rules[{i}].id: duplicate {rid!r}")
            seen.add(rid)
    if issues:
        raise RulesConfigError(path, issues)
    return [r for r in rules if r is not None]


def load_rules(path: Path | str, cameras: list[CameraSpec]) -> list[Rule]:
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RulesConfigError(path, [f"cannot read file: {exc}"]) from exc
    except yaml.YAMLError as exc:
        raise RulesConfigError(path, [f"unparseable YAML: {exc}"]) from exc
    return parse_rules(data, cameras, path)
