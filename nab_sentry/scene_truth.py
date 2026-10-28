"""Ground truth of a continuous recording: a generated scene or real labelled footage (V2 task 17).

Both ``scripts/make_scene.py`` output and a real recording prepared by
``scripts/prepare_footage.py`` are a folder with ``videos/*.mkv`` Segments, ``cameras.yaml``,
``rules.yaml`` and ``ground_truth.yaml``::

    camera: CAM01
    start: '2026-10-26T18:00:00'      # start of the first Segment, local time
    duration_s: 3600
    objects:                          # every person / vehicle that appears
      - name: neighbour walks to the gate
        class: person
        visible: ['18:04:10', '18:04:31']
    expected_alerts:                  # what each rule should have raised
      - {rule: gate-in, t: '18:04:22'}

Times are either seconds from ``start`` (numbers, as make_scene.py writes them) or local clock
times ``'HH:MM:SS'`` on the day of ``start``; a clock time earlier than ``start`` is on the next
day (recordings that run past midnight). :func:`load_scene_truth` turns every time into seconds
from ``start``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from nab_sentry.ingest.metadata import parse_filename

CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck")
_CLOCK_RE = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})$")


class SceneTruthError(ValueError):
    pass


def to_offset(value: Any, start: datetime, where: str) -> float:
    """Seconds from ``start`` for a number or an ``'HH:MM:SS'`` clock time."""
    if isinstance(value, bool):
        raise SceneTruthError(f"{where}: expected seconds or 'HH:MM:SS', got {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    m = _CLOCK_RE.match(str(value).strip())
    if m is None:
        raise SceneTruthError(f"{where}: expected seconds or 'HH:MM:SS', got {value!r}")
    h, mi, s = (int(g) for g in m.groups())
    if h > 23 or mi > 59 or s > 59:
        raise SceneTruthError(f"{where}: {value!r} is not a time of day")
    t = start.replace(hour=h, minute=mi, second=s, microsecond=0)
    if t < start - timedelta(seconds=1):
        t += timedelta(days=1)
    return (t - start).total_seconds()


def parse_scene_truth(data: Any, rule_ids: Sequence[str] | None = None) -> dict:
    if not isinstance(data, dict):
        raise SceneTruthError("ground_truth.yaml must be a mapping")
    for key in ("camera", "start", "duration_s"):
        if key not in data:
            raise SceneTruthError(f"missing '{key}'")
    try:
        start = datetime.fromisoformat(str(data["start"]))
    except ValueError:
        raise SceneTruthError(f"start: not an ISO date-time: {data['start']!r}") from None
    out = dict(data)
    out["start"] = start.isoformat()
    out["duration_s"] = float(data["duration_s"])
    objects = []
    for i, o in enumerate(data.get("objects") or []):
        where = f"objects[{i}]"
        if not isinstance(o, dict):
            raise SceneTruthError(f"{where}: must be a mapping")
        if o.get("class") not in CLASSES:
            raise SceneTruthError(f"{where}.class: one of {', '.join(CLASSES)}, got {o.get('class')!r}")
        vis = o.get("visible")
        if vis:
            if not isinstance(vis, (list, tuple)) or len(vis) != 2:
                raise SceneTruthError(f"{where}.visible: [from, to]")
            a, b = (to_offset(v, start, f"{where}.visible") for v in vis)
            if b < a:
                raise SceneTruthError(f"{where}.visible: ends before it starts")
            vis = [a, b]
        objects.append({**o, "name": str(o.get("name") or f"object {i + 1}"), "visible": vis or None})
    alerts = []
    for i, a in enumerate(data.get("expected_alerts") or []):
        where = f"expected_alerts[{i}]"
        if not isinstance(a, dict) or "rule" not in a or "t" not in a:
            raise SceneTruthError(f"{where}: needs rule and t")
        if rule_ids is not None and a["rule"] not in rule_ids:
            raise SceneTruthError(f"{where}.rule: {a['rule']!r} is not a rule in rules.yaml")
        alerts.append({**a, "t": to_offset(a["t"], start, f"{where}.t")})
    out["objects"] = objects
    out["expected_alerts"] = alerts
    return out


def load_scene_truth(path: Path, rule_ids: Sequence[str] | None = None) -> dict:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SceneTruthError(f"{path}: not valid YAML: {exc}") from None
    try:
        return parse_scene_truth(data, rule_ids)
    except SceneTruthError as exc:
        raise SceneTruthError(f"{path}: {exc}") from None


def segment_offsets(paths: Sequence[Path], start: datetime) -> list[tuple[Path, float]]:
    """``(path, seconds from start)`` from each Segment's file name, in time order. Real
    recordings have Segments of uneven length and gaps, so the names are the only reliable
    clock."""
    out = []
    for p in paths:
        parsed = parse_filename(Path(p).name)
        if parsed is None:
            raise SceneTruthError(f"{Path(p).name}: not a CAMERA_YYYYMMDDTHHMMSS Segment name")
        out.append((Path(p), (parsed[1] - start.replace(tzinfo=None)).total_seconds()))
    return sorted(out, key=lambda x: x[1])
