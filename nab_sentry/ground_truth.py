"""Ground_Truth_File loading and validation (V2 Requirement 2).

``eval/ground_truth/<CAMERA_ID>.yaml`` labels one video::

    video: CAM-SEC01_20261007T230010.mp4
    duration_s: 387.0            # optional; enables the "past the end" check
    scene_cuts: [0.0, 23.0, 61.0]
    incidents:
      - {start: 2.0, end: 14.5, classes: [person], description: "man opens gate", zone: gate}

Offsets are seconds from the start of the video. :func:`load_ground_truth` reports every bad
entry at once in a :class:`GroundTruthError` (one issue string per problem, naming the entry),
mirroring ``ConfigError``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from nab_sentry.ingest.detector import TARGET_CLASSES

CLASS_NAMES = frozenset(TARGET_CLASSES.values())


class GroundTruthError(ValueError):
    def __init__(self, path: Path | str, issues: list[str]) -> None:
        self.issues = list(issues)
        lines = "\n".join(f"  - {i}" for i in self.issues)
        super().__init__(f"invalid ground truth {path} ({len(self.issues)} issue(s)):\n{lines}")


@dataclass(frozen=True)
class Incident:
    start: float
    end: float
    classes: tuple[str, ...]
    description: str
    zone: str | None = None

    def contains(self, offset_s: float) -> bool:
        return self.start <= offset_s <= self.end


@dataclass(frozen=True)
class GroundTruth:
    video: str
    scene_cuts: tuple[float, ...]
    incidents: tuple[Incident, ...]
    duration_s: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def parse_ground_truth(data: Any, path: Path | str = "<data>") -> GroundTruth:
    """Validate a parsed YAML document; raise :class:`GroundTruthError` listing every issue."""
    issues: list[str] = []
    if not isinstance(data, dict):
        raise GroundTruthError(path, ["top level must be a mapping"])

    video = data.get("video")
    if not isinstance(video, str) or not video.strip():
        issues.append("video: must be a non-empty file name")
        video = ""

    duration = None
    if data.get("duration_s") is not None:
        duration = _num(data["duration_s"])
        if duration is None or duration <= 0:
            issues.append("duration_s: must be a positive number")
            duration = None

    cuts: list[float] = []
    raw_cuts = data.get("scene_cuts") or []  # an empty key parses as None
    if not isinstance(raw_cuts, list):
        issues.append("scene_cuts: must be a list of offsets")
        raw_cuts = []
    for i, c in enumerate(raw_cuts):
        v = _num(c)
        if v is None or v < 0:
            issues.append(f"scene_cuts[{i}]: must be a number >= 0, got {c!r}")
        elif duration is not None and v > duration:
            issues.append(f"scene_cuts[{i}]: {v} is past the video duration {duration}")
        else:
            cuts.append(v)

    incidents: list[Incident] = []
    raw_inc = data.get("incidents") or []
    if not isinstance(raw_inc, list):
        issues.append("incidents: must be a list")
        raw_inc = []
    for i, item in enumerate(raw_inc):
        name = f"incidents[{i}]"
        if not isinstance(item, dict):
            issues.append(f"{name}: must be a mapping")
            continue
        start, end = _num(item.get("start")), _num(item.get("end"))
        bad = len(issues)
        if start is None or start < 0:
            issues.append(f"{name}.start: must be a number >= 0, got {item.get('start')!r}")
        if end is None or end < 0:
            issues.append(f"{name}.end: must be a number >= 0, got {item.get('end')!r}")
        if start is not None and end is not None and start >= 0 and end >= 0 and end <= start:
            issues.append(f"{name}: end {end} must be later than start {start}")
        if end is not None and duration is not None and end > duration:
            issues.append(f"{name}.end: {end} is past the video duration {duration}")
        classes = item.get("classes")
        if not isinstance(classes, list) or not classes:
            issues.append(f"{name}.classes: must be a non-empty list")
            classes = []
        unknown = [c for c in classes if c not in CLASS_NAMES]
        if unknown:
            issues.append(f"{name}.classes: unknown class(es) {unknown}; allowed {sorted(CLASS_NAMES)}")
        desc = item.get("description", "")
        if not isinstance(desc, str):
            issues.append(f"{name}.description: must be a string")
            desc = ""
        zone = item.get("zone")
        if zone is not None and (not isinstance(zone, str) or not zone.strip()):
            issues.append(f"{name}.zone: must be a non-empty string when given")
        if len(issues) == bad:
            incidents.append(Incident(start, end, tuple(classes), desc.strip(), zone))  # type: ignore[arg-type]

    if issues:
        raise GroundTruthError(path, issues)
    return GroundTruth(video, tuple(sorted(cuts)), tuple(incidents), duration)


def load_ground_truth(path: Path | str) -> GroundTruth:
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GroundTruthError(path, [f"cannot read file: {exc}"]) from exc
    except yaml.YAMLError as exc:
        raise GroundTruthError(path, [f"unparseable YAML: {exc}"]) from exc
    return parse_ground_truth(data, path)
