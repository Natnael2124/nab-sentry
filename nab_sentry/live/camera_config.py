"""Camera_Config: ``config/cameras.yaml`` (V2 Requirement 6).

::

    cameras:
      - camera_id: CAM01                 # Filename_Convention-valid: [A-Za-z0-9-]{1,64}
        label: Front gate
        source: rtsp://user:pass@192.168.1.20:554/stream2   # or http(s)://..., dshow:<device>
        segment_s: 10                    # 5..300, default 10
        encoder: auto                    # auto | copy | qsv | x264 (auto: copy for rtsp, qsv otherwise)
        roi: [0.0, 0.3, 1.0, 1.0]        # optional motion ROI x1, y1, x2, y2 in 0..1
        zones:                           # name -> polygon, >= 3 points in 0..1
          yard: [[0.1, 0.5], [0.9, 0.5], [0.9, 1.0], [0.1, 1.0]]
        lines:                           # name -> two distinct points in 0..1
          gate-line: [[0.3, 0.6], [0.7, 0.6]]

:func:`load_cameras` reports every invalid field at once in a :class:`CameraConfigError`
(one issue per field, named like ``cameras[0].roi``), mirroring ``ConfigError``. The file holds
camera passwords, so it is git-ignored; :func:`mask_url` must wrap every source URL that is
logged or printed.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import yaml

from nab_sentry.ingest.metadata import CAMERA_ID_RE

DEFAULT_SEGMENT_S = 10
SEGMENT_S_MIN, SEGMENT_S_MAX = 5, 300
LABEL_MAX_LEN = 128
ENCODERS = ("auto", "copy", "qsv", "x264")
URL_SCHEMES = ("rtsp", "rtsps", "http", "https")
DSHOW_PREFIX = "dshow:"
NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")

Point = tuple[float, float]


class CameraConfigError(ValueError):
    def __init__(self, path: Path | str, issues: list[str]) -> None:
        self.issues = list(issues)
        lines = "\n".join(f"  - {i}" for i in self.issues)
        super().__init__(f"invalid camera config {path} ({len(self.issues)} issue(s)):\n{lines}")


@dataclass(frozen=True)
class CameraSpec:
    camera_id: str
    label: str
    source: str
    segment_s: int = DEFAULT_SEGMENT_S
    encoder: str = "auto"
    roi: tuple[float, float, float, float] | None = None
    zones: dict[str, tuple[Point, ...]] = field(default_factory=dict)
    lines: dict[str, tuple[Point, Point]] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        """``rtsp`` (rtsp/rtsps), ``http`` (http/https) or ``dshow``."""
        if self.source.startswith(DSHOW_PREFIX):
            return "dshow"
        scheme = urlsplit(self.source).scheme.lower()
        return "rtsp" if scheme.startswith("rtsp") else "http"

    @property
    def resolved_encoder(self) -> str:
        """``auto`` -> ``copy`` for RTSP (already H.264/H.265), ``qsv`` for raw/MJPEG sources."""
        if self.encoder != "auto":
            return self.encoder
        return "copy" if self.kind == "rtsp" else "qsv"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"camera_id": self.camera_id, "label": self.label, "source": self.source,
                             "segment_s": self.segment_s, "encoder": self.encoder}
        if self.roi is not None:
            d["roi"] = list(self.roi)
        if self.zones:
            d["zones"] = {k: [list(p) for p in v] for k, v in self.zones.items()}
        if self.lines:
            d["lines"] = {k: [list(p) for p in v] for k, v in self.lines.items()}
        return d


# ---------------------------------------------------------------- URL masking


def mask_url(text: str) -> str:
    """Replace the password (and user) of every URL in ``text`` with ``***``. Idempotent."""
    def repl(m: re.Match[str]) -> str:
        url = m.group(0)
        try:
            parts = urlsplit(url)
        except ValueError:
            return url
        if "@" not in parts.netloc:
            return url
        host = parts.netloc.rsplit("@", 1)[1]
        return urlunsplit(parts._replace(netloc=f"***@{host}"))

    return re.sub(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s'\"<>]+", repl, text)


# ---------------------------------------------------------------- validation helpers


def _unit(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if math.isfinite(v) and 0.0 <= v <= 1.0 else None


def _point(v: Any) -> Point | None:
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        return None
    x, y = _unit(v[0]), _unit(v[1])
    return None if x is None or y is None else (x, y)


def _source_issue(source: Any) -> str | None:
    if not isinstance(source, str) or not source.strip():
        return "must be a non-empty string"
    if source.startswith(DSHOW_PREFIX):
        return None if source[len(DSHOW_PREFIX):].strip() else "dshow: needs a device name"
    parts = urlsplit(source)
    if parts.scheme.lower() not in URL_SCHEMES or not parts.hostname:
        return f"must be an {'/'.join(URL_SCHEMES)} URL or dshow:<device name>"
    return None


def _parse_camera(i: int, raw: Any, issues: list[str]) -> CameraSpec | None:
    name = f"cameras[{i}]"
    if not isinstance(raw, dict):
        issues.append(f"{name}: must be a mapping")
        return None
    before = len(issues)

    cam = raw.get("camera_id")
    if not isinstance(cam, str) or not CAMERA_ID_RE.fullmatch(cam):
        issues.append(f"{name}.camera_id: must be 1-64 letters, digits or hyphens, got {cam!r}")
    label = raw.get("label")
    if not isinstance(label, str) or not label.strip() or len(label) > LABEL_MAX_LEN:
        issues.append(f"{name}.label: must be a non-empty string of at most {LABEL_MAX_LEN} characters")
    source = raw.get("source")
    if (why := _source_issue(source)) is not None:
        issues.append(f"{name}.source: {why}")

    seg = raw.get("segment_s", DEFAULT_SEGMENT_S)
    if isinstance(seg, bool) or not isinstance(seg, int) or not SEGMENT_S_MIN <= seg <= SEGMENT_S_MAX:
        issues.append(f"{name}.segment_s: must be an integer {SEGMENT_S_MIN} to {SEGMENT_S_MAX}, got {seg!r}")
    enc = raw.get("encoder", "auto")
    if enc not in ENCODERS:
        issues.append(f"{name}.encoder: must be one of {', '.join(ENCODERS)}, got {enc!r}")

    roi = None
    if raw.get("roi") is not None:
        r = raw["roi"]
        vals = [_unit(v) for v in r] if isinstance(r, (list, tuple)) and len(r) == 4 else None
        if vals is None or any(v is None for v in vals) or not (vals[0] < vals[2] and vals[1] < vals[3]):
            issues.append(f"{name}.roi: must be [x1, y1, x2, y2] in 0..1 with x1 < x2 and y1 < y2")
        else:
            roi = tuple(vals)  # type: ignore[assignment]

    zones: dict[str, tuple[Point, ...]] = {}
    raw_zones = raw.get("zones") or {}
    if not isinstance(raw_zones, dict):
        issues.append(f"{name}.zones: must be a mapping of name -> polygon")
        raw_zones = {}
    for zname, poly in raw_zones.items():
        zkey = f"{name}.zones.{zname}"
        if not isinstance(zname, str) or not NAME_RE.fullmatch(zname):
            issues.append(f"{zkey}: name must be 1-64 letters, digits, '_' or '-'")
            continue
        pts = [_point(p) for p in poly] if isinstance(poly, list) else None
        if pts is None or len(pts) < 3 or any(p is None for p in pts):
            issues.append(f"{zkey}: must be a list of at least 3 [x, y] points in 0..1")
        else:
            zones[zname] = tuple(pts)  # type: ignore[arg-type]

    lines: dict[str, tuple[Point, Point]] = {}
    raw_lines = raw.get("lines") or {}
    if not isinstance(raw_lines, dict):
        issues.append(f"{name}.lines: must be a mapping of name -> two points")
        raw_lines = {}
    for lname, seg_pts in raw_lines.items():
        lkey = f"{name}.lines.{lname}"
        if not isinstance(lname, str) or not NAME_RE.fullmatch(lname):
            issues.append(f"{lkey}: name must be 1-64 letters, digits, '_' or '-'")
            continue
        pts = [_point(p) for p in seg_pts] if isinstance(seg_pts, list) and len(seg_pts) == 2 else None
        if pts is None or any(p is None for p in pts) or pts[0] == pts[1]:
            issues.append(f"{lkey}: must be two distinct [x, y] points in 0..1")
        else:
            lines[lname] = (pts[0], pts[1])  # type: ignore[assignment]

    if len(issues) != before:
        return None
    return CameraSpec(cam, label.strip(), source, seg, enc, roi, zones, lines)


def parse_cameras(data: Any, path: Path | str = "<data>") -> list[CameraSpec]:
    if not isinstance(data, dict) or not isinstance(data.get("cameras"), list) or not data["cameras"]:
        raise CameraConfigError(path, ["cameras: must be a non-empty list"])
    issues: list[str] = []
    specs = [_parse_camera(i, raw, issues) for i, raw in enumerate(data["cameras"])]
    seen: set[str] = set()
    for i, raw in enumerate(data["cameras"]):
        cam = raw.get("camera_id") if isinstance(raw, dict) else None
        if isinstance(cam, str) and cam in seen:
            issues.append(f"cameras[{i}].camera_id: duplicate {cam!r}")
        if isinstance(cam, str):
            seen.add(cam)
    if issues:
        raise CameraConfigError(path, issues)
    return [s for s in specs if s is not None]


def load_cameras(path: Path | str) -> list[CameraSpec]:
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CameraConfigError(path, [f"cannot read file: {exc}"]) from exc
    except yaml.YAMLError as exc:
        raise CameraConfigError(path, [f"unparseable YAML: {mask_url(str(exc))}"]) from exc
    return parse_cameras(data, path)


def dump_cameras(specs: list[CameraSpec]) -> str:
    return yaml.safe_dump({"cameras": [s.to_dict() for s in specs]}, sort_keys=False)
