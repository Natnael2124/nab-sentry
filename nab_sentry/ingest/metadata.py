"""Camera metadata resolution for video files (Requirement 1) and Src_Hash (Requirement 7).

Metadata comes from a Sidecar_File (``<base>.json`` next to the video) when it is valid,
otherwise from the Filename_Convention ``<CAMERA_ID>_<YYYYMMDDTHHMMSS>.<ext>``, otherwise the
video is unresolved and the pipeline skips it.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from nab_sentry.startup import sha256_file

CAMERA_ID_RE = re.compile(r"[A-Za-z0-9-]{1,64}")
# re.ASCII keeps \d to 0-9 so non-ASCII digits never parse as a timestamp.
FILENAME_RE = re.compile(
    r"^(?P<cam>[A-Za-z0-9-]{1,64})_(?P<ts>\d{8}T\d{6})\.(?P<ext>[A-Za-z0-9]{1,10})$",
    re.ASCII,
)

LABEL_MAX_LEN = 128
SIDECAR_FIELDS = ("camera_id", "label", "start_time")


@dataclass(frozen=True)
class Sidecar:
    camera_id: str
    label: str
    start_time: datetime  # aware or naive, as written


@dataclass(frozen=True)
class SidecarParse:
    sidecar: Sidecar | None
    errors: list[str] = field(default_factory=list)  # field names, or ["unparseable"], or ["missing"]


@dataclass(frozen=True)
class ResolvedMetadata:
    camera_id: str
    label: str
    start_time: datetime  # always aware (naive -> local zone)
    source: Literal["sidecar", "filename"]


# --------------------------------------------------------------------------- filename


def format_filename(camera_id: str, start: datetime, ext: str) -> str:
    """Build ``<camera_id>_<YYYYMMDDTHHMMSS>.<ext>``.

    Formatted manually because ``strftime`` is unreliable for years below 1000 on Windows.
    """
    ext = ext[1:] if ext.startswith(".") else ext
    ts = (
        f"{start.year:04d}{start.month:02d}{start.day:02d}"
        f"T{start.hour:02d}{start.minute:02d}{start.second:02d}"
    )
    return f"{camera_id}_{ts}.{ext}"


def parse_filename(name: str) -> tuple[str, datetime] | None:
    """Return ``(camera_id, naive start time)`` or ``None`` if the name does not match
    the Filename_Convention or the timestamp is not a real calendar date-time."""
    m = FILENAME_RE.fullmatch(name)
    if m is None:
        return None
    ts = m.group("ts")
    try:
        start = datetime(
            int(ts[0:4]), int(ts[4:6]), int(ts[6:8]),
            int(ts[9:11]), int(ts[11:13]), int(ts[13:15]),
        )
    except ValueError:  # year 0000, month 13, Feb 30, hour 24, ...
        return None
    return m.group("cam"), start


# --------------------------------------------------------------------------- sidecar


def serialize_sidecar(s: Sidecar) -> str:
    return json.dumps(
        {"camera_id": s.camera_id, "label": s.label, "start_time": s.start_time.isoformat()}
    )


def _valid_camera_id(v: Any) -> bool:
    return isinstance(v, str) and CAMERA_ID_RE.fullmatch(v) is not None


def _valid_label(v: Any) -> bool:
    return isinstance(v, str) and 1 <= len(v) <= LABEL_MAX_LEN and v.strip() != ""


def _parse_start_time(v: Any) -> datetime | None:
    # Must be an ISO 8601 date-time: date-only strings are rejected.
    if not isinstance(v, str) or ("T" not in v and " " not in v):
        return None
    try:
        return datetime.fromisoformat(v)
    except ValueError:
        return None


def parse_sidecar_text(text: str) -> SidecarParse:
    """Validate sidecar JSON text. Collects every invalid or missing field name;
    returns ``["unparseable"]`` when the text is not a single JSON object."""
    try:
        obj = json.loads(text)
    except (ValueError, RecursionError):
        return SidecarParse(None, ["unparseable"])
    if not isinstance(obj, dict):
        return SidecarParse(None, ["unparseable"])

    errors: list[str] = []
    camera_id = obj.get("camera_id")
    if not _valid_camera_id(camera_id):
        errors.append("camera_id")
    label = obj.get("label")
    if not _valid_label(label):
        errors.append("label")
    start_time = _parse_start_time(obj.get("start_time"))
    if start_time is None:
        errors.append("start_time")

    if errors:
        return SidecarParse(None, errors)
    return SidecarParse(Sidecar(camera_id, label, start_time), [])


def sidecar_path_for(video_path: Path) -> Path:
    """Same directory, same base name, ``.json`` extension."""
    return video_path.with_suffix(".json")


def read_sidecar(video_path: Path) -> SidecarParse:
    """Read and validate the video's Sidecar_File; ``["missing"]`` when there is none."""
    path = sidecar_path_for(video_path)
    if not path.exists():
        return SidecarParse(None, ["missing"])
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return SidecarParse(None, ["unparseable"])
    return parse_sidecar_text(text)


# --------------------------------------------------------------------------- resolution


def to_aware_local(dt: datetime) -> datetime:
    """Naive -> interpreted in the machine's local zone (Requirement 1.12); aware unchanged."""
    if dt.tzinfo is not None and dt.utcoffset() is not None:
        return dt
    try:
        return dt.astimezone()
    except (OverflowError, OSError, ValueError):
        # Windows localtime() rejects dates outside its range (e.g. year 0001);
        # fall back to the current local UTC offset.
        return dt.replace(tzinfo=datetime.now().astimezone().tzinfo)


def resolve_metadata(
    video_path: Path, log: logging.Logger | logging.LoggerAdapter
) -> ResolvedMetadata | None:
    """Sidecar first, then file name, else ``None``.

    Logs ``"invalid sidecar: <path>: ..."`` for a present but invalid sidecar (1.6) and
    ``"unresolved camera metadata: <path>"`` when nothing resolves (1.5); callers need
    not log the unresolved case again.
    """
    video_path = Path(video_path)
    parsed = read_sidecar(video_path)
    if parsed.sidecar is not None:
        s = parsed.sidecar
        return ResolvedMetadata(s.camera_id, s.label, to_aware_local(s.start_time), "sidecar")
    if parsed.errors != ["missing"]:
        sc_path = sidecar_path_for(video_path)
        if parsed.errors == ["unparseable"]:
            log.warning("invalid sidecar: %s: unparseable", sc_path)
        else:
            log.warning("invalid sidecar: %s: invalid fields: %s", sc_path, ", ".join(parsed.errors))

    from_name = parse_filename(video_path.name)
    if from_name is not None:
        camera_id, start = from_name
        return ResolvedMetadata(camera_id, camera_id, to_aware_local(start), "filename")

    log.warning("unresolved camera metadata: %s", video_path)
    return None


# --------------------------------------------------------------------------- hash


def src_hash(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 hex of the file bytes (content-addressed: independent of path/name)."""
    return sha256_file(Path(path), chunk)
