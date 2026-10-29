"""Start clock time and camera name of a video from its file name (design: cliptime.py).

DVRs name exports in many ways. Recognised, first match wins (the camera is the text before the
time, cleaned; else the parent folder's name):

- ``CAM01_20261028T232546.mkv`` (NAB Sentry segments)
- ``ch01_20261028143000_20261028150000.dav`` (Dahua-style start_end)
- ``CH01_20261028143000.mp4`` / ``20261028143000.mp4`` (14 digits)
- ``Front 20261028_143000.mp4`` / ``2026-10-28_14-30-00.mp4`` / ``2026-10-28 14.30.00``

Without a match: modified time minus duration, marked estimated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

_PATTERNS = [
    re.compile(r"(?P<y>\d{4})(?P<mo>\d{2})(?P<d>\d{2})T(?P<h>\d{2})(?P<mi>\d{2})(?P<s>\d{2})"),
    re.compile(r"(?<!\d)(?P<y>\d{4})(?P<mo>\d{2})(?P<d>\d{2})(?P<h>\d{2})(?P<mi>\d{2})(?P<s>\d{2})(?!\d)"),
    re.compile(r"(?<!\d)(?P<y>\d{4})[-_.]?(?P<mo>\d{2})[-_.]?(?P<d>\d{2})[ _T-]+"
               r"(?P<h>\d{2})[-_.:h]?(?P<mi>\d{2})[-_.:m]?(?P<s>\d{2})(?!\d)"),
]


@dataclass(frozen=True)
class NameTime:
    start: datetime
    camera: str
    estimated: bool


def _valid(m: re.Match) -> datetime | None:
    try:
        dt = datetime(int(m["y"]), int(m["mo"]), int(m["d"]), int(m["h"]), int(m["mi"]), int(m["s"]))
    except ValueError:
        return None
    return dt if 1990 <= dt.year <= 2100 else None


def clean_camera(text: str) -> str:
    text = re.sub(r"[_.\s]+", " ", text).strip(" -")
    return text[:40]


def from_name(path: Path | str) -> tuple[datetime, str] | None:
    """(start, camera) from the file name, or None."""
    path = Path(path)
    stem = path.stem
    for pat in _PATTERNS:
        for m in pat.finditer(stem):
            dt = _valid(m)
            if dt is None:
                continue
            cam = clean_camera(stem[:m.start()]) or clean_camera(path.parent.name) or "Camera"
            return dt, cam
    return None


def start_time(path: Path | str, duration_s: float) -> NameTime:
    path = Path(path)
    got = from_name(path)
    if got:
        return NameTime(got[0], got[1], False)
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return NameTime((mtime - timedelta(seconds=duration_s)).replace(microsecond=0),
                    clean_camera(path.parent.name) or "Camera", True)
