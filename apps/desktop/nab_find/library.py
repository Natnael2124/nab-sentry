"""Adding videos to a case: expand folders, read each file, work out its clock time and camera."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from nab_find.cliptime import start_time
from nab_find.media import VIDEO_EXTS, MediaError, probe
from nab_find.store import CaseStore

log = logging.getLogger("nab_find.library")


@dataclass
class Added:
    path: str
    video_id: int | None
    error: str | None


def expand(paths: Iterable[str | Path]) -> list[Path]:
    """Files as given, plus every video file inside given folders (recursively), sorted, unique."""
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out.extend(sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in VIDEO_EXTS))
        else:
            out.append(p)
    seen, uniq = set(), []
    for p in out:
        k = str(p.resolve()).lower()
        if k not in seen:
            seen.add(k)
            uniq.append(p)
    return uniq


def add_videos(store: CaseStore, paths: Iterable[str | Path]) -> list[Added]:
    known = {str(Path(v.path).resolve()).lower() for v in store.videos()}
    res = []
    for p in expand(paths):
        key = str(p.resolve()).lower()
        if key in known:
            continue
        try:
            info = probe(p)
        except MediaError as exc:
            res.append(Added(str(p), None, str(exc)))
            continue
        except Exception as exc:  # noqa: BLE001 - one bad file never stops the others
            log.warning("probe failed for %s", p, exc_info=True)
            res.append(Added(str(p), None, f"this file could not be opened ({type(exc).__name__})"))
            continue
        nt = start_time(p, info.duration_s)
        vid = store.add_video(str(p.resolve()), nt.camera, nt.start.timestamp(), nt.estimated, info.duration_s,
                              info.fps, info.width, info.height, info.keyframe_gap_s)
        known.add(key)
        res.append(Added(str(p), vid, None))
    return res
