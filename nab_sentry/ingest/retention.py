"""Retention of live footage (V2 Requirement 15.3, task 16.2).

Run by the Watcher once an hour. Only cameras in the Camera_Config (live cameras) are touched:
videos imported by hand are never deleted automatically. For each video that ended more than
``retention_days`` ago: remove its vectors from the Vector_Index, delete its rows (frames,
vectors and track observations cascade), its thumbnails and playback copy, and its source
Segment when that file lies inside ``data/videos/``; tracks left without observations go too.
The index is saved once at the end; a crash before that leaves only orphan index IDs, which
``reconcile()`` removes on the next start. Alert rows are kept; their snapshot and clip files
are deleted after ``alert_retention_days``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from nab_sentry.config import Config
from nab_sentry.ingest.pipeline import playback_name
from nab_sentry.logging_setup import get_logger
from nab_sentry.store.db import MetadataStore
from nab_sentry.store.vector_index import VectorIndex

log = get_logger("retention")
DAY_S = 86400


@dataclass
class RetentionReport:
    videos: list[int] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    vectors: int = 0
    tracks: int = 0
    alert_files: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.videos or self.alert_files or self.tracks)


def _unlink(p: Path) -> bool:
    try:
        p.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        log.error("retention: could not delete %s: %s", p, exc)
        return False


def _inside(path: Path, folder: Path) -> bool:
    try:
        return path.resolve().is_relative_to(folder.resolve())
    except OSError:
        return False


def apply_retention(cfg: Config, db: MetadataStore, index: VectorIndex, live_cameras: Iterable[str],
                    now_s: float) -> RetentionReport:
    rep = RetentionReport()
    cutoff_ms = round((now_s - cfg.retention_days * DAY_S) * 1000)
    for video_id, src in db.videos_ended_before(cutoff_ms, sorted(set(live_cameras))):
        ids = db.vector_ids_for_video(video_id)
        if len(ids):
            index.remove(np.asarray(ids, dtype=np.int64))
            rep.vectors += len(ids)
        names = db.delete_video(video_id)
        for name in names:
            folder = cfg.playback_dir if name == playback_name(video_id) else cfg.thumbs_dir
            if _unlink(folder / name):
                rep.files.append(name)
        src_path = Path(src)
        if _inside(src_path, cfg.videos_dir) and _unlink(src_path):
            rep.files.append(src_path.name)
        rep.videos.append(video_id)
    if rep.videos:
        index.save(cfg.index_path)
        rep.tracks = db.delete_empty_tracks()
    alert_cutoff = round((now_s - cfg.alert_retention_days * DAY_S) * 1000)
    for name in db.expire_alert_media(alert_cutoff):
        if _unlink(cfg.alerts_dir / name):
            rep.alert_files.append(name)
    if rep.changed:
        log.info("retention: deleted %d videos (%d vectors, %d files), %d tracks, %d alert files",
                 len(rep.videos), rep.vectors, len(rep.files), rep.tracks, len(rep.alert_files))
    return rep
