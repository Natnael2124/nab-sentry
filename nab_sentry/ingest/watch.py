"""Watch-mode ingest of live Segments (V2 Requirement 8).

:func:`closed_segments` (pure) decides which Segment files are finished: a file is closed when
a later Segment of the same camera exists, or when it has been idle (unchanged size, mtime at
least ``segment_s + IDLE_GRACE_S`` old) across two polls. Only names that follow the
Filename_Convention are considered. :class:`Watcher` polls ``data/videos/`` and ingests closed
Segments in start-time order per camera through the V1 :class:`IngestPipeline`, retrying a
failed Segment at most ``max_retries`` times.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from nab_sentry.ingest.metadata import FILENAME_RE
from nab_sentry.ingest.pipeline import VIDEO_EXTENSIONS, IngestPipeline, VideoResult
from nab_sentry.logging_setup import get_logger

log = get_logger("watch")

IDLE_GRACE_S = 5.0
RETRY_STATUSES = frozenset({"failed", "unreadable"})


@dataclass(frozen=True)
class SegmentFile:
    path: Path
    camera_id: str
    ts: str  # YYYYMMDDTHHMMSS from the name; sorts chronologically
    size: int
    mtime: float


def parse_listing(entries: Iterable[tuple[Path, int, float]]) -> list[SegmentFile]:
    """``(path, size, mtime)`` -> SegmentFiles, keeping only Filename_Convention video names."""
    out = []
    for path, size, mtime in entries:
        m = FILENAME_RE.match(path.name)
        if m and path.suffix.lower() in VIDEO_EXTENSIONS:
            out.append(SegmentFile(path, m.group("cam"), m.group("ts"), int(size), float(mtime)))
    return out


def closed_segments(
    files: Iterable[SegmentFile],
    now: float,
    segment_s_for: Callable[[str], float],
    prev_sizes: dict[Path, int],
    done: set[Path],
) -> list[SegmentFile]:
    """Closed, not yet handled Segments, sorted by (start time, camera)."""
    by_cam: dict[str, list[SegmentFile]] = {}
    for f in files:
        by_cam.setdefault(f.camera_id, []).append(f)
    closed: list[SegmentFile] = []
    for cam, fs in by_cam.items():
        fs.sort(key=lambda f: f.ts)
        idle_s = float(segment_s_for(cam)) + IDLE_GRACE_S
        for i, f in enumerate(fs):
            if f.path in done:
                continue
            has_later = any(g.ts > f.ts for g in fs[i + 1:])
            idle = now - f.mtime >= idle_s and prev_sizes.get(f.path) == f.size
            if has_later or idle:
                closed.append(f)
    closed.sort(key=lambda f: (f.ts, f.camera_id))
    return closed


def scan_dir(directory: Path) -> list[SegmentFile]:
    entries = []
    try:
        with os.scandir(directory) as it:
            for e in it:
                if e.is_file():
                    st = e.stat()
                    entries.append((Path(e.path), st.st_size, st.st_mtime))
    except FileNotFoundError:
        return []
    return parse_listing(entries)


@dataclass
class Watcher:
    pipeline: IngestPipeline
    videos_dir: Path
    segment_s_for: Callable[[str], float]
    poll_s: float = 1.0
    max_retries: int = 2
    clock: Callable[[], float] = time.time
    scan: Callable[[Path], list[SegmentFile]] = scan_dir
    done: set[Path] = field(default_factory=set)
    failures: dict[Path, int] = field(default_factory=dict)
    # Hourly housekeeping (retention, V2 task 16.2), called with the current time.
    maintenance: Callable[[float], object] | None = None
    maintenance_every_s: float = 3600.0
    _sizes: dict[Path, int] = field(default_factory=dict)
    _last_maintenance: float | None = None

    def poll_once(self) -> list[VideoResult]:
        files = self.scan(self.videos_dir)
        ready = closed_segments(files, self.clock(), self.segment_s_for, self._sizes, self.done)
        self._sizes = {f.path: f.size for f in files}
        results = []
        for seg in ready:
            res = self.pipeline.ingest_video(seg.path)
            results.append(res)
            if res.status in RETRY_STATUSES:
                n = self.failures.get(seg.path, 0) + 1
                self.failures[seg.path] = n
                if n <= self.max_retries:
                    log.warning("watch: %s %s (attempt %d); will retry", seg.path.name, res.status, n)
                    continue
                log.error("watch: giving up on %s after %d attempts: %s", seg.path.name, n, res.reason)
            self.done.add(seg.path)
            self.failures.pop(seg.path, None)
        return results

    def maintain(self) -> None:
        now = self.clock()
        if self.maintenance is None:
            return
        if self._last_maintenance is not None and now - self._last_maintenance < self.maintenance_every_s:
            return
        self._last_maintenance = now
        try:
            self.maintenance(now)
        except Exception:  # noqa: BLE001 - housekeeping must never stop ingest
            log.exception("watch: maintenance failed")

    def run(self, stop: threading.Event) -> None:
        while not stop.is_set():
            self.poll_once()
            self.maintain()
            stop.wait(self.poll_s)
