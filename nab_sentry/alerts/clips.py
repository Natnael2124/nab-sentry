"""Alert clips cut from the stored Segments (V2 Requirement 13, task 14.1).

The window is ``[trigger - pre_s, trigger + post_s]``. :func:`covering` (pure) picks the
Segments of the camera that overlap it, with in/out points inside each; :func:`clip_command`
(pure) builds one ffmpeg run that joins them with the concat demuxer and re-encodes once to a
small H.264 MP4 (at most ``width`` px wide, no audio, ``faststart``) that Telegram plays inline.

:func:`clip_status` decides whether to cut now: a window that runs past the last stored
Segment waits for the next one, at most ``wait_s`` after the window ends; after that the clip
ends at the last available Segment (V2 Requirement 13.2).
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from nab_sentry.ingest.transcode import ffmpeg_exe, scale_filter

CONTIGUOUS_GAP_S = 1.5  # Segments closer than this count as continuous footage


@dataclass(frozen=True)
class Seg:
    path: Path
    start_s: float  # epoch seconds of the first frame
    duration_s: float

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s


@dataclass(frozen=True)
class Part:
    path: Path
    inpoint: float  # seconds from the start of the file
    outpoint: float


def covering(segs: Sequence[Seg], start_s: float, end_s: float) -> list[Part]:
    """Parts of the Segments overlapping ``[start_s, end_s]``, in time order."""
    parts = []
    for s in sorted(segs, key=lambda s: s.start_s):
        a, b = max(start_s, s.start_s), min(end_s, s.end_s)
        if b - a > 1e-3:
            parts.append(Part(s.path, round(a - s.start_s, 3), round(b - s.start_s, 3)))
    return parts


def clip_status(segs: Sequence[Seg], window_end_s: float, now_s: float, wait_s: float) -> str:
    """``ready`` (the window is covered, or waiting longer would not help), ``wait`` (the next
    Segment may still arrive) or ``none`` (no footage at all in the window)."""
    if not segs:
        return "none" if now_s >= window_end_s + wait_s else "wait"
    covered_to = max(s.end_s for s in segs)
    if covered_to >= window_end_s - 1e-3 or now_s >= window_end_s + wait_s:
        return "ready"
    return "wait"


def concat_list(parts: Sequence[Part]) -> str:
    """ffconcat file text with in/out points (paths quoted for the demuxer)."""
    lines = ["ffconcat version 1.0"]
    for p in parts:
        path = str(Path(p.path).resolve()).replace("\\", "/").replace("'", "'\\''")
        lines += [f"file '{path}'", f"inpoint {p.inpoint:.3f}", f"outpoint {p.outpoint:.3f}"]
    return "\n".join(lines) + "\n"


def clip_command(list_file: Path, out: Path, width: int, ffmpeg: str | None = None) -> list[str]:
    exe = ffmpeg if ffmpeg is not None else ffmpeg_exe()
    return [exe, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "concat", "-safe", "0", "-i", str(list_file),
            "-map", "0:v:0", "-vf", scale_filter(width), "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "28", "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(out)]


def make_clip(parts: Sequence[Part], out: Path, width: int, *, timeout_s: float = 120.0) -> Path:
    """Cut and encode the clip; raises ``RuntimeError`` (with ffmpeg's message) on failure."""
    if not parts:
        raise RuntimeError("no footage in the clip window")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lst = out.with_suffix(".ffconcat")
    tmp = out.with_suffix(".part.mp4")
    lst.write_text(concat_list(parts), encoding="utf-8")
    try:
        proc = subprocess.run(clip_command(lst, tmp, width), capture_output=True, timeout=timeout_s)
        if proc.returncode != 0 or not tmp.is_file() or tmp.stat().st_size == 0:
            err = proc.stderr.decode("utf-8", errors="replace").strip()[-300:]
            raise RuntimeError(f"ffmpeg clip failed ({proc.returncode}): {err}")
        tmp.replace(out)
        return out
    finally:
        lst.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)
