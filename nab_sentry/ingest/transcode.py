"""Thumbnails and Playback_Files (Requirements 6.4, 6.5).

Playback uses the ffmpeg binary bundled with imageio-ffmpeg. A source that is already H.264
``yuv420p`` with a browser-safe profile is remuxed (``-c copy``); anything else is transcoded with
libx264 and scaled down to at most ``max_width`` pixels. Both paths write ``+faststart`` MP4 to a
``.part`` file that the pipeline renames with :meth:`PlaybackJob.finalize` after a clean exit.

Every ffmpeg call uses a list argv (no shell), so file names with spaces or shell
metacharacters are passed through unchanged.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PART_SUFFIX = ".part"
BROWSER_SAFE_PROFILES = frozenset({"constrained baseline", "baseline", "main", "high"})
THUMB_JPEG_QUALITY = 85
JPEG_MAX_DIM = 65535
STDERR_TAIL_CHARS = 2000
PROBE_TIMEOUT_S = 60.0

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d{1,2}):(\d{1,2}(?:\.\d+)?)")
_VIDEO_RE = re.compile(r"Stream #[^\n]*?Video:\s*([A-Za-z0-9_]+)([^\n]*)")
_PIX_FMT_RE = re.compile(r"[a-z0-9_]+")
# A codec tag such as "(avc1 / 0x31637661)" is not a profile.
_CODEC_TAG_RE = re.compile(r"/\s*0x[0-9A-Fa-f]+")


class TranscodeError(RuntimeError):
    """Raised when ffmpeg fails to produce a Playback_File."""


class ThumbnailError(RuntimeError):
    """Raised when a Thumbnail cannot be encoded or written."""


@dataclass(frozen=True)
class ProbeInfo:
    codec: str | None
    profile: str | None
    pix_fmt: str | None
    duration_s: float | None


def ffmpeg_exe() -> str:
    """Absolute path of the ffmpeg binary bundled with imageio-ffmpeg."""
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _split_top_level(text: str) -> list[str]:
    """Split on commas that are not inside parentheses."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(ch)
    parts.append("".join(current))
    return parts


def parse_ffmpeg_probe(stderr: str) -> ProbeInfo:
    """Parse ``ffmpeg -i`` stderr into a ProbeInfo (pure).

    Reads the container ``Duration: hh:mm:ss.xx`` and the first video stream line, for example
    ``Stream #0:0: Video: h264 (High) (avc1 / 0x31637661), yuv420p(progressive), 1920x1080``.
    Fields that are absent or unparseable are ``None``.
    """
    duration: float | None = None
    m = _DURATION_RE.search(stderr)
    if m:
        h, mi, s = m.groups()
        duration = int(h) * 3600 + int(mi) * 60 + float(s)

    codec = profile = pix_fmt = None
    v = _VIDEO_RE.search(stderr)
    if v:
        codec = v.group(1).lower()
        fields = _split_top_level(v.group(2))
        head = fields[0]  # " (High) (avc1 / 0x...)" - parenthesised groups after the codec name
        groups = re.findall(r"\(([^()]*)\)", head)
        if groups and not _CODEC_TAG_RE.search(groups[0]):
            profile = groups[0].strip() or None
        if len(fields) > 1:
            token = fields[1].strip().split("(", 1)[0].strip().lower()
            if _PIX_FMT_RE.fullmatch(token):
                pix_fmt = token
    return ProbeInfo(codec=codec, profile=profile, pix_fmt=pix_fmt, duration_s=duration)


def probe_video(path: Path) -> ProbeInfo:
    """Run ``ffmpeg -hide_banner -i path`` and parse its stderr.

    ffmpeg exits non-zero because no output is given; only the stderr text matters. A missing or
    unreadable file yields a ProbeInfo of ``None`` fields, which selects the transcode path.
    """
    argv = [ffmpeg_exe(), "-hide_banner", "-nostdin", "-i", str(path)]
    try:
        proc = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=PROBE_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TranscodeError(f"ffmpeg probe failed for {path}: {exc}") from exc
    return parse_ffmpeg_probe(proc.stderr.decode("utf-8", errors="replace"))


def is_browser_safe(probe: ProbeInfo) -> bool:
    """True when the source can be remuxed without re-encoding."""
    return (
        probe.codec == "h264"
        and probe.pix_fmt == "yuv420p"
        and probe.profile is not None
        and probe.profile.strip().lower() in BROWSER_SAFE_PROFILES
    )


def part_path(dst: Path) -> Path:
    """The temporary output path ffmpeg writes to before the rename."""
    return dst.with_name(dst.name + PART_SUFFIX)


def scale_filter(max_width: int) -> str:
    """Scale to at most ``max_width`` wide, never upscale, keep even dimensions for yuv420p."""
    if max_width < 2:
        raise ValueError(f"max_width must be >= 2, got {max_width}")
    # Commas inside expressions are escaped so the filtergraph parser keeps them.
    return f"scale=w=max(2\\,trunc(min(iw\\,{int(max_width)})/2)*2):h=-2"


def playback_command(
    src: Path, dst: Path, probe: ProbeInfo, max_width: int, ffmpeg: str | None = None
) -> list[str]:
    """Build the ffmpeg argv that writes the Playback_File for ``src`` to ``dst.part`` (pure).

    ``ffmpeg`` defaults to the bundled binary; tests pass a fixed name.
    """
    exe = ffmpeg if ffmpeg is not None else ffmpeg_exe()
    argv = [exe, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-i", str(src), "-map", "0:v:0"]
    if is_browser_safe(probe):
        argv += ["-c", "copy"]
    else:
        argv += ["-vf", scale_filter(max_width), "-c:v", "libx264", "-preset", "veryfast",
                 "-crf", "23", "-pix_fmt", "yuv420p"]
    # Output name ends in .part, so the muxer is named explicitly.
    argv += ["-movflags", "+faststart", "-an", "-f", "mp4", str(part_path(dst))]
    return argv


class PlaybackJob:
    """A running ffmpeg process writing a ``.part`` Playback_File.

    The last argv element must be the ``.part`` output path; :meth:`finalize` renames it to the
    final name after :meth:`wait` returns.
    """

    def __init__(self, argv: list[str]):
        if not argv or not str(argv[-1]).endswith(PART_SUFFIX):
            raise ValueError("PlaybackJob argv must end with a .part output path")
        self.argv = [str(a) for a in argv]
        self.part = Path(self.argv[-1])
        self.final = self.part.with_name(self.part.name[: -len(PART_SUFFIX)])
        # Req 13.7: capture stderr next to the output (under data/), never in the OS temp dir.
        # TemporaryFile is anonymous / delete-on-close, and the ".ffmpeg-" prefix matches
        # neither of the reconcile orphan patterns nor the media-serving name pattern.
        try:
            self._log = tempfile.TemporaryFile(prefix=".ffmpeg-", suffix=".log", dir=self.part.parent)
        except OSError as exc:
            raise TranscodeError(f"could not create ffmpeg log in {self.part.parent}: {exc}") from exc
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self._proc = subprocess.Popen(
                self.argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=self._log,
                shell=False,
                creationflags=flags,
            )
        except OSError as exc:
            self._log.close()
            raise TranscodeError(f"could not start ffmpeg: {exc}") from exc

    def _stderr_tail(self) -> str:
        try:
            self._log.seek(0)
            text = self._log.read().decode("utf-8", errors="replace")
        except (OSError, ValueError):
            return ""
        return text[-STDERR_TAIL_CHARS:].strip()

    def wait(self, timeout: float | None = None) -> None:
        """Wait for ffmpeg; raise TranscodeError (with stderr tail) on failure."""
        try:
            code = self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            self.kill()
            raise TranscodeError(f"ffmpeg timed out after {timeout} s") from exc
        tail = self._stderr_tail()
        self._log.close()
        if code != 0:
            raise TranscodeError(f"ffmpeg exited with code {code}: {tail}")
        if not self.part.is_file():
            raise TranscodeError(f"ffmpeg exited cleanly but wrote no output: {self.part}")

    def kill(self) -> None:
        """Stop ffmpeg if still running and remove the partial output."""
        if self._proc.poll() is None:
            self._proc.kill()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        self._log.close()
        try:
            self.part.unlink(missing_ok=True)
        except OSError:
            pass

    def finalize(self) -> Path:
        """Rename ``.part`` to the final ``.mp4`` and return the final path."""
        try:
            os.replace(self.part, self.final)
        except OSError as exc:
            raise TranscodeError(f"could not finalize {self.part}: {exc}") from exc
        return self.final


def default_playback(src: Path, dst: Path, max_width: int = 1280) -> PlaybackJob:
    """Probe ``src`` and start the remux or transcode job writing ``dst`` via ``dst.part``."""
    probe = probe_video(src)
    return PlaybackJob(playback_command(src, dst, probe, max_width))


def thumbnail_size(w: int, h: int, width: int = 320) -> tuple[int, int]:
    """(width, height) of a Thumbnail for a ``w`` x ``h`` image, keeping the aspect ratio."""
    return width, max(1, round(width * h / w))


def write_thumbnail(image: np.ndarray, path: Path, width: int = 320) -> None:
    """Write ``image`` (BGR or grayscale uint8) as a JPEG ``width`` px wide, quality 85.

    Uses ``cv2.imencode`` plus ``Path.write_bytes`` so non-ASCII paths work on Windows.
    """
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3) or image.dtype != np.uint8:
        raise ThumbnailError("thumbnail source must be a 2-D or 3-D uint8 array")
    h, w = image.shape[:2]
    if h < 1 or w < 1:
        raise ThumbnailError(f"thumbnail source has zero size: {w}x{h}")
    tw, th = thumbnail_size(w, h, width)
    if th > JPEG_MAX_DIM:
        raise ThumbnailError(f"thumbnail height {th} exceeds the JPEG limit of {JPEG_MAX_DIM}")
    try:
        interp = cv2.INTER_AREA if tw <= w and th <= h else cv2.INTER_LINEAR
        small = cv2.resize(image, (tw, th), interpolation=interp)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, THUMB_JPEG_QUALITY])
    except cv2.error as exc:
        raise ThumbnailError(f"thumbnail encode failed: {exc}") from exc
    if not ok:
        raise ThumbnailError("thumbnail encode failed")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(buf.tobytes())
    except OSError as exc:
        raise ThumbnailError(f"thumbnail write failed for {path}: {exc}") from exc
