"""Segment Recorder (V2 Requirement 7).

One ffmpeg process per camera writes ``<CAMERA_ID>_<YYYYMMDDTHHMMSS>.mkv`` Segments of
``segment_s`` seconds into ``data/videos/`` with the segment muxer. Matroska is used because a
file whose writer dies stays readable up to its last cluster (an MP4 without its ``moov`` atom
does not). RTSP H.264/H.265 is stream-copied; DirectShow / HTTP (MJPEG) sources are encoded
with Intel Quick Sync (``h264_qsv``) or ``libx264``, with keyframes forced on segment
boundaries so every Segment starts with a keyframe.

Segment names come from the wall clock (``-strftime``) with one-second resolution, so the
source must run in real time (live cameras do; :func:`replay_command` uses ``-re``).

:class:`Recorder` supervises one camera: it restarts ffmpeg with exponential backoff
(:func:`backoff`), copies ffmpeg's stderr into ``data/logs/recorder_<CAMERA_ID>.log`` with
URL credentials masked, and writes a heartbeat ``data/live/<CAMERA_ID>.json`` every few seconds.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Callable

from nab_sentry.ingest.metadata import FILENAME_RE
from nab_sentry.live.camera_config import DSHOW_PREFIX, CameraSpec, mask_url
from nab_sentry.logging_setup import get_logger

log = get_logger("recorder")

BACKOFF_MAX_S = 60.0
HEALTHY_RESET_S = 60.0
RTSP_TIMEOUT_US = 10_000_000
SEGMENT_EXT = ".mkv"


# ---------------------------------------------------------------- pure command builders


def segment_pattern(out_dir: Path, camera_id: str) -> str:
    return str(Path(out_dir) / f"{camera_id}_%Y%m%dT%H%M%S{SEGMENT_EXT}")


def _encode_args(encoder: str, segment_s: int, fps_hint: int = 15) -> list[str]:
    if encoder == "copy":
        return ["-c:v", "copy"]
    gop = str(max(1, fps_hint * 2))
    keys = ["-force_key_frames", f"expr:gte(t,n_forced*{segment_s})"]
    if encoder == "qsv":
        return ["-c:v", "h264_qsv", "-preset", "veryfast", "-g", gop, *keys]
    if encoder == "x264":
        return ["-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
                "-pix_fmt", "yuv420p", "-g", gop, *keys]
    raise ValueError(f"unknown encoder {encoder!r}")


def _segment_args(out_dir: Path, camera_id: str, segment_s: int) -> list[str]:
    return ["-map", "0:v:0", "-an", "-f", "segment", "-segment_time", str(segment_s),
            "-segment_format", "matroska", "-reset_timestamps", "1", "-strftime", "1",
            segment_pattern(out_dir, camera_id)]


def input_args(spec: CameraSpec) -> list[str]:
    """ffmpeg input options + ``-i`` for a camera source (shared by recorder and snapshot)."""
    if spec.kind == "rtsp":
        return ["-rtsp_transport", "tcp", "-timeout", str(RTSP_TIMEOUT_US), "-i", spec.source]
    if spec.kind == "dshow":
        device = spec.source[len(DSHOW_PREFIX):]
        return ["-f", "dshow", "-rtbufsize", "64M", "-i", f"video={device}"]
    return ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "10",
            "-i", spec.source]


def snapshot_command(spec: CameraSpec, out: Path, ffmpeg: str) -> list[str]:
    """ffmpeg argv saving one frame of ``spec`` as a JPEG."""
    return [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", *input_args(spec),
            "-frames:v", "1", "-q:v", "2", "-y", str(out)]


def recorder_command(spec: CameraSpec, out_dir: Path, ffmpeg: str) -> list[str]:
    """ffmpeg argv recording ``spec`` into Segments under ``out_dir`` (list argv, no shell).

    stdin stays open (no ``-nostdin``) so :class:`Recorder` can stop ffmpeg with ``q``, which
    finalises the current Segment instead of cutting it off.
    """
    argv = [ffmpeg, "-hide_banner", "-loglevel", "warning", *input_args(spec)]
    return argv + _encode_args(spec.resolved_encoder, spec.segment_s) + _segment_args(
        out_dir, spec.camera_id, spec.segment_s)


def replay_command(video: Path, camera_id: str, out_dir: Path, segment_s: int, ffmpeg: str,
                   *, encode: bool = False, loop: bool = False) -> list[str]:
    """Feed a file through ffmpeg in real time (``-re``) as if it were a live camera."""
    argv = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "warning", "-re"]
    if loop:
        argv += ["-stream_loop", "-1"]
    argv += ["-i", str(video)]
    enc = _encode_args("x264", segment_s) if encode else ["-c:v", "copy"]
    return argv + enc + _segment_args(out_dir, camera_id, segment_s)


def backoff(n: int) -> float:
    """Restart delay after the ``n``-th consecutive failure: 1, 2, 4, ... capped at 60 s."""
    return float(min(BACKOFF_MAX_S, 2 ** max(0, n)))


def newest_segment(out_dir: Path, camera_id: str) -> Path | None:
    """The latest Segment of ``camera_id`` by the start time in its name, or ``None``."""
    best: tuple[str, Path] | None = None
    if not Path(out_dir).is_dir():
        return None
    for f in Path(out_dir).iterdir():
        m = FILENAME_RE.match(f.name)
        if m and m.group("cam") == camera_id and f.suffix.lower() == SEGMENT_EXT:
            if best is None or m.group("ts") > best[0]:
                best = (m.group("ts"), f)
    return best[1] if best else None


# ---------------------------------------------------------------- heartbeat


@dataclass
class Heartbeat:
    camera_id: str
    state: str  # starting | running | backoff | stopped
    restarts: int
    pid: int | None
    last_segment: str | None
    last_segment_mtime: float | None
    updated_at: float

    def write(self, path: Path) -> None:
        """Atomic write (tmp + replace) so readers never see half a file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.__dict__), encoding="utf-8")
        os.replace(tmp, path)

    @staticmethod
    def read(path: Path) -> Heartbeat | None:
        try:
            return Heartbeat(**json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return None


def heartbeat_path(live_dir: Path, camera_id: str) -> Path:
    return Path(live_dir) / f"{camera_id}.json"


# ---------------------------------------------------------------- supervisor


class Recorder:
    """Run, supervise and restart the ffmpeg recorder of one camera until :meth:`stop`."""

    def __init__(self, spec: CameraSpec, videos_dir: Path, live_dir: Path, logs_dir: Path,
                 ffmpeg: str, *, popen: Callable[..., subprocess.Popen] = subprocess.Popen,
                 sleep: Callable[[float], None] | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.spec = spec
        self.videos_dir = Path(videos_dir)
        self.live_dir = Path(live_dir)
        self.log_path = Path(logs_dir) / f"recorder_{spec.camera_id}.log"
        self.ffmpeg = ffmpeg
        self._popen = popen
        self._stop = threading.Event()
        self._sleep = sleep or (lambda s: self._stop.wait(s))
        self._clock = clock
        self._proc: subprocess.Popen | None = None
        self.restarts = 0
        self.state = "starting"

    # -- heartbeat ---------------------------------------------------------------------------

    def heartbeat(self) -> Heartbeat:
        seg = newest_segment(self.videos_dir, self.spec.camera_id)
        mtime = None
        if seg is not None:
            try:
                mtime = seg.stat().st_mtime
            except OSError:
                seg = None
        return Heartbeat(self.spec.camera_id, self.state, self.restarts,
                         self._proc.pid if self._proc else None,
                         seg.name if seg else None, mtime, self._clock())

    def write_heartbeat(self) -> None:
        try:
            self.heartbeat().write(heartbeat_path(self.live_dir, self.spec.camera_id))
        except OSError as exc:
            log.warning("heartbeat write failed for %s: %s", self.spec.camera_id, exc)

    # -- process -----------------------------------------------------------------------------

    def _pump_stderr(self, stream: IO[bytes], sink: IO[str]) -> None:
        for raw in iter(stream.readline, b""):
            line = mask_url(raw.decode("utf-8", errors="replace").rstrip())
            if line:
                sink.write(f"{datetime.now(timezone.utc).isoformat()} {line}\n")
                sink.flush()

    def run(self) -> None:
        """Blocking supervisor loop."""
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        argv = recorder_command(self.spec, self.videos_dir, self.ffmpeg)
        failures = 0
        cam = self.spec.camera_id
        with self.log_path.open("a", encoding="utf-8") as sink:
            while not self._stop.is_set():
                self.state = "running"
                started = self._clock()
                log.info("recorder %s: starting %s", cam, mask_url(self.spec.source))
                self._proc = self._popen(argv, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                         stderr=subprocess.PIPE)
                pump = threading.Thread(target=self._pump_stderr, args=(self._proc.stderr, sink),
                                        daemon=True)
                pump.start()
                while self._proc.poll() is None and not self._stop.is_set():
                    self.write_heartbeat()
                    self._sleep(min(5.0, float(self.spec.segment_s)))
                if self._stop.is_set():
                    self._terminate()
                    pump.join(timeout=5)
                    break
                pump.join(timeout=5)
                code = self._proc.returncode
                failures = 0 if self._clock() - started >= HEALTHY_RESET_S else failures + 1
                delay = backoff(failures)
                self.restarts += 1
                self.state = "backoff"
                self.write_heartbeat()
                log.warning("recorder %s: ffmpeg exited with %s; restarting in %.0f s", cam, code, delay)
                self._sleep(delay)
        self.state = "stopped"
        self._proc = None
        self.write_heartbeat()
        log.info("recorder %s: stopped", cam)

    def _terminate(self) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.stdin.write(b"q")  # graceful: ffmpeg closes the current Segment properly
            proc.stdin.flush()
        except (OSError, AttributeError, ValueError):
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

    def stop(self) -> None:
        self._stop.set()
