"""Reading footage with FFmpeg: probe, frame reader, clip cutter (design: media.py).

FFmpeg runs as a subprocess and hands over raw BGR frames, already scaled, on a pipe. Measured on
the development laptop (bench/decode_bench.py): decoding every frame 42x real time (OpenCV's
capture: 8x), keyframes only 254x.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import numpy as np

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".asf", ".wmv", ".ts", ".m2ts", ".mts", ".flv", ".dav",
              ".264", ".265", ".h264", ".h265", ".hevc", ".mpg", ".mpeg", ".m4v", ".3gp", ".webm"}
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class MediaError(RuntimeError):
    """FFmpeg could not read or write a file; ``str(exc)`` is a plain reason for the user."""


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


@dataclass(frozen=True)
class VideoInfo:
    duration_s: float
    fps: float
    width: int
    height: int
    codec: str
    keyframe_gap_s: float | None  # mean spacing of keyframes in the first minute; None: unknown


_DUR = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_VID = re.compile(r"Stream #\S+.*?: Video: (\w+)[^\n]*?,\s*(\d{2,5})x(\d{2,5})")
_FPS = re.compile(r"(\d+(?:\.\d+)?)\s*fps")
_TBR = re.compile(r"(\d+(?:\.\d+)?)\s*tbr")
_PTS = re.compile(r"pts_time:\s*([0-9.]+)")


def parse_probe(stderr: str) -> tuple[float, float, int, int, str]:
    """(duration_s, fps, width, height, codec) from ``ffmpeg -i`` output; raises MediaError."""
    v = _VID.search(stderr)
    if not v:
        raise MediaError("no video stream found - the file may be damaged or in an encrypted DVR format")
    line = stderr[v.start():stderr.find("\n", v.start()) if "\n" in stderr[v.start():] else len(stderr)]
    m = _FPS.search(line) or _TBR.search(line)
    fps = float(m.group(1)) if m else 0.0
    d = _DUR.search(stderr)
    dur = int(d.group(1)) * 3600 + int(d.group(2)) * 60 + float(d.group(3)) if d else 0.0
    return dur, fps, int(v.group(2)), int(v.group(3)), v.group(1)


def _run(argv: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, timeout=timeout, creationflags=_NO_WINDOW)


def keyframe_times(path: Path, seconds: float = 60.0) -> list[float]:
    """Presentation times of the keyframes in the first ``seconds`` of the video."""
    r = _run([ffmpeg_exe(), "-hide_banner", "-skip_frame", "nokey", "-t", str(seconds), "-i", str(path),
              "-an", "-vf", "showinfo", "-f", "null", "-"], timeout=120)
    return [float(x) for x in re.findall(r"pts_time:\s*([0-9.]+)", r.stderr.decode(errors="replace"))]


def count_duration(path: Path) -> float:
    """Duration by reading the whole stream (raw .264/.dav files often have no header duration)."""
    r = _run([ffmpeg_exe(), "-hide_banner", "-i", str(path), "-map", "0:v:0", "-c", "copy", "-f", "null", "-"],
             timeout=1800)
    times = re.findall(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)", r.stderr.decode(errors="replace"))
    if not times:
        return 0.0
    h, m, s = times[-1]
    return int(h) * 3600 + int(m) * 60 + float(s)


def probe(path: Path | str) -> VideoInfo:
    path = Path(path)
    if not path.is_file():
        raise MediaError("file not found")
    r = _run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], timeout=60)
    dur, fps, w, h, codec = parse_probe(r.stderr.decode(errors="replace"))
    if dur <= 0:
        dur = count_duration(path)
    if dur <= 0:
        raise MediaError("could not tell how long this video is - it may be damaged")
    kt = keyframe_times(path)
    gap = (kt[-1] - kt[0]) / (len(kt) - 1) if len(kt) >= 3 else None
    return VideoInfo(dur, fps if fps > 0 else 25.0, w, h, codec, gap)


def scaled_size(w: int, h: int, max_width: int) -> tuple[int, int]:
    """Even-sized output no wider than ``max_width``, aspect kept, never upscaled."""
    if w <= max_width:
        ow, oh = w, h
    else:
        ow, oh = max_width, round(h * max_width / w)
    return max(2, ow - ow % 2), max(2, oh - oh % 2)


Mode = Literal["keyframes", "rate"]


class FrameReader:
    """Frames ``(offset_s, bgr)`` of ``[start_s, end_s)`` from FFmpeg.

    ``mode="keyframes"`` decodes keyframes only; ``mode="rate"`` decodes everything and outputs
    ``rate`` frames per second. Offsets are video time (seconds from the file start). Use as a
    context manager or call :meth:`close`; closing kills FFmpeg.
    """

    def __init__(self, path: Path | str, info: VideoInfo, start_s: float = 0.0, end_s: float | None = None,
                 mode: Mode = "rate", rate: float = 2.0, max_width: int = 1280) -> None:
        self.path, self.info = Path(path), info
        self.start_s = max(0.0, float(start_s))
        self.end_s = info.duration_s if end_s is None else min(float(end_s), info.duration_s + 1)
        self.mode, self.rate = mode, float(rate)
        self.width, self.height = scaled_size(info.width, info.height, max_width)
        self._proc: subprocess.Popen | None = None
        self._err: list[str] = []  # last FFmpeg messages (not frame info)
        self._times: list[float] = []  # keyframe times from showinfo, in output order
        self._cv = threading.Condition()

    def argv(self) -> list[str]:
        # keyframe times come from showinfo on stderr, which needs the info log level
        a = [ffmpeg_exe(), "-hide_banner", "-v", "info" if self.mode == "keyframes" else "error", "-nostdin"]
        if self.mode == "keyframes":
            a += ["-skip_frame", "nokey"]
        if self.start_s > 0:
            a += ["-ss", f"{self.start_s:.3f}"]
        a += ["-i", str(self.path), "-t", f"{max(0.0, self.end_s - self.start_s):.3f}", "-an", "-sn", "-dn",
              "-map", "0:v:0"]
        scale = f"scale={self.width}:{self.height}:flags=area"
        if self.mode == "keyframes":
            a += ["-vf", f"{scale},showinfo", "-fps_mode", "passthrough"]
        else:
            a += ["-vf", f"fps={self.rate}:start_time=0,{scale}"]
        return a + ["-pix_fmt", "bgr24", "-f", "rawvideo", "-"]

    def __enter__(self) -> "FrameReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _drain_stderr(self, pipe) -> None:
        for raw in iter(pipe.readline, b""):
            line = raw.decode(errors="replace")
            m = _PTS.search(line)
            with self._cv:
                if m:
                    self._times.append(float(m.group(1)))
                    self._cv.notify_all()
                elif "Parsed_showinfo" not in line:
                    self._err = (self._err + [line.strip()])[-50:]
        with self._cv:
            self._done = True
            self._cv.notify_all()

    def frames(self) -> Iterator[tuple[float, np.ndarray]]:
        frame_bytes = self.width * self.height * 3
        self._proc = subprocess.Popen(self.argv(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      creationflags=_NO_WINDOW, bufsize=frame_bytes * 4)
        self._done = False
        t = threading.Thread(target=self._drain_stderr, args=(self._proc.stderr,), daemon=True)
        t.start()
        n = 0
        try:
            while True:
                buf = self._proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                img = np.frombuffer(buf, np.uint8).reshape(self.height, self.width, 3)
                if self.mode == "keyframes":
                    off = self._keyframe_time(n)
                else:
                    off = self.start_s + n / self.rate
                n += 1
                yield off, img
        finally:
            self.close()
            t.join(timeout=2)
        if n == 0 and self._proc is not None and self._proc.returncode != 0:
            raise MediaError(self.error_text() or "the video could not be decoded")

    def _keyframe_time(self, n: int) -> float:
        """Video time of keyframe ``n``: showinfo reports it on stderr before the frame reaches stdout."""
        with self._cv:
            self._cv.wait_for(lambda: len(self._times) > n or self._done, timeout=5.0)
            t = self._times[n] if len(self._times) > n else (self._times[-1] if self._times else 0.0)
        return self.start_s + t

    def error_text(self) -> str:
        return self._err[-1] if self._err else ""

    def close(self) -> None:
        p = self._proc
        if p is not None and p.poll() is None:
            p.kill()
            p.wait()


def cut_clip(src: Path | str, start_s: float, dur_s: float, dst: Path | str, overlay: str | None = None,
             max_width: int = 1280) -> Path:
    """H.264 MP4 of ``[start_s, start_s + dur_s)``; ``overlay`` text burned in at the top left."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".part.mp4")
    vf = f"scale='min({max_width},iw)':-2"
    if overlay:
        text = overlay.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’")
        vf += (f",drawtext=text='{text}':x=12:y=12:fontsize=h/24:fontcolor=white:"
               "box=1:boxcolor=black@0.55:boxborderw=8")
    argv = [ffmpeg_exe(), "-hide_banner", "-v", "error", "-nostdin", "-y", "-ss", f"{max(0.0, start_s):.3f}",
            "-i", str(src), "-t", f"{dur_s:.3f}", "-map", "0:v:0", "-an", "-vf", vf, "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)]
    r = _run(argv, timeout=600)
    if r.returncode != 0 and overlay:
        # some FFmpeg builds have no drawtext (no fontconfig): fall back to no overlay
        return cut_clip(src, start_s, dur_s, dst, None, max_width)
    if r.returncode != 0 or not tmp.is_file():
        tmp.unlink(missing_ok=True)
        raise MediaError("could not cut the clip: " + r.stderr.decode(errors="replace").strip()[-200:])
    tmp.replace(dst)
    return dst


def info_json(info: VideoInfo) -> str:
    return json.dumps(info.__dict__)


def grab_jpeg(src: Path | str, offset_s: float, max_side: int = 384, quality: int = 4) -> bytes:
    """One frame at ``offset_s`` as JPEG bytes, longest side at most ``max_side`` (MJPEG q 2-31)."""
    vf = f"scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'"
    argv = [ffmpeg_exe(), "-hide_banner", "-v", "error", "-nostdin", "-ss", f"{max(0.0, offset_s):.3f}",
            "-i", str(src), "-frames:v", "1", "-map", "0:v:0", "-an", "-vf", vf, "-q:v", str(quality),
            "-f", "image2pipe", "-vcodec", "mjpeg", "-"]
    r = _run(argv, timeout=60)
    if r.returncode != 0 or not r.stdout:
        raise MediaError("could not read a frame: " + r.stderr.decode(errors="replace").strip()[-200:])
    return r.stdout
