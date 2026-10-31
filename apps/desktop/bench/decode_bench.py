"""Decode speed per reading mode (desktop speed experiment, step 1).

Usage::

    venv\\Scripts\\python.exe apps\\desktop\\bench\\decode_bench.py VIDEO [VIDEO ...]

Times each way of getting frames out of a video, as seconds of wall time per hour of footage:
OpenCV decoding every frame, FFmpeg decoding every frame, FFmpeg keyframes only, and FFmpeg
decoding every frame but handing over only 2 frames per second scaled to 640 px wide (software
and hardware decoders). Prints a Markdown table.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import cv2
import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
OUT_2FPS = ["-an", "-vf", "fps=2,scale=640:-2", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
MODES = {
    "ffmpeg all frames": ["-i", "{v}", "-an", "-f", "null", "-"],
    "ffmpeg keyframes only": ["-skip_frame", "nokey", "-i", "{v}", "-an", "-f", "null", "-"],
    "ffmpeg 2 fps @640 (cpu)": ["-i", "{v}", *OUT_2FPS],
    "ffmpeg 2 fps @640 (d3d11va)": ["-hwaccel", "d3d11va", "-i", "{v}", *OUT_2FPS],
    "ffmpeg 2 fps @640 (qsv)": ["-hwaccel", "qsv", "-i", "{v}", *OUT_2FPS],
    "ffmpeg keyframes @640": ["-skip_frame", "nokey", "-i", "{v}", "-an", "-vf", "scale=640:-2",
                              "-fps_mode", "passthrough", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"],
}


def duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return n / fps


def run_ffmpeg(args: list[str], video: Path) -> tuple[float, int, str]:
    argv = [FF, "-hide_banner", "-v", "error", *[a.replace("{v}", str(video)) for a in args]]
    t0 = time.perf_counter()
    p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    nbytes = 0
    while chunk := p.stdout.read(1 << 20):
        nbytes += len(chunk)
    err = p.stderr.read().decode(errors="replace").strip().splitlines()
    p.wait()
    return time.perf_counter() - t0, nbytes, (err[-1] if p.returncode else "")


def opencv_all(video: Path) -> float:
    cap = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
    t0 = time.perf_counter()
    while cap.grab():
        cap.retrieve()
    cap.release()
    return time.perf_counter() - t0


def main(paths: list[str]) -> int:
    print("| Video | Mode | Wall s | s per footage hour | x real time | note |")
    print("|---|---|---|---|---|---|")
    for p in map(Path, paths):
        dur = duration(p)
        rows = [("opencv all frames", opencv_all(p), "")]
        for name, args in MODES.items():
            wall, nbytes, err = run_ffmpeg(args, p)
            rows.append((name, wall, err[:60]))
        for name, wall, note in rows:
            print(f"| {p.name} | {name} | {wall:.1f} | {wall / dur * 3600:.0f} | {dur / wall:.0f}x | {note} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
