"""Segment Recorder (V2 Requirement 7): command builders, backoff, heartbeat, supervisor loop
with a fake ffmpeg, and (slow) a real-time replay into readable Segments."""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from nab_sentry.ingest.metadata import FILENAME_RE
from nab_sentry.live.camera_config import CameraSpec
from nab_sentry.live.recorder import (
    Heartbeat,
    Recorder,
    backoff,
    heartbeat_path,
    newest_segment,
    recorder_command,
    replay_command,
)

OUT = Path("data/videos")


def _after(argv, flag):
    return argv[argv.index(flag) + 1]


def test_rtsp_copy_command():
    spec = CameraSpec("CAM01", "Gate", "rtsp://u:p@10.0.0.2:554/s", segment_s=10)
    argv = recorder_command(spec, OUT, "ffmpeg")
    assert argv[0] == "ffmpeg" and "-nostdin" not in argv
    assert _after(argv, "-rtsp_transport") == "tcp" and _after(argv, "-i") == spec.source
    assert _after(argv, "-c:v") == "copy" and "-force_key_frames" not in argv
    assert _after(argv, "-segment_time") == "10" and _after(argv, "-segment_format") == "matroska"
    assert "-an" in argv and "-strftime" in argv
    assert argv[-1] == str(OUT / "CAM01_%Y%m%dT%H%M%S.mkv")


def test_dshow_qsv_command_forces_keyframes_on_segment_boundaries():
    spec = CameraSpec("CAM02", "Door", "dshow:Integrated Webcam", segment_s=15)
    argv = recorder_command(spec, OUT, "ffmpeg")
    assert _after(argv, "-f") == "dshow" and _after(argv, "-i") == "video=Integrated Webcam"
    assert _after(argv, "-c:v") == "h264_qsv"
    assert _after(argv, "-force_key_frames") == "expr:gte(t,n_forced*15)"


def test_http_x264_command():
    spec = CameraSpec("CAM03", "Phone", "http://192.168.1.9:8080/video", encoder="x264")
    argv = recorder_command(spec, OUT, "ffmpeg")
    assert "-reconnect" in argv and _after(argv, "-c:v") == "libx264"


def test_replay_command():
    argv = replay_command(Path("v.mp4"), "CAMR", OUT, 10, "ffmpeg", loop=True)
    assert "-re" in argv and _after(argv, "-stream_loop") == "-1" and _after(argv, "-c:v") == "copy"
    assert _after(replay_command(Path("v.mp4"), "CAMR", OUT, 10, "ffmpeg", encode=True), "-c:v") == "libx264"


@given(st.integers(0, 100))
def test_backoff_doubles_and_caps(n):
    assert backoff(n) == min(60.0, 2.0 ** n)
    assert backoff(n + 1) >= backoff(n)


def test_newest_segment_uses_name_time_and_camera(tmp_path: Path):
    for name in ("CAM01_20260101T000010.mkv", "CAM01_20260101T000000.mkv",
                 "CAM02_20260101T000020.mkv", "CAM01_20260101T000030.mp4", "junk.mkv"):
        (tmp_path / name).write_bytes(b"x")
    assert newest_segment(tmp_path, "CAM01").name == "CAM01_20260101T000010.mkv"
    assert newest_segment(tmp_path, "CAM09") is None
    assert newest_segment(tmp_path / "missing", "CAM01") is None


def test_heartbeat_round_trip(tmp_path: Path):
    hb = Heartbeat("CAM01", "running", 2, 1234, "CAM01_20260101T000000.mkv", 1.5, 2.5)
    p = heartbeat_path(tmp_path, "CAM01")
    hb.write(p)
    assert Heartbeat.read(p) == hb
    p.write_text("{broken", encoding="utf-8")
    assert Heartbeat.read(p) is None


class FakeProc:
    """Exits immediately with code 1 after printing a line that contains credentials."""

    def __init__(self, argv, **kw):
        self.argv, self.pid, self.returncode = argv, 4242, None
        self.stderr = io.BytesIO(b"[rtsp] Could not open rtsp://admin:hunter2@10.0.0.2/s\n")
        self.stdin = io.BytesIO()

    def poll(self):
        self.returncode = 1
        return 1

    def wait(self, timeout=None):
        return 1


def test_supervisor_restarts_with_backoff_and_masks_stderr(tmp_path: Path):
    spec = CameraSpec("CAM01", "Gate", "rtsp://admin:hunter2@10.0.0.2/s", segment_s=10)
    delays: list[float] = []
    t = {"now": 0.0}

    rec: Recorder

    def sleep(s):
        delays.append(s)
        t["now"] += s
        if len(delays) == 4:
            rec.stop()

    rec = Recorder(spec, tmp_path / "videos", tmp_path / "live", tmp_path / "logs", "ffmpeg",
                   popen=FakeProc, sleep=sleep, clock=lambda: t["now"])
    rec.run()
    assert delays == [2.0, 4.0, 8.0, 16.0]  # failures 1..4, each run shorter than 60 s
    assert rec.restarts == 4 and rec.state == "stopped"
    log_text = (tmp_path / "logs" / "recorder_CAM01.log").read_text(encoding="utf-8")
    assert "hunter2" not in log_text and "rtsp://***@10.0.0.2/s" in log_text
    hb = Heartbeat.read(heartbeat_path(tmp_path / "live", "CAM01"))
    assert hb.state == "stopped" and hb.restarts == 4


@pytest.mark.slow
def test_real_time_replay_produces_readable_segments(tmp_path: Path):
    from nab_sentry.ingest.transcode import ffmpeg_exe

    src = tmp_path / "src.mp4"
    w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    for i in range(130):  # 13 s
        w.write(np.full((120, 160, 3), (i * 7) % 255, np.uint8))
    w.release()
    out = tmp_path / "videos"
    out.mkdir()
    argv = replay_command(src, "CAMR", out, 5, ffmpeg_exe(), encode=True)
    subprocess.run(argv, check=True, timeout=60)
    segs = sorted(out.iterdir())
    assert len(segs) >= 2
    for s in segs:
        assert FILENAME_RE.match(s.name) and s.suffix == ".mkv"
        cap = cv2.VideoCapture(str(s))
        ok, _ = cap.read()
        cap.release()
        assert ok, s
