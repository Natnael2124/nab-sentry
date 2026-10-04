"""Fast sanity tests for nab_sentry.ingest.transcode (property/slow tests live in 9.9/9.10)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from nab_sentry.ingest.transcode import (
    PlaybackJob,
    ProbeInfo,
    ThumbnailError,
    TranscodeError,
    ffmpeg_exe,
    parse_ffmpeg_probe,
    playback_command,
    probe_video,
    write_thumbnail,
)

HIGH_SAMPLE = """Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'cam 1.mp4':
  Duration: 00:01:02.50, start: 0.000000, bitrate: 5000 kb/s
  Stream #0:0[0x1](und): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1920x1080 [SAR 1:1 DAR 16:9], 4999 kb/s, 30 fps
  Stream #0:1[0x2](und): Audio: aac (LC) (mp4a / 0x6134706D), 48000 Hz, stereo
"""
LOSSLESS_SAMPLE = """  Duration: 00:00:10.00, start: 0.000000, bitrate: 900 kb/s
  Stream #0:0: Video: h264 (High 4:4:4 Predictive) (avc1 / 0x31637661), yuv420p(progressive), 1280x720, 25 fps
"""
NO_PROFILE_SAMPLE = """  Duration: N/A, bitrate: N/A
  Stream #0:0: Video: mpeg4 (FMP4 / 0x34504D46), yuv420p, 640x480, 25 fps
"""


def test_parse_high_profile():
    assert parse_ffmpeg_probe(HIGH_SAMPLE) == ProbeInfo("h264", "High", "yuv420p", 62.5)


def test_parse_lossless_and_missing_fields():
    assert parse_ffmpeg_probe(LOSSLESS_SAMPLE) == ProbeInfo(
        "h264", "High 4:4:4 Predictive", "yuv420p", 10.0
    )
    assert parse_ffmpeg_probe(NO_PROFILE_SAMPLE) == ProbeInfo("mpeg4", None, "yuv420p", None)
    assert parse_ffmpeg_probe("garbage") == ProbeInfo(None, None, None, None)


def test_playback_command_remux_vs_transcode():
    src, dst = Path("in dir/a b;&.mp4"), Path("out/v1.mp4")
    remux = playback_command(src, dst, parse_ffmpeg_probe(HIGH_SAMPLE), 1280, ffmpeg="ffmpeg")
    assert remux[remux.index("-c") + 1] == "copy"
    assert "libx264" not in remux and "-vf" not in remux
    assert str(src) in remux  # passed as one argv element, unquoted
    assert remux[-3:] == ["-f", "mp4", str(Path("out/v1.mp4.part"))]
    assert "+faststart" in remux and "-an" in remux

    tx = playback_command(src, dst, parse_ffmpeg_probe(LOSSLESS_SAMPLE), 1280, ffmpeg="ffmpeg")
    for a, b in [("-c:v", "libx264"), ("-preset", "veryfast"), ("-crf", "23"),
                 ("-pix_fmt", "yuv420p"), ("-movflags", "+faststart")]:
        assert tx[tx.index(a) + 1] == b
    assert "-an" in tx and "1280" in tx[tx.index("-vf") + 1]
    assert tx[-1].endswith(".mp4.part")


@pytest.mark.parametrize("shape", [(480, 640, 3), (1, 4000, 3), (4000, 20), (37, 53, 3)])
def test_write_thumbnail_dimensions(tmp_path, shape):
    img = np.random.default_rng(0).integers(0, 256, size=shape, dtype=np.uint8)
    out = tmp_path / "thümb ñ" / "t.jpg"  # non-ASCII directory
    write_thumbnail(img, out)
    decoded = cv2.imdecode(np.frombuffer(out.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    h, w = shape[:2]
    assert decoded.shape[1] == 320
    assert abs(decoded.shape[0] - max(1, 320 * h / w)) <= 1


def test_write_thumbnail_errors(tmp_path):
    with pytest.raises(ThumbnailError):
        write_thumbnail(np.zeros((0, 10, 3), np.uint8), tmp_path / "x.jpg")
    with pytest.raises(ThumbnailError):
        write_thumbnail(np.zeros((10, 10, 3), np.float32), tmp_path / "x.jpg")
    with pytest.raises(ThumbnailError, match="JPEG limit"):  # 320 x 1,280,000 is not encodable
        write_thumbnail(np.zeros((4000, 1), np.uint8), tmp_path / "x.jpg")


def test_playback_job_transcodes_tiny_clip(tmp_path):
    src = tmp_path / "src clip.mp4"
    gen = PlaybackJob([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                       "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=1",
                       "-c:v", "mpeg4", "-f", "mp4", str(src) + ".part"])
    gen.wait(timeout=60)
    assert gen.finalize() == src

    probe = probe_video(src)
    assert probe.codec == "mpeg4" and probe.duration_s == pytest.approx(1.0, abs=0.2)

    dst = tmp_path / "v1.mp4"
    job = PlaybackJob(playback_command(src, dst, probe, 1280))
    job.wait(timeout=60)
    assert job.finalize() == dst and dst.is_file()
    out = probe_video(dst)
    assert (out.codec, out.pix_fmt) == ("h264", "yuv420p")


def test_playback_job_failure_and_kill(tmp_path):
    job = PlaybackJob(playback_command(tmp_path / "missing.mp4", tmp_path / "v2.mp4",
                                       ProbeInfo(None, None, None, None), 1280))
    with pytest.raises(TranscodeError, match="exited with code"):
        job.wait(timeout=60)

    job = PlaybackJob(playback_command(tmp_path / "missing.mp4", tmp_path / "v3.mp4",
                                       ProbeInfo(None, None, None, None), 1280))
    job.kill()
    assert not (tmp_path / "v3.mp4.part").exists()
