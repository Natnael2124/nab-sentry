"""Playback_File tests (Requirements 6.5, 6.6).

Fast tests cover ``parse_ffmpeg_probe`` / ``playback_command`` on fixed ffmpeg stderr samples and
the pure top-level MP4 box scanner. Tests marked ``slow`` generate clips with the bundled ffmpeg
(lavfi ``testsrc``) and check the real remux and transcode outputs: H.264, ``moov`` before
``mdat``, duration within 0.5 s of the source, and downscaling to <= 1280 px with even dimensions.
"""

from __future__ import annotations

import io
import re
import struct
import subprocess
from pathlib import Path
from typing import BinaryIO

import pytest

from nab_sentry.ingest.transcode import (
    PlaybackJob,
    ProbeInfo,
    ffmpeg_exe,
    is_browser_safe,
    parse_ffmpeg_probe,
    playback_command,
    probe_video,
    scale_filter,
)

DURATION_TOLERANCE_S = 0.5
MAX_WIDTH = 1280


# ---------------------------------------------------------------------------
# Pure helper: top-level MP4 box scan
# ---------------------------------------------------------------------------

def scan_top_level_boxes(f: BinaryIO) -> list[tuple[str, int, int]]:
    """Return ``(type, offset, size)`` for each top-level ISO-BMFF box in ``f``.

    Handles the 64-bit ``largesize`` form (size field == 1) and the "extends to end of file" form
    (size field == 0). Raises ValueError on a truncated or malformed box header.
    """
    f.seek(0, io.SEEK_END)
    end = f.tell()
    boxes: list[tuple[str, int, int]] = []
    offset = 0
    while offset < end:
        f.seek(offset)
        header = f.read(8)
        if len(header) < 8:
            raise ValueError(f"truncated box header at offset {offset}")
        size, raw_type = struct.unpack(">I4s", header)
        box_type = raw_type.decode("latin-1")
        header_len = 8
        if size == 1:
            large = f.read(8)
            if len(large) < 8:
                raise ValueError(f"truncated largesize at offset {offset}")
            (size,) = struct.unpack(">Q", large)
            header_len = 16
        elif size == 0:
            size = end - offset
        if size < header_len or offset + size > end:
            raise ValueError(f"bad size {size} for box {box_type!r} at offset {offset}")
        boxes.append((box_type, offset, size))
        offset += size
    return boxes


def box_types(path: Path) -> list[str]:
    with path.open("rb") as f:
        return [t for t, _, _ in scan_top_level_boxes(f)]


def _box(box_type: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + box_type + payload


def _large_box(box_type: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 1) + box_type + struct.pack(">Q", 16 + len(payload)) + payload


def test_box_scan_regular_large_and_to_end():
    data = (
        _box(b"ftyp", b"isom\x00\x00\x02\x00")
        + _box(b"moov", b"\x00" * 20)
        + _large_box(b"free", b"\x01" * 5)
        + struct.pack(">I", 0) + b"mdat" + b"\xff" * 33  # size 0: runs to end of file
    )
    boxes = scan_top_level_boxes(io.BytesIO(data))
    assert [t for t, _, _ in boxes] == ["ftyp", "moov", "free", "mdat"]
    assert boxes[0] == ("ftyp", 0, 16)
    assert boxes[1] == ("moov", 16, 28)
    assert boxes[2] == ("free", 44, 21)
    assert boxes[3] == ("mdat", 65, len(data) - 65)


@pytest.mark.parametrize(
    "data",
    [
        b"\x00\x00\x00\x10ftyp",  # declared size runs past end
        _box(b"ftyp", b"") + b"\x00\x00",  # trailing partial header
        struct.pack(">I", 4) + b"moov",  # size smaller than header
        struct.pack(">I", 1) + b"mdat" + b"\x00\x00",  # truncated largesize
    ],
)
def test_box_scan_rejects_malformed(data):
    with pytest.raises(ValueError):
        scan_top_level_boxes(io.BytesIO(data))


# ---------------------------------------------------------------------------
# Fast: parse_ffmpeg_probe and playback_command on fixed stderr samples
# (complements tests/test_transcode_sanity.py, which covers High / 4:4:4 / mpeg4 samples)
# ---------------------------------------------------------------------------

MAIN_SAMPLE = """Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'C:\\cams\\lobby, east.mp4':
  Metadata:
    major_brand     : isom
  Duration: 01:02:03.04, start: 0.000000, bitrate: 2100 kb/s
  Stream #0:0[0x1](eng): Video: h264 (Main) (avc1 / 0x31637661), yuv420p(tv, bt709), 1280x720, 2000 kb/s, 25 fps
"""
CBASELINE_SAMPLE = """  Duration: 00:00:05.20, start: 0.000000, bitrate: 300 kb/s
  Stream #0:0: Video: h264 (Constrained Baseline) (avc1 / 0x31637661), yuv420p, 320x240, 15 fps
"""
HIGH10_SAMPLE = """  Duration: 00:00:07.00, start: 0.000000, bitrate: 1000 kb/s
  Stream #0:0: Video: h264 (High 10) (avc1 / 0x31637661), yuv420p10le(progressive), 1920x1080, 30 fps
"""
YUV444_SAMPLE = """  Duration: 00:00:03.00, start: 0.000000, bitrate: 800 kb/s
  Stream #0:0: Video: h264 (High 4:4:4 Predictive) (avc1 / 0x31637661), yuv444p(progressive), 2560x1440, 25 fps
"""
HEVC_SAMPLE = """  Duration: 00:00:09.50, start: 0.000000, bitrate: 1500 kb/s
  Stream #0:0: Video: hevc (Main) (hvc1 / 0x31637668), yuv420p(tv), 1920x1080, 30 fps
"""
AUDIO_FIRST_SAMPLE = """  Duration: 00:00:04.00, start: 0.000000, bitrate: 600 kb/s
  Stream #0:0: Audio: aac (LC), 44100 Hz, stereo, fltp, 128 kb/s
  Stream #0:1: Video: h264 (High), yuv420p(progressive), 640x360, 25 fps
"""


@pytest.mark.parametrize(
    "stderr, expected, safe",
    [
        (MAIN_SAMPLE, ProbeInfo("h264", "Main", "yuv420p", 3723.04), True),
        (CBASELINE_SAMPLE, ProbeInfo("h264", "Constrained Baseline", "yuv420p", 5.2), True),
        (AUDIO_FIRST_SAMPLE, ProbeInfo("h264", "High", "yuv420p", 4.0), True),
        (HIGH10_SAMPLE, ProbeInfo("h264", "High 10", "yuv420p10le", 7.0), False),
        (YUV444_SAMPLE, ProbeInfo("h264", "High 4:4:4 Predictive", "yuv444p", 3.0), False),
        (HEVC_SAMPLE, ProbeInfo("hevc", "Main", "yuv420p", 9.5), False),
    ],
)
def test_parse_probe_samples_and_path_choice(stderr, expected, safe):
    probe = parse_ffmpeg_probe(stderr)
    assert probe.codec == expected.codec
    assert probe.profile == expected.profile
    assert probe.pix_fmt == expected.pix_fmt
    assert probe.duration_s == pytest.approx(expected.duration_s)
    assert is_browser_safe(probe) is safe

    argv = playback_command(Path("src.mp4"), Path("out/v.mp4"), probe, MAX_WIDTH, ffmpeg="ffmpeg")
    assert argv[0] == "ffmpeg"
    assert argv[argv.index("-map") + 1] == "0:v:0"
    assert argv[argv.index("-movflags") + 1] == "+faststart"
    assert argv[-3:] == ["-f", "mp4", str(Path("out/v.mp4.part"))]
    if safe:
        assert argv[argv.index("-c") + 1] == "copy"
        assert "-vf" not in argv and "libx264" not in argv
    else:
        assert "-c" not in argv
        assert argv[argv.index("-c:v") + 1] == "libx264"
        assert argv[argv.index("-pix_fmt") + 1] == "yuv420p"
        assert argv[argv.index("-vf") + 1] == scale_filter(MAX_WIDTH)


def test_playback_command_uses_max_width_and_unknown_probe_transcodes():
    argv = playback_command(Path("a.avi"), Path("b.mp4"), ProbeInfo(None, None, None, None),
                            640, ffmpeg="ffmpeg")
    vf = argv[argv.index("-vf") + 1]
    assert vf == scale_filter(640)
    assert "640" in vf and "h=-2" in vf  # -2 keeps the height even


# ---------------------------------------------------------------------------
# Slow: real ffmpeg on generated clips
# ---------------------------------------------------------------------------

_DIMS_RE = re.compile(r"Video:[^\n]*?\b(\d{2,5})x(\d{2,5})\b")


def _ffmpeg_stderr(path: Path) -> str:
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-nostdin", "-i", str(path)],
                          stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                          stderr=subprocess.PIPE, timeout=60, check=False)
    return proc.stderr.decode("utf-8", errors="replace")


def _dims(path: Path) -> tuple[int, int]:
    m = _DIMS_RE.search(_ffmpeg_stderr(path))
    assert m, f"no video dimensions in probe of {path}"
    return int(m.group(1)), int(m.group(2))


def _make_clip(path: Path, size: str, duration: float, codec_args: list[str]) -> Path:
    """Generate a lavfi testsrc clip with the bundled ffmpeg (list argv, no shell)."""
    argv = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-f", "lavfi", "-i", f"testsrc=size={size}:rate=25:duration={duration}",
            *codec_args, str(path)]
    subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE, timeout=120, check=True)
    assert path.is_file()
    return path


def _make_playback(src: Path, dst: Path) -> tuple[ProbeInfo, list[str], ProbeInfo]:
    src_probe = probe_video(src)
    argv = playback_command(src, dst, src_probe, MAX_WIDTH)
    job = PlaybackJob(argv)
    job.wait(timeout=180)
    assert job.finalize() == dst
    assert dst.is_file() and not Path(str(dst) + ".part").exists()
    return src_probe, argv, probe_video(dst)


def _assert_playback_file(dst: Path, src_probe: ProbeInfo, out_probe: ProbeInfo) -> None:
    # Requirement 6.5: H.264 with moov before mdat (faststart).
    assert out_probe.codec == "h264"
    assert out_probe.pix_fmt == "yuv420p"
    types = box_types(dst)
    assert "moov" in types and "mdat" in types, types
    assert types.index("moov") < types.index("mdat"), types
    # Requirement 6.6: duration within 0.5 s of the source.
    assert src_probe.duration_s is not None and out_probe.duration_s is not None
    assert abs(out_probe.duration_s - src_probe.duration_s) <= DURATION_TOLERANCE_S


@pytest.mark.slow
def test_remux_browser_safe_h264(tmp_path):
    src = _make_clip(tmp_path / "cam 1 (lobby).mp4", "320x240", 3,
                     # ultrafast disables High-profile tools and is signalled as Constrained
                     # Baseline, so use veryfast to get a genuine High-profile source.
                     ["-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p",
                      "-preset", "veryfast"])
    dst = tmp_path / "playback" / "v1.mp4"
    dst.parent.mkdir()
    src_probe, argv, out_probe = _make_playback(src, dst)

    assert (src_probe.codec, src_probe.profile, src_probe.pix_fmt) == ("h264", "High", "yuv420p")
    assert argv[argv.index("-c") + 1] == "copy"  # remux path chosen
    _assert_playback_file(dst, src_probe, out_probe)
    assert out_probe.profile == "High"  # stream copied, not re-encoded
    assert _dims(dst) == (320, 240)


@pytest.mark.slow
@pytest.mark.parametrize(
    "name, size, codec_args, src_codec",
    [
        ("mpeg4 clip.avi", "320x240", ["-c:v", "mpeg4", "-q:v", "5"], "mpeg4"),
        ("h264 444.mkv", "320x240",
         ["-c:v", "libx264", "-pix_fmt", "yuv444p", "-preset", "ultrafast"], "h264"),
    ],
)
def test_transcode_non_browser_safe(tmp_path, name, size, codec_args, src_codec):
    src = _make_clip(tmp_path / name, size, 3, codec_args)
    dst = tmp_path / "v2.mp4"
    src_probe, argv, out_probe = _make_playback(src, dst)

    assert src_probe.codec == src_codec and not is_browser_safe(src_probe)
    assert argv[argv.index("-c:v") + 1] == "libx264"  # transcode path chosen
    _assert_playback_file(dst, src_probe, out_probe)
    assert is_browser_safe(out_probe)
    assert _dims(dst) == (320, 240)  # never upscaled, dimensions kept


@pytest.mark.slow
def test_transcode_downscales_wide_source(tmp_path):
    # 1922x1082 scales to 1280 wide; the exact height (720.6) must be rounded to an even value.
    src = _make_clip(tmp_path / "wide.mp4", "1922x1082", 2,
                     ["-c:v", "libx264", "-pix_fmt", "yuv444p", "-preset", "ultrafast"])
    dst = tmp_path / "v3.mp4"
    src_probe, _, out_probe = _make_playback(src, dst)

    _assert_playback_file(dst, src_probe, out_probe)
    w, h = _dims(dst)
    assert w <= MAX_WIDTH and w == MAX_WIDTH
    assert w % 2 == 0 and h % 2 == 0
    assert abs(h - w * 1082 / 1922) <= 2
