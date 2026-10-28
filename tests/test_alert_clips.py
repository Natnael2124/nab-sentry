"""Alert clips (V2 Requirement 13, task 14.2)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import pytest
from hypothesis import given
from hypothesis import strategies as st

from nab_sentry.alerts.clips import Part, Seg, clip_status, concat_list, covering, make_clip
from nab_sentry.ingest.transcode import ffmpeg_exe


@st.composite
def segment_runs(draw):
    """Contiguous 5-15 s Segments with occasional gaps."""
    t, segs = 1000.0, []
    for i in range(draw(st.integers(0, 6))):
        d = draw(st.floats(5, 15))
        segs.append(Seg(Path(f"s{i}.mkv"), t, d))
        t += d + draw(st.sampled_from([0.0, 0.0, 0.5, 20.0]))
    return segs


@given(segment_runs(), st.floats(990, 1100), st.floats(0.5, 20))
def test_covering_parts_lie_inside_window_and_segments(segs, start, length):
    end = start + length
    parts = covering(segs, start, end)
    by_path = {s.path: s for s in segs}
    total = 0.0
    for p in parts:
        s = by_path[p.path]
        assert 0 <= p.inpoint < p.outpoint <= s.duration_s + 1e-3
        assert s.start_s + p.inpoint >= start - 1e-3 and s.start_s + p.outpoint <= end + 1e-3
        total += p.outpoint - p.inpoint
    overlap = sum(max(0.0, min(end, s.end_s) - max(start, s.start_s)) for s in segs)
    assert total == pytest.approx(overlap, abs=0.01)
    starts = [by_path[p.path].start_s for p in parts]
    assert starts == sorted(starts)


def test_clip_status():
    segs = [Seg(Path("a"), 1000.0, 10.0)]
    assert clip_status(segs, 1008.0, now_s=1012.0, wait_s=25) == "ready"  # covered
    assert clip_status(segs, 1013.0, now_s=1014.0, wait_s=25) == "wait"  # next Segment may come
    assert clip_status(segs, 1013.0, now_s=1040.0, wait_s=25) == "ready"  # waited long enough
    assert clip_status([], 1013.0, now_s=1014.0, wait_s=25) == "wait"
    assert clip_status([], 1013.0, now_s=1040.0, wait_s=25) == "none"


def test_concat_list_quotes_paths(tmp_path: Path):
    text = concat_list([Part(tmp_path / "it's.mkv", 1.0, 2.5)])
    assert text.splitlines()[0] == "ffconcat version 1.0"
    assert "it'\\''s.mkv'" in text and "inpoint 1.000" in text and "outpoint 2.500" in text


def _segment(path: Path, seconds: int, colour: int) -> None:
    subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    f"color=c=0x{colour:06x}:s=320x240:r=10:d={seconds}", "-c:v", "libx264",
                    "-preset", "veryfast", "-g", "10", "-pix_fmt", "yuv420p", str(path)], check=True)


def test_make_clip_joins_two_segments(tmp_path: Path):
    a, b = tmp_path / "CAM_20260101T000000.mkv", tmp_path / "CAM_20260101T000005.mkv"
    _segment(a, 5, 0xFF0000)
    _segment(b, 5, 0x0000FF)
    segs = [Seg(a, 0.0, 5.0), Seg(b, 5.0, 5.0)]
    out = make_clip(covering(segs, 3.0, 8.0), tmp_path / "clip.mp4", width=160)
    cap = cv2.VideoCapture(str(out))
    n, w = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ok, first = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, n - 2)
    ok2, last = cap.read()
    cap.release()
    assert ok and ok2 and w == 160
    assert 45 <= n <= 55  # 5 s at 10 fps
    assert first[..., 2].mean() > 200 and last[..., 0].mean() > 200  # red, then blue (BGR)
    assert not list(tmp_path.glob("*.ffconcat")) and not list(tmp_path.glob("*.part.mp4"))


def test_make_clip_without_footage_raises(tmp_path: Path):
    with pytest.raises(RuntimeError, match="no footage"):
        make_clip([], tmp_path / "x.mp4", 320)
