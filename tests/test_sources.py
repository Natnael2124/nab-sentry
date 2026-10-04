"""Unit tests for ``FileSource`` and ``FakeSource`` (task 5.9).

Requirements 1.1, 1.11, 2.9, 2.10.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

from nab_sentry.ingest.sources import (
    DecodedFrame,
    FileSource,
    UnknownFrameRate,
    UnreadableVideo,
    VideoSource,
)
from tests.fakes import FakeSource


class StubCapture:
    """Minimal ``cv2.VideoCapture`` stand-in."""

    def __init__(self, *, fps: float = 10.0, n_frames: int = 5, opened: bool = True,
                 fail_indices: frozenset[int] = frozenset(), width: int = 8,
                 height: int = 6) -> None:
        self.fps = fps
        self.n_frames = n_frames
        self.opened = opened
        self.fail_indices = fail_indices
        self.width = width
        self.height = height
        self._pos = -1
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - mirrors OpenCV API
        return self.opened

    def get(self, prop: int) -> float:
        return {
            cv2.CAP_PROP_FPS: self.fps,
            cv2.CAP_PROP_FRAME_WIDTH: float(self.width),
            cv2.CAP_PROP_FRAME_HEIGHT: float(self.height),
            cv2.CAP_PROP_FRAME_COUNT: float(self.n_frames),
        }.get(prop, 0.0)

    def grab(self) -> bool:
        if self._pos + 1 >= self.n_frames:
            return False
        self._pos += 1
        return True

    def retrieve(self):
        if self._pos in self.fail_indices:
            return False, None
        return True, np.full((self.height, self.width, 3), self._pos, dtype=np.uint8)

    def release(self) -> None:
        self.released = True


def _write_clip(path: Path, n_frames: int = 12, fps: float = 10.0) -> Path:
    w, h = 64, 48
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert writer.isOpened(), "OpenCV could not open a VideoWriter for mp4v"
    for i in range(n_frames):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.rectangle(img, (i * 4 % w, 10), (i * 4 % w + 8, 30), (255, 255, 255), -1)
        writer.write(img)
    writer.release()
    return path


# --- real files ------------------------------------------------------------

def test_garbage_bytes_file_raises_unreadable(tmp_path: Path) -> None:
    bad = tmp_path / "garbage.mp4"
    bad.write_bytes(b"\x00\x13not a video at all\xff" * 64)
    with pytest.raises(UnreadableVideo):
        FileSource.open(bad)


def test_missing_file_raises_unreadable(tmp_path: Path) -> None:
    with pytest.raises(UnreadableVideo):
        FileSource.open(tmp_path / "does-not-exist.mp4")


def test_opencv_clip_offsets_start_at_zero_and_non_decreasing(tmp_path: Path) -> None:
    clip = _write_clip(tmp_path / "clip.mp4", n_frames=12, fps=10.0)
    with FileSource.open(clip) as src:
        assert isinstance(src, VideoSource)
        assert src.fps > 0
        frames = list(src.frames())
        assert frames, "expected at least one decoded frame"
        assert frames[0].offset_s == 0.0
        offsets = [f.offset_s for f in frames]
        assert all(a <= b for a, b in zip(offsets, offsets[1:]))
        for f in frames:
            assert isinstance(f, DecodedFrame)
            assert f.offset_s == pytest.approx(f.index / src.fps)
            assert f.image.size > 0
        assert src.decoded_index_count == len(frames) + src.failed_frames
        assert src.width == 64 and src.height == 48


# --- stubbed captures ------------------------------------------------------

@pytest.mark.parametrize("fps", [0.0, float("nan"), -5.0, float("inf"), None])
def test_invalid_fps_raises_unknown_frame_rate(fps) -> None:
    stub = StubCapture(fps=fps)  # type: ignore[arg-type]
    with pytest.raises(UnknownFrameRate):
        FileSource.open(Path("stub.mp4"), capture_factory=lambda _p: stub)
    assert stub.released


def test_not_opened_capture_raises_unreadable() -> None:
    stub = StubCapture(opened=False)
    with pytest.raises(UnreadableVideo):
        FileSource.open(Path("stub.mp4"), capture_factory=lambda _p: stub)
    assert stub.released


def test_capture_factory_exception_raises_unreadable() -> None:
    def boom(_p: Path):
        raise cv2.error("boom")

    with pytest.raises(UnreadableVideo):
        FileSource.open(Path("stub.mp4"), capture_factory=boom)


def test_failed_retrieve_is_skipped_logged_and_counted(caplog) -> None:
    stub = StubCapture(fps=4.0, n_frames=5, fail_indices=frozenset({2}))
    src = FileSource.open(Path("stub.mp4"), capture_factory=lambda _p: stub)
    with caplog.at_level(logging.WARNING, logger="nab_sentry"):
        frames = list(src.frames())
    assert [f.index for f in frames] == [0, 1, 3, 4]
    assert [f.offset_s for f in frames] == [0.0, 0.25, 0.75, 1.0]
    assert src.failed_frames == 1
    assert src.decoded_index_count == 5
    assert any("frame decode failed" in r.getMessage() and "0.500" in r.getMessage()
               for r in caplog.records)
    src.close()
    assert stub.released


def test_close_is_idempotent_and_stops_iteration() -> None:
    stub = StubCapture(n_frames=3)
    src = FileSource.open(Path("stub.mp4"), capture_factory=lambda _p: stub)
    src.close()
    src.close()
    assert stub.released
    assert list(src.frames()) == []


# --- FakeSource ------------------------------------------------------------

def test_fake_source_implements_protocol_and_skips_undecodable() -> None:
    fake = FakeSource(n_frames=4, fps=2.0, undecodable={1}, empty={3})
    assert isinstance(fake, VideoSource)
    frames = list(fake.frames())
    assert [f.index for f in frames] == [0, 2, 3]
    assert [f.offset_s for f in frames] == [0.0, 1.0, 1.5]
    assert frames[-1].image.size == 0
    assert fake.failed_frames == 1
    assert fake.decoded_index_count == 4
