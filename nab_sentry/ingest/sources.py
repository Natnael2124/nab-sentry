"""Video sources: ``DecodedFrame``, the ``VideoSource`` protocol and ``FileSource``.

Requirements 1.1, 1.11, 2.7, 2.9, 2.10.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol, runtime_checkable

import cv2
import numpy as np

from nab_sentry.logging_setup import get_logger

log = get_logger("ingest.sources")


class UnreadableVideo(Exception):
    """The file cannot be opened, or yields zero decoded / passed frames (1.11, 2.10)."""


class UnknownFrameRate(Exception):
    """The container reports a missing, zero, negative or NaN frame rate (2.9)."""


@dataclass(frozen=True)
class DecodedFrame:
    index: int  # zero-based decode index (counts failed frames too)
    offset_s: float  # index / fps
    image: np.ndarray  # may be empty (size 0) -> MotionGate discards it


@runtime_checkable
class VideoSource(Protocol):
    path: Path
    fps: float
    width: int
    height: int
    est_frame_count: int  # container estimate, used only for progress + initial duration

    def frames(self) -> Iterator[DecodedFrame]: ...

    @property
    def decoded_index_count(self) -> int: ...  # frames advanced so far (after iteration: total)

    @property
    def failed_frames(self) -> int: ...

    def close(self) -> None: ...


def _default_capture(path: Path) -> Any:
    return cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)


def _valid_fps(fps: Any) -> bool:
    try:
        f = float(fps)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f) and f > 0


def _as_nonneg_int(value: Any) -> int:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(f) or f < 0:
        return 0
    return int(f)


class FileSource:
    """OpenCV/FFmpeg-backed ``VideoSource``. Construct via :meth:`open`."""

    def __init__(self, path: Path, capture: Any, fps: float, width: int, height: int,
                 est_frame_count: int) -> None:
        self.path = Path(path)
        self.fps = float(fps)
        self.width = width
        self.height = height
        self.est_frame_count = est_frame_count
        self._cap = capture
        self._decoded_index_count = 0
        self._failed_frames = 0
        self._closed = False

    @classmethod
    def open(cls, path: Path,
             capture_factory: Callable[[Path], Any] = _default_capture) -> FileSource:
        """Open ``path``; raise ``UnreadableVideo`` or ``UnknownFrameRate``.

        ``capture_factory`` is injectable so tests can stub a capture (e.g. zero fps).
        """
        path = Path(path)
        try:
            cap = capture_factory(path)
        except Exception as exc:  # cv2 can raise on odd inputs
            raise UnreadableVideo(f"unreadable video: {path}") from exc
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            raise UnreadableVideo(f"unreadable video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not _valid_fps(fps):
            cap.release()
            raise UnknownFrameRate(f"unknown frame rate: {path}")
        return cls(
            path=path,
            capture=cap,
            fps=float(fps),
            width=_as_nonneg_int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=_as_nonneg_int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            est_frame_count=_as_nonneg_int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    def frames(self) -> Iterator[DecodedFrame]:
        """Yield decoded frames in order; failed decodes are logged and skipped (2.7)."""
        if self._closed:
            return
        while True:
            if not self._cap.grab():
                break
            index = self._decoded_index_count
            self._decoded_index_count += 1
            offset = index / self.fps
            try:
                ok, image = self._cap.retrieve()
            except cv2.error:
                ok, image = False, None
            if not ok or image is None:
                self._failed_frames += 1
                log.warning("frame decode failed: %s at offset %.3f s", self.path, offset)
                continue
            yield DecodedFrame(index=index, offset_s=offset, image=image)

    @property
    def decoded_index_count(self) -> int:
        return self._decoded_index_count

    @property
    def failed_frames(self) -> int:
        return self._failed_frames

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._cap.release()

    def __enter__(self) -> FileSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
