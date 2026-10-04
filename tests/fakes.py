"""Deterministic fakes for fast tests (FakeSource, FakeEncoder, FakeDetector, ...).

Later tasks add FakeDetector, FakePlaybackJob and FailureInjector.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from nab_sentry.ingest.sources import DecodedFrame
from nab_sentry.logging_setup import get_logger

# ---------------------------------------------------------------------------
# FakeSource (VideoSource)
# ---------------------------------------------------------------------------

_fake_source_log = get_logger("tests.fake_source")


def _default_frame(index: int, width: int, height: int) -> np.ndarray:
    """Small deterministic BGR frame whose pixel value depends on the index."""
    return np.full((height, width, 3), index % 256, dtype=np.uint8)


class FakeSource:
    """Scripted ``VideoSource``.

    - ``images``: explicit frames, one per decode index; or ``n_frames`` to generate
      deterministic ``width``x``height`` frames.
    - ``undecodable``: indices that fail to decode (counted, logged, not yielded).
    - ``empty``: indices yielded with a size-0 image.
    """

    def __init__(
        self,
        images: Sequence[np.ndarray] | None = None,
        *,
        n_frames: int | None = None,
        fps: float = 10.0,
        undecodable: Iterable[int] = (),
        empty: Iterable[int] = (),
        width: int = 16,
        height: int = 12,
        path: Path | str = Path("fake.mp4"),
        est_frame_count: int | None = None,
    ) -> None:
        if images is None and n_frames is None:
            raise ValueError("give images or n_frames")
        self._images = list(images) if images is not None else None
        self._n = len(self._images) if self._images is not None else int(n_frames)  # type: ignore[arg-type]
        self.path = Path(path)
        self.fps = float(fps)
        self.width = width
        self.height = height
        self.est_frame_count = self._n if est_frame_count is None else est_frame_count
        self.undecodable = frozenset(undecodable)
        self.empty = frozenset(empty)
        self._decoded_index_count = 0
        self._failed_frames = 0
        self.closed = False

    def frames(self) -> Iterator[DecodedFrame]:
        for index in range(self._n):
            self._decoded_index_count = index + 1
            offset = index / self.fps
            if index in self.undecodable:
                self._failed_frames += 1
                _fake_source_log.warning(
                    "frame decode failed: %s at offset %.3f s", self.path, offset)
                continue
            if index in self.empty:
                image = np.empty((0, 0, 3), dtype=np.uint8)
            elif self._images is not None:
                image = self._images[index]
            else:
                image = _default_frame(index, self.width, self.height)
            yield DecodedFrame(index=index, offset_s=offset, image=image)

    @property
    def decoded_index_count(self) -> int:
        return self._decoded_index_count

    @property
    def failed_frames(self) -> int:
        return self._failed_frames

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# FakeEncoder (Encoder)
# ---------------------------------------------------------------------------

import hashlib as _hashlib

from nab_sentry.embed.clip_encoder import (
    EMBED_DIM,
    EmptyQueryError,
    batched,
    l2_normalize,
)

RED_DIRECTION = np.zeros(EMBED_DIM, dtype=np.float32)
RED_DIRECTION[0] = 1.0
BLUE_DIRECTION = np.zeros(EMBED_DIM, dtype=np.float32)
BLUE_DIRECTION[1] = 1.0


def _hash_unit_vector(payload: bytes, zero_dims: int = 0) -> np.ndarray:
    """Deterministic unit vector seeded by SHA-256 of ``payload``.

    ``zero_dims`` leading components are zeroed so the vector is orthogonal to the
    colour directions in colour mode.
    """
    seed = int.from_bytes(_hashlib.sha256(payload).digest()[:8], "little")
    v = np.random.default_rng(seed).standard_normal(EMBED_DIM)
    v[:zero_dims] = 0.0
    return l2_normalize(v)


def _image_payload(image: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(image)
    return f"{arr.shape}|{arr.dtype}|".encode() + arr.tobytes()


class FakeEncoder:
    """Deterministic ``Encoder`` for fast tests.

    Default mode: every image/text maps to a unit vector derived from a hash of its
    bytes (images) or stripped text, so equal inputs give equal vectors.

    ``colour_mode=True``: images with at least ``min_colour_fraction`` saturated red
    (or blue) pixels map to ``RED_DIRECTION`` (or ``BLUE_DIRECTION``), whichever has
    more pixels; queries containing the word "red"/"blue" (e.g. "red square",
    "blue circle") map to the same directions. Everything else gets a hash vector
    orthogonal to both directions.

    ``batch_sizes`` records the size of every image batch encoded.
    """

    dim = EMBED_DIM

    def __init__(self, batch_size: int = 8, *, colour_mode: bool = False,
                 min_colour_fraction: float = 0.002) -> None:
        self.batch_size = batch_size
        self.colour_mode = colour_mode
        self.min_colour_fraction = min_colour_fraction
        self.batch_sizes: list[int] = []
        self.text_calls: list[str] = []

    def _colour_of(self, image: np.ndarray) -> str | None:
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] != 3 or arr.size == 0:
            return None
        b = arr[:, :, 0].astype(np.int16)
        g = arr[:, :, 1].astype(np.int16)
        r = arr[:, :, 2].astype(np.int16)
        red = int(np.count_nonzero((r >= 150) & (g <= 100) & (b <= 100)))
        blue = int(np.count_nonzero((b >= 150) & (g <= 100) & (r <= 100)))
        threshold = max(1, int(self.min_colour_fraction * arr.shape[0] * arr.shape[1]))
        if max(red, blue) < threshold:
            return None
        return "red" if red >= blue else "blue"

    def _encode_image(self, image: np.ndarray) -> np.ndarray:
        if self.colour_mode:
            colour = self._colour_of(image)
            if colour == "red":
                return RED_DIRECTION.copy()
            if colour == "blue":
                return BLUE_DIRECTION.copy()
            return _hash_unit_vector(_image_payload(image), zero_dims=2)
        return _hash_unit_vector(_image_payload(image))

    def encode_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for chunk in batched(list(images), self.batch_size):
            self.batch_sizes.append(len(chunk))
            rows.extend(self._encode_image(img) for img in chunk)
        if not rows:
            return np.empty((0, EMBED_DIM), dtype=np.float32)
        return np.stack(rows).astype(np.float32)

    def encode_text(self, query: str) -> np.ndarray:
        q = query.strip()
        if not q:
            raise EmptyQueryError()
        self.text_calls.append(q)
        if self.colour_mode:
            words = q.lower().split()
            if "red" in words:
                return RED_DIRECTION.copy()
            if "blue" in words:
                return BLUE_DIRECTION.copy()
            return _hash_unit_vector(b"text|" + q.encode("utf-8"), zero_dims=2)
        return _hash_unit_vector(b"text|" + q.encode("utf-8"))


# ---------------------------------------------------------------------------
# FakeDetector (Detector)
# ---------------------------------------------------------------------------

from typing import Callable as _Callable, Mapping as _Mapping

from nab_sentry.ingest.detector import Detection


class FakeDetectorError(RuntimeError):
    """Default exception raised by ``FakeDetector`` to simulate inference errors."""


class FakeDetector:
    """Scripted ``Detector``.

    ``script`` selects detections per call (0-based call index):
    - ``None``: always no detections;
    - a sequence: ``script[i]`` for call ``i``, no detections past the end;
    - a mapping ``{call_idx: [Detection, ...]}``: missing indices give none;
    - a callable ``(frame, call_idx) -> list[Detection]``.

    ``raise_on``: call indices at which ``detect`` raises ``error`` (an exception
    instance or class; default ``FakeDetectorError``) instead of returning.

    ``calls`` records ``(call_idx, frame.shape)`` for every call, including raising
    ones; ``frames`` keeps references to the frames passed in.
    """

    def __init__(
        self,
        script: (Sequence[Sequence[Detection]]
                 | _Mapping[int, Sequence[Detection]]
                 | _Callable[[np.ndarray, int], Sequence[Detection]]
                 | None) = None,
        *,
        raise_on: Iterable[int] = (),
        error: BaseException | type[BaseException] = FakeDetectorError,
    ) -> None:
        self.script = script
        self.raise_on = frozenset(raise_on)
        self.error = error
        self.calls: list[tuple[int, tuple[int, ...]]] = []
        self.frames: list[np.ndarray] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        idx = len(self.calls)
        self.calls.append((idx, tuple(np.asarray(frame).shape)))
        self.frames.append(frame)
        if idx in self.raise_on:
            err = self.error
            if isinstance(err, type):
                raise err(f"fake inference error at call {idx}")
            raise err
        script = self.script
        if script is None:
            return []
        if callable(script):
            return list(script(frame, idx))
        if isinstance(script, _Mapping):
            return list(script.get(idx, ()))
        return list(script[idx]) if idx < len(script) else []


# ---------------------------------------------------------------------------
# FakePlaybackJob (PlaybackJob) and FakePlayback (start_playback factory)
# ---------------------------------------------------------------------------

import os as _os

from nab_sentry.ingest.transcode import TranscodeError, part_path

# Minimal ISO-BMFF 'ftyp' box: enough for "an MP4 file exists" checks in fast tests.
FAKE_MP4_BYTES = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2"


class FakePlaybackJob:
    """Stand-in for ``PlaybackJob``: writes a tiny MP4 placeholder to ``dst.part`` at start.

    ``fail_on_wait=True`` makes ``wait()`` raise ``TranscodeError`` (encode failure).
    ``kill()`` removes the ``.part`` file, like the real job.
    """

    def __init__(self, src: Path, dst: Path, *, fail_on_wait: bool = False) -> None:
        self.src = Path(src)
        self.final = Path(dst)
        self.part = part_path(self.final)
        self.fail_on_wait = fail_on_wait
        self.waited = False
        self.killed = False
        self.finalized = False
        self.part.parent.mkdir(parents=True, exist_ok=True)
        self.part.write_bytes(FAKE_MP4_BYTES)

    def wait(self, timeout: float | None = None) -> None:
        self.waited = True
        if self.fail_on_wait:
            raise TranscodeError(f"fake ffmpeg failure for {self.src}")

    def kill(self) -> None:
        self.killed = True
        self.part.unlink(missing_ok=True)

    def finalize(self) -> Path:
        _os.replace(self.part, self.final)
        self.finalized = True
        return self.final


class FakePlayback:
    """``start_playback`` factory recording every ``FakePlaybackJob`` it starts.

    ``fail_for``: source file names (``Path.name``) whose job fails on ``wait()``;
    ``fail_all=True`` fails every job.
    """

    def __init__(self, *, fail_for: Iterable[str] = (), fail_all: bool = False) -> None:
        self.fail_for = frozenset(fail_for)
        self.fail_all = fail_all
        self.jobs: list[FakePlaybackJob] = []

    def __call__(self, src: Path, dst: Path) -> FakePlaybackJob:
        fail = self.fail_all or Path(src).name in self.fail_for
        job = FakePlaybackJob(src, dst, fail_on_wait=fail)
        self.jobs.append(job)
        return job


# ---------------------------------------------------------------------------
# FailureInjector (pipeline ``checkpoint`` hook)
# ---------------------------------------------------------------------------

import sqlite3 as _sqlite3

from nab_sentry.embed.clip_encoder import EmbeddingError
from nab_sentry.ingest.pipeline import CHECKPOINTS
from nab_sentry.ingest.transcode import ThumbnailError


class SimulatedCrash(BaseException):
    """Process death: the pipeline runs no compensation for it (only SQLite's rollback)."""


class InjectedFailure(RuntimeError):
    """Generic injected failure."""


def _default_error(step: str) -> BaseException:
    msg = f"injected failure at {step}"
    if step == "frame":
        return InjectedFailure(f"injected decode error at {step}")
    if step == "thumbnail":
        return ThumbnailError(msg)
    if step == "embed":
        return EmbeddingError(msg)
    if step == "playback_wait":
        return TranscodeError(msg)
    if step == "index_save":
        return OSError(msg)
    if step == "commit":
        return _sqlite3.OperationalError(msg)
    return InjectedFailure(msg)


class FailureInjector:
    """Pass as ``IngestPipeline(checkpoint=...)``; raises at a chosen pipeline step.

    - ``step``: one of ``nab_sentry.ingest.pipeline.CHECKPOINTS``.
    - ``nth``: fire on the n-th time the step is reached (1-based), counting only calls for
      ``video`` when it is given.
    - ``video``: only fire for the video whose file name (``Path.name``) equals this.
    - ``error``: exception instance or class to raise (default: a realistic error per step).
    - ``once``: fire at most once (default True).

    ``FailureInjector.crash_after_index_save()`` raises ``SimulatedCrash`` after the index
    has been saved and before COMMIT, i.e. orphan FAISS IDs on disk and no committed rows.
    """

    def __init__(
        self,
        step: str,
        *,
        nth: int = 1,
        video: str | None = None,
        error: BaseException | type[BaseException] | None = None,
        once: bool = True,
    ) -> None:
        if step not in CHECKPOINTS:
            raise ValueError(f"unknown pipeline step {step!r}; expected one of {CHECKPOINTS}")
        if nth < 1:
            raise ValueError("nth must be >= 1")
        self.step = step
        self.nth = nth
        self.video = video
        self.error = error
        self.once = once
        self.hits = 0
        self.fired = 0
        self.seen: list[tuple[str, str]] = []

    @classmethod
    def crash_after_index_save(cls, *, video: str | None = None) -> FailureInjector:
        return cls("after_index_save", video=video, error=SimulatedCrash)

    def _make_error(self) -> BaseException:
        err = self.error
        if err is None:
            return _default_error(self.step)
        if isinstance(err, type):
            return err(f"injected failure at {self.step}")
        return err

    def __call__(self, step: str, path: Path) -> None:
        self.seen.append((step, Path(path).name))
        if step != self.step or (self.video is not None and Path(path).name != self.video):
            return
        self.hits += 1
        if self.hits < self.nth or (self.once and self.fired):
            return
        self.fired += 1
        raise self._make_error()
