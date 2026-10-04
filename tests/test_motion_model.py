"""Property 13: the motion gate matches a small reference decision model.

Requirements 3.2, 3.3, 3.4, 3.7, 3.9.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from nab_sentry.ingest.motion import EPS, MotionGate, changed_fraction, downscale_gray
from nab_sentry.ingest.sources import DecodedFrame


# ---------------------------------------------------------------- reference model


class ReferenceGate:
    """Independent re-statement of the gate rules (diff method).

    Per video: empty frames are discarded without touching state; the first valid
    frame passes; a frame >= K after the last passed offset passes as keyframe; a
    frame whose changed fraction vs the previous *valid* frame is >= threshold
    passes as motion; everything else is static.
    """

    def __init__(self, threshold: float, k: float, gate_width: int, pixel_delta: int) -> None:
        self.threshold = threshold
        self.k = k
        self.gate_width = gate_width
        self.pixel_delta = pixel_delta
        self.prev: Optional[np.ndarray] = None
        self.last_passed: Optional[float] = None

    def reset(self) -> None:
        self.prev = None
        self.last_passed = None

    def step(self, offset: float, image: np.ndarray) -> Tuple[bool, str, float]:
        if image.size == 0:
            return False, "empty", 0.0
        small = downscale_gray(image, self.gate_width)
        if self.prev is None:
            self.prev = small
            self.last_passed = offset
            return True, "first", 0.0
        fraction = changed_fraction(self.prev, small, self.pixel_delta)
        self.prev = small
        assert self.last_passed is not None
        if offset - self.last_passed >= self.k - EPS:
            self.last_passed = offset
            return True, "keyframe", fraction
        if fraction >= self.threshold:
            self.last_passed = offset
            return True, "motion", fraction
        return False, "static", fraction


# ---------------------------------------------------------------- generators

FRAME_KINDS = ("empty", "same", "small", "big", "resize")


@dataclass(frozen=True)
class VideoSpec:
    seed: int
    height: int
    width: int
    color: bool
    fps: float
    gaps: List[int]  # index increments (>= 1) between sampled frames
    kinds: List[str]


@st.composite
def video_specs(draw) -> VideoSpec:
    n = draw(st.integers(min_value=1, max_value=25))
    return VideoSpec(
        seed=draw(st.integers(min_value=0, max_value=2**32 - 1)),
        height=draw(st.integers(min_value=8, max_value=48)),
        width=draw(st.integers(min_value=8, max_value=64)),
        color=draw(st.booleans()),
        fps=draw(st.sampled_from([1.0, 2.0, 5.0, 10.0, 24.0, 25.0, 29.97, 30.0])),
        gaps=draw(st.lists(st.integers(min_value=1, max_value=30), min_size=n, max_size=n)),
        kinds=draw(st.lists(
            st.sampled_from(FRAME_KINDS),
            min_size=n, max_size=n,
        )),
    )


def build_frames(spec: VideoSpec) -> List[DecodedFrame]:
    """Materialise a frame sequence: empties, repeats, small/big changes, resizes."""
    rng = np.random.default_rng(spec.seed)

    def rand_image(h: int, w: int) -> np.ndarray:
        shape = (h, w, 3) if spec.color else (h, w)
        return rng.integers(0, 256, size=shape, dtype=np.uint8)

    current = rand_image(spec.height, spec.width)
    frames: List[DecodedFrame] = []
    index = -1
    for gap, kind in zip(spec.gaps, spec.kinds):
        index += gap
        if kind == "empty":
            image = np.zeros((0, 0, 3), dtype=np.uint8) if spec.color else np.empty((0,), np.uint8)
        else:
            if kind == "small":
                current = current.copy()
                h, w = current.shape[:2]
                y, x = int(rng.integers(0, h)), int(rng.integers(0, w))
                dh, dw = int(rng.integers(1, max(2, h // 3))), int(rng.integers(1, max(2, w // 3)))
                delta = int(rng.integers(-120, 121))
                patch = current[y:y + dh, x:x + dw].astype(np.int16) + delta
                current[y:y + dh, x:x + dw] = np.clip(patch, 0, 255).astype(np.uint8)
            elif kind == "big":
                h, w = current.shape[:2]
                current = rand_image(h, w)
            elif kind == "resize":
                current = rand_image(int(rng.integers(8, 49)), int(rng.integers(8, 65)))
            image = current.copy()
        frames.append(DecodedFrame(index=index, offset_s=index / spec.fps, image=image))
    return frames


gate_params = st.fixed_dictionaries({
    "threshold": st.one_of(
        st.sampled_from([0.0, 0.001, 0.01, 0.05, 0.5, 1.0]),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    ),
    "keyframe_interval_s": st.one_of(
        st.sampled_from([0.5, 1.0, 2.0, 5.0, 10.0]),
        st.floats(min_value=0.05, max_value=20.0, allow_nan=False),
    ),
    "gate_width": st.sampled_from([8, 16, 32, 64, 320]),
    "pixel_delta": st.integers(min_value=0, max_value=80),
})


# ---------------------------------------------------------------- property


# Feature: nab-sentry, Property 13: Motion gate matches the reference decision model
@given(params=gate_params, videos=st.lists(video_specs(), min_size=1, max_size=3))
def test_motion_gate_matches_reference_model(params, videos):
    """**Validates: Requirements 3.2, 3.3, 3.4, 3.7, 3.9**"""
    gate = MotionGate(method="diff", **params)
    ref = ReferenceGate(params["threshold"], params["keyframe_interval_s"],
                        params["gate_width"], params["pixel_delta"])

    for v, spec in enumerate(videos):
        gate.reset()  # one gate reused across videos, reset between them
        ref.reset()
        for frame in build_frames(spec):
            got = gate.evaluate(frame)
            exp_passed, exp_reason, exp_fraction = ref.step(frame.offset_s, frame.image)
            where = f"video {v}, frame index {frame.index}, offset {frame.offset_s:.4f}"
            assert got.reason == exp_reason, where
            assert got.passed == exp_passed, where
            assert got.fraction == exp_fraction, where
            assert 0.0 <= got.fraction <= 1.0


# ---------------------------------------------------------------- examples


def _frame(i: int, offset: float, image: np.ndarray) -> DecodedFrame:
    return DecodedFrame(index=i, offset_s=offset, image=image)


def test_reset_makes_next_video_start_with_first():
    img = np.full((24, 32), 100, np.uint8)
    gate = MotionGate(threshold=0.01, keyframe_interval_s=10.0, gate_width=32, pixel_delta=25)
    assert gate.evaluate(_frame(0, 0.0, img)).reason == "first"
    assert gate.evaluate(_frame(1, 1.0, img)).reason == "static"
    gate.reset()
    # Earlier offset than the last video's frames; must still be the first of a new video.
    d = gate.evaluate(_frame(0, 0.0, img))
    assert d.passed and d.reason == "first"


def test_empty_frames_leave_state_untouched():
    a = np.full((24, 32), 50, np.uint8)
    gate = MotionGate(threshold=0.01, keyframe_interval_s=5.0, gate_width=32, pixel_delta=25)
    empty = np.empty((0, 0, 3), np.uint8)
    assert gate.evaluate(_frame(0, 0.0, empty)).reason == "empty"  # not consumed as "first"
    assert gate.evaluate(_frame(1, 1.0, a)).reason == "first"
    assert gate.evaluate(_frame(2, 2.0, empty)).reason == "empty"
    # Same image after the empty frame: compared to `a`, so static.
    assert gate.evaluate(_frame(3, 3.0, a)).reason == "static"
    # Keyframe counted from the first pass at 1.0, not disturbed by empties.
    assert gate.evaluate(_frame(4, 6.0, a)).reason == "keyframe"
