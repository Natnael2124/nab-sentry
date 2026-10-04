"""Motion gate: skip static sampled frames before the expensive models run.

Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.7, 3.9 (design key decision 4: frame
differencing is the default; MOG2 is optional for real footage with lighting drift).

``evaluate`` decision order:

1. Empty / zero-size image -> ``empty``, not passed, state untouched (3.9).
2. No valid frame seen yet in this video -> ``first``, passed (3.4).
3. ``offset - last_passed_offset >= keyframe_interval_s - EPS`` -> ``keyframe`` (3.3).
4. Changed fraction ``>= threshold`` -> ``motion``, passed (3.2).
5. Otherwise ``static``, discarded (3.7).

The previous-frame reference (and the MOG2 model) is updated on every valid
frame; ``last_passed_offset`` only on passes.

Resolution change: if the downscaled shape of the current frame differs from
the previous reference (a mid-stream resolution change), the frames cannot be
compared pixel-for-pixel, so the changed fraction is defined as 1.0 (the whole
scene is treated as changed) and, for MOG2, the background model is rebuilt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import cv2
import numpy as np

from nab_sentry.ingest.sources import DecodedFrame

EPS = 1e-9

Reason = Literal["first", "motion", "keyframe", "static", "empty"]
Method = Literal["diff", "mog2"]


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    reason: Reason
    fraction: float  # 0.0..1.0; 0.0 for "first"/"empty"


def _is_empty(image: Any) -> bool:
    return image is None or not isinstance(image, np.ndarray) or image.size == 0


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Grayscale uint8 copy of a 2-D gray, HxWx1, BGR, or BGRA image."""
    img = image
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        channels = img.shape[2]
        if channels == 1:
            return img[:, :, 0]
        if channels == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if channels == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"unsupported image shape for motion gate: {image.shape}")


def downscale_gray(image: np.ndarray, gate_width: int) -> np.ndarray:
    """Grayscale, downscaled to ``gate_width`` (never upscaled, aspect kept), 5x5 blur.

    Frames whose width is at or below ``gate_width`` keep their size (3.1).
    """
    gray = _to_gray(image)
    h, w = gray.shape[:2]
    if w > gate_width:
        new_w = int(gate_width)
        new_h = max(1, int(round(h * new_w / w)))
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def changed_fraction(prev: np.ndarray, cur: np.ndarray, pixel_delta: int) -> float:
    """Fraction of pixels whose absolute grey-level change exceeds ``pixel_delta``.

    Returns a value in [0.0, 1.0]; exactly 0.0 for identical inputs. Differing
    shapes (resolution change) return 1.0; empty inputs return 0.0.
    """
    if prev.shape != cur.shape:
        return 1.0
    if cur.size == 0:
        return 0.0
    diff = cv2.absdiff(prev, cur)
    return float(np.mean(diff > pixel_delta))


class MotionGate:
    def __init__(self, threshold: float, keyframe_interval_s: float, gate_width: int,
                 pixel_delta: int, method: Method = "diff") -> None:
        if method not in ("diff", "mog2"):
            raise ValueError(f"motion method must be 'diff' or 'mog2', got {method!r}")
        self.threshold = float(threshold)
        self.keyframe_interval_s = float(keyframe_interval_s)
        self.gate_width = int(gate_width)
        self.pixel_delta = int(pixel_delta)
        self.method: Method = method
        self._prev: Optional[np.ndarray] = None
        self._last_passed_offset: Optional[float] = None
        self._mog2: Any = None
        self.reset()

    def reset(self) -> None:
        """Clear all motion and keyframe state; call at the start of every video (3.4)."""
        self._prev = None
        self._last_passed_offset = None
        self._mog2 = self._new_mog2() if self.method == "mog2" else None

    @staticmethod
    def _new_mog2() -> Any:
        return cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    def _mog2_fraction(self, small: np.ndarray, shape_changed: bool) -> float:
        if shape_changed:
            self._mog2 = self._new_mog2()
            self._mog2.apply(small)
            return 1.0
        mask = self._mog2.apply(small)
        return float(np.mean(mask > 0))

    def evaluate(self, frame: DecodedFrame) -> GateDecision:
        image = frame.image
        # 1. empty: discard, keep state from the last valid frame (3.9)
        if _is_empty(image):
            return GateDecision(passed=False, reason="empty", fraction=0.0)

        small = downscale_gray(image, self.gate_width)
        offset = float(frame.offset_s)

        # 2. first valid frame of this video (3.4)
        if self._prev is None or self._last_passed_offset is None:
            if self._mog2 is not None:
                self._mog2.apply(small)  # seed the background model
            self._prev = small
            self._last_passed_offset = offset
            return GateDecision(passed=True, reason="first", fraction=0.0)

        # Fraction against the previous valid sampled frame (diff) or MOG2 model.
        shape_changed = self._prev.shape != small.shape
        if self.method == "mog2":
            fraction = self._mog2_fraction(small, shape_changed)
        else:
            fraction = changed_fraction(self._prev, small, self.pixel_delta)
        fraction = min(1.0, max(0.0, fraction))
        self._prev = small  # reference updated on every valid frame

        # 3. keyframe, measured in video time (3.3)
        if offset - self._last_passed_offset >= self.keyframe_interval_s - EPS:
            self._last_passed_offset = offset
            return GateDecision(passed=True, reason="keyframe", fraction=fraction)

        # 4. motion (3.2)
        if fraction >= self.threshold:
            self._last_passed_offset = offset
            return GateDecision(passed=True, reason="motion", fraction=fraction)

        # 5. static (3.7)
        return GateDecision(passed=False, reason="static", fraction=fraction)
