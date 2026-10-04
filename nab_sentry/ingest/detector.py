"""Object detection: ``Detector`` protocol, ``Detection``, YOLO11n ONNX detector.

Pure helpers (``clamp_box``, ``letterbox``, ``postprocess``) are testable without
the model. ``onnxruntime`` is imported lazily inside ``OnnxYoloDetector.__init__``
so importing this module never loads the runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import cv2
import numpy as np

from nab_sentry.config import ConfigError, ConfigIssue
from nab_sentry.errors import ModelMissingError

# COCO class index -> name for the Target_Classes kept by the Detector (Req 4.2).
TARGET_CLASSES: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass(frozen=True)
class Detection:
    """One kept detection in source-frame pixel coordinates."""

    cls: str
    conf: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in source-frame pixels


@runtime_checkable
class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


def clamp_box(
    x1: float, y1: float, x2: float, y2: float, w: int, h: int
) -> tuple[int, int, int, int] | None:
    """Convert a float box to an integer-pixel box inside a ``w`` x ``h`` frame.

    - ``x1``/``y1`` are floored and ``x2``/``y2`` ceiled, so the integer box covers
      the float box; then each coordinate is clamped to ``[0, w]`` / ``[0, h]``.
    - Swapped coordinates (``x1 > x2`` or ``y1 > y2``) are reordered first, so the
      box is treated as the same rectangle given by its two corners.
    - Returns ``None`` when any input is NaN/infinite, when the frame has no area
      (``w < 1`` or ``h < 1``), or when the clamped box is narrower or shorter than
      1 pixel (Req 4.3, 5.10).

    A non-``None`` result always satisfies ``0 <= x1 < x2 <= w`` and
    ``0 <= y1 < y2 <= h``.
    """
    coords = (x1, y1, x2, y2)
    try:
        if not all(math.isfinite(float(c)) for c in coords):
            return None
    except (TypeError, ValueError):
        return None
    w = int(w)
    h = int(h)
    if w < 1 or h < 1:
        return None

    lo_x, hi_x = sorted((float(x1), float(x2)))
    lo_y, hi_y = sorted((float(y1), float(y2)))

    ix1 = min(max(math.floor(lo_x), 0), w)
    iy1 = min(max(math.floor(lo_y), 0), h)
    ix2 = min(max(math.ceil(hi_x), 0), w)
    iy2 = min(max(math.ceil(hi_y), 0), h)

    if ix2 - ix1 < 1 or iy2 - iy1 < 1:
        return None
    return (ix1, iy1, ix2, iy2)


# ---------------------------------------------------------------------------
# Letterbox pre-processing and YOLO11n post-processing (pure)
# ---------------------------------------------------------------------------

LETTERBOX_PAD = 114
NUM_COCO_CLASSES = 80


class DetectorInferenceError(RuntimeError):
    """ONNX Runtime raised while running inference on one frame (Req 4.8)."""


@dataclass(frozen=True)
class LetterboxMeta:
    """Maps letterboxed model-input coordinates back to source-frame pixels.

    ``frame_xy = (input_xy - pad) / scale``.
    """

    scale: float
    pad_x: float
    pad_y: float


def _to_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.ndim == 3 and frame.shape[2] == 1:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def letterbox(frame: np.ndarray, size: int) -> tuple[np.ndarray, LetterboxMeta]:
    """BGR uint8 HxWx3 frame -> ``(1, 3, size, size)`` float32 RGB in [0, 1].

    The frame is resized keeping its aspect ratio so it fits in ``size`` x ``size``,
    centred, and padded with value 114. The returned ``LetterboxMeta`` undoes it.
    """
    if frame is None or frame.size == 0 or frame.ndim < 2:
        raise ValueError("letterbox: empty frame")
    size = int(size)
    if size < 1:
        raise ValueError(f"letterbox: size must be >= 1, got {size}")
    bgr = _to_bgr(frame)
    if bgr.dtype != np.uint8:
        bgr = np.clip(bgr, 0, 255).astype(np.uint8)
    h, w = bgr.shape[:2]
    scale = min(size / w, size / h)
    nw = min(size, max(1, round(w * scale)))
    nh = min(size, max(1, round(h * scale)))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), LETTERBOX_PAD, dtype=np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas[py : py + nh, px : px + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
    return tensor.astype(np.float32, copy=False), LetterboxMeta(float(scale), float(px), float(py))


def _nms_indices(boxes_xywh: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    """Deterministic greedy NMS over ``(x, y, w, h)`` boxes; returns kept indices.

    Boxes are visited highest score first (stable on ties, i.e. lower index first).
    A box is suppressed when its IoU with any already-kept box is ``> iou`` (it
    survives when IoU ``<= iou``), matching ``cv2.dnn.NMSBoxes``' suppression rule.

    Unlike ``cv2.dnn.NMSBoxes`` there is no score filter: OpenCV keeps only scores
    strictly greater than its threshold, which would drop detections whose score
    equals ``det_conf`` (Req 4.2 keeps scores at or above it). ``postprocess``
    applies the ``>=`` threshold itself before calling this.
    """
    n = len(scores)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    b = np.asarray(boxes_xywh, dtype=np.float64).reshape(n, 4)
    x1, y1, bw, bh = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    x2, y2 = x1 + bw, y1 + bh
    area = bw * bh
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    kept: list[int] = []
    for i in order:
        if kept:
            k = np.asarray(kept, dtype=np.int64)
            iw = np.minimum(x2[i], x2[k]) - np.maximum(x1[i], x1[k])
            ih = np.minimum(y2[i], y2[k]) - np.maximum(y1[i], y1[k])
            inter = np.where((iw > 0) & (ih > 0), iw * ih, 0.0)
            union = area[i] + area[k] - inter
            with np.errstate(divide="ignore", invalid="ignore"):
                ious = np.where(union > 0, inter / union, 0.0)
            if np.any(ious > iou):
                continue
        kept.append(int(i))
    return np.asarray(kept, dtype=np.int64)


def postprocess(
    raw: np.ndarray,
    meta: LetterboxMeta,
    frame_w: int,
    frame_h: int,
    conf: float,
    iou: float,
    max_det: int,
) -> list[Detection]:
    """Decode a YOLO11n ``(1, 84, N)`` output into kept ``Detection`` objects.

    Steps: transpose to ``(N, 84)``; best class per anchor; keep Target_Classes with
    score >= ``conf``; undo the letterbox; class-wise greedy NMS (``_nms_indices``,
    no extra score filter); ``clamp_box`` (drop boxes
    under 1 px); sort by confidence desc (ties by ``x1`` then ``y1``); take ``max_det``.
    """
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(f"postprocess: expected batch size 1, got shape {arr.shape}")
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[0] < 4 + NUM_COCO_CLASSES:
        raise ValueError(f"postprocess: unexpected output shape {np.shape(raw)}")
    preds = arr[: 4 + NUM_COCO_CLASSES].T  # (N, 84)
    if preds.shape[0] == 0 or max_det < 1:
        return []

    class_scores = preds[:, 4:]
    best_cls = np.argmax(np.nan_to_num(class_scores, nan=-np.inf), axis=1)
    best_score = class_scores[np.arange(len(preds)), best_cls]

    target_ids = np.array(sorted(TARGET_CLASSES), dtype=np.int64)
    cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    keep = (
        np.isin(best_cls, target_ids)
        & np.isfinite(best_score)
        & (best_score >= conf)
        & np.isfinite(preds[:, :4]).all(axis=1)
        & (bw > 0)
        & (bh > 0)
    )
    if not keep.any():
        return []

    scale = meta.scale if meta.scale > 0 else 1.0
    x1 = (cx[keep] - bw[keep] / 2 - meta.pad_x) / scale
    y1 = (cy[keep] - bh[keep] / 2 - meta.pad_y) / scale
    w = bw[keep] / scale
    h = bh[keep] / scale
    cls_k = best_cls[keep]
    score_k = best_score[keep].astype(np.float64)

    candidates: list[tuple[float, int, int, int, int, str]] = []
    xywh = np.stack([x1, y1, w, h], axis=1).astype(np.float64)
    for cid in np.unique(cls_k):
        sel = np.flatnonzero(cls_k == cid)
        for j in _nms_indices(xywh[sel], score_k[sel], iou):
            i = sel[j]
            box = clamp_box(x1[i], y1[i], x1[i] + w[i], y1[i] + h[i], frame_w, frame_h)
            if box is None:
                continue
            c = float(min(max(score_k[i], 0.0), 1.0))
            candidates.append((c, *box, TARGET_CLASSES[int(cid)]))

    candidates.sort(key=lambda t: (-t[0], t[1], t[2]))
    return [Detection(cls=name, conf=c, box=(bx1, by1, bx2, by2))
            for c, bx1, by1, bx2, by2, name in candidates[: int(max_det)]]


# ---------------------------------------------------------------------------
# OnnxYoloDetector
# ---------------------------------------------------------------------------


def _check_range(name: str, value: Any, lo: float, hi: float, integer: bool) -> ConfigIssue | None:
    allowed = f"{lo}..{hi}"
    if isinstance(value, bool):
        return ConfigIssue(name, value, allowed, "wrong type")
    if integer:
        if not isinstance(value, (int, np.integer)):
            return ConfigIssue(name, value, allowed, "not an integer")
    elif not isinstance(value, (int, float, np.integer, np.floating)):
        return ConfigIssue(name, value, allowed, "not a number")
    if not math.isfinite(float(value)) or not lo <= value <= hi:
        return ConfigIssue(name, value, allowed, "out of range")
    return None


class OnnxYoloDetector:
    """YOLO11n detector on ONNX Runtime (CPUExecutionProvider). Implements ``Detector``."""

    def __init__(
        self,
        model_path: Path | str,
        conf: float,
        iou: float,
        max_det: int,
        input_size: int = 640,
        threads: int = 0,
    ) -> None:
        issues = [
            issue
            for issue in (
                _check_range("det_conf", conf, 0.0, 1.0, integer=False),
                _check_range("det_iou", iou, 0.0, 1.0, integer=False),
                _check_range("det_max_per_frame", max_det, 1, 100, integer=True),
                _check_range("det_input_size", input_size, 32, 1920, integer=True),
                _check_range("num_threads", threads, 0, 256, integer=True),
            )
            if issue is not None
        ]
        if issues:
            raise ConfigError(issues)

        self.model_path = Path(model_path)
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.input_size = int(input_size)

        if not self.model_path.is_file():
            raise ModelMissingError(self.model_path, "missing")

        import onnxruntime as ort  # lazy: keep module import free of onnxruntime

        opts = ort.SessionOptions()
        if threads > 0:
            opts.intra_op_num_threads = int(threads)
        opts.inter_op_num_threads = 1
        try:
            self._session = ort.InferenceSession(
                str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # noqa: BLE001 - corrupt/unreadable weights
            raise ModelMissingError(self.model_path, f"unreadable ({exc})") from exc
        self._input_name = self._session.get_inputs()[0].name

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if frame is None or frame.size == 0 or frame.ndim < 2:
            return []
        frame_h, frame_w = frame.shape[:2]
        tensor, meta = letterbox(frame, self.input_size)
        try:
            outputs = self._session.run(None, {self._input_name: tensor})
        except Exception as exc:  # noqa: BLE001 - surfaced to the pipeline as a warning
            raise DetectorInferenceError(f"ONNX inference failed: {exc}") from exc
        return postprocess(outputs[0], meta, frame_w, frame_h,
                           self.conf, self.iou, self.max_det)
