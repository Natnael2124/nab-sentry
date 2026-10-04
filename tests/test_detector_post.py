"""Property 15: Detector postprocess keeps only top target-class detections.

**Validates: Requirements 4.1, 4.2, 4.4**

Two properties, both checking Property 15:

* ``test_postprocess_matches_reference`` compares ``postprocess`` on random raw
  ``(1, 84, N)`` tensors against an independent per-anchor reference pipeline with
  its own greedy NMS (not ``cv2.dnn.NMSBoxes``, whose score filter is a strict ``>``
  and so cannot express "at or above" the threshold), and checks the invariants:
  Target_Class names, confidence >= threshold, at most ``max_det`` results, sorted
  by confidence desc, boxes inside the frame.
* ``test_postprocess_without_overlap_keeps_top_anchors`` places boxes on disjoint grid
  cells so NMS suppresses nothing; the result must equal the top ``max_det``
  target-class anchors whose score is at or above the threshold.

Box coordinates, pads, and scales are dyadic (multiples of 1/4, powers of two) so
undoing the letterbox is exact in both float32 and float64 and the reference can be
compared for exact equality.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from nab_sentry.ingest.detector import TARGET_CLASSES, Detection, LetterboxMeta, postprocess

NUM_CLASSES = 80
TARGET_IDS = sorted(TARGET_CLASSES)
TARGET_NAMES = set(TARGET_CLASSES.values())

# ---------------------------------------------------------------------------- strategies

quarter = lambda lo, hi: st.integers(lo * 4, hi * 4).map(lambda v: v / 4)  # noqa: E731
f32_unit = st.floats(0.0, 1.0, width=32)
class_ids = st.one_of(st.sampled_from(TARGET_IDS), st.integers(0, NUM_CLASSES - 1))


@st.composite
def anchors(draw):
    """One anchor: (cx, cy, w, h, best_class, best_score)."""
    return (
        draw(quarter(-50, 700)),
        draw(quarter(-50, 700)),
        draw(quarter(0, 300)),  # 0-sized boxes must be dropped
        draw(quarter(0, 300)),
        draw(class_ids),
        draw(f32_unit),
    )


@st.composite
def cases(draw):
    anc = draw(st.lists(anchors(), min_size=0, max_size=60))
    n = max(len(anc), draw(st.integers(1, 80)))  # trailing all-zero anchors
    raw = np.zeros((1, 4 + NUM_CLASSES, n), dtype=np.float32)
    noise_seed = draw(st.integers(0, 2**32 - 1))
    noise_hi = draw(st.sampled_from([0.0, 0.2, 0.6]))
    if noise_hi > 0:  # background scores on other classes, may beat the "best" class
        raw[0, 4:, :] = np.random.default_rng(noise_seed).uniform(0, noise_hi, (NUM_CLASSES, n))
    for i, (cx, cy, w, h, cid, score) in enumerate(anc):
        raw[0, :4, i] = (cx, cy, w, h)
        raw[0, 4 + cid, i] = score
    scores_present = [a[5] for a in anc]
    if scores_present and draw(st.booleans()):
        conf = draw(st.sampled_from(scores_present))  # exercise score == threshold
    else:
        conf = draw(f32_unit)
    meta = LetterboxMeta(
        scale=draw(st.sampled_from([0.125, 0.25, 0.5, 1.0, 2.0, 4.0])),
        pad_x=draw(quarter(0, 320)),
        pad_y=draw(quarter(0, 320)),
    )
    frame_w = draw(st.integers(1, 2000))
    frame_h = draw(st.integers(1, 2000))
    iou = draw(st.floats(0.0, 1.0))
    max_det = draw(st.one_of(st.integers(1, 10), st.integers(1, 100)))
    return raw, meta, frame_w, frame_h, float(conf), iou, max_det


# ---------------------------------------------------------------------------- reference


def _ref_clamp(x1, y1, x2, y2, w, h):
    bx1 = min(max(math.floor(min(x1, x2)), 0), w)
    by1 = min(max(math.floor(min(y1, y2)), 0), h)
    bx2 = min(max(math.ceil(max(x1, x2)), 0), w)
    by2 = min(max(math.ceil(max(y1, y2)), 0), h)
    if bx2 - bx1 < 1 or by2 - by1 < 1:
        return None
    return (bx1, by1, bx2, by2)


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = min(ax + aw, bx + bw) - max(ax, bx)
    ih = min(ay + ah, by + bh) - max(ay, by)
    inter = iw * ih if iw > 0 and ih > 0 else 0.0
    return inter / (aw * ah + bw * bh - inter)


def _greedy_nms(boxes, scores, iou):
    """Standard greedy NMS: highest score first (stable on ties); a box survives
    when its IoU with every already-kept box is at most ``iou``. No score filter."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    kept: list[int] = []
    for i in order:
        if all(_iou(boxes[i], boxes[k]) <= iou for k in kept):
            kept.append(i)
    return kept


def reference(raw, meta, frame_w, frame_h, conf, iou):
    """Independent pipeline; returns ALL survivors sorted (not truncated to max_det)."""
    preds = raw[0].astype(np.float64)
    per_class: dict[int, list[tuple[float, list[float]]]] = {}
    for i in range(preds.shape[1]):
        cx, cy, bw, bh = (float(v) for v in preds[:4, i])
        scores = preds[4:, i]
        cid = int(np.argmax(scores))
        score = float(scores[cid])
        if cid not in TARGET_CLASSES or not score >= conf or bw <= 0 or bh <= 0:
            continue
        x1 = (cx - bw / 2 - meta.pad_x) / meta.scale
        y1 = (cy - bh / 2 - meta.pad_y) / meta.scale
        per_class.setdefault(cid, []).append((score, [x1, y1, bw / meta.scale, bh / meta.scale]))

    out = []
    for cid, items in per_class.items():
        for j in _greedy_nms([b for _, b in items], [s for s, _ in items], iou):
            s, (x, y, w, h) = items[j]
            box = _ref_clamp(x, y, x + w, y + h, frame_w, frame_h)
            if box is not None:
                out.append(Detection(TARGET_CLASSES[cid], s, box))
    out.sort(key=lambda d: (-d.conf, d.box[0], d.box[1]))
    return out


def _key(d: Detection):
    return (d.conf, d.box[0], d.box[1])


def _check_invariants(dets, frame_w, frame_h, conf, max_det):
    assert len(dets) <= max_det
    for d in dets:
        assert d.cls in TARGET_NAMES
        assert conf <= d.conf <= 1.0
        x1, y1, x2, y2 = d.box
        assert all(isinstance(v, int) for v in d.box)
        assert 0 <= x1 < x2 <= frame_w and 0 <= y1 < y2 <= frame_h
    confs = [d.conf for d in dets]
    assert confs == sorted(confs, reverse=True)


# ---------------------------------------------------------------------------- properties


@settings(max_examples=200)
@given(cases())
def test_postprocess_matches_reference(case):
    raw, meta, frame_w, frame_h, conf, iou, max_det = case
    dets = postprocess(raw, meta, frame_w, frame_h, conf, iou, max_det)
    _check_invariants(dets, frame_w, frame_h, conf, max_det)

    expected = reference(raw, meta, frame_w, frame_h, conf, iou)
    k = min(max_det, len(expected))
    assert len(dets) == k, f"expected {k} detections, got {dets!r} vs {expected!r}"
    # Same ordering keys as the top-k reference survivors (no higher-confidence
    # survivor was discarded in favour of a lower one).
    assert [_key(d) for d in dets] == [_key(d) for d in expected[:k]]
    # Same detections, allowing only the order of exact sort-key ties to differ.
    assert not Counter(dets) - Counter(expected)
    if k:
        boundary = _key(expected[k - 1])
        strictly_above = lambda ds: Counter(d for d in ds if _key(d) != boundary)  # noqa: E731
        assert strictly_above(dets) == strictly_above(expected[:k])


@st.composite
def grid_cases(draw):
    """Anchors on disjoint 20x20 cells of a 640x640 frame: NMS never suppresses."""
    cells = draw(st.lists(st.integers(0, 31 * 32 - 1), min_size=0, max_size=60, unique=True))
    raw = np.zeros((1, 4 + NUM_CLASSES, max(len(cells), 1)), dtype=np.float32)
    specs = []
    for i, cell in enumerate(cells):
        gx, gy = divmod(cell, 32)
        cid = draw(class_ids)
        score = draw(f32_unit)
        raw[0, :4, i] = (gx * 20 + 10, gy * 20 + 10, 16, 16)  # 2 px gap between boxes
        raw[0, 4 + cid, i] = score
        specs.append((gx * 20 + 2, gy * 20 + 2, cid, float(score)))
    scores = [s for *_, s in specs]
    conf = float(draw(st.sampled_from(scores)) if scores and draw(st.booleans()) else draw(f32_unit))
    iou = draw(st.floats(0.0, 1.0))
    max_det = draw(st.integers(1, 100))
    return raw, specs, conf, iou, max_det


@settings(max_examples=150)
@given(grid_cases())
def test_postprocess_without_overlap_keeps_top_anchors(case):
    raw, specs, conf, iou, max_det = case
    dets = postprocess(raw, LetterboxMeta(1.0, 0.0, 0.0), 640, 640, conf, iou, max_det)
    _check_invariants(dets, 640, 640, conf, max_det)

    # Zero-score classes only win argmax at index 0 ("person") when every score is 0.
    wanted = []
    for x1, y1, cid, score in specs:
        best = 0 if score == 0.0 else cid
        if best in TARGET_CLASSES and score >= conf:
            wanted.append(Detection(TARGET_CLASSES[best], score, (x1, y1, x1 + 16, y1 + 16)))
    wanted.sort(key=lambda d: (-d.conf, d.box[0], d.box[1]))
    assert dets == wanted[:max_det]
