"""Property 47: Synthetic generation is deterministic.

**Validates: Requirements 15.1, 15.4**
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from nab_sentry import synthetic
from nab_sentry.synthetic import (
    OBJECT_SIZE_PX,
    SyntheticObject,
    SyntheticSpec,
    active_objects,
    background,
    ground_truth_document,
    object_position,
    render_frame,
    validate_spec,
    write_synthetic,
)

_SHAPES = {"red square": ("square", "red"), "blue circle": ("circle", "blue")}


@st.composite
def synthetic_specs(draw: st.DrawFn) -> SyntheticSpec:
    """Small valid specs: 100-200 px even dimensions, at most ~30 frames."""
    fps = draw(st.sampled_from([1, 2, 5, 10]))
    frames = draw(st.integers(min_value=1, max_value=30))
    duration = frames / fps
    width = draw(st.integers(min_value=50, max_value=100)) * 2
    height = draw(st.integers(min_value=50, max_value=100)) * 2
    objects = []
    for label in draw(st.lists(st.sampled_from(sorted(_SHAPES)), max_size=2, unique=True)):
        start = draw(st.integers(min_value=0, max_value=frames - 1)) / fps
        end = draw(st.integers(min_value=int(round(start * fps)) + 1, max_value=frames)) / fps
        shape, colour = _SHAPES[label]
        objects.append(SyntheticObject(label, shape, colour, start, end))
    spec = SyntheticSpec(
        seed=draw(st.integers(min_value=0, max_value=2**32 - 1)),
        duration_s=duration,
        fps=fps,
        width=width,
        height=height,
        camera_id="CAM-SYN01",
        label="Synthetic Yard",
        start_time=datetime(2025, 1, 1, 8, 0, 0),
        objects=tuple(objects),
    )
    validate_spec(spec)  # generator only yields valid settings
    return spec


def _render_all(spec: SyntheticSpec) -> list[np.ndarray]:
    # Clear the background cache so each run recomputes from the seed.
    synthetic._background.cache_clear()
    return [render_frame(i, spec) for i in range(spec.frame_count)]


@settings(max_examples=50)
@given(spec=synthetic_specs())
def test_render_and_ground_truth_are_deterministic(spec: SyntheticSpec) -> None:
    first = _render_all(spec)
    gt_first = json.dumps(ground_truth_document(spec), sort_keys=True)
    second = _render_all(spec)
    gt_second = json.dumps(ground_truth_document(spec), sort_keys=True)

    assert len(first) == len(second) == spec.frame_count
    for a, b in zip(first, second):
        assert a.dtype == np.uint8 and a.shape == (spec.height, spec.width, 3)
        assert a.tobytes() == b.tobytes()
    assert gt_first == gt_second

    # Background outside every active object's bounding box is identical in every frame.
    synthetic._background.cache_clear()
    bg = background(spec)
    for i, frame in enumerate(first):
        outside = np.ones((spec.height, spec.width), dtype=bool)
        for obj in active_objects(i, spec):
            x, y = object_position(i, obj, spec)
            outside[y : y + OBJECT_SIZE_PX, x : x + OBJECT_SIZE_PX] = False
        assert np.array_equal(frame[outside], bg[outside])


@settings(max_examples=50)
@given(
    seeds=st.lists(st.integers(min_value=0, max_value=2**32 - 1), min_size=2, max_size=2, unique=True),
    width=st.integers(min_value=50, max_value=100).map(lambda v: v * 2),
    height=st.integers(min_value=50, max_value=100).map(lambda v: v * 2),
)
def test_different_seeds_give_different_backgrounds(
    seeds: list[int], width: int, height: int
) -> None:
    a = SyntheticSpec(seed=seeds[0], width=width, height=height, objects=())
    b = SyntheticSpec(seed=seeds[1], width=width, height=height, objects=())
    synthetic._background.cache_clear()
    assert not np.array_equal(background(a), background(b))


def _decode(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), f"cannot open {path}"
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def test_written_video_decodes_identically_across_runs(tmp_path: Path) -> None:
    spec = SyntheticSpec(
        seed=7,
        duration_s=1.0,
        fps=10,
        width=160,
        height=120,
        objects=(
            SyntheticObject("red square", "square", "red", 0.2, 0.5),
            SyntheticObject("blue circle", "circle", "blue", 0.6, 0.9),
        ),
    )
    outputs = []
    for run in ("a", "b"):
        synthetic._background.cache_clear()
        video, _sidecar, gt = write_synthetic(spec, tmp_path / run)
        outputs.append((_decode(video), gt.read_text(encoding="utf-8")))

    (frames_a, gt_a), (frames_b, gt_b) = outputs
    assert len(frames_a) == len(frames_b) == spec.frame_count
    for fa, fb in zip(frames_a, frames_b):
        assert fa.tobytes() == fb.tobytes()
    assert gt_a == gt_b
