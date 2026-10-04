"""Slow end-to-end synthetic tests (Requirements 15.4, 15.7).

- Determinism: the default Synthetic_Video generated twice with the same seed decodes to
  byte-identical frames (runs wherever the bundled ffmpeg is available).
- End-to-end: ingest the synthetic video with the real ``OpenClipEncoder`` (detector off),
  query "red square" and "blue circle", and check the top Event overlaps the matching
  ground-truth interval by at least 1 s. Skipped when the CLIP weights are not in ``models/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from nab_sentry.config import Config
from nab_sentry.synthetic import SyntheticSpec, write_synthetic

pytestmark = pytest.mark.slow

CLIP_WEIGHTS_REL = Path("open_clip/ViT-B-32-laion2b_s34b_b79k/open_clip_pytorch_model.bin")
MIN_OVERLAP_S = 1.0


def _decode_all(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), f"cannot open {path}"
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    return frames


def overlap_s(start: float, end: float, gt_start: float, gt_end: float) -> float:
    return max(0.0, min(end, gt_end) - max(start, gt_start))


def test_same_seed_generates_identical_decoded_frames(tmp_path: Path) -> None:
    spec = SyntheticSpec()
    video_a, _, gt_a = write_synthetic(spec, tmp_path / "a")
    video_b, _, gt_b = write_synthetic(spec, tmp_path / "b")

    assert gt_a.read_bytes() == gt_b.read_bytes()

    frames_a = _decode_all(video_a)
    frames_b = _decode_all(video_b)
    assert len(frames_a) == spec.frame_count
    assert len(frames_a) == len(frames_b)
    for i, (fa, fb) in enumerate(zip(frames_a, frames_b)):
        assert fa.shape == (spec.height, spec.width, 3), f"frame {i} shape {fa.shape}"
        assert fa.tobytes() == fb.tobytes(), f"decoded frame {i} differs between runs"


def test_end_to_end_queries_find_ground_truth(tmp_path: Path, network_guard) -> None:
    weights = Config().models_dir / CLIP_WEIGHTS_REL
    if not weights.is_file():
        pytest.skip(f"OpenCLIP weights not present: {weights}")

    from nab_sentry.embed.clip_encoder import OpenClipEncoder
    from nab_sentry.ingest.pipeline import IngestPipeline
    from nab_sentry.search.engine import SearchEngine, SearchFilter
    from nab_sentry.store.db import MetadataStore
    from nab_sentry.store.vector_index import VectorIndex

    spec = SyntheticSpec()
    videos_dir = tmp_path / "videos"
    _, _, gt_path = write_synthetic(spec, videos_dir)
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    intervals = {o["label"]: (o["start_s"], o["end_s"]) for o in gt["videos"][0]["objects"]}

    cfg = Config(root=tmp_path / "ws", enable_detector=False)
    encoder = OpenClipEncoder(weights, batch_size=cfg.batch_size, threads=cfg.num_threads)
    db = MetadataStore(cfg.db_path)
    try:
        index = VectorIndex()
        report = IngestPipeline(cfg, db, index, encoder, None).ingest_paths([videos_dir])
        assert report.ingested == 1, report
        assert report.vectors > 0

        engine = SearchEngine(cfg, db, index, encoder)
        for query in ("red square", "blue circle"):
            events = engine.search(query, SearchFilter(), 10)
            assert events, f"no Events for {query!r}"
            top = events[0]
            gt_start, gt_end = intervals[query]
            ov = overlap_s(top.start_s, top.end_s, gt_start, gt_end)
            assert ov >= MIN_OVERLAP_S, (
                f"{query!r}: top Event [{top.start_s}, {top.end_s}] overlaps ground truth "
                f"[{gt_start}, {gt_end}) by {ov:.2f} s"
            )
    finally:
        db.close()

    network_guard.assert_no_attempts()
