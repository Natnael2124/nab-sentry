"""Model file locations (manifest-relative) and default model factories.

Shared by the API_Server startup (``nab_sentry.api.app.serve``). Paths are relative to
``models/`` with forward slashes, as listed in ``models/manifest.json``. Model libraries
(torch/open_clip, onnxruntime) are imported lazily inside the factories.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from nab_sentry.config import Config

EMBEDDER_REL = "open_clip/ViT-B-32-laion2b_s34b_b79k/open_clip_pytorch_model.bin"
DETECTOR_REL = "yolo11n.onnx"  # square 640 x 640 input (V1)
# ONNX input shapes (height, width) exported by scripts/fetch_models.py. ONNX exports are
# static, so each shape is its own file (V2 task 9.0; 384 x 640 suits 16:9 cameras).
DETECTOR_SHAPES: tuple[tuple[int, int], ...] = ((640, 640), (384, 640))

# (cfg, manifest files rel->abs) -> Encoder / Detector
EncoderFactory = Callable[["Config", dict[str, Path]], Any]
DetectorFactory = Callable[["Config", dict[str, Path]], Any]


def embedder_path(cfg: Config, files: dict[str, Path]) -> Path:
    return Path(files.get(EMBEDDER_REL, cfg.models_dir / EMBEDDER_REL))


def detector_rel(height: int, width: int) -> str:
    """Manifest path of the YOLO11n ONNX export for a ``height`` x ``width`` input."""
    if (height, width) == (640, 640):
        return DETECTOR_REL
    return f"yolo11n_{height}x{width}.onnx"


def detector_input(cfg: Config) -> tuple[int, int]:
    """Configured detector input ``(height, width)``."""
    return int(cfg.det_input_h), int(cfg.det_input_size)


def detector_path(cfg: Config, files: dict[str, Path]) -> Path:
    rel = detector_rel(*detector_input(cfg))
    return Path(files.get(rel, cfg.models_dir / rel))


def default_encoder_factory(cfg: Config, files: dict[str, Path]) -> Any:
    from nab_sentry.embed.clip_encoder import OpenClipEncoder

    return OpenClipEncoder(embedder_path(cfg, files), batch_size=cfg.batch_size,
                           threads=cfg.num_threads)


def default_detector_factory(cfg: Config, files: dict[str, Path]) -> Any:
    from nab_sentry.ingest.detector import OnnxYoloDetector

    return OnnxYoloDetector(
        detector_path(cfg, files),
        conf=cfg.det_conf,
        iou=cfg.det_iou,
        max_det=cfg.det_max_per_frame,
        input_size=detector_input(cfg),
        threads=cfg.det_threads,
    )
