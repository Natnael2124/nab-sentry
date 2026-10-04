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
DETECTOR_REL = "yolo11n.onnx"

# (cfg, manifest files rel->abs) -> Encoder / Detector
EncoderFactory = Callable[["Config", dict[str, Path]], Any]
DetectorFactory = Callable[["Config", dict[str, Path]], Any]


def embedder_path(cfg: Config, files: dict[str, Path]) -> Path:
    return Path(files.get(EMBEDDER_REL, cfg.models_dir / EMBEDDER_REL))


def detector_path(cfg: Config, files: dict[str, Path]) -> Path:
    return Path(files.get(DETECTOR_REL, cfg.models_dir / DETECTOR_REL))


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
        input_size=cfg.det_input_size,
        threads=cfg.num_threads,
    )
