"""Model runtime: the Intel GPU through OpenVINO when it works, else ONNX Runtime on the CPU
(design: models.py).

Measured on the development laptop (bench/embed_bench.py), ms per image: CLIP image tower 13 on the
Iris Xe GPU vs 30-75 on the CPU; YOLO11n 384x640 7.8 vs 18-31. The GPU is used only after a
self-test shows it gives the CPU's result (cosine >= 0.999), so a broken driver falls back to the CPU.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from nab_find.embed import OnnxClip, l2_normalize

log = logging.getLogger("nab_find.models")

GPU_BATCH = 8
YOLO_FILE = "yolo11n_384x640.onnx"
YOLO_HW = (384, 640)


def _v2_path() -> None:
    root = Path(__file__).resolve().parents[3]
    if (root / "nab_sentry").is_dir() and str(root) not in sys.path:
        sys.path.insert(0, str(root))


_v2_path()
from nab_sentry.ingest.detector import Detection, letterbox, postprocess  # noqa: E402


@dataclass
class Models:
    clip: "ClipModel"
    yolo: "Yolo"
    device: str  # "GPU" or "CPU"


def _ov_core(cache_dir: Path | None):
    import openvino as ov

    core = ov.Core()
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        core.set_property({"CACHE_DIR": str(cache_dir)})
    return core


class ClipModel:
    """CLIP image and text towers. Images: OpenVINO GPU (fixed batch of 8, padded) or ONNX Runtime."""

    def __init__(self, model_dir: Path, device: str = "CPU", threads: int = 0, core=None) -> None:
        self.cpu = OnnxClip(model_dir, threads=threads)
        self.model_id = self.cpu.model_id
        self.device = device
        self._gpu = None
        if device == "GPU":
            m = core.read_model(str(model_dir / "clip_image.onnx"))
            m.reshape({m.inputs[0].any_name: [GPU_BATCH, 3, self.cpu.size, self.cpu.size]})
            self._gpu = core.compile_model(m, "GPU", {"PERFORMANCE_HINT": "THROUGHPUT"}).create_infer_request()

    def _run_gpu(self, pix: np.ndarray) -> np.ndarray:
        n = len(pix)
        if n < GPU_BATCH:
            pix = np.concatenate([pix, np.zeros((GPU_BATCH - n, *pix.shape[1:]), np.float32)])
        out = self._gpu.infer({0: pix})
        return np.asarray(next(iter(out.values())))[:n]

    def encode_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        if self._gpu is None:
            return self.cpu.encode_images(images)
        out = []
        for i in range(0, len(images), GPU_BATCH):
            pix = np.stack([self.cpu.preprocess(im) for im in images[i:i + GPU_BATCH]])
            out.append(self._run_gpu(pix))
        return l2_normalize(np.concatenate(out)) if out else np.empty((0, 512), np.float32)

    def encode_text(self, query: str) -> np.ndarray:
        return self.cpu.encode_text(query)

    def encode_one_text(self, text: str) -> np.ndarray:
        return self.cpu.encode_one_text(text)


class Yolo:
    """YOLO11n at 384x640 with V2's letterbox and post-processing."""

    def __init__(self, model_dir: Path, device: str = "CPU", threads: int = 0, core=None,
                 conf: float = 0.35, low_conf: float = 0.1, iou: float = 0.5, max_det: int = 50) -> None:
        path = model_dir / YOLO_FILE
        self.conf, self.low_conf, self.iou, self.max_det = conf, low_conf, iou, max_det
        self.device = device
        if device == "GPU":
            self._req = core.compile_model(str(path), "GPU", {"PERFORMANCE_HINT": "LATENCY"}).create_infer_request()
            self._run = lambda x: np.asarray(next(iter(self._req.infer({0: x}).values())))
        else:
            import onnxruntime as ort

            o = ort.SessionOptions()
            if threads > 0:
                o.intra_op_num_threads = threads
            s = ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])
            name = s.get_inputs()[0].name
            self._run = lambda x: s.run(None, {name: x})[0]

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        tensor, meta = letterbox(frame, YOLO_HW)
        return postprocess(self._run(tensor), meta, w, h, self.conf, self.iou, self.max_det, self.low_conf)


def _gpu_ok(model_dir: Path, core) -> bool:
    if "GPU" not in core.available_devices:
        return False
    try:
        test = ClipModel(model_dir, "GPU", core=core)
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)
        a, b = test.encode_images([img])[0], test.cpu.encode_images([img])[0]
        cos = float(a @ b)
        log.info("GPU self-test cosine %.5f", cos)
        return cos >= 0.999
    except Exception:  # noqa: BLE001 - any driver problem means CPU
        log.warning("GPU unusable; using the CPU", exc_info=True)
        return False


def load_models(model_dir: Path, cache_dir: Path | None = None, prefer: str = "auto", threads: int = 0) -> Models:
    """``prefer``: "auto" (GPU if it passes the self-test), "GPU" or "CPU"."""
    core = None
    device = "CPU"
    if prefer in ("auto", "GPU"):
        try:
            core = _ov_core(cache_dir)
            if _gpu_ok(model_dir, core):
                device = "GPU"
        except Exception:  # noqa: BLE001 - OpenVINO missing or broken
            log.warning("OpenVINO unavailable; using the CPU", exc_info=True)
    log.info("models on %s", device)
    return Models(ClipModel(model_dir, device, threads, core), Yolo(model_dir, device, threads, core), device)
