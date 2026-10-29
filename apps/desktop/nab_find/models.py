"""Model runtime: OpenVINO, on the Intel GPU when it works, else the CPU (design: models.py).

The app ships OpenVINO IR files with fp16-stored weights (tools/convert_ir.py; half the download of
the ONNX files, computed in fp32 on the CPU). Measured on the development laptop, ms per image: CLIP
image tower 11.5 on the Iris Xe GPU, 27 on the CPU (ONNX Runtime fp32: 38; PyTorch in V2: ~127);
YOLO11n 384x640 7.8 GPU, 14 CPU. The GPU is used only after a self-test shows it gives the CPU's
result (cosine >= 0.999), so a broken driver falls back to the CPU.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from nab_find.clip_tokenizer import ClipTokenizer
from nab_find.embed import PROMPT_TEMPLATES, l2_normalize

log = logging.getLogger("nab_find.models")

CLIP_BATCH = 8
YOLO_NAME = "yolo11n_384x640"
YOLO_HW = (384, 640)


def _v2_path() -> None:
    root = Path(__file__).resolve().parents[3]
    if (root / "nab_sentry").is_dir() and str(root) not in sys.path:
        sys.path.insert(0, str(root))


_v2_path()
from nab_sentry.ingest.detector import Detection, letterbox, postprocess  # noqa: E402


def _hint(device: str) -> dict:
    # measured: on the CPU the LATENCY hint is ~2x faster than THROUGHPUT for batches of 8
    return {"PERFORMANCE_HINT": "THROUGHPUT" if device == "GPU" else "LATENCY"}


class _Net:
    """A compiled model with one infer request, serialised by a lock (search and ingest share it)."""

    def __init__(self, core, path: Path, device: str) -> None:
        self._req = core.compile_model(str(path), device, _hint(device)).create_infer_request()
        self._lock = threading.Lock()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with self._lock:
            out = self._req.infer([x])
            return np.array(next(iter(out.values())))


@dataclass
class Models:
    clip: "ClipModel"
    yolo: "Yolo"
    device: str  # "GPU" or "CPU"


class ClipModel:
    def __init__(self, model_dir: Path, device: str, core) -> None:
        meta = json.loads((model_dir / "clip.json").read_text(encoding="utf-8"))
        self.model_id: str = meta["embedder_id"]
        self.size = int(meta["image_size"])
        self.mean = np.asarray(meta["mean"], np.float32).reshape(3, 1, 1)
        self.std = np.asarray(meta["std"], np.float32).reshape(3, 1, 1)
        self.tokenizer = ClipTokenizer(model_dir / "bpe_simple_vocab_16e6.txt.gz", int(meta["context_length"]))
        self.device = device
        self._image = _Net(core, model_dir / "clip_image.xml", device)
        self._text = _Net(core, model_dir / "clip_text.xml", "CPU")  # one query at a time: CPU is plenty

    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        """open_clip's preprocessing: bicubic resize of the short side, centre crop, normalise."""
        img = Image.fromarray(np.ascontiguousarray(bgr[:, :, ::-1]))
        w, h = img.size
        s = self.size / min(w, h)
        nw, nh = max(self.size, round(w * s)), max(self.size, round(h * s))
        img = img.resize((nw, nh), Image.BICUBIC)
        left, top = int(round((nw - self.size) / 2.0)), int(round((nh - self.size) / 2.0))
        img = img.crop((left, top, left + self.size, top + self.size))
        arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return (arr - self.mean) / self.std

    def encode_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """(n, 512) unit rows; batches padded to the model's fixed batch of 8."""
        out = []
        for i in range(0, len(images), CLIP_BATCH):
            pix = np.stack([self.preprocess(im) for im in images[i:i + CLIP_BATCH]])
            n = len(pix)
            if n < CLIP_BATCH:
                pix = np.concatenate([pix, np.zeros((CLIP_BATCH - n, *pix.shape[1:]), np.float32)])
            out.append(self._image(pix)[:n])
        return l2_normalize(np.concatenate(out)) if out else np.empty((0, 512), np.float32)

    def encode_one_text(self, text: str) -> np.ndarray:
        return l2_normalize(self._text(self.tokenizer([text]))[0])

    def encode_text(self, query: str) -> np.ndarray:
        """Prompt-ensembled query, as the product's search: (512,) unit."""
        q = query.strip()
        if not q:
            raise ValueError("empty query")
        feats = self._text(self.tokenizer([t.format(q=q) for t in PROMPT_TEMPLATES]))
        return l2_normalize(l2_normalize(feats).mean(axis=0))


class Yolo:
    """YOLO11n at 384x640 with V2's letterbox and post-processing."""

    def __init__(self, model_dir: Path, device: str, core, conf: float = 0.35, low_conf: float = 0.1,
                 iou: float = 0.5, max_det: int = 50) -> None:
        self.conf, self.low_conf, self.iou, self.max_det = conf, low_conf, iou, max_det
        self.device = device
        self._net = _Net(core, model_dir / f"{YOLO_NAME}.xml", device)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        tensor, meta = letterbox(frame, YOLO_HW)
        return postprocess(self._net(tensor), meta, w, h, self.conf, self.iou, self.max_det, self.low_conf)


def _core(cache_dir: Path | None):
    import openvino as ov

    core = ov.Core()
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        core.set_property({"CACHE_DIR": str(cache_dir)})
    return core


def _gpu_ok(model_dir: Path, core, cpu: ClipModel) -> bool:
    if "GPU" not in core.available_devices:
        return False
    try:
        gpu = ClipModel(model_dir, "GPU", core)
        img = np.random.default_rng(0).integers(0, 255, (300, 400, 3), dtype=np.uint8)
        cos = float(gpu.encode_images([img])[0] @ cpu.encode_images([img])[0])
        log.info("GPU self-test cosine %.5f", cos)
        return cos >= 0.999
    except Exception:  # noqa: BLE001 - any driver problem means CPU
        log.warning("GPU unusable; using the CPU", exc_info=True)
        return False


def load_models(model_dir: Path, cache_dir: Path | None = None, prefer: str = "auto") -> Models:
    """``prefer``: "auto" (GPU if it passes the self-test), "GPU" or "CPU"."""
    core = _core(cache_dir)
    cpu = ClipModel(model_dir, "CPU", core)
    device = "GPU" if prefer in ("auto", "GPU") and _gpu_ok(model_dir, core, cpu) else "CPU"
    log.info("models on %s", device)
    clip = cpu if device == "CPU" else ClipModel(model_dir, "GPU", core)
    return Models(clip, Yolo(model_dir, device, core), device)
