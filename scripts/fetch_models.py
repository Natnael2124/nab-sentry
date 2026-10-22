"""Model_Fetcher: download model weights into ``models/`` once, on a connected machine.

Usage::

    venv\\Scripts\\python.exe scripts\\fetch_models.py [--models-dir models] [--export-only]

Steps (Requirements 13.1, 13.2, 13.8):

1. Delete any existing ``models/manifest.json`` (and a stale ``manifest.json.tmp``).
2. Download ``open_clip_pytorch_model.bin`` (OpenCLIP ViT-B/32 ``laion2b_s34b_b79k``) with
   ``huggingface_hub.hf_hub_download`` and ``yolo11n.pt`` via Ultralytics, each with up to
   3 attempts and exponential backoff.
3. Export YOLO11n once per input shape in ``DETECTOR_SHAPES`` (``dynamic=False, nms=False``):
   ``models/yolo11n.onnx`` (640 x 640) and ``models/yolo11n_384x640.onnx`` (V2 task 9.0).
4. Smoke-load every runtime model file from its local path (network disabled); each detector
   export must have a static ``(1, 3, h, w)`` input and the matching ``(1, 84, anchors)`` output.
5. Write ``manifest.json.tmp`` and rename it to ``manifest.json``.

``--export-only`` skips step 2 and re-exports from the ``yolo11n.pt`` and OpenCLIP weights
already in ``models/``, so an offline machine can add new detector shapes.

Any failure prints a message naming the failing file and exits 1 with no manifest.
The downloader, exporter, smoke loader and sleep function are injectable so tests can
stub them (see :func:`main`).

``huggingface_hub`` and ``ultralytics`` are imported only here, lazily, inside the
default implementations. Importing ``nab_sentry`` forces ``HF_HUB_OFFLINE=1``; the
default downloader re-enables network access only for the duration of the download.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nab_sentry.model_files import DETECTOR_SHAPES, detector_rel  # noqa: E402  (stdlib-only)
from nab_sentry.startup import MANIFEST_NAME, sha256_file  # noqa: E402  (stdlib-only module)

DEFAULT_MODELS_DIR = _ROOT / "models"

CLIP_REPO_ID = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
CLIP_FILENAME = "open_clip_pytorch_model.bin"
CLIP_REL = f"open_clip/ViT-B-32-laion2b_s34b_b79k/{CLIP_FILENAME}"
YOLO_PT_REL = "yolo11n.pt"
YOLO_ONNX_REL = "yolo11n.onnx"  # the 640 x 640 export
YOLO_ONNX_RELS = tuple(detector_rel(h, w) for h, w in DETECTOR_SHAPES)

ROLE_EMBEDDER = "embedder"
ROLE_DETECTOR = "detector"

MAX_ATTEMPTS = 3
BACKOFF_BASE_S = 2.0  # waits 2 s, then 4 s between the 3 attempts

MANIFEST_VERSION = 1
EXIT_OK = 0
EXIT_FAIL = 1

# Callable signatures
Downloader = Callable[[str, Path], Path]  # (relative path under models/, models_dir) -> local path
Exporter = Callable[[Path, Path, tuple[int, int]], Path]  # (yolo11n.pt, target onnx, (h, w)) -> onnx
SmokeLoader = Callable[[str, Path], None]  # (role, absolute path) -> raises on failure

_OFFLINE_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")


class FetchError(RuntimeError):
    """A model file could not be provisioned. ``rel`` names the failing file."""

    def __init__(self, rel: str, reason: str) -> None:
        super().__init__(f"{rel}: {reason}")
        self.rel = rel
        self.reason = reason


# ------------------------------------------------------------------ network toggle


@contextlib.contextmanager
def _network_enabled() -> Iterator[None]:
    """Temporarily lift the offline barrier set by ``nab_sentry`` for the download step."""
    saved = {k: os.environ.get(k) for k in _OFFLINE_VARS}
    hf_constants = sys.modules.get("huggingface_hub.constants")
    saved_const = getattr(hf_constants, "HF_HUB_OFFLINE", None) if hf_constants else None
    try:
        for k in _OFFLINE_VARS:
            os.environ[k] = "0"
        if hf_constants is not None:
            hf_constants.HF_HUB_OFFLINE = False
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if hf_constants is not None and saved_const is not None:
            hf_constants.HF_HUB_OFFLINE = saved_const


# ------------------------------------------------------------------ default implementations


def default_downloader(rel: str, models_dir: Path) -> Path:
    """Download one model file (one attempt) to ``models_dir / rel`` and return its path."""
    target = Path(models_dir) / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    with _network_enabled():
        if rel == CLIP_REL:
            import huggingface_hub  # Model_Fetcher only
            import huggingface_hub.constants  # lazy submodule: must be imported explicitly

            # Constants may have been read with HF_HUB_OFFLINE=1 at first import.
            huggingface_hub.constants.HF_HUB_OFFLINE = False
            path = huggingface_hub.hf_hub_download(
                repo_id=CLIP_REPO_ID, filename=CLIP_FILENAME, local_dir=str(target.parent)
            )
            return Path(path)
        if rel == YOLO_PT_REL:
            from ultralytics.utils.downloads import attempt_download_asset  # Model_Fetcher only

            # retry=1: retries are handled by _with_retries so attempts are counted once.
            path = Path(attempt_download_asset(str(target), retry=1))
            if path.resolve() != target.resolve() and path.is_file():
                shutil.copyfile(path, target)  # found in Ultralytics weights_dir; keep a copy here
                path = target
            return path
    raise FetchError(rel, "no download source configured")


def default_exporter(pt_path: Path, onnx_path: Path, hw: tuple[int, int]) -> Path:
    """Export YOLO11n to ONNX with a fixed ``hw`` input and no NMS in the graph.

    Ultralytics writes the export next to the ``.pt`` file, so it runs on a copy in a scratch
    folder: two shapes exported in a row would otherwise overwrite each other's output.
    """
    from ultralytics import YOLO  # Model_Fetcher only

    work = Path(onnx_path).parent / f".export_{hw[0]}x{hw[1]}"
    work.mkdir(parents=True, exist_ok=True)
    try:
        local_pt = work / Path(pt_path).name
        shutil.copy2(pt_path, local_pt)
        imgsz = hw[0] if hw[0] == hw[1] else list(hw)
        with _network_enabled():  # Ultralytics may auto-install the `onnx` package
            out = YOLO(str(local_pt)).export(format="onnx", imgsz=imgsz, dynamic=False, nms=False)
        out_path = Path(out)
        Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
        os.replace(out_path, onnx_path)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return Path(onnx_path)


def anchors(h: int, w: int) -> int:
    """YOLO11 output anchors for an ``h`` x ``w`` input (strides 8, 16, 32)."""
    return sum((h // s) * (w // s) for s in (8, 16, 32))


def default_smoke_loader(role: str, path: Path) -> None:
    """Load a model file from its absolute local path, as the runtime would."""
    path = Path(path).resolve()
    if role == ROLE_EMBEDDER:
        import open_clip
        import torch

        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained=str(path))
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        model.eval()
        with torch.no_grad():
            vec = model.encode_text(tokenizer(["a red car"]))
        if tuple(vec.shape) != (1, 512):
            raise RuntimeError(f"unexpected text embedding shape {tuple(vec.shape)}")
    elif role == ROLE_DETECTOR:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        shape = list(inp.shape)
        if len(shape) != 4 or shape[:2] != [1, 3] or not all(isinstance(d, int) for d in shape[2:]):
            raise RuntimeError(f"unexpected ONNX input shape {inp.shape}")
        h, w = shape[2], shape[3]
        (out,) = sess.run(None, {inp.name: np.zeros((1, 3, h, w), dtype=np.float32)})
        if out.shape != (1, 84, anchors(h, w)):
            raise RuntimeError(f"unexpected ONNX output shape {out.shape} for a {h}x{w} input")
    else:
        raise ValueError(f"unknown role {role!r}")


# ------------------------------------------------------------------ steps


def _with_retries(
    rel: str,
    downloader: Downloader,
    models_dir: Path,
    sleep: Callable[[float], None],
    log: Callable[[str], None],
) -> Path:
    """Call ``downloader`` up to MAX_ATTEMPTS times with exponential backoff."""
    target = models_dir / rel
    last: BaseException | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            path = Path(downloader(rel, models_dir))
            if path.resolve() != target.resolve():
                raise RuntimeError(f"downloaded to unexpected location {path}")
            if not target.is_file() or target.stat().st_size == 0:
                raise RuntimeError("file missing or empty after download")
            return target
        except Exception as exc:  # noqa: BLE001 - any downloader failure counts as an attempt
            last = exc
            log(f"download attempt {attempt}/{MAX_ATTEMPTS} failed for {rel}: {exc}")
            if attempt < MAX_ATTEMPTS:
                sleep(BACKOFF_BASE_S * 2 ** (attempt - 1))
    raise FetchError(rel, f"download failed after {MAX_ATTEMPTS} attempts: {last}")


def _remove_manifest(models_dir: Path) -> None:
    for name in (MANIFEST_NAME, MANIFEST_NAME + ".tmp"):
        p = models_dir / name
        if p.exists():
            p.unlink()


def _write_manifest(models_dir: Path, entries: list[tuple[str, str]]) -> Path:
    files = [
        {"role": role, "path": rel, "sha256": sha256_file(models_dir / rel)} for role, rel in entries
    ]
    data = {
        "manifest_version": MANIFEST_VERSION,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": files,
    }
    tmp = models_dir / (MANIFEST_NAME + ".tmp")
    final = models_dir / MANIFEST_NAME
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, final)
    return final


def fetch(
    models_dir: Path,
    downloader: Downloader = default_downloader,
    exporter: Exporter = default_exporter,
    smoke_loader: SmokeLoader = default_smoke_loader,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
    *,
    export_only: bool = False,
) -> Path:
    """Provision all model files and write the manifest. Raises FetchError on failure."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    _remove_manifest(models_dir)

    for rel in (CLIP_REL, YOLO_PT_REL):
        if export_only:
            if not (models_dir / rel).is_file():
                raise FetchError(rel, "missing; run without --export-only on a connected machine")
            continue
        log(f"downloading {rel} ...")
        _with_retries(rel, downloader, models_dir, sleep, log)

    for hw, rel in zip(DETECTOR_SHAPES, YOLO_ONNX_RELS):
        log(f"exporting {rel} ...")
        onnx_target = models_dir / rel
        try:
            out = Path(exporter(models_dir / YOLO_PT_REL, onnx_target, hw))
            if out.resolve() != onnx_target.resolve() or not onnx_target.is_file():
                raise RuntimeError("export produced no file at the expected path")
        except Exception as exc:  # noqa: BLE001
            raise FetchError(rel, f"ONNX export failed: {exc}") from exc

    runtime_files = [(ROLE_EMBEDDER, CLIP_REL)] + [(ROLE_DETECTOR, rel) for rel in YOLO_ONNX_RELS]
    for role, rel in runtime_files:
        log(f"smoke-loading {rel} ...")
        try:
            smoke_loader(role, (models_dir / rel).resolve())
        except Exception as exc:  # noqa: BLE001
            raise FetchError(rel, f"smoke load failed: {exc}") from exc

    try:
        return _write_manifest(models_dir, runtime_files)
    except Exception as exc:  # noqa: BLE001
        _remove_manifest(models_dir)
        raise FetchError(MANIFEST_NAME, f"could not write manifest: {exc}") from exc


def main(
    argv: Sequence[str] | None = None,
    *,
    downloader: Downloader = default_downloader,
    exporter: Exporter = default_exporter,
    smoke_loader: SmokeLoader = default_smoke_loader,
    sleep: Callable[[float], None] = time.sleep,
    models_dir: Path | None = None,
) -> int:
    """CLI entry point. Returns 0 on success, 1 on any failure (no manifest left behind)."""
    parser = argparse.ArgumentParser(description="Download NAB Sentry model files into models/.")
    parser.add_argument("--models-dir", type=Path, default=None, help="target directory (default: models/)")
    parser.add_argument("--export-only", action="store_true",
                        help="no downloads: re-export ONNX from the weights already in models/")
    args = parser.parse_args(list(argv) if argv is not None else None)
    target = Path(models_dir or args.models_dir or DEFAULT_MODELS_DIR)

    try:
        manifest = fetch(target, downloader, exporter, smoke_loader, sleep, export_only=args.export_only)
    except FetchError as exc:
        with contextlib.suppress(OSError):
            _remove_manifest(target)
        print(f"Model_Fetcher FAILED: {exc.rel}: {exc.reason}", file=sys.stderr)
        return EXIT_FAIL
    except Exception as exc:  # noqa: BLE001 - e.g. cannot delete the old manifest
        with contextlib.suppress(OSError):
            _remove_manifest(target)
        print(f"Model_Fetcher FAILED: {MANIFEST_NAME}: {exc}", file=sys.stderr)
        return EXIT_FAIL
    print(f"Model_Fetcher OK: wrote {manifest}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
