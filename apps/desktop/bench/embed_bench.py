"""Model speed per runtime (desktop speed experiment, step 2).

Usage::

    venv-desktop\\Scripts\\python.exe apps\\desktop\\bench\\embed_bench.py [--models models]

Times the CLIP image tower (batch 8) and the YOLO11n detector (one 384x640 frame) on ONNX Runtime
(fp32, and int8 for CLIP) and on OpenVINO (CPU and the Intel GPU), and reports the cosine
similarity of each CLIP variant to ONNX Runtime fp32 on real frames (accuracy cost). Prints a
Markdown table of median milliseconds per image.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "apps" / "desktop"))

from nab_find.embed import OnnxClip, l2_normalize  # noqa: E402

VIDEO = _ROOT / "data" / "videos" / "CAM-SEC01_20261007T230010.mp4"
REPS = 6


def frames(n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(VIDEO))
    out = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 300 + i * 997)
        ok, img = cap.read()
        if ok:
            out.append(img)
    cap.release()
    return out


def timed(fn, reps: int = REPS) -> float:
    fn()  # warm-up
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return statistics.median(t)


def quantize(src: Path, dst: Path) -> None:
    if dst.is_file():
        return
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=Path, default=_ROOT / "models")
    args = p.parse_args(argv)
    import onnxruntime as ort
    import openvino as ov

    mdir = args.models / "desktop"
    imgs = frames(16)
    clip = OnnxClip(mdir)
    pix = np.stack([clip.preprocess(im) for im in imgs[:8]])
    pre = timed(lambda: [clip.preprocess(im) for im in imgs[:8]]) / 8 * 1000
    ref = l2_normalize(np.concatenate([clip._image.run(None, {"pixels": pix})[0]]))
    rows = [("CLIP preprocess (PIL)", f"{pre:.1f}", "")]

    quantize(mdir / "clip_image.onnx", mdir / "clip_image_int8.onnx")
    for name, path in (("fp32", "clip_image.onnx"), ("int8", "clip_image_int8.onnx")):
        for threads in (4, 8):
            o = ort.SessionOptions()
            o.intra_op_num_threads = threads
            s = ort.InferenceSession(str(mdir / path), o, providers=["CPUExecutionProvider"])
            ms = timed(lambda: s.run(None, {"pixels": pix})) / 8 * 1000
            cos = float(np.min(np.sum(l2_normalize(s.run(None, {"pixels": pix})[0]) * ref, axis=1)))
            rows.append((f"CLIP onnxruntime {name}, {threads} threads", f"{ms:.1f}", f"{cos:.4f}"))

    core = ov.Core()
    for dev in [d for d in ("CPU", "GPU") if d in core.available_devices]:
        for name, path in (("fp32", "clip_image.onnx"), ("int8", "clip_image_int8.onnx")):
            try:
                cm = core.compile_model(str(mdir / path), dev, {"PERFORMANCE_HINT": "THROUGHPUT"})
                req = cm.create_infer_request()
                ms = timed(lambda: req.infer({0: pix})) / 8 * 1000
                out = l2_normalize(next(iter(req.infer({0: pix}).values())))
                cos = float(np.min(np.sum(out * ref, axis=1)))
                rows.append((f"CLIP openvino {name}, {dev}", f"{ms:.1f}", f"{cos:.4f}"))
            except Exception as exc:  # noqa: BLE001 - report and go on
                rows.append((f"CLIP openvino {name}, {dev}", "failed", str(exc).splitlines()[0][:60]))

    sys.path.insert(0, str(_ROOT))
    from nab_sentry.ingest.detector import letterbox

    yolo = args.models / "yolo11n_384x640.onnx"
    tensor, _ = letterbox(imgs[0], (384, 640))
    for threads in (4, 8):
        o = ort.SessionOptions()
        o.intra_op_num_threads = threads
        s = ort.InferenceSession(str(yolo), o, providers=["CPUExecutionProvider"])
        name = s.get_inputs()[0].name
        rows.append((f"YOLO11n 384x640 onnxruntime, {threads} threads",
                     f"{timed(lambda: s.run(None, {name: tensor}), 20) * 1000:.1f}", ""))
    for dev in [d for d in ("CPU", "GPU") if d in core.available_devices]:
        try:
            cm = core.compile_model(str(yolo), dev, {"PERFORMANCE_HINT": "LATENCY"})
            req = cm.create_infer_request()
            rows.append((f"YOLO11n 384x640 openvino, {dev}",
                         f"{timed(lambda: req.infer({0: tensor}), 20) * 1000:.1f}", ""))
        except Exception as exc:  # noqa: BLE001
            rows.append((f"YOLO11n 384x640 openvino, {dev}", "failed", str(exc).splitlines()[0][:60]))

    print(f"OpenVINO devices: {core.available_devices}")
    print("| Model / runtime | ms per image (median) | min cosine to fp32 |")
    print("|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
