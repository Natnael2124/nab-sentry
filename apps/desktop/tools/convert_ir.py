"""Convert the desktop ONNX models to OpenVINO IR with fp16-stored weights (what the app ships).

Usage (after tools/export_clip.py and with models/desktop/yolo11n_384x640.onnx in place)::

    venv-desktop\\Scripts\\python.exe apps\\desktop\\tools\\convert_ir.py [--dir models\\desktop]

Writes ``clip_image.xml/.bin`` (input fixed to a batch of 8), ``clip_text.xml/.bin`` and
``yolo11n_384x640.xml/.bin``. Weights are stored as fp16 (half the download) and computed in fp32 on
the CPU; checks every converted model against ONNX Runtime fp32 on the same input (cosine >= 0.9999
for CLIP, max abs difference for YOLO) and prints the sizes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "apps" / "desktop"))

from nab_find.models import CLIP_BATCH  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dir", type=Path, default=_ROOT / "models" / "desktop")
    args = p.parse_args(argv)
    import onnxruntime as ort
    import openvino as ov

    d = args.dir
    core = ov.Core()
    img = core.read_model(str(d / "clip_image.onnx"))
    img.reshape({img.inputs[0].any_name: [CLIP_BATCH, 3, 224, 224]})
    ov.save_model(img, str(d / "clip_image.xml"), compress_to_fp16=True)
    ov.save_model(core.read_model(str(d / "clip_text.onnx")), str(d / "clip_text.xml"), compress_to_fp16=True)
    ov.save_model(core.read_model(str(d / "yolo11n_384x640.onnx")), str(d / "yolo11n_384x640.xml"),
                  compress_to_fp16=True)

    rng = np.random.default_rng(0)
    worst = 1.0
    checks = {
        "clip_image": {"pixels": rng.normal(size=(CLIP_BATCH, 3, 224, 224)).astype(np.float32)},
        "clip_text": {"tokens": np.array([[49406, 320, 786, 530, 320, 736, 49407] + [0] * 70,
                                          [49406, 1579, 2451, 49407] + [0] * 73], np.int64)},
    }
    for name, feed in checks.items():
        ref = ort.InferenceSession(str(d / f"{name}.onnx"), providers=["CPUExecutionProvider"]).run(None, feed)[0]
        got = core.compile_model(str(d / f"{name}.xml"), "CPU").create_infer_request().infer(list(feed.values()))
        got = np.asarray(next(iter(got.values())))
        cos = np.sum(ref * got, 1) / np.linalg.norm(ref, axis=1) / np.linalg.norm(got, axis=1)
        worst = min(worst, float(cos.min()))
        print(f"{name}: min cosine to ONNX fp32 {cos.min():.6f}")
    x = rng.random((1, 3, 384, 640), dtype=np.float32)
    s = ort.InferenceSession(str(d / "yolo11n_384x640.onnx"), providers=["CPUExecutionProvider"])
    ref = s.run(None, {s.get_inputs()[0].name: x})[0]
    got = np.asarray(next(iter(core.compile_model(str(d / "yolo11n_384x640.xml"), "CPU")
                               .create_infer_request().infer([x]).values())))
    print(f"yolo: max abs difference {np.abs(ref - got).max():.4f} (boxes in pixels, scores 0-1)")
    for f in sorted(d.glob("*.bin")):
        print(f"{f.name}: {f.stat().st_size / 1e6:.0f} MB")
    return 0 if worst >= 0.9999 else 1


if __name__ == "__main__":
    raise SystemExit(main())
