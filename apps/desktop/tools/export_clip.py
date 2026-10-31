"""Export the product's OpenCLIP embedder to ONNX for the desktop app (no PyTorch at runtime).

Usage (in the development venv, which has torch and open_clip)::

    venv\\Scripts\\python.exe apps\\desktop\\tools\\export_clip.py [--out models\\desktop]

Writes ``clip_image.onnx`` (``pixels`` float32 (n, 3, 224, 224) -> ``embedding`` (n, 512)),
``clip_text.onnx`` (``tokens`` int64 (n, 77) -> ``embedding`` (n, 512)), the tokenizer's BPE
vocabulary ``bpe_simple_vocab_16e6.txt.gz`` and ``clip.json`` (embedder id, image size,
normalisation constants). Embeddings are *not* normalised in the graph; the runtime normalises.

Then checks the ONNX towers against PyTorch on real frames and text and prints the lowest cosine
similarity (must be >= 0.999).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

from nab_sentry.model_files import EMBEDDER_ARCH, EMBEDDER_ID, EMBEDDER_REL  # noqa: E402

OPSET = 17


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path, default=_ROOT / "models" / "desktop")
    args = p.parse_args(argv)
    import open_clip
    import torch
    import torchvision.transforms as T

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    model, _, preprocess = open_clip.create_model_and_transforms(
        EMBEDDER_ARCH, pretrained=str(_ROOT / "models" / EMBEDDER_REL), device="cpu")
    model.eval()
    norm = next(t for t in preprocess.transforms if isinstance(t, T.Normalize))
    size = next(t for t in preprocess.transforms if isinstance(t, T.CenterCrop)).size
    size = int(size[0] if isinstance(size, (tuple, list)) else size)

    class Image(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixels):
            return self.m.encode_image(pixels)

    class Text(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, tokens):
            return self.m.encode_text(tokens)

    tok = open_clip.get_tokenizer(EMBEDDER_ARCH)
    torch.backends.mha.set_fastpath_enabled(False)  # the fused attention kernel has no ONNX export
    with torch.no_grad():
        torch.onnx.export(Image(model), (torch.zeros(2, 3, size, size),), str(out / "clip_image.onnx"),
                          input_names=["pixels"], output_names=["embedding"], opset_version=OPSET,
                          dynamic_axes={"pixels": {0: "n"}, "embedding": {0: "n"}}, dynamo=False)
        torch.onnx.export(Text(model), (tok(["a", "b"]),), str(out / "clip_text.onnx"),
                          input_names=["tokens"], output_names=["embedding"], opset_version=OPSET,
                          dynamic_axes={"tokens": {0: "n"}, "embedding": {0: "n"}}, dynamo=False)
    vocab = Path(open_clip.__file__).parent / "bpe_simple_vocab_16e6.txt.gz"
    shutil.copyfile(vocab, out / vocab.name)
    (out / "clip.json").write_text(json.dumps({
        "embedder_id": EMBEDDER_ID, "image_size": size, "context_length": 77,
        "mean": [float(x) for x in norm.mean], "std": [float(x) for x in norm.std],
    }, indent=2), encoding="utf-8")

    # parity check against PyTorch
    sys.path.insert(0, str(_ROOT / "apps" / "desktop"))
    import cv2
    from nab_find.embed import OnnxClip

    clip = OnnxClip(out)
    cap = cv2.VideoCapture(str(_ROOT / "data" / "videos" / "CAM-SEC01_20261007T230010.mp4"))
    frames = []
    for i in range(0, 9000, 1100):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = cap.read()
        if ok:
            frames.append(img)
            frames.append(img[100:400, 300:500].copy())
    cap.release()
    from PIL import Image as PILImage
    with torch.inference_mode():
        ref = model.encode_image(torch.stack([preprocess(PILImage.fromarray(f[:, :, ::-1].copy())) for f in frames]))
    ref = ref.numpy() / np.linalg.norm(ref.numpy(), axis=1, keepdims=True)
    got = clip.encode_images(frames)
    texts = ["a person in a red jacket", "a white van", "a CCTV photo of a man running"]
    with torch.inference_mode():
        tref = model.encode_text(tok(texts)).numpy()
    tref /= np.linalg.norm(tref, axis=1, keepdims=True)
    tgot = np.stack([clip.encode_one_text(t) for t in texts])
    img_cos = float(np.min(np.sum(ref * got, axis=1)))
    txt_cos = float(np.min(np.sum(tref * tgot, axis=1)))
    print(f"exported to {out}; min cosine vs PyTorch: images {img_cos:.5f} ({len(frames)}), text {txt_cos:.5f}")
    return 0 if min(img_cos, txt_cos) >= 0.999 else 1


if __name__ == "__main__":
    raise SystemExit(main())
