"""The product's CLIP embedder on ONNX Runtime (no PyTorch): same vectors as
:class:`nab_sentry.embed.clip_encoder.OpenClipEncoder`, from ``apps/desktop/tools/export_clip.py``.

Images are BGR uint8 arrays; preprocessing matches open_clip's (bicubic resize of the short side,
centre crop, normalise). Text uses the same prompt ensemble as the product's search.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from nab_find.clip_tokenizer import ClipTokenizer

PROMPT_TEMPLATES = ("{q}", "a CCTV photo of {q}")  # = nab_sentry.embed.clip_encoder.PROMPT_TEMPLATES


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


class OnnxClip:
    dim = 512

    def __init__(self, model_dir: Path | str, threads: int = 0, batch_size: int = 8,
                 image_file: str = "clip_image.onnx") -> None:
        import onnxruntime as ort

        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "clip.json").read_text(encoding="utf-8"))
        self.model_id: str = meta["embedder_id"]
        self.size = int(meta["image_size"])
        self.mean = np.asarray(meta["mean"], np.float32).reshape(3, 1, 1)
        self.std = np.asarray(meta["std"], np.float32).reshape(3, 1, 1)
        self.batch_size = batch_size
        opts = ort.SessionOptions()
        if threads > 0:
            opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self._image = ort.InferenceSession(str(model_dir / image_file), opts, providers=["CPUExecutionProvider"])
        self._text_path = model_dir / "clip_text.onnx"
        self._text = None  # loaded on the first query: ingest never needs it
        self._opts = opts
        self.tokenizer = ClipTokenizer(model_dir / "bpe_simple_vocab_16e6.txt.gz", int(meta["context_length"]))

    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
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
        """(n, 512) float32 unit rows."""
        out = []
        for i in range(0, len(images), self.batch_size):
            batch = np.stack([self.preprocess(im) for im in images[i:i + self.batch_size]])
            out.append(self._image.run(None, {"pixels": batch})[0])
        if not out:
            return np.empty((0, self.dim), np.float32)
        return l2_normalize(np.concatenate(out))

    def _text_session(self):
        if self._text is None:
            import onnxruntime as ort

            self._text = ort.InferenceSession(str(self._text_path), self._opts, providers=["CPUExecutionProvider"])
        return self._text

    def encode_one_text(self, text: str) -> np.ndarray:
        """One text, no prompt templates: (512,) unit."""
        return l2_normalize(self._text_session().run(None, {"tokens": self.tokenizer([text])})[0][0])

    def encode_text(self, query: str) -> np.ndarray:
        """Prompt-ensembled query, as the product's search: (512,) unit."""
        q = query.strip()
        if not q:
            raise ValueError("empty query")
        feats = self._text_session().run(None, {"tokens": self.tokenizer([t.format(q=q) for t in PROMPT_TEMPLATES])})[0]
        return l2_normalize(l2_normalize(feats).mean(axis=0))
