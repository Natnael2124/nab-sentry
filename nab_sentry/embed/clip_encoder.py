"""OpenCLIP ViT-B/32 image and text Embedder (Requirement 5).

Importing this module must stay light: ``torch``, ``open_clip`` and ``PIL`` are imported
lazily inside :class:`OpenClipEncoder` so that fast tests and ``import nab_sentry`` never
pull in the model stack.

Offline loading: open_clip is given the *absolute local checkpoint path* as
``pretrained`` (never a hub tag), and the path is checked before open_clip is called,
because a non-existent ``pretrained`` string would be treated as a tag and may trigger
a download (open_clip#968). ``HF_HUB_OFFLINE=1`` (set by ``nab_sentry.startup``) is a
second barrier.

Test seam: :meth:`OpenClipEncoder.from_components` builds an encoder around an injected
``model`` (with ``encode_image``/``encode_text``), ``preprocess`` and ``tokenizer``,
skipping weight loading. ``model.encode_*`` may return torch tensors or NumPy arrays.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol, Sequence, TypeVar

import numpy as np

from nab_sentry.errors import ModelMissingError

__all__ = [
    "PROMPT_TEMPLATES",
    "EMBED_DIM",
    "MODEL_NAME",
    "MIN_BATCH_SIZE",
    "MAX_BATCH_SIZE",
    "Encoder",
    "EmbeddingError",
    "EmptyQueryError",
    "ModelMissingError",
    "batched",
    "l2_normalize",
    "OpenClipEncoder",
]

PROMPT_TEMPLATES: tuple[str, str] = ("{q}", "a CCTV photo of {q}")
EMBED_DIM = 512
MODEL_NAME = "ViT-B-32"
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 64
_NORM_EPS = 1e-12

T = TypeVar("T")


class EmbeddingError(RuntimeError):
    """A vector was zero-norm, non-finite or of the wrong shape (Requirement 5.3)."""


class EmptyQueryError(ValueError):
    """The query is empty or whitespace-only after trimming (Requirement 5.9)."""

    def __init__(self, message: str = "empty query") -> None:
        super().__init__(message)


class Encoder(Protocol):
    dim: int
    batch_size: int

    def encode_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """(n, 512) float32, unit rows."""
        ...

    def encode_text(self, query: str) -> np.ndarray:
        """(512,) float32, unit."""
        ...


def batched(items: Sequence[T], n: int) -> Iterator[Sequence[T]]:
    """Yield consecutive slices of ``items`` of length ``n``; only the last may be shorter (5.5)."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"batch size must be a positive integer, got {n!r}")
    for start in range(0, len(items), n):
        yield items[start:start + n]


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """L2-normalise along ``axis`` as float32; raise EmbeddingError on zero or non-finite input."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        raise EmbeddingError("cannot normalise an empty vector")
    if not np.all(np.isfinite(arr)):
        raise EmbeddingError("embedding contains non-finite values")
    norm = np.linalg.norm(arr, axis=axis, keepdims=True)
    if not np.all(np.isfinite(norm)) or np.any(norm <= _NORM_EPS):
        raise EmbeddingError("embedding has zero or non-finite norm")
    out = (arr / norm).astype(np.float32)
    if not np.all(np.isfinite(out)):
        raise EmbeddingError("normalised embedding contains non-finite values")
    return out


def _check_batch_size(batch_size: Any) -> int:
    if (isinstance(batch_size, bool) or not isinstance(batch_size, int)
            or not MIN_BATCH_SIZE <= batch_size <= MAX_BATCH_SIZE):
        raise ValueError(
            f"batch_size must be an integer from {MIN_BATCH_SIZE} to {MAX_BATCH_SIZE}, "
            f"got {batch_size!r}")
    return batch_size


def _to_numpy(t: Any) -> np.ndarray:
    """Torch tensor or array-like -> float32 ndarray."""
    if hasattr(t, "detach"):
        t = t.detach().cpu().float().numpy()
    return np.asarray(t, dtype=np.float32)


def _check_weights_readable(path: Path) -> None:
    if not path.is_file():
        raise ModelMissingError(path, "missing")
    try:
        with open(path, "rb") as fh:
            fh.read(1)
    except OSError as exc:
        raise ModelMissingError(path, "unreadable") from exc


class OpenClipEncoder:
    """OpenCLIP ViT-B/32 (``laion2b_s34b_b79k``) loaded from a local checkpoint only."""

    dim: int = EMBED_DIM

    def __init__(self, weights_path: Path | str, batch_size: int = 8, threads: int = 0) -> None:
        batch_size = _check_batch_size(batch_size)
        path = Path(weights_path).expanduser().resolve()
        _check_weights_readable(path)  # before open_clip sees the string (5.8)

        import torch  # lazy: keep module import light
        import open_clip

        if threads and threads > 0:
            torch.set_num_threads(int(threads))
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                MODEL_NAME, pretrained=str(path), device="cpu")
        except Exception as exc:  # corrupt / truncated / wrong checkpoint
            raise ModelMissingError(path, f"unreadable ({type(exc).__name__})") from exc
        tokenizer = open_clip.get_tokenizer(MODEL_NAME)  # bundled BPE, no download
        model.eval()
        self._setup(model, preprocess, tokenizer, batch_size, torch)
        self.weights_path = path

    @classmethod
    def from_components(
        cls,
        model: Any,
        preprocess: Callable[[Any], Any],
        tokenizer: Callable[[list[str]], Any],
        batch_size: int = 8,
        torch_module: Any | None = None,
    ) -> "OpenClipEncoder":
        """Build an encoder from injected parts (tests stub the towers / spy on the tokenizer).

        Without ``torch_module`` preprocessed images are stacked with NumPy and no
        inference-mode context is used.
        """
        self = cls.__new__(cls)
        self._setup(model, preprocess, tokenizer, _check_batch_size(batch_size), torch_module)
        self.weights_path = None
        return self

    def _setup(self, model: Any, preprocess: Callable[[Any], Any],
               tokenizer: Callable[[list[str]], Any], batch_size: int, torch_module: Any) -> None:
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self._torch = torch_module

    # ------------------------------------------------------------------ helpers

    def _inference(self) -> contextlib.AbstractContextManager[Any]:
        if self._torch is not None:
            return self._torch.inference_mode()
        return contextlib.nullcontext()

    def _stack(self, tensors: list[Any]) -> Any:
        if self._torch is not None:
            return self._torch.stack(tensors)
        return np.stack([np.asarray(t) for t in tensors])

    @staticmethod
    def _to_pil(image: np.ndarray) -> Any:
        from PIL import Image  # lazy

        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] != 3 or arr.shape[0] < 1 or arr.shape[1] < 1:
            raise ValueError(f"expected a non-empty BGR HxWx3 image, got shape {arr.shape}")
        if arr.dtype != np.uint8:
            raise ValueError(f"expected uint8 image, got {arr.dtype}")
        rgb = np.ascontiguousarray(arr[:, :, ::-1])  # BGR -> RGB
        return Image.fromarray(rgb)

    def _check_rows(self, vecs: np.ndarray, n: int, what: str) -> np.ndarray:
        if vecs.shape != (n, self.dim):
            raise EmbeddingError(f"{what} output shape {vecs.shape}, expected ({n}, {self.dim})")
        return vecs

    # ------------------------------------------------------------------ Encoder API

    def encode_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Encode BGR uint8 images in batches of ``batch_size``; returns (n, 512) unit rows."""
        out: list[np.ndarray] = []
        for chunk in batched(list(images), self.batch_size):
            batch = self._stack([self.preprocess(self._to_pil(img)) for img in chunk])
            with self._inference():
                feats = self.model.encode_image(batch)
            vecs = self._check_rows(_to_numpy(feats), len(chunk), "image encoder")
            out.append(l2_normalize(vecs, axis=1))
        if not out:
            return np.empty((0, self.dim), dtype=np.float32)
        return np.concatenate(out, axis=0).astype(np.float32, copy=False)

    def encode_text(self, query: str) -> np.ndarray:
        """Prompt-ensembled query embedding (5.4); blank queries raise EmptyQueryError (5.9)."""
        if not isinstance(query, str):
            raise TypeError(f"query must be a str, got {type(query).__name__}")
        q = query.strip()
        if not q:
            raise EmptyQueryError()
        texts = [t.format(q=q) for t in PROMPT_TEMPLATES]
        tokens = self.tokenizer(texts)  # open_clip tokenizer truncates to 77 tokens
        with self._inference():
            feats = self.model.encode_text(tokens)
        per_template = l2_normalize(
            self._check_rows(_to_numpy(feats), len(PROMPT_TEMPLATES), "text encoder"), axis=1)
        return l2_normalize(per_template.mean(axis=0))
