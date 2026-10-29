from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "models" / "desktop"
VIDEO = ROOT / "data" / "videos" / "CAM-SEC01_20261007T230010.mp4"


def unit(v) -> np.ndarray:
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


class FakeClip:
    """Text -> a fixed unit vector per word set; images -> vectors from their mean colour."""

    model_id = "fake"

    def __init__(self, table: dict[str, np.ndarray] | None = None) -> None:
        self.table = table or {}

    def encode_text(self, q: str) -> np.ndarray:
        if q in self.table:
            return self.table[q]
        rng = np.random.default_rng(abs(hash(q)) % (2 ** 32))
        return unit(rng.normal(size=512))

    encode_one_text = encode_text

    def encode_images(self, images) -> np.ndarray:
        out = []
        for im in images:
            v = np.zeros(512, np.float32)
            v[:3] = np.asarray(im, np.float32).reshape(-1, 3).mean(axis=0) + 1
            v[3] = 1
            out.append(unit(v))
        return np.stack(out) if out else np.empty((0, 512), np.float32)


@pytest.fixture
def fake_clip() -> FakeClip:
    return FakeClip()


def need(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"missing {path}")
