"""Property 18: Batching shape.

**Validates: Requirements 5.5**
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from nab_sentry.embed.clip_encoder import (
    EMBED_DIM,
    MAX_BATCH_SIZE,
    MIN_BATCH_SIZE,
    OpenClipEncoder,
    batched,
)

batch_sizes = st.integers(min_value=MIN_BATCH_SIZE, max_value=MAX_BATCH_SIZE)


def _expected_sizes(length: int, n: int) -> list[int]:
    full, rem = divmod(length, n)
    return [n] * full + ([rem] if rem else [])


@settings(max_examples=200)
@given(items=st.lists(st.integers(), max_size=300), n=batch_sizes)
def test_batched_shape(items: list[int], n: int) -> None:
    """Feature: nab-sentry, Property 18: Batching shape."""
    batches = list(batched(items, n))

    assert len(batches) == math.ceil(len(items) / n)
    assert [x for b in batches for x in b] == items
    for b in batches[:-1]:
        assert len(b) == n
    if batches:
        assert 1 <= len(batches[-1]) <= n
    assert [len(b) for b in batches] == _expected_sizes(len(items), n)


class _RecordingModel:
    """Stub image tower: records batch sizes, returns a distinct non-zero row per image."""

    def __init__(self) -> None:
        self.calls: list[int] = []

    def encode_image(self, batch: np.ndarray) -> np.ndarray:
        k = len(batch)
        self.calls.append(k)
        out = np.zeros((k, EMBED_DIM), dtype=np.float32)
        out[:, 0] = 1.0
        out[:, 1] = np.asarray(batch, dtype=np.float32).reshape(k, -1)[:, 0]
        return out


def _preprocess(pil_image) -> np.ndarray:
    # Tag each image by its red value so ordering through the batches can be checked.
    return np.asarray([np.asarray(pil_image)[0, 0, 0]], dtype=np.float32)


@settings(max_examples=50, deadline=None)
@given(length=st.integers(min_value=0, max_value=150), n=batch_sizes)
def test_encode_images_uses_batches(length: int, n: int) -> None:
    """Feature: nab-sentry, Property 18: Batching shape (encode_images)."""
    model = _RecordingModel()
    enc = OpenClipEncoder.from_components(model, _preprocess, lambda texts: texts, batch_size=n)
    # BGR images; red channel (index 2) carries the image index mod 256.
    images = []
    for i in range(length):
        img = np.zeros((2, 2, 3), dtype=np.uint8)
        img[:, :, 2] = i % 256
        images.append(img)

    vecs = enc.encode_images(images)

    assert model.calls == _expected_sizes(length, n)
    assert vecs.shape == (length, EMBED_DIM)
    assert vecs.dtype == np.float32
    # Order preserved: row i corresponds to image i.
    tags = vecs[:, 1] / vecs[:, 0] if length else np.empty(0)
    np.testing.assert_allclose(tags, [i % 256 for i in range(length)], rtol=1e-5)


def test_batched_examples() -> None:
    assert list(batched([], 3)) == []
    assert list(batched([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert list(batched([1, 2, 3, 4], 4)) == [[1, 2, 3, 4]]
