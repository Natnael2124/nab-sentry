"""Fast unit tests for nab_sentry.embed.clip_encoder and FakeEncoder (no model weights)."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from nab_sentry.embed.clip_encoder import (
    EMBED_DIM,
    PROMPT_TEMPLATES,
    EmbeddingError,
    EmptyQueryError,
    ModelMissingError,
    OpenClipEncoder,
    batched,
    l2_normalize,
)
from nab_sentry.errors import ModelMissingError as SharedModelMissingError
from nab_sentry.startup import FETCH_HINT
from tests.fakes import BLUE_DIRECTION, RED_DIRECTION, FakeEncoder


def test_import_does_not_pull_torch():
    code = ("import sys, nab_sentry.embed.clip_encoder; "
            "sys.exit(1 if ('torch' in sys.modules or 'open_clip' in sys.modules) else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_missing_weights_raise_model_missing_with_fetch_hint(tmp_path):
    missing = tmp_path / "nope" / "open_clip_pytorch_model.bin"
    with pytest.raises(ModelMissingError) as exc:
        OpenClipEncoder(missing)
    assert ModelMissingError is SharedModelMissingError
    assert FETCH_HINT in str(exc.value)
    assert str(missing.resolve()) in str(exc.value)


def test_directory_instead_of_file_is_missing(tmp_path):
    with pytest.raises(ModelMissingError):
        OpenClipEncoder(tmp_path)


def test_invalid_batch_size_rejected(tmp_path):
    with pytest.raises(ValueError):
        OpenClipEncoder(tmp_path / "x.bin", batch_size=0)
    with pytest.raises(ValueError):
        OpenClipEncoder(tmp_path / "x.bin", batch_size=65)


def test_l2_normalize_unit_rows_and_errors():
    out = l2_normalize(np.array([[3.0, 4.0], [0.0, 2.0]]), axis=1)
    assert out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)
    for bad in (np.zeros(4), np.array([1.0, np.nan]), np.array([np.inf, 1.0]), np.array([])):
        with pytest.raises(EmbeddingError):
            l2_normalize(bad)


def test_batched_sizes():
    assert [len(b) for b in batched(list(range(10)), 4)] == [4, 4, 2]
    assert list(batched([], 3)) == []
    with pytest.raises(ValueError):
        list(batched([1], 0))


class _StubModel:
    def __init__(self):
        self.image_batches: list[int] = []

    def encode_image(self, batch):
        self.image_batches.append(len(batch))
        return np.ones((len(batch), EMBED_DIM)) * np.arange(1, len(batch) + 1)[:, None]

    def encode_text(self, tokens):
        rng = np.random.default_rng(0)
        return rng.standard_normal((len(tokens), EMBED_DIM))


def test_injected_components_batches_and_ensembles():
    seen: list[list[str]] = []

    def tokenizer(texts):
        seen.append(list(texts))
        return texts

    model = _StubModel()
    enc = OpenClipEncoder.from_components(
        model, preprocess=lambda pil: np.asarray(pil, dtype=np.float32), tokenizer=tokenizer,
        batch_size=3)
    imgs = [np.full((4, 5, 3), i, dtype=np.uint8) for i in range(7)]
    out = enc.encode_images(imgs)
    assert out.shape == (7, EMBED_DIM) and out.dtype == np.float32
    assert model.image_batches == [3, 3, 1]
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)

    v = enc.encode_text("  red square \n")
    assert seen == [[t.format(q="red square") for t in PROMPT_TEMPLATES]]
    raw = model.encode_text(seen[0])
    per = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    expected = per.mean(axis=0)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(v, expected, atol=1e-5)

    for blank in ("", "   ", "\t\n"):
        with pytest.raises(EmptyQueryError):
            enc.encode_text(blank)
    assert len(seen) == 1  # tokenizer never called for blank queries


def test_fake_encoder_deterministic_unit_vectors():
    enc = FakeEncoder(batch_size=2)
    imgs = [np.full((6, 6, 3), i, dtype=np.uint8) for i in range(5)]
    a, b = enc.encode_images(imgs), FakeEncoder().encode_images(imgs)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (5, EMBED_DIM)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)
    assert enc.batch_sizes == [2, 2, 1]
    np.testing.assert_array_equal(enc.encode_text(" hello "), enc.encode_text("hello"))
    with pytest.raises(EmptyQueryError):
        enc.encode_text("  ")


def test_fake_encoder_colour_mode():
    enc = FakeEncoder(colour_mode=True)
    red = np.zeros((40, 40, 3), dtype=np.uint8)
    red[10:20, 10:20] = (0, 0, 255)  # BGR red square
    blue = np.zeros((40, 40, 3), dtype=np.uint8)
    blue[5:15, 5:15] = (255, 0, 0)  # BGR blue
    grey = np.full((40, 40, 3), 128, dtype=np.uint8)
    out = enc.encode_images([red, blue, grey])
    np.testing.assert_array_equal(out[0], RED_DIRECTION)
    np.testing.assert_array_equal(out[1], BLUE_DIRECTION)
    assert abs(out[2] @ RED_DIRECTION) < 1e-6 and abs(out[2] @ BLUE_DIRECTION) < 1e-6
    np.testing.assert_array_equal(enc.encode_text("red square"), RED_DIRECTION)
    np.testing.assert_array_equal(enc.encode_text("blue circle"), BLUE_DIRECTION)
