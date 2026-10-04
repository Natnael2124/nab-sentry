"""Property 19: Embeddings are finite unit vectors.

Feature: nab-sentry, Property 19: Embeddings are finite unit vectors
**Validates: Requirements 5.3**

The ``l2_normalize`` part is fast. The real-encoder part loads the OpenCLIP ViT-B/32
checkpoint from ``models/`` and is marked ``slow``; it skips when the weights are absent.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nab_sentry.config import Config
from nab_sentry.embed.clip_encoder import EMBED_DIM, EmbeddingError, l2_normalize

NORM_TOL = 1e-4

# --------------------------------------------------------------------------- strategies


@st.composite
def nonzero_rows(draw: st.DrawFn, n: int, d: int) -> np.ndarray:
    """(n, d) float64 rows, each with one entry |v| >= 0.1, scaled by 10**[-10, 145].

    The exponent range spans tiny and huge magnitudes while keeping every row norm
    well above the zero-norm epsilon and the sum of squares far below float64 max.
    """
    rows = []
    for _ in range(n):
        base = np.array(draw(st.lists(
            st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False),
            min_size=d, max_size=d)), dtype=np.float64)
        idx = draw(st.integers(0, d - 1))
        base[idx] = draw(st.sampled_from([-1.0, 1.0])) * draw(st.floats(0.1, 1.0))
        scale = 10.0 ** draw(st.integers(-10, 145))
        rows.append(base * scale)
    return np.stack(rows)


@st.composite
def batches(draw: st.DrawFn) -> np.ndarray:
    n = draw(st.integers(1, 6))
    d = draw(st.sampled_from([1, 2, 3, 16, EMBED_DIM]))
    arr = draw(nonzero_rows(n, d))
    if draw(st.booleans()):
        # Float32 input too, as produced by the encoders: rescale each row into float32
        # range (max |v| = 10**[-10, 30]) so no row underflows to the zero-norm epsilon.
        exps = np.array(draw(st.lists(st.integers(-10, 30), min_size=n, max_size=n)))
        row_max = np.max(np.abs(arr), axis=1, keepdims=True)
        arr = (arr / row_max * (10.0 ** exps)[:, None]).astype(np.float32)
    return arr


def assert_unit_rows(out: np.ndarray, shape: tuple[int, ...]) -> None:
    assert out.dtype == np.float32
    assert out.shape == shape
    assert np.all(np.isfinite(out))
    norms = np.linalg.norm(out.astype(np.float64), axis=-1)
    assert np.all(np.abs(norms - 1.0) <= NORM_TOL), norms


# --------------------------------------------------------------------------- fast part


@given(batches())
def test_l2_normalize_rows_are_finite_unit_vectors(arr: np.ndarray) -> None:
    out = l2_normalize(arr, axis=1)
    assert_unit_rows(out, arr.shape)


@given(batches())
def test_l2_normalize_single_vector_is_unit(arr: np.ndarray) -> None:
    vec = arr[0]
    out = l2_normalize(vec)
    assert_unit_rows(out, vec.shape)


@given(batches(), st.data())
def test_l2_normalize_rejects_zero_or_non_finite_rows(arr: np.ndarray, data: st.DataObject) -> None:
    bad = arr.astype(np.float64)
    row = data.draw(st.integers(0, bad.shape[0] - 1))
    kind = data.draw(st.sampled_from(["zero", "nan", "inf", "-inf"]))
    if kind == "zero":
        bad[row, :] = 0.0
    else:
        col = data.draw(st.integers(0, bad.shape[1] - 1))
        bad[row, col] = {"nan": np.nan, "inf": np.inf, "-inf": -np.inf}[kind]
    with pytest.raises(EmbeddingError):
        l2_normalize(bad, axis=1)


def test_l2_normalize_rejects_empty() -> None:
    with pytest.raises(EmbeddingError):
        l2_normalize(np.empty((0,), dtype=np.float32))


# --------------------------------------------------------------------------- slow part

WEIGHTS_REL = ("open_clip", "ViT-B-32-laion2b_s34b_b79k", "open_clip_pytorch_model.bin")


@pytest.fixture(scope="module")
def real_encoder():
    weights = Config().models_dir.joinpath(*WEIGHTS_REL)
    if not weights.is_file():
        pytest.skip(f"OpenCLIP weights not present at {weights}; run the Model_Fetcher")
    from nab_sentry.embed.clip_encoder import OpenClipEncoder

    return OpenClipEncoder(weights, batch_size=4)


@st.composite
def random_images(draw: st.DrawFn) -> list[np.ndarray]:
    count = draw(st.integers(1, 5))
    seed = draw(st.integers(0, 2**32 - 1))
    rng = np.random.default_rng(seed)
    images = []
    for _ in range(count):
        h = draw(st.integers(1, 1080))
        w = draw(st.integers(1, 1920))
        images.append(rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8))
    return images


non_blank_queries = st.text(min_size=1, max_size=256).filter(lambda s: s.strip() != "")


@pytest.mark.slow
@settings(max_examples=20, deadline=None)
@given(images=random_images())
def test_real_encoder_images_are_unit_vectors(real_encoder, images: list[np.ndarray]) -> None:
    out = real_encoder.encode_images(images)
    assert_unit_rows(out, (len(images), EMBED_DIM))


@pytest.mark.slow
@settings(max_examples=50, deadline=None)
@given(query=non_blank_queries)
def test_real_encoder_text_is_unit_vector(real_encoder, query: str) -> None:
    out = real_encoder.encode_text(query)
    assert_unit_rows(out, (EMBED_DIM,))
