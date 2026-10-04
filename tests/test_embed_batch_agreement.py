"""Property 22: Batched and single-image embeddings agree (slow, real OpenCLIP weights).

**Validates: Requirements 5.6**

Skips cleanly when the local ViT-B/32 checkpoint is absent (fetch it with the
scripts/ fetch script). The weights are loaded once per module via a cached loader
function (not a function-scoped fixture, which Hypothesis health checks reject).
"""

from __future__ import annotations

import functools

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nab_sentry.config import Config

pytestmark = pytest.mark.slow

WEIGHTS_PATH = (Config().models_dir / "open_clip" / "ViT-B-32-laion2b_s34b_b79k"
                / "open_clip_pytorch_model.bin")

MIN_COSINE = 0.999


@functools.lru_cache(maxsize=1)
def _load_encoder():
    """Load the real encoder once; skip the module's tests if weights are missing."""
    from nab_sentry.embed.clip_encoder import OpenClipEncoder

    return OpenClipEncoder(WEIGHTS_PATH, batch_size=1)


@pytest.fixture(scope="module", autouse=True)
def _require_weights():
    if not WEIGHTS_PATH.is_file():
        pytest.skip(f"OpenCLIP weights not found at {WEIGHTS_PATH}")


def _encoder_with_batch_size(batch_size: int):
    """Share the loaded model/preprocess/tokenizer, varying only the batch size."""
    import torch

    from nab_sentry.embed.clip_encoder import OpenClipEncoder

    base = _load_encoder()
    return OpenClipEncoder.from_components(
        base.model, base.preprocess, base.tokenizer, batch_size=batch_size, torch_module=torch)


@st.composite
def bgr_images(draw) -> np.ndarray:
    """Random BGR uint8 images, 32-256 px per side (noise, cheap to generate, varied content)."""
    h = draw(st.integers(min_value=32, max_value=256))
    w = draw(st.integers(min_value=32, max_value=256))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    return np.random.default_rng(seed).integers(0, 256, size=(h, w, 3), dtype=np.uint8)


# Feature: nab-sentry, Property 22: Batched and single-image embeddings agree
@settings(max_examples=12, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
@given(images=st.lists(bgr_images(), min_size=1, max_size=16),
       batch_size=st.integers(min_value=1, max_value=16))
def test_batched_and_single_embeddings_agree(images, batch_size):
    single_enc = _encoder_with_batch_size(1)
    batched_enc = _encoder_with_batch_size(batch_size)

    batched = batched_enc.encode_images(images)
    assert batched.shape == (len(images), 512)

    for i, img in enumerate(images):
        single = single_enc.encode_images([img])[0]
        # Rows are unit-norm, so the dot product is the cosine similarity.
        cos = float(np.dot(single.astype(np.float64), batched[i].astype(np.float64)))
        assert cos >= MIN_COSINE, f"image {i} {img.shape}: cosine {cos:.6f} < {MIN_COSINE}"
