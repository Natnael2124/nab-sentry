"""Property 20: query encoding ensembles both prompt templates (Requirement 5.4).

A stub text tower maps each templated string to a deterministic random vector (seeded
from a SHA-256 of the string), and a spy tokenizer records what it was called with, so
the expected ensembled embedding can be computed independently of the encoder.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from nab_sentry.embed.clip_encoder import EMBED_DIM, PROMPT_TEMPLATES, OpenClipEncoder


def _vec_for(text: str) -> np.ndarray:
    """Deterministic, non-unit float64 vector for a string (independent of the encoder)."""
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    # Scale varies per string so per-template normalisation actually matters.
    return rng.standard_normal(EMBED_DIM) * rng.uniform(0.1, 50.0)


class SpyTokenizer:
    """Records each call; 'tokens' are just the list of strings passed through."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> list[str]:
        self.calls.append(list(texts))
        return list(texts)


class StubTextModel:
    """Text tower stub: one deterministic vector per 'token' row (a string)."""

    def __init__(self) -> None:
        self.text_calls = 0

    def encode_text(self, tokens: list[str]) -> np.ndarray:
        self.text_calls += 1
        return np.stack([_vec_for(t) for t in tokens]).astype(np.float32)

    def encode_image(self, batch):  # pragma: no cover - unused here
        raise AssertionError("image tower must not be used by encode_text")


def _normalise(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def _make_encoder(tokenizer=None):
    tok = tokenizer if tokenizer is not None else SpyTokenizer()
    model = StubTextModel()
    enc = OpenClipEncoder.from_components(model, preprocess=lambda x: x, tokenizer=tok)
    return enc, model, tok


# Non-blank core text (including braces and format-like fragments), wrapped in whitespace.
_core = st.one_of(
    st.text(min_size=1, max_size=60),
    st.sampled_from(["{q}", "{0}", "{}", "{x}", "{{q}}", "red {q} car", "}{", "%s", "a CCTV photo of {q}"]),
).filter(lambda s: s.strip() != "")
_ws = st.text(alphabet=" \t\n\r\x0b\x0c", max_size=4)
queries = st.tuples(_ws, _core, _ws).map(lambda t: t[0] + t[1] + t[2])


@given(query=queries)
def test_query_encoding_ensembles_both_templates(query: str) -> None:
    # Feature: nab-sentry, Property 20: Query encoding ensembles both prompt templates
    # **Validates: Requirements 5.4**
    enc, model, tok = _make_encoder()
    q = query.strip()

    out = enc.encode_text(query)

    # Tokenizer called exactly once with both templated strings, in template order.
    expected_texts = [q, "a CCTV photo of " + q]
    assert [t.replace("{q}", q) for t in PROMPT_TEMPLATES] == expected_texts
    assert tok.calls == [expected_texts]
    assert model.text_calls == 1

    # Output = normalise(mean(normalise(t(f1)), normalise(t(f2)))) computed independently.
    v1 = _normalise(_vec_for(expected_texts[0]).astype(np.float32))
    v2 = _normalise(_vec_for(expected_texts[1]).astype(np.float32))
    expected = _normalise((v1 + v2) / 2.0)

    assert out.shape == (EMBED_DIM,)
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, expected, atol=1e-6, rtol=0)
    assert abs(float(np.linalg.norm(out.astype(np.float64))) - 1.0) < 1e-5


def test_braces_in_query_are_passed_verbatim() -> None:
    enc, _, tok = _make_encoder()
    enc.encode_text("  {0} near {q}  ")
    assert tok.calls == [["{0} near {q}", "a CCTV photo of {0} near {q}"]]


def test_ensemble_differs_from_single_template() -> None:
    """Both templates contribute: the result is not just either per-template vector."""
    enc, _, _ = _make_encoder()
    out = enc.encode_text("red car")
    v1 = _normalise(_vec_for("red car").astype(np.float32))
    v2 = _normalise(_vec_for("a CCTV photo of red car").astype(np.float32))
    assert not np.allclose(out, v1, atol=1e-3)
    assert not np.allclose(out, v2, atol=1e-3)
    # The ensemble sits symmetrically between the two template directions.
    assert abs(float(out @ v1) - float(out @ v2)) < 1e-5


@pytest.mark.slow
def test_real_tokenizer_truncates_long_queries_to_77_tokens(monkeypatch) -> None:
    """The bundled open_clip BPE tokenizer (no download) yields (2, 77) tokens for long queries."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    open_clip = pytest.importorskip("open_clip")
    real_tok = open_clip.get_tokenizer("ViT-B-32")
    seen: list = []

    def spy(texts):
        tokens = real_tok(texts)
        seen.append(tokens)
        return tokens

    class ShapeModel:
        def encode_text(self, tokens):
            arr = np.asarray(tokens.numpy() if hasattr(tokens, "numpy") else tokens, dtype=np.float64)
            rng = np.random.default_rng(int(arr.sum()) % (2**32))
            return rng.standard_normal((arr.shape[0], EMBED_DIM)).astype(np.float32)

    enc = OpenClipEncoder.from_components(ShapeModel(), preprocess=lambda x: x, tokenizer=spy)
    long_query = " ".join(["suspicious person walking near the loading dock"] * 40)
    out = enc.encode_text(long_query)

    assert len(seen) == 1
    assert tuple(seen[0].shape) == (2, 77)
    assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-5
