"""Property 21 (encoder half): blank queries are rejected before any tokenizer/model call.

**Validates: Requirements 5.9, 10.7**
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nab_sentry.embed.clip_encoder import EMBED_DIM, EmptyQueryError, OpenClipEncoder
from tests.fakes import FakeEncoder

# Every code point str.strip() removes (exactly those with str.isspace() True).
WHITESPACE_CHARS = "".join(chr(c) for c in range(sys.maxunicode + 1) if chr(c).isspace())

blank_strings = st.text(alphabet=st.sampled_from(WHITESPACE_CHARS), min_size=0, max_size=20)
non_blank_core = st.text(min_size=1, max_size=30).filter(lambda s: s.strip() == s and s != "")


class _Spy:
    """Records calls; ``fail`` makes any call an immediate test failure."""

    def __init__(self, fail: bool) -> None:
        self.fail = fail
        self.calls = 0

    def tokenizer(self, texts):
        self.calls += 1
        if self.fail:
            pytest.fail(f"tokenizer called for blank query: {texts!r}")
        return np.zeros((len(texts), 77), dtype=np.int64)

    def encode_text(self, tokens):
        self.calls += 1
        if self.fail:
            pytest.fail("model.encode_text called for blank query")
        n = len(tokens)
        out = np.zeros((n, EMBED_DIM), dtype=np.float32)
        out[:, 0] = 1.0
        return out

    def encode_image(self, batch):  # pragma: no cover - not used here
        pytest.fail("encode_image must not be called")


def _encoder(spy: _Spy) -> OpenClipEncoder:
    return OpenClipEncoder.from_components(
        model=spy, preprocess=lambda img: img, tokenizer=spy.tokenizer)


def test_whitespace_alphabet_matches_strip():
    assert WHITESPACE_CHARS.strip() == ""
    for ch in (" ", "\t", "\n", "\r", "\x0b", "\x0c", "\x1c", "\x85", "\xa0", "\u2028", "\u3000"):
        assert ch in WHITESPACE_CHARS


# Feature: nab-sentry, Property 21: Blank queries are rejected
@settings(max_examples=200)
@given(query=blank_strings)
def test_blank_query_rejected_without_model_call(query: str):
    spy = _Spy(fail=True)
    enc = _encoder(spy)
    result = None
    with pytest.raises(EmptyQueryError, match="empty query"):
        result = enc.encode_text(query)
    assert result is None
    assert spy.calls == 0
    assert issubclass(EmptyQueryError, ValueError)


# Feature: nab-sentry, Property 21: Blank queries are rejected (FakeEncoder agrees)
@settings(max_examples=200)
@given(query=blank_strings)
def test_fake_encoder_rejects_blank(query: str):
    fake = FakeEncoder()
    with pytest.raises(EmptyQueryError):
        fake.encode_text(query)
    assert fake.text_calls == []


# Contrast: non-blank queries with surrounding whitespace are accepted.
@settings(max_examples=100)
@given(core=non_blank_core, left=blank_strings, right=blank_strings)
def test_non_blank_with_padding_is_encoded(core: str, left: str, right: str):
    spy = _Spy(fail=False)
    enc = _encoder(spy)
    vec = enc.encode_text(left + core + right)
    assert vec.shape == (EMBED_DIM,)
    assert vec.dtype == np.float32
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)
    assert spy.calls == 2  # one tokenizer call, one model call

    fake = FakeEncoder()
    fvec = fake.encode_text(left + core + right)
    assert fvec.shape == (EMBED_DIM,)
    assert fake.text_calls == [core]
