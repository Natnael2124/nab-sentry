"""Attribute tags (Step 3, task 4.5): scoring, voting, phrases."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from nab_sentry.describe.attributes import (
    DEFAULT_GROUPS,
    AttributeTagger,
    Group,
    phrase,
    softmax_scores,
    vote,
)

TAGS = ("red", "blue", "black")


def onehot(k: int, p: float) -> np.ndarray:
    v = np.full(3, (1 - p) / 2)
    v[k] = p
    return v


@given(st.lists(st.tuples(st.integers(0, 2), st.floats(0.34, 1.0)), min_size=1, max_size=12))
def test_vote_returns_a_tag_only_when_it_wins_clearly(crops):
    probs = [onehot(k, p) for k, p in crops]
    tag, mean = vote(probs, TAGS, min_share=0.5, min_prob=0.4)
    weights = {t: sum(p for k, p in crops if TAGS[k] == t) for t in TAGS}
    if tag is not None:
        assert weights[tag] > 0.5 * sum(weights.values()) - 1e-9 and mean >= 0.4 - 1e-9
        assert weights[tag] == max(weights.values())


def test_vote_examples():
    assert vote([onehot(0, 0.9), onehot(0, 0.8), onehot(1, 0.6)], TAGS)[0] == "red"
    assert vote([onehot(0, 0.9), onehot(1, 0.9)], TAGS)[0] is None  # a tie is unknown
    assert vote([onehot(0, 0.5), onehot(1, 0.5), onehot(2, 0.5)], TAGS)[0] is None  # no majority
    assert vote([onehot(0, 0.35)], TAGS, min_prob=0.4)[0] is None  # not sure enough
    assert vote([], TAGS) == (None, 0.0)


@given(st.integers(0, 2), st.floats(-0.3, 0.3))
def test_softmax_prefers_the_aligned_prompt(k, noise):
    text = np.eye(3, dtype=np.float32)
    vec = text[k] + noise * text[(k + 1) % 3]
    vec /= np.linalg.norm(vec)
    p = softmax_scores(vec, text)
    assert p.sum() == pytest.approx(1.0) and int(np.argmax(p)) == k


def test_tagger_with_a_fake_encoder():
    space = {}

    def encode(text: str) -> np.ndarray:  # each prompt gets its own axis
        if text not in space:
            space[text] = len(space)
        v = np.zeros(64, np.float32)
        v[space[text]] = 1
        return v
    groups = [Group("top colour", frozenset({"person"}), ("red", "blue"),
                    ("a person wearing a red top", "a person wearing a blue top"))]
    tagger = AttributeTagger(encode, groups)
    red = encode("a person wearing a red top")
    assert tagger.tag("person", [red, red]) == {"top colour": ("red", 1.0)}
    assert tagger.tag("car", [red]) == {}
    assert tagger.tag("person", []) == {}


def test_phrases_and_default_groups():
    assert phrase("person", {"top colour": ("red", 0.8)}) == "person in a red top"
    assert phrase("person", {}) == "person"
    assert phrase("car", {"colour": ("white", 0.9), "type": ("SUV", 0.9)}) == "white SUV"
    assert phrase("truck", {}) == "truck"
    assert {g.name for g in DEFAULT_GROUPS} == {"top colour", "colour", "type"}
    assert all(len(g.tags) == len(g.prompts) for g in DEFAULT_GROUPS)


@given(st.lists(st.floats(0.0, 1.0), min_size=6, max_size=6).filter(lambda x: sum(x) > 0))
def test_merged_tags_keep_the_probability_mass(raw):
    from nab_sentry.describe.attributes import merge_by_tag

    p = np.asarray(raw) / sum(raw)
    tags = ("sedan", "pickup", "SUV/van", "SUV/van", "police car", "bus")
    m = merge_by_tag(p, tags)
    assert m.shape == (5,) and abs(m.sum() - 1.0) < 1e-9
    assert abs(m[2] - (p[2] + p[3])) < 1e-12 and m[0] == p[0]


def test_suv_and_van_are_one_tag():
    from nab_sentry.describe.attributes import DEFAULT_GROUPS

    g = {x.name: x for x in DEFAULT_GROUPS}["type"]
    assert g.unique_tags == ("sedan", "pickup", "SUV/van", "police car", "bus")
    assert len(g.tags) == len(g.prompts)
