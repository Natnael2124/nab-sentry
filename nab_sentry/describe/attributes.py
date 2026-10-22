"""Zero-shot appearance tags from the crop embeddings ingest already stores (Step 3, task 4.5).

Each attribute group is a list of ``(tag, prompt)`` pairs for one object class. A crop
embedding is scored against the group's prompt embeddings with a softmax at CLIP's logit scale
(100); :func:`vote` combines the crops of one track - weighting each crop by its own winning
probability - and returns a tag only when it wins more than ``min_share`` of the total weight and
its mean probability reaches ``min_prob``, otherwise nothing ("unknown" is better than wrong).

Measured on the Compilation_Video (single crops, checked by eye): vehicle colour and type about
80% correct, person top colour about 60% - single crops with several people in them were the
usual failure, which per-track voting is meant to fix. Groups are configured, not hard-coded,
so a group that measures below 70% (Requirement 4A.3) can be switched off.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

LOGIT_SCALE = 100.0


@dataclass(frozen=True)
class Group:
    name: str  # e.g. "top colour"
    classes: frozenset[str]  # object classes the group applies to
    tags: tuple[str, ...]
    prompts: tuple[str, ...]


def _colours(subject: str, noun: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    cols = ("red", "blue", "black", "white", "grey", "green", "yellow", "brown")
    return cols, tuple(f"a {subject} wearing a {c} {noun}" if subject == "person" else f"a {c} {subject}"
                       for c in cols)


_TOP_TAGS, _TOP_PROMPTS = _colours("person", "top")
_CAR_COLOURS = ("white", "black", "silver", "grey", "red", "blue")

DEFAULT_GROUPS: tuple[Group, ...] = (
    Group("top colour", frozenset({"person"}), _TOP_TAGS, _TOP_PROMPTS),
    Group("colour", frozenset({"car", "truck", "bus"}), _CAR_COLOURS,
          tuple(f"a {c} vehicle" for c in _CAR_COLOURS)),
    Group("type", frozenset({"car", "truck", "bus"}),
          ("sedan", "pickup", "SUV", "van", "police car", "bus"),
          ("a sedan car", "a pickup truck", "an SUV", "a van", "a police car", "a bus")),
)


def softmax_scores(vec: np.ndarray, text: np.ndarray) -> np.ndarray:
    """Probabilities of one unit image embedding over a group's unit text embeddings."""
    logits = LOGIT_SCALE * (text @ vec)
    p = np.exp(logits - logits.max())
    return p / p.sum()


def vote(probs: Sequence[np.ndarray], tags: Sequence[str], *, min_share: float = 0.5,
         min_prob: float = 0.4) -> tuple[str | None, float]:
    """Combine per-crop probability vectors into one tag (or None) and its mean probability.

    Each crop votes for its argmax with weight = that probability. The winner needs more than
    ``min_share`` of the total weight (a tie is unknown) and a mean probability (over the crops that voted for it)
    of at least ``min_prob``.
    """
    if not probs:
        return None, 0.0
    weight = np.zeros(len(tags))
    count = np.zeros(len(tags))
    for p in probs:
        k = int(np.argmax(p))
        weight[k] += float(p[k])
        count[k] += 1
    k = int(np.argmax(weight))
    total = float(weight.sum())
    mean = float(weight[k] / count[k]) if count[k] else 0.0
    if total <= 0 or weight[k] / total <= min_share or mean < min_prob:
        return None, mean
    return tags[k], mean


class AttributeTagger:
    """Tags tracks from their crop embeddings; prompt embeddings are encoded once."""

    def __init__(self, encode_text: Callable[[str], np.ndarray],
                 groups: Sequence[Group] = DEFAULT_GROUPS, *, min_share: float = 0.5,
                 min_prob: float = 0.4) -> None:
        self.groups = list(groups)
        self.min_share = min_share
        self.min_prob = min_prob
        self._text = {g.name + "|" + ",".join(sorted(g.classes)):
                      np.stack([np.asarray(encode_text(p), np.float32) for p in g.prompts])
                      for g in self.groups}

    def _key(self, g: Group) -> str:
        return g.name + "|" + ",".join(sorted(g.classes))

    def tag(self, det_class: str, crop_vectors: Sequence[np.ndarray]) -> dict[str, tuple[str, float]]:
        """{group name: (tag, mean probability)} for the groups of this class that are sure."""
        out: dict[str, tuple[str, float]] = {}
        for g in self.groups:
            if det_class not in g.classes or not len(crop_vectors):
                continue
            text = self._text[self._key(g)]
            probs = [softmax_scores(np.asarray(v, np.float32), text) for v in crop_vectors]
            tag, p = vote(probs, g.tags, min_share=self.min_share, min_prob=self.min_prob)
            if tag is not None:
                out[g.name] = (tag, round(p, 3))
        return out


def phrase(det_class: str, tags: dict[str, tuple[str, float]]) -> str:
    """'person in a red top', 'white SUV', 'car' - from whatever tags are sure."""
    if det_class == "person":
        top = tags.get("top colour")
        return f"person in a {top[0]} top" if top else "person"
    colour = tags.get("colour")
    kind = tags.get("type")
    noun = kind[0] if kind else det_class
    return f"{colour[0]} {noun}" if colour else noun
