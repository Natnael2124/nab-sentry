"""CLIP's byte-level BPE tokenizer without PyTorch.

A port of OpenAI CLIP's ``simple_tokenizer.py`` (MIT licence, Copyright (c) 2021 OpenAI), with
open_clip's ``tokenize`` framing: ``<start>`` + tokens + ``<end>``, truncated to
``context_length`` (the last kept token becomes ``<end>``), zero padded.
"""

from __future__ import annotations

import gzip
import html
from functools import lru_cache
from pathlib import Path

import ftfy
import numpy as np
import regex as re


@lru_cache()
def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) \
        + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    return {(a, b) for a, b in zip(word, word[1:])}


def clean(text: str) -> str:
    text = html.unescape(html.unescape(ftfy.fix_text(text))).strip()
    return re.sub(r"\s+", " ", text).strip().lower()


class ClipTokenizer:
    def __init__(self, bpe_path: Path | str, context_length: int = 77) -> None:
        self.context_length = context_length
        self.byte_encoder = bytes_to_unicode()
        with gzip.open(bpe_path) as fh:
            merges = fh.read().decode("utf-8").split("\n")[1:49152 - 256 - 2 + 1]
        merges = [tuple(m.split()) for m in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab] + ["".join(m) for m in merges]
        vocab += ["<|startoftext|>", "<|endoftext|>"]
        self.encoder = {v: i for i, v in enumerate(vocab)}
        self.bpe_ranks = {m: i for i, m in enumerate(merges)}
        self.sot, self.eot = self.encoder["<|startoftext|>"], self.encoder["<|endoftext|>"]
        self.cache: dict[str, str] = {}
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE)

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)
        if not pairs:
            return token + "</w>"
        while True:
            bigram = min(pairs, key=lambda pr: self.bpe_ranks.get(pr, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new: list[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new.extend(word[i:])
                    break
                new.extend(word[i:j])
                i = j
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new.append(first + second)
                    i += 2
                else:
                    new.append(word[i])
                    i += 1
            word = tuple(new)
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        out = " ".join(word)
        self.cache[token] = out
        return out

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for token in re.findall(self.pat, clean(text)):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            ids.extend(self.encoder[t] for t in self.bpe(token).split(" "))
        return ids

    def __call__(self, texts: list[str]) -> np.ndarray:
        """(len(texts), context_length) int64, framed as open_clip's ``tokenize``."""
        out = np.zeros((len(texts), self.context_length), dtype=np.int64)
        for row, text in enumerate(texts):
            ids = [self.sot, *self.encode(text), self.eot][: self.context_length]
            ids[-1] = self.eot
            out[row, : len(ids)] = ids
        return out
