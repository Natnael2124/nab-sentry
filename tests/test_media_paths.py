"""Property 45 (resolver part): media paths cannot escape their directories.

The HTTP 404 half of Property 45 is covered by task 14.12.
"""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nab_sentry.api.media import resolve_playback, resolve_thumb

# Real files that live directly inside thumbs/ and playback/.
REAL_THUMBS = ("a.jpg", "frame_001.jpg", "x-y.jpg", "secret.jpg")
REAL_PLAYBACK = ("v1.mp4", "v42.mp4")
# Decoy names that exist *outside* the media dirs, reachable by traversal.
DECOY_NAMES = ("secret.jpg", "a.jpg", "v1.mp4", "v7.mp4", "passwd.txt")


@dataclass(frozen=True)
class Layout:
    root: Path
    thumbs: Path
    playback: Path
    real_thumbs: frozenset[Path]
    real_playback: frozenset[Path]
    decoys: frozenset[Path]


@contextmanager
def media_layout() -> Iterator[Layout]:
    """Fresh tree per example::

        root/{decoys}
        root/data/{decoys}
        root/data/thumbs/{REAL_THUMBS}
        root/data/playback/{REAL_PLAYBACK}
        root/data/other/{decoys}
        root/data/thumbs_evil/{decoys}     (prefix-sharing sibling)
        root/data/playback2/{decoys}       (prefix-sharing sibling)
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "root"
        data = root / "data"
        thumbs = data / "thumbs"
        playback = data / "playback"
        decoy_dirs = [root, data, data / "other", data / "thumbs_evil", data / "playback2"]
        for d in [thumbs, playback, *decoy_dirs]:
            d.mkdir(parents=True, exist_ok=True)
        decoys = set()
        for d in decoy_dirs:
            for n in DECOY_NAMES:
                p = d / n
                p.write_bytes(b"SECRET " + str(p).encode())
                decoys.add(p.resolve())
        for n in REAL_THUMBS:
            (thumbs / n).write_bytes(b"\xff\xd8thumb")
        for n in REAL_PLAYBACK:
            (playback / n).write_bytes(b"mp4data")
        yield Layout(
            root=root,
            thumbs=thumbs,
            playback=playback,
            real_thumbs=frozenset((thumbs / n).resolve() for n in REAL_THUMBS),
            real_playback=frozenset((playback / n).resolve() for n in REAL_PLAYBACK),
            decoys=frozenset(decoys),
        )


# Sentinels substituted with absolute paths of the per-example tree.
ABS_ROOT = "<ABS_ROOT>"
ABS_DATA = "<ABS_DATA>"

TOKENS = [
    ".", "..", "/", "\\", "//", "\\\\", "%2e", "%2E", "%2f", "%2F", "%5c", "%5C",
    "%2e%2e", "%252e", ":", "C:", "c:", "D:", "C:\\", "C:/", "\\\\?\\", "\\\\.\\",
    "~", "\x00", "\n", " ", "data", "thumbs", "playback", "other", "thumbs_evil",
    "playback2", *DECOY_NAMES, *REAL_THUMBS, *REAL_PLAYBACK, ".jpg", ".mp4", "v",
    "secret", ABS_ROOT, ABS_DATA,
]

segments = st.one_of(
    st.sampled_from(TOKENS),
    st.text(alphabet="abv0123456789_-./\\:%2ecfCF", min_size=1, max_size=6),
)
raw_names = st.lists(segments, min_size=1, max_size=8).map("".join)


def _materialise(name: str, layout: Layout) -> str:
    return name.replace(ABS_ROOT, str(layout.root)).replace(ABS_DATA, str(layout.root / "data"))


def _check(
    resolver: Callable[[Path, str], Path | None],
    base: Path,
    real: frozenset[Path],
    layout: Layout,
    name: str,
) -> None:
    result = resolver(base, name)
    if result is None:
        return
    # Contained: resolved parent is exactly the resolved base, and it is a regular
    # file that we placed there (never a decoy outside).
    assert result.parent == base.resolve(), (name, result)
    assert result.is_file()
    assert result in real, (name, result)
    assert result not in layout.decoys
    # Only bare names can resolve; anything carrying path syntax must be rejected.
    for bad in ("/", "\\", ":", "%", "\x00", ".."):
        assert bad not in name, (name, result)


# Feature: nab-sentry, Property 45: Media paths cannot escape their directories
# **Validates: Requirements 11.8, 18.6**
@settings(max_examples=200)
@given(raw=raw_names)
def test_resolve_thumb_never_escapes(raw: str) -> None:
    with media_layout() as layout:
        name = _materialise(raw, layout)
        _check(resolve_thumb, layout.thumbs, layout.real_thumbs, layout, name)


# Feature: nab-sentry, Property 45: Media paths cannot escape their directories
# **Validates: Requirements 11.8, 18.6**
@settings(max_examples=200)
@given(raw=raw_names)
def test_resolve_playback_never_escapes(raw: str) -> None:
    with media_layout() as layout:
        name = _materialise(raw, layout)
        _check(resolve_playback, layout.playback, layout.real_playback, layout, name)


# Feature: nab-sentry, Property 45: Media paths cannot escape their directories
# **Validates: Requirements 11.8, 18.6**
@given(
    prefix=st.sampled_from(["../", "..\\", "../../", "..%2f", "%2e%2e/", "other/", "../other/",
                            "../thumbs_evil/", "../playback2/", "./../", ABS_ROOT + os.sep,
                            ABS_DATA + os.sep, "/", "\\", "C:"]),
    target=st.sampled_from(DECOY_NAMES),
)
def test_traversal_to_decoy_is_rejected(prefix: str, target: str) -> None:
    with media_layout() as layout:
        name = _materialise(prefix + target, layout)
        assert resolve_thumb(layout.thumbs, name) is None
        assert resolve_playback(layout.playback, name) is None


# --------------------------------------------------------------- examples


def test_legit_names_resolve() -> None:
    with media_layout() as layout:
        for n in REAL_THUMBS:
            assert resolve_thumb(layout.thumbs, n) == (layout.thumbs / n).resolve()
        for n in REAL_PLAYBACK:
            assert resolve_playback(layout.playback, n) == (layout.playback / n).resolve()


def test_valid_but_missing_names_are_none() -> None:
    with media_layout() as layout:
        assert resolve_thumb(layout.thumbs, "missing.jpg") is None
        assert resolve_playback(layout.playback, "v999.mp4") is None
        assert resolve_playback(layout.playback, "v7.mp4") is None  # exists only outside
        assert resolve_playback(layout.playback, None) is None


def test_directory_with_valid_name_is_none() -> None:
    with media_layout() as layout:
        (layout.thumbs / "dir.jpg").mkdir()
        (layout.playback / "v5.mp4").mkdir()
        assert resolve_thumb(layout.thumbs, "dir.jpg") is None
        assert resolve_playback(layout.playback, "v5.mp4") is None


def test_symlink_escape_is_rejected() -> None:
    with media_layout() as layout:
        outside = layout.root / "secret.jpg"
        try:
            os.symlink(outside, layout.thumbs / "link.jpg")
            os.symlink(layout.root / "data" / "other" / "v7.mp4", layout.playback / "v9.mp4")
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not permitted on this system")
        assert resolve_thumb(layout.thumbs, "link.jpg") is None
        assert resolve_playback(layout.playback, "v9.mp4") is None
