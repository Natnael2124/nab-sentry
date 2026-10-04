"""Property 25: Source hash is content-addressed.

**Validates: Requirements 7.2**
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from nab_sentry.ingest.metadata import src_hash

# Portable file/dir name segments. The alphabet excludes reserved chars, dots and spaces
# (so no trailing dot/space), and the fixed "s_" prefix guarantees no segment can ever be a
# Windows reserved device name (CON, PRN, AUX, NUL, COM1-9, LPT1-9, any case, any extension).
_segment = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"),
    min_size=1,
    max_size=12,
).map(lambda s: f"s_{s}")
_ext = st.sampled_from(["mp4", "MOV", "avi", "mkv", "bin", "json"])
_rel_path = st.tuples(st.lists(_segment, min_size=0, max_size=3), _segment, _ext).map(
    lambda t: Path(*t[0], f"{t[1]}.{t[2]}")
)
# Small chunk sizes exercise chunk boundaries; None uses the 1 MiB default.
_chunk = st.one_of(st.none(), st.integers(min_value=1, max_value=64))


def _write(root: Path, sub: str, rel: Path, data: bytes) -> Path:
    p = root / sub / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def _hash(p: Path, chunk: int | None) -> str:
    return src_hash(p) if chunk is None else src_hash(p, chunk)


@given(
    a=st.binary(max_size=512),
    b=st.binary(max_size=512),
    same=st.booleans(),
    path_a=_rel_path,
    path_b=_rel_path,
    chunk_a=_chunk,
    chunk_b=_chunk,
)
def test_src_hash_is_content_addressed(a, b, same, path_a, path_b, chunk_a, chunk_b):
    # Feature: nab-sentry, Property 25: Source hash is content-addressed
    if same:
        b = a  # force the equal-content case to occur often
    root = Path(tempfile.mkdtemp(prefix="srchash-"))
    try:
        fa = _write(root, "a", path_a, a)
        fb = _write(root, "b", path_b, b)
        ha, hb = _hash(fa, chunk_a), _hash(fb, chunk_b)

        # Equal iff the bytes are equal, regardless of name, directory, or chunk size.
        assert (ha == hb) == (a == b)
        # And it is the SHA-256 hex digest of the content.
        assert ha == hashlib.sha256(a).hexdigest()
        assert hb == hashlib.sha256(b).hexdigest()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_src_hash_empty_file_and_rename(tmp_path: Path):
    p = tmp_path / "CAM-01_20240101T000000.mp4"
    p.write_bytes(b"")
    assert src_hash(p) == hashlib.sha256(b"").hexdigest()

    data = b"x" * ((1 << 20) + 3)  # spans a default-chunk boundary
    p.write_bytes(data)
    before = src_hash(p)
    moved = tmp_path / "other" / "renamed.mov"
    moved.parent.mkdir()
    p.rename(moved)
    assert src_hash(moved) == before == hashlib.sha256(data).hexdigest()
