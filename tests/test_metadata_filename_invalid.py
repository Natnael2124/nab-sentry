"""Property test: invalid Filename_Convention names are rejected by ``parse_filename``."""

from __future__ import annotations

import calendar
import string

from hypothesis import given
from hypothesis import strategies as st

from nab_sentry.ingest.metadata import parse_filename
from tests.strategies import camera_ids

EXTENSIONS = st.text(alphabet=string.ascii_letters + string.digits, min_size=1, max_size=10)


def _is_real_datetime(y: int, mo: int, d: int, h: int, mi: int, s: int) -> bool:
    """Reference calendar check, independent of ``datetime``."""
    if not (1 <= y <= 9999 and 1 <= mo <= 12):
        return False
    if not (1 <= d <= calendar.monthrange(y, mo)[1]):
        return False
    return h < 24 and mi < 60 and s < 60


def _ts(y: int, mo: int, d: int, h: int, mi: int, s: int) -> str:
    return f"{y:04d}{mo:02d}{d:02d}T{h:02d}{mi:02d}{s:02d}"


_two = st.integers(0, 99)
_valid_parts = st.tuples(
    st.integers(1, 9999), st.integers(1, 12), st.integers(1, 28),
    st.integers(0, 23), st.integers(0, 59), st.integers(0, 59),
)


@st.composite
def _targeted_invalid(draw) -> tuple[int, ...]:
    """Start from a valid date-time and break exactly one field."""
    y, mo, d, h, mi, s = draw(_valid_parts)
    field = draw(st.sampled_from(["year", "month", "day", "hour", "minute", "second"]))
    if field == "year":
        y = 0
    elif field == "month":
        mo = draw(st.one_of(st.just(0), st.integers(13, 99)))
    elif field == "day":
        d = draw(st.one_of(st.just(0), st.integers(calendar.monthrange(y, mo)[1] + 1, 99)))
    elif field == "hour":
        h = draw(st.integers(24, 99))
    elif field == "minute":
        mi = draw(st.integers(60, 99))
    else:
        s = draw(st.integers(60, 99))
    return (y, mo, d, h, mi, s)


# Arbitrary digit combinations, kept only when the reference says they are not real.
_random_invalid = st.tuples(st.integers(0, 9999), _two, _two, _two, _two, _two).filter(
    lambda p: not _is_real_datetime(*p)
)

invalid_timestamps = st.one_of(_targeted_invalid(), _random_invalid).map(lambda p: _ts(*p))


# Feature: nab-sentry, Property 2: Invalid filename timestamps are rejected
@given(cam=camera_ids, ts=invalid_timestamps, ext=EXTENSIONS)
def test_invalid_calendar_timestamp_rejected(cam: str, ts: str, ext: str) -> None:
    """**Validates: Requirements 1.3**"""
    assert parse_filename(f"{cam}_{ts}.{ext}") is None


# Feature: nab-sentry, Property 2: Invalid filename timestamps are rejected
@given(left=camera_ids, right=camera_ids, parts=_valid_parts, ext=EXTENSIONS)
def test_underscore_in_camera_part_rejected(left: str, right: str, parts, ext: str) -> None:
    """**Validates: Requirements 1.3**"""
    assert parse_filename(f"{left}_{right}_{_ts(*parts)}.{ext}") is None


# Feature: nab-sentry, Property 2: Invalid filename timestamps are rejected
@given(
    cam=camera_ids,
    parts=_valid_parts,
    ext=EXTENSIONS,
    missing=st.sampled_from(["underscore", "T", "dot"]),
)
def test_missing_separator_rejected(cam: str, parts, ext: str, missing: str) -> None:
    """**Validates: Requirements 1.3**"""
    ts = _ts(*parts)
    if missing == "underscore":
        name = f"{cam}{ts}.{ext}"
    elif missing == "T":
        name = f"{cam}_{ts.replace('T', '')}.{ext}"
    else:
        name = f"{cam}_{ts}{ext}"
    assert parse_filename(name) is None
