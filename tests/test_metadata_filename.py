"""Property 1: Filename round trip."""

from __future__ import annotations

import string
from datetime import datetime

from hypothesis import given, settings
from hypothesis import strategies as st

from nab_sentry.ingest.metadata import format_filename, parse_filename
from tests.strategies import camera_ids, naive_datetimes

# Extensions: 1-10 ASCII letters/digits (FILENAME_RE ext group).
extensions = st.text(alphabet=string.ascii_letters + string.digits, min_size=1, max_size=10)


# Feature: nab-sentry, Property 1: Filename round trip
# **Validates: Requirements 1.3, 1.7**
@settings(max_examples=200)
@given(cam=camera_ids, dt=naive_datetimes, ext=extensions)
def test_filename_round_trip(cam: str, dt: datetime, ext: str) -> None:
    name = format_filename(cam, dt, ext)
    parsed = parse_filename(name)
    assert parsed is not None, name
    parsed_cam, parsed_dt = parsed
    assert parsed_cam == cam
    assert parsed_dt == dt
    assert parsed_dt.tzinfo is None


def test_filename_round_trip_year_bounds() -> None:
    for dt in (datetime(1, 1, 1, 0, 0, 0), datetime(9999, 12, 31, 23, 59, 59)):
        assert parse_filename(format_filename("cam-01", dt, ".mp4")) == ("cam-01", dt)
