"""Property 3: Sidecar round trip.

**Validates: Requirements 1.8**
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from hypothesis import given, settings
from hypothesis import strategies as st

from nab_sentry.ingest.metadata import Sidecar, parse_sidecar_text, serialize_sidecar
from tests.strategies import camera_ids, datetimes_any, labels

# Beyond the shared whole-second / whole-minute strategies: microsecond times and UTC
# offsets with a seconds component. Sub-second offsets are excluded: ISO 8601 has none, and
# CPython 3.12's fromisoformat parses "+00:00:00.000001" back as UTC.
_fine_offsets = st.integers(min_value=-(24 * 3600 - 1), max_value=24 * 3600 - 1).map(
    lambda secs: timezone(timedelta(seconds=secs))
)
_fine_datetimes = st.one_of(
    st.datetimes(min_value=datetime(1, 1, 1), max_value=datetime(9999, 12, 31, 23, 59, 59, 999999)),
    st.builds(
        lambda dt, tz: dt.replace(tzinfo=tz),
        st.datetimes(min_value=datetime(1, 1, 2), max_value=datetime(9999, 12, 30, 23, 59, 59, 999999)),
        _fine_offsets,
    ),
)

sidecars = st.builds(
    Sidecar,
    camera_id=camera_ids,
    label=labels,
    start_time=st.one_of(datetimes_any, _fine_datetimes),
)


def _assert_same_start(got: datetime, want: datetime) -> None:
    # Aware `==` compares instants only, so check wall-clock fields and offset separately.
    assert got.replace(tzinfo=None) == want.replace(tzinfo=None)
    assert (got.tzinfo is None) == (want.tzinfo is None)
    assert got.utcoffset() == want.utcoffset()


@settings(max_examples=300)
@given(sidecars)
def test_sidecar_round_trip(s: Sidecar) -> None:
    parsed = parse_sidecar_text(serialize_sidecar(s))
    assert parsed.errors == []
    assert parsed.sidecar is not None
    assert parsed.sidecar.camera_id == s.camera_id
    assert parsed.sidecar.label == s.label
    _assert_same_start(parsed.sidecar.start_time, s.start_time)


def test_sidecar_round_trip_examples() -> None:
    cases = [
        Sidecar("CAM-01", "Front Door", datetime(2024, 3, 5, 14, 30, 0)),
        Sidecar("a", " x ", datetime(1, 1, 1, 0, 0, 0)),
        Sidecar("Z" * 64, "é" * 128, datetime(9999, 12, 31, 23, 59, 59, 999999)),
        Sidecar("cam", "utc", datetime(2024, 1, 1, tzinfo=timezone.utc)),
        Sidecar("cam", "neg", datetime(2024, 1, 1, 8, tzinfo=timezone(timedelta(hours=-5, minutes=-30)))),
        Sidecar("cam", "secs", datetime(2024, 1, 1, tzinfo=timezone(timedelta(seconds=3723)))),
    ]
    for s in cases:
        parsed = parse_sidecar_text(serialize_sidecar(s))
        assert parsed.errors == [], s
        assert parsed.sidecar.camera_id == s.camera_id
        assert parsed.sidecar.label == s.label
        _assert_same_start(parsed.sidecar.start_time, s.start_time)
