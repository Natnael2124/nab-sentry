"""Shared Hypothesis strategies. Later tasks add sidecars, frames, YOLO tensors, hits, filters, etc."""

from __future__ import annotations

import string
from datetime import datetime, timedelta, timezone

from hypothesis import strategies as st

CAMERA_ID_ALPHABET = string.ascii_letters + string.digits + "-"
CAMERA_ID_MAX_LEN = 64
LABEL_MAX_LEN = 128

# Valid camera IDs: 1-64 ASCII letters, digits, and hyphens (Requirement 1.2, 1.3).
camera_ids = st.text(alphabet=CAMERA_ID_ALPHABET, min_size=1, max_size=CAMERA_ID_MAX_LEN)

# Valid labels: 1-128 characters, not only whitespace (Requirement 1.2).
# Letters, numbers, punctuation, symbols, and spaces; excludes control characters and surrogates.
_label_chars = st.characters(categories=("L", "N", "P", "S", "Zs"))
labels = st.text(alphabet=_label_chars, min_size=1, max_size=LABEL_MAX_LEN).filter(
    lambda s: s.strip() != ""
)

# Whole-second datetimes in years 0001-9999 (Requirement 1.7).
naive_datetimes = st.datetimes(
    min_value=datetime(1, 1, 1), max_value=datetime(9999, 12, 31, 23, 59, 59)
).map(lambda dt: dt.replace(microsecond=0))

# Fixed UTC offsets in whole minutes within ISO 8601's +/-23:59 range.
utc_offsets = st.integers(min_value=-(23 * 60 + 59), max_value=23 * 60 + 59).map(
    lambda minutes: timezone(timedelta(minutes=minutes))
)

# Aware datetimes kept a day away from the year bounds so UTC conversion cannot overflow.
aware_datetimes = st.builds(
    lambda dt, tz: dt.replace(tzinfo=tz),
    st.datetimes(
        min_value=datetime(1, 1, 2), max_value=datetime(9999, 12, 30, 23, 59, 59)
    ).map(lambda dt: dt.replace(microsecond=0)),
    utc_offsets,
)

# Either kind; useful where both naive and aware start times are accepted.
datetimes_any = st.one_of(naive_datetimes, aware_datetimes)
