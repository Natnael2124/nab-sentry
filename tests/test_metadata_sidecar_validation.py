"""Property 4: Sidecar validation reports exactly the invalid fields.

**Validates: Requirements 1.2, 1.6**
"""

from __future__ import annotations

import json
import string

from hypothesis import assume, given
from hypothesis import strategies as st

from nab_sentry.ingest.metadata import SIDECAR_FIELDS, parse_sidecar_text
from tests.strategies import (
    CAMERA_ID_ALPHABET,
    CAMERA_ID_MAX_LEN,
    LABEL_MAX_LEN,
    camera_ids,
    datetimes_any,
    labels,
)

# Sentinel meaning "remove the key from the object".
REMOVE = object()

_wrong_types = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.lists(st.integers(), max_size=3),
    st.dictionaries(st.text(max_size=3), st.integers(), max_size=2),
)

_forbidden_cam_char = st.characters(
    blacklist_characters=CAMERA_ID_ALPHABET, blacklist_categories=("Cs",)
)

invalid_camera_ids = st.one_of(
    st.just(REMOVE),
    _wrong_types,
    st.just(""),
    st.text(alphabet=CAMERA_ID_ALPHABET, min_size=CAMERA_ID_MAX_LEN + 1, max_size=CAMERA_ID_MAX_LEN + 20),
    # A valid-looking ID with at least one forbidden character spliced in.
    st.builds(
        lambda a, c, b: a + c + b,
        st.text(alphabet=CAMERA_ID_ALPHABET, max_size=20),
        _forbidden_cam_char,
        st.text(alphabet=CAMERA_ID_ALPHABET, max_size=20),
    ),
)

invalid_labels = st.one_of(
    st.just(REMOVE),
    _wrong_types,
    st.just(""),
    st.text(alphabet=" \t\r\n\u00a0\u2003", min_size=1, max_size=LABEL_MAX_LEN),  # whitespace only
    st.text(alphabet=string.ascii_letters, min_size=LABEL_MAX_LEN + 1, max_size=LABEL_MAX_LEN + 40),
)

invalid_start_times = st.one_of(
    st.just(REMOVE),
    _wrong_types,
    st.just(""),
    # Date-only ISO strings are rejected (must be a date-time).
    datetimes_any.map(lambda dt: dt.date().isoformat()),
    # Letters only: never an ISO date-time even if it contains "T" or a space.
    st.text(alphabet=string.ascii_letters + " ", min_size=1, max_size=30),
    st.sampled_from(
        ["2024-13-01T00:00:00", "2024-02-30T12:00:00", "2024-01-01T25:00:00", "yesterday at noon"]
    ),
)

_invalid_by_field = {
    "camera_id": invalid_camera_ids,
    "label": invalid_labels,
    "start_time": invalid_start_times,
}


@st.composite
def corrupted_sidecars(draw):
    obj = {
        "camera_id": draw(camera_ids),
        "label": draw(labels),
        "start_time": draw(datetimes_any).isoformat(),
    }
    corrupted = draw(
        st.sets(st.sampled_from(SIDECAR_FIELDS), min_size=1, max_size=len(SIDECAR_FIELDS))
    )
    for name in corrupted:
        value = draw(_invalid_by_field[name])
        if value is REMOVE:
            del obj[name]
        else:
            obj[name] = value
    return obj, corrupted


@given(corrupted_sidecars())
def test_corrupted_fields_are_reported_exactly(case):
    obj, corrupted = case
    result = parse_sidecar_text(json.dumps(obj))
    assert result.sidecar is None
    assert set(result.errors) == corrupted
    # Reported in the fixed field order, each once.
    assert result.errors == [f for f in SIDECAR_FIELDS if f in corrupted]


_json_non_objects = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats(allow_nan=False) | st.text(),
    lambda children: st.lists(children, max_size=4),
    max_leaves=10,
)


def _is_json_object(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except (ValueError, RecursionError):
        return False


@given(
    st.one_of(
        _json_non_objects.map(json.dumps),
        st.text(),
        st.sampled_from(["", "{", '{"camera_id": "a"', "[" * 100_000, "{'label': 'x'}", "NaN-ish"]),
    )
)
def test_non_object_text_is_unparseable(text):
    assume(not _is_json_object(text))
    result = parse_sidecar_text(text)
    assert result.sidecar is None
    assert result.errors == ["unparseable"]


@given(camera_ids, labels, datetimes_any)
def test_valid_sidecar_has_no_errors(cam, label, start):
    # Control case: the uncorrupted base object is accepted.
    obj = {"camera_id": cam, "label": label, "start_time": start.isoformat()}
    result = parse_sidecar_text(json.dumps(obj))
    assert result.errors == []
    assert result.sidecar is not None
    assert (result.sidecar.camera_id, result.sidecar.label, result.sidecar.start_time) == (
        cam,
        label,
        start,
    )
