"""Property 5: Metadata source precedence.

For any video file name and optional sidecar in a temporary directory: if the sidecar is
valid, ``resolve_metadata`` returns the sidecar values with ``source="sidecar"`` regardless
of the file name; otherwise, if the name matches the Filename_Convention, it returns the
name's camera ID as both ID and label with ``source="filename"``; otherwise it returns
``None``.

**Validates: Requirements 1.2, 1.3, 1.4, 1.5, 1.6**
"""

from __future__ import annotations

import json
import logging
import string
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from nab_sentry.ingest.metadata import (
    Sidecar,
    format_filename,
    resolve_metadata,
    serialize_sidecar,
    to_aware_local,
)
from tests.strategies import camera_ids, datetimes_any, labels, naive_datetimes

# --------------------------------------------------------------------------- helpers


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _fresh_logger() -> tuple[logging.Logger, _ListHandler]:
    log = logging.getLogger("tests.metadata_resolve")
    for h in list(log.handlers):
        log.removeHandler(h)
    handler = _ListHandler()
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    return log, handler


# Video extensions: ASCII alphanumerics, never "json" in any case (the sidecar path would
# equal the video path, and Windows is case-insensitive).
extensions = st.sampled_from(["mp4", "MP4", "avi", "mkv", "mov", "ts", "m4v", "h264", "3gp"])


# --------------------------------------------------------------------------- video names


@dataclass(frozen=True)
class VideoName:
    name: str
    # (camera_id, naive start) when the name matches the Filename_Convention, else None.
    expected: tuple[str, datetime] | None


matching_names = st.builds(
    lambda cam, dt, ext: VideoName(format_filename(cam, dt, ext), (cam, dt)),
    camera_ids,
    naive_datetimes,
    extensions,
)

_bad_timestamps = st.sampled_from(
    [
        "00000101T000000",  # year 0000
        "20200001T000000",  # month 00
        "20201301T000000",  # month 13
        "20200100T000000",  # day 00
        "20200132T000000",  # day 32
        "20210230T000000",  # Feb 30
        "20210229T000000",  # Feb 29 in a non-leap year
        "20200101T240000",  # hour 24
        "20200101T006000",  # minute 60
        "20200101T000060",  # second 60
    ]
)

_safe_stem_chars = string.ascii_letters + string.digits + "-_"


def _cam_with_bad_char(cam: str, bad: str, pos: int) -> str:
    pos = pos % (len(cam) + 1)
    return cam[:pos] + bad + cam[pos:]


non_matching_names = st.one_of(
    # Real-calendar violation in the timestamp.
    st.builds(lambda c, ts, e: f"{c}_{ts}.{e}", camera_ids, _bad_timestamps, extensions),
    # Camera ID longer than 64 characters.
    st.builds(
        lambda c, dt, e: format_filename(c, dt, e),
        st.text(alphabet=string.ascii_letters + string.digits + "-", min_size=65, max_size=80),
        naive_datetimes,
        extensions,
    ),
    # Camera ID containing a character outside letters, digits, hyphens.
    st.builds(
        lambda c, bad, pos, dt, e: format_filename(_cam_with_bad_char(c, bad, pos), dt, e),
        camera_ids.filter(lambda c: len(c) <= 60),
        st.sampled_from(["_", " ", "+", "~", "é", "!"]),
        st.integers(min_value=0, max_value=64),
        naive_datetimes,
        extensions,
    ),
    # Empty camera ID.
    st.builds(lambda dt, e: format_filename("", dt, e), naive_datetimes, extensions),
    # No file extension.
    st.builds(lambda c, dt: format_filename(c, dt, "x")[:-2], camera_ids, naive_datetimes),
    # Timestamp with the wrong shape (no "T", too short, or a date-only stamp).
    st.builds(
        lambda c, dt, e, kind: f"{c}_"
        + {
            "noT": f"{dt.year:04d}{dt.month:02d}{dt.day:02d}{dt.hour:02d}{dt.minute:02d}{dt.second:02d}",
            "date": f"{dt.year:04d}{dt.month:02d}{dt.day:02d}",
            "short": f"{dt.year:04d}{dt.month:02d}{dt.day:02d}T{dt.hour:02d}{dt.minute:02d}",
        }[kind]
        + f".{e}",
        camera_ids,
        naive_datetimes,
        extensions,
        st.sampled_from(["noT", "date", "short"]),
    ),
    # Arbitrary names with no underscore-timestamp part at all.
    st.builds(
        lambda stem, e: f"video-{stem}.{e}",
        st.text(alphabet=string.ascii_letters + string.digits + "-", min_size=1, max_size=30),
        extensions,
    ),
    # Valid name with trailing junk after the extension separator.
    st.builds(
        lambda c, dt, e: format_filename(c, dt, e + "_x"), camera_ids, naive_datetimes, extensions
    ),
).map(lambda n: VideoName(n, None))

video_names = st.one_of(matching_names, non_matching_names)


# --------------------------------------------------------------------------- sidecars


@dataclass(frozen=True)
class SidecarCase:
    kind: str  # "none" | "valid" | "unparseable" | "fields"
    text: str | None = None
    bom: bool = False
    sidecar: Sidecar | None = None  # for "valid"
    bad_fields: frozenset[str] = frozenset()  # for "fields"


no_sidecar = st.just(SidecarCase("none"))

valid_sidecars = st.builds(
    lambda s, bom: SidecarCase("valid", serialize_sidecar(s), bom, sidecar=s),
    st.builds(Sidecar, camera_ids, labels, datetimes_any),
    st.booleans(),
)

unparseable_sidecars = st.builds(
    lambda t, bom: SidecarCase("unparseable", t, bom),
    st.sampled_from(["", "not json", "{", '{"camera_id": "A"', "[]", "[1, 2]", '"text"', "42", "null", "true"]),
    st.booleans(),
)

_bad_values = {
    "camera_id": st.sampled_from(
        ["", "a_b", "has space", "é", "x" * 65, 7, None, ["CAM"], {"id": "CAM"}, True]
    ),
    "label": st.sampled_from(["", " ", "   \t ", "x" * 129, 3, None, ["L"], False]),
    "start_time": st.sampled_from(
        ["", "2020-01-01", "garbage", "2020-13-01T00:00:00", "2021-02-30T10:00:00",
         "2020-01-01T25:00:00", 20200101, None, ["2020-01-01T00:00:00"]]
    ),
}
_MISSING = object()


@st.composite
def field_invalid_sidecars(draw: st.DrawFn) -> SidecarCase:
    base = {
        "camera_id": draw(camera_ids),
        "label": draw(labels),
        "start_time": draw(datetimes_any).isoformat(),
    }
    bad = draw(
        st.sets(st.sampled_from(sorted(base)), min_size=1, max_size=3).map(frozenset)
    )
    obj: dict[str, object] = dict(base)
    for f in bad:
        value = draw(st.one_of(st.just(_MISSING), _bad_values[f]))
        if value is _MISSING:
            del obj[f]
        else:
            obj[f] = value
    return SidecarCase("fields", json.dumps(obj), draw(st.booleans()), bad_fields=bad)


sidecar_cases = st.one_of(
    no_sidecar, valid_sidecars, unparseable_sidecars, field_invalid_sidecars()
)


# --------------------------------------------------------------------------- property


@given(video=video_names, case=sidecar_cases)
def test_metadata_source_precedence(video: VideoName, case: SidecarCase) -> None:
    log, handler = _fresh_logger()
    with tempfile.TemporaryDirectory() as tmp:
        video_path = Path(tmp) / video.name
        video_path.write_bytes(b"\x00not-really-a-video")
        sidecar_path = video_path.with_suffix(".json")
        assert sidecar_path != video_path
        if case.text is not None:
            sidecar_path.write_text(case.text, encoding="utf-8-sig" if case.bom else "utf-8")

        result = resolve_metadata(video_path, log)

    messages = [r.getMessage() for r in handler.records]
    invalid_msgs = [m for m in messages if m.startswith("invalid sidecar:")]
    unresolved_msgs = [m for m in messages if m.startswith("unresolved camera metadata:")]

    if case.kind == "valid":
        # 1.2 / 1.4: a valid sidecar wins regardless of the file name.
        s = case.sidecar
        assert s is not None and result is not None
        assert result.source == "sidecar"
        assert result.camera_id == s.camera_id
        assert result.label == s.label
        assert result.start_time.tzinfo is not None
        assert result.start_time.replace(tzinfo=None) == s.start_time.replace(tzinfo=None)
        if s.start_time.tzinfo is not None:
            assert result.start_time.utcoffset() == s.start_time.utcoffset()
        else:
            assert result.start_time == to_aware_local(s.start_time)
        assert invalid_msgs == [] and unresolved_msgs == []
        return

    # 1.6: a present but invalid sidecar is logged with its path and the reason.
    if case.kind == "none":
        assert invalid_msgs == []
    else:
        assert len(invalid_msgs) == 1, messages
        prefix = f"invalid sidecar: {sidecar_path}: "
        assert invalid_msgs[0].startswith(prefix), invalid_msgs[0]
        detail = invalid_msgs[0][len(prefix):]
        if case.kind == "unparseable":
            assert "unparseable" in detail
        else:
            for f in ("camera_id", "label", "start_time"):
                assert (f in detail) == (f in case.bad_fields), (detail, case.bad_fields)

    if video.expected is not None:
        # 1.3: fall back to the file name; camera ID doubles as the label.
        cam, start = video.expected
        assert result is not None
        assert result.source == "filename"
        assert result.camera_id == cam
        assert result.label == cam
        assert result.start_time.tzinfo is not None
        assert result.start_time.replace(tzinfo=None) == start
        assert result.start_time == to_aware_local(start)
        assert unresolved_msgs == []
    else:
        # 1.5: unresolved -> None, logged with the video path.
        assert result is None
        assert unresolved_msgs == [f"unresolved camera metadata: {video_path}"]
