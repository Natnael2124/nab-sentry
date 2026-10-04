"""Example-based sanity tests for nab_sentry.ingest.metadata (Requirements 1.2-1.8, 1.12, 7.2)."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone

from nab_sentry.ingest.metadata import (
    Sidecar,
    format_filename,
    parse_filename,
    parse_sidecar_text,
    resolve_metadata,
    serialize_sidecar,
    src_hash,
    to_aware_local,
)

LOG = logging.getLogger("nab_sentry.test.metadata")


def test_filename_format_and_parse_year_one():
    start = datetime(1, 2, 3, 4, 5, 6)
    name = format_filename("CAM-01", start, "mp4")
    assert name == "CAM-01_00010203T040506.mp4"
    assert parse_filename(name) == ("CAM-01", start)


def test_parse_filename_rejects_bad_values():
    assert parse_filename("CAM01_20250230T080000.mp4") is None  # Feb 30
    assert parse_filename("CAM01_20251301T080000.mp4") is None  # month 13
    assert parse_filename("CAM01_20250101T240000.mp4") is None  # hour 24
    assert parse_filename("CAM_01_20250101T080000.mp4") is None  # underscore in camera id
    assert parse_filename("CAM01_20250101T080000.mp4\n") is None
    assert parse_filename("CAM01-20250101T080000.mp4") is None


def test_sidecar_roundtrip_keeps_offset_or_naivety():
    for st in (datetime(2025, 1, 1, 8), datetime(2025, 1, 1, 8, tzinfo=timezone(timedelta(hours=5, minutes=30)))):
        s = Sidecar("CAM01", "Front door", st)
        p = parse_sidecar_text(serialize_sidecar(s))
        assert p.errors == []
        assert p.sidecar == s
        assert p.sidecar.start_time.utcoffset() == st.utcoffset()


def test_sidecar_collects_every_invalid_field():
    bad = json.dumps({"camera_id": "CAM 01", "label": "   ", "start_time": "2025-01-01"})
    p = parse_sidecar_text(bad)
    assert p.sidecar is None
    assert set(p.errors) == {"camera_id", "label", "start_time"}
    assert parse_sidecar_text(json.dumps({"label": "x" * 129, "camera_id": "A", "start_time": 5})).errors == [
        "label",
        "start_time",
    ]
    assert parse_sidecar_text("[1, 2]").errors == ["unparseable"]
    assert parse_sidecar_text("{not json").errors == ["unparseable"]


def test_resolve_prefers_sidecar_then_filename_then_none(tmp_path, caplog):
    video = tmp_path / "CAM01_20250101T080000.mp4"
    video.write_bytes(b"x")
    sidecar = tmp_path / "CAM01_20250101T080000.json"
    sidecar.write_text(serialize_sidecar(Sidecar("SIDE-1", "Lobby", datetime(2024, 6, 1, 12, tzinfo=timezone.utc))))

    r = resolve_metadata(video, LOG)
    assert (r.camera_id, r.label, r.source) == ("SIDE-1", "Lobby", "sidecar")
    assert r.start_time == datetime(2024, 6, 1, 12, tzinfo=timezone.utc)

    sidecar.write_text('{"camera_id": "bad id"}')
    with caplog.at_level(logging.WARNING, logger=LOG.name):
        r = resolve_metadata(video, LOG)
    assert (r.camera_id, r.label, r.source) == ("CAM01", "CAM01", "filename")
    assert r.start_time.tzinfo is not None
    assert "invalid sidecar" in caplog.text and "camera_id" in caplog.text and "label" in caplog.text

    other = tmp_path / "random.mp4"
    other.write_bytes(b"x")
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=LOG.name):
        assert resolve_metadata(other, LOG) is None
    assert "unresolved camera metadata" in caplog.text and str(other) in caplog.text


def test_to_aware_local():
    aware = datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert to_aware_local(aware) is aware
    naive = datetime(2025, 1, 1, 8)
    local = to_aware_local(naive)
    assert local.tzinfo is not None and local.replace(tzinfo=None) == naive


def test_src_hash_is_content_addressed(tmp_path):
    data = b"abc" * 1_000_000  # spans multiple 1 MiB chunks
    a, b = tmp_path / "a.mp4", tmp_path / "b.bin"
    a.write_bytes(data)
    b.write_bytes(data)
    assert src_hash(a) == src_hash(b) == hashlib.sha256(data).hexdigest()
    b.write_bytes(data + b"!")
    assert src_hash(a) != src_hash(b)
