"""Sanity tests for nab_sentry.api.media (properties live in 14.4/14.5/14.10)."""

from __future__ import annotations

import os

import pytest

from nab_sentry.api.media import (
    FullBody,
    PartialBody,
    Unsatisfiable,
    iter_file,
    media_response_plan,
    parse_range,
    resolve_playback,
    resolve_thumb,
)


@pytest.mark.parametrize(
    "header,size,expected",
    [
        (None, 100, FullBody()),
        ("bytes=0-9", 100, PartialBody(0, 9)),
        ("bytes=90-500", 100, PartialBody(90, 99)),
        ("bytes=10-", 100, PartialBody(10, 99)),
        ("bytes=-10", 100, PartialBody(90, 99)),
        ("bytes=-500", 100, PartialBody(0, 99)),
        ("bytes=100-", 100, Unsatisfiable()),
        ("bytes=100-200", 100, Unsatisfiable()),
        ("bytes=-0", 100, Unsatisfiable()),
        ("bytes=-", 100, FullBody()),
        ("bytes=9-3", 100, FullBody()),
        ("bytes=0-1,5-6", 100, FullBody()),
        ("bytes=0-1\n", 100, FullBody()),
        ("items=0-1", 100, FullBody()),
        ("bytes=\u0661-2", 100, FullBody()),  # non-ASCII digit
        ("bytes=0-0", 0, Unsatisfiable()),
        ("bytes=-5", 0, Unsatisfiable()),
        (None, 0, FullBody()),
    ],
)
def test_parse_range_table(header, size, expected):
    assert parse_range(header, size) == expected


def test_response_plans():
    p = media_response_plan(PartialBody(2, 5), 10)
    assert p.status == 206
    assert p.headers["Content-Range"] == "bytes 2-5/10"
    assert p.headers["Content-Length"] == "4"
    assert p.headers["Accept-Ranges"] == "bytes"
    u = media_response_plan(Unsatisfiable(), 10)
    assert u.status == 416 and u.headers["Content-Range"] == "bytes */10"
    assert u.start is None
    f = media_response_plan(FullBody(), 10)
    assert f.status == 200 and f.headers["Content-Length"] == "10"
    assert (f.start, f.end) == (0, 9)
    assert media_response_plan(FullBody(), 0).start is None


def test_iter_file_ranges(tmp_path):
    data = os.urandom(1000)
    p = tmp_path / "f.bin"
    p.write_bytes(data)
    assert b"".join(iter_file(p, 0, 999, chunk=64)) == data
    assert b"".join(iter_file(p, 10, 20, chunk=3)) == data[10:21]
    assert all(len(c) <= 7 for c in iter_file(p, 0, 999, chunk=7))
    assert b"".join(iter_file(p, 990, 5000)) == data[990:]


def test_resolve_thumb(tmp_path):
    thumbs = tmp_path / "thumbs"
    thumbs.mkdir()
    (thumbs / "v12_f0000034.jpg").write_bytes(b"jpg")
    (tmp_path / "secret.jpg").write_bytes(b"secret")
    assert resolve_thumb(thumbs, "v12_f0000034.jpg") == (thumbs / "v12_f0000034.jpg").resolve()
    for bad in [
        "missing.jpg",
        "../secret.jpg",
        "..\\secret.jpg",
        "%2e%2e%2fsecret.jpg",
        "C:secret.jpg",
        str(tmp_path / "secret.jpg"),
        "a.jpg\x00",
        "a.jpg\n",
        "v12_f0000034.png",
        "",
    ]:
        assert resolve_thumb(thumbs, bad) is None, bad
    (thumbs / "dir.jpg").mkdir()
    assert resolve_thumb(thumbs, "dir.jpg") is None


def test_resolve_thumb_symlink_escape(tmp_path):
    thumbs = tmp_path / "thumbs"
    thumbs.mkdir()
    outside = tmp_path / "secret.jpg"
    outside.write_bytes(b"secret")
    try:
        (thumbs / "link.jpg").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted on this system")
    assert resolve_thumb(thumbs, "link.jpg") is None


def test_resolve_playback(tmp_path):
    pb = tmp_path / "playback"
    pb.mkdir()
    (pb / "v12.mp4").write_bytes(b"mp4")
    assert resolve_playback(pb, "v12.mp4") == (pb / "v12.mp4").resolve()
    for bad in [None, "v13.mp4", "../v12.mp4", "x.mp4", "v12.mp4/", "V12.mp4", "v.mp4"]:
        assert resolve_playback(pb, bad) is None, bad
