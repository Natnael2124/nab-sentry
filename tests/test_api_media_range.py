"""Property tests for HTTP Range serving of ``GET /media/video/{id}`` (task 14.10).

Property 44: Range reassembly round trip.
Also covers the HTTP part of Property 43 (status, Content-Range, Content-Length,
Accept-Ranges and body bytes) against an independent reference model.
"""

from __future__ import annotations

import re
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nab_sentry.api.app import Services, create_app
from nab_sentry.config import Config
from nab_sentry.querylog import QueryLog
from nab_sentry.store.db import MetadataStore, NewVideo
from nab_sentry.store.vector_index import VectorIndex

T0 = datetime(2025, 1, 1, 8, 0, tzinfo=timezone(timedelta(hours=1)))
MAX_SIZE = 65_536

PBT = settings(max_examples=60, deadline=None,
               suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])


@contextmanager
def _video_client(data: bytes) -> Iterator[tuple[TestClient, int]]:
    """Fresh temp root, metadata store with one finalized video whose
    Playback_File holds ``data``, and a TestClient over ``create_app``."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cfg = Config(root=root)
        cfg.thumbs_dir.mkdir(parents=True)
        cfg.playback_dir.mkdir(parents=True)
        db = MetadataStore(":memory:")
        try:
            db.upsert_camera("CAM01", "Reception")
            vid = db.insert_video(NewVideo("CAM01", "a.mp4", "h1", T0, 10.0, 16, 12, 60.0))
            db.finalize_video(vid, sampled=1, passed=1, duration_s=60.0, playback_path=f"v{vid}.mp4")
            (cfg.playback_dir / f"v{vid}.mp4").write_bytes(data)
            svc = Services(cfg, db, VectorIndex(), None, False, None, QueryLog(cfg.query_log_path))
            web = root / "web"
            web.mkdir()
            with TestClient(create_app(svc, web_dir=web)) as client:
                yield client, vid
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Property 44: Range reassembly round trip
# ---------------------------------------------------------------------------

@st.composite
def file_and_split(draw):
    """Random file of 1..65,536 bytes plus a split of [0, size-1] into
    consecutive non-overlapping inclusive ranges."""
    size = draw(st.integers(min_value=1, max_value=MAX_SIZE))
    data = draw(st.binary(min_size=size, max_size=size))
    n_cuts = draw(st.integers(min_value=0, max_value=min(size - 1, 12)))
    cuts = sorted(draw(st.sets(st.integers(min_value=1, max_value=size - 1),
                               min_size=n_cuts, max_size=n_cuts))) if n_cuts else []
    bounds = [0, *cuts, size]
    ranges = [(bounds[i], bounds[i + 1] - 1) for i in range(len(bounds) - 1)]
    # Optionally express the last range open-ended (bytes=a-) to exercise 11.3.
    open_last = draw(st.booleans())
    return data, ranges, open_last


@PBT
@given(file_and_split())
def test_range_reassembly_round_trip(case):
    """Feature: nab-sentry, Property 44: Range reassembly round trip

    **Validates: Requirements 11.1, 11.2, 11.3, 11.4**
    """
    data, ranges, open_last = case
    size = len(data)
    with _video_client(data) as (client, vid):
        parts = []
        for i, (a, e) in enumerate(ranges):
            last = i == len(ranges) - 1
            header = f"bytes={a}-" if (last and open_last) else f"bytes={a}-{e}"
            r = client.get(f"/media/video/{vid}", headers={"Range": header})
            assert r.status_code == 206, header
            assert r.headers["content-range"] == f"bytes {a}-{e}/{size}"
            assert r.headers["content-length"] == str(e - a + 1)
            assert r.headers["content-type"] == "video/mp4"
            assert len(r.content) == e - a + 1
            parts.append(r.content)
        assert b"".join(parts) == data


# ---------------------------------------------------------------------------
# Property 43 (HTTP part): response matches an independent Requirement 11 model
# ---------------------------------------------------------------------------

_REF_RE = re.compile(r"bytes=([0-9]*)-([0-9]*)")


def _reference(header: str | None, size: int) -> tuple[int, int | None, int | None]:
    """Requirement 11 rules -> (status, start, end_inclusive)."""
    if header is None or "," in header:
        return 200, None, None
    m = _REF_RE.fullmatch(header)
    if m is None or (m.group(1) == "" and m.group(2) == ""):
        return 200, None, None
    first, last = m.group(1), m.group(2)
    if first == "":
        n = int(last)
        if n == 0:
            return 416, None, None
        return 206, max(0, size - n), size - 1
    a = int(first)
    if last != "" and a > int(last):
        return 200, None, None
    if a >= size:
        return 416, None, None
    e = size - 1 if last == "" else min(int(last), size - 1)
    return 206, a, e


def _num(size: int) -> st.SearchStrategy[int]:
    return st.one_of(st.integers(0, size + 2), st.integers(0, 10**9),
                     st.sampled_from([0, size - 1, size, size + 1]).filter(lambda x: x >= 0))


# Arbitrary printable-ASCII header text without leading/trailing whitespace
# (HTTP header values cannot carry those through the client unchanged).
_ARBITRARY = st.text(alphabet=st.characters(min_codepoint=0x21, max_codepoint=0x7E),
                     max_size=30)


@st.composite
def file_and_header(draw):
    size = draw(st.integers(min_value=1, max_value=MAX_SIZE))
    data = draw(st.binary(min_size=size, max_size=size))
    n = _num(size)
    header = draw(st.one_of(
        st.none(),
        st.tuples(n, n).map(lambda t: f"bytes={t[0]}-{t[1]}"),             # single / inverted
        n.map(lambda a: f"bytes={a}-"),                                     # open-ended
        n.map(lambda k: f"bytes=-{k}"),                                     # suffix (incl. -0)
        st.just("bytes=-"),
        st.lists(st.tuples(n, n), min_size=2, max_size=3).map(              # multi-range
            lambda rs: "bytes=" + ",".join(f"{a}-{b}" for a, b in rs)),
        st.tuples(n, n).map(lambda t: f"items={t[0]}-{t[1]}"),              # wrong unit
        _ARBITRARY,
    ))
    return data, header


@PBT
@given(file_and_header())
def test_range_http_responses_match_model(case):
    """Feature: nab-sentry, Property 43: Range header model (HTTP response part)

    **Validates: Requirements 11.1, 11.2, 11.3, 11.5, 11.9, 11.10**
    """
    data, header = case
    size = len(data)
    status, a, e = _reference(header, size)
    with _video_client(data) as (client, vid):
        headers = {} if header is None else {"Range": header}
        r = client.get(f"/media/video/{vid}", headers=headers)

    assert r.status_code == status, header
    assert r.headers["accept-ranges"] == "bytes"
    assert r.headers["content-length"] == str(len(r.content))
    if status == 200:
        assert r.content == data
        assert "content-range" not in r.headers
        assert r.headers["content-type"] == "video/mp4"
    elif status == 206:
        assert r.content == data[a:e + 1]
        assert r.headers["content-range"] == f"bytes {a}-{e}/{size}"
        assert r.headers["content-type"] == "video/mp4"
    else:
        assert r.content == b""
        assert r.headers["content-range"] == f"bytes */{size}"
