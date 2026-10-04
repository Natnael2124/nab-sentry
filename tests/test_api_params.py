"""Property 41: Search parameter validation (plus the API half of Property 21, blank ``q``).

Feature: nab-sentry, Property 41 / Property 21
**Validates: Requirements 10.7, 10.8, 10.9**

Length of ``q`` (10.7): the requirement says ``q`` is invalid if it is "missing, empty, contains
only whitespace, or is longer than 256 characters". The length bound therefore applies to the raw
parameter value (not the stripped text): a value of 257+ characters is rejected even when its
non-whitespace core is short, and a value of <= 256 characters with surrounding whitespace is
accepted. The Console trims before sending (12.3), so the API never needs to trim for length.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from hypothesis import given
from hypothesis import strategies as st

from nab_sentry.api.app import (
    PARAM_ORDER,
    ParamError,
    SearchRequest,
    Services,
    create_app,
    validate_search_params,
)
from nab_sentry.config import Config
from nab_sentry.querylog import QueryLog
from nab_sentry.search.engine import VALID_CLASSES, SearchEngine
from nab_sentry.store.db import MetadataStore
from nab_sentry.store.vector_index import VectorIndex
from tests.fakes import FakeEncoder

CFG = Config(root=Path("."))
MAX_LEN = int(CFG.max_query_len)  # 256
MAX_LIMIT = int(CFG.max_limit)  # 100
DEFAULT_LIMIT = int(CFG.default_limit)  # 20
CLASSES = sorted(VALID_CLASSES)

# Every character Python considers whitespace (str.isspace), across all of Unicode.
WHITESPACE = "".join(chr(c) for c in range(sys.maxunicode + 1) if chr(c).isspace())


# ---------------------------------------------------------------------------------------------
# Reference model
# ---------------------------------------------------------------------------------------------


def _model_time(raw: str) -> datetime | None:
    """Parsed aware datetime, or None if ``raw`` is not ISO 8601. Naive -> local time zone."""
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return dt if dt.tzinfo is not None else dt.astimezone()


def model(params: dict[str, str]) -> str | dict[str, Any]:
    """Name of the first invalid parameter, or the expected SearchRequest fields."""
    q = params.get("q")
    if q is None or len(q) == 0 or all(c.isspace() for c in q) or len(q) > MAX_LEN:
        return "q"
    start = end = None
    if "start" in params:
        start = _model_time(params["start"])
        if start is None:
            return "start"
    if "end" in params:
        end = _model_time(params["end"])
        if end is None:
            return "end"
    if start is not None and end is not None and start > end:
        return "end"
    if "cls" in params and params["cls"] not in VALID_CLASSES:
        return "cls"
    limit = DEFAULT_LIMIT
    if "limit" in params:
        raw = params["limit"]
        if not (raw.isascii() and raw.isdigit()) or not 1 <= int(raw) <= MAX_LIMIT:
            return "limit"
        limit = int(raw)
    return {
        "q": q,
        "camera": params.get("camera"),
        "start": start,
        "end": end,
        "cls": params.get("cls"),
        "limit": limit,
    }


# ---------------------------------------------------------------------------------------------
# Generators: each field is absent, valid, or invalid
# ---------------------------------------------------------------------------------------------

ws_text = st.text(alphabet=WHITESPACE, max_size=8)
core_text = st.text(min_size=1, max_size=40).filter(lambda s: not all(c.isspace() for c in s))


@st.composite
def valid_q(draw) -> str:
    q = draw(ws_text) + draw(core_text) + draw(ws_text)
    if draw(st.booleans()) and len(q) < MAX_LEN:  # sometimes pad exactly to the bound
        q = q + "a" * (MAX_LEN - len(q))
    return q[:MAX_LEN] if len(q) > MAX_LEN else q


@st.composite
def too_long_q(draw) -> str:
    core = draw(core_text)
    extra = draw(st.integers(min_value=1, max_value=20))
    pad_char = draw(st.sampled_from(["a", " ", "\u3000"]))  # whitespace padding still too long
    return core + pad_char * (MAX_LEN - len(core) + extra)


q_values = st.one_of(
    valid_q(),
    st.just(""),
    st.text(alphabet=WHITESPACE, min_size=1, max_size=20),
    too_long_q(),
)

_TZ = st.sampled_from(
    [None, timezone.utc, timezone(timedelta(hours=1)), timezone(timedelta(hours=-5, minutes=-30))]
)


@st.composite
def valid_time(draw) -> str:
    # 1971-2100 keeps naive -> local conversion inside the platform's supported range.
    dt = draw(st.datetimes(min_value=datetime(1971, 1, 2), max_value=datetime(2100, 1, 1)))
    tz = draw(_TZ)
    if tz is not None:
        dt = dt.replace(tzinfo=tz)
    spec = draw(st.sampled_from(["auto", "seconds", "milliseconds", "minutes"]))
    return dt.isoformat(timespec=spec)


INVALID_TIMES = ["", "x", "yesterday", "2025-13-01T00:00:00", "2025-02-30", "2025-01-01T25:00",
                 "01/02/2025", "2025-01-01T08:00:00+25:00", "now"]
time_values = st.one_of(valid_time(), st.sampled_from(INVALID_TIMES))

INVALID_CLASSES = ["", "dog", " person", "person ", *[c.upper() for c in CLASSES],
                   *[c.capitalize() for c in CLASSES if c.capitalize() != c]]
cls_values = st.one_of(st.sampled_from(CLASSES), st.sampled_from(INVALID_CLASSES))

INVALID_LIMITS = ["0", "000", "101", "999", "-1", "+5", "5.0", " 5", "5 ", "", "abc", "1e2",
                  "\uff15", "\u0665"]  # fullwidth / Arabic-Indic digits are not ASCII
valid_limit = st.integers(min_value=1, max_value=MAX_LIMIT).flatmap(
    lambda n: st.sampled_from([str(n), str(n).zfill(3), str(n).zfill(4)])
)
limit_values = st.one_of(valid_limit, st.sampled_from(INVALID_LIMITS))


@st.composite
def param_maps(draw, q_strategy=q_values) -> dict[str, str]:
    fields = {
        "q": q_strategy,
        "camera": st.text(max_size=12),
        "start": time_values,
        "end": time_values,
        "cls": cls_values,
        "limit": limit_values,
    }
    out: dict[str, str] = {}
    for key, strat in fields.items():
        if draw(st.integers(0, 4)) > 0:  # present ~80% of the time
            out[key] = draw(strat)
    return out


def _check_pure(params: dict[str, str]) -> str | dict[str, Any]:
    expected = model(params)
    got = validate_search_params(params, CFG)
    if isinstance(expected, str):
        assert isinstance(got, ParamError), (params, got)
        assert got.param == expected, (params, got)
        assert got.param in PARAM_ORDER and got.message
    else:
        assert isinstance(got, SearchRequest), (params, got)
        assert got.q == expected["q"]
        assert got.limit == expected["limit"]
        f = got.filter
        assert f.camera_id == expected["camera"]
        assert f.cls == expected["cls"]
        for name in ("start", "end"):
            value = getattr(f, name)
            if expected[name] is None:
                assert value is None
            else:
                assert value is not None and value.tzinfo is not None
                assert value == expected[name]
    return expected


# ---------------------------------------------------------------------------------------------
# Pure validator
# ---------------------------------------------------------------------------------------------


@given(param_maps())
def test_validate_search_params_matches_model(params):
    """First invalid parameter in order q, start, end, cls, limit; else the right SearchRequest."""
    _check_pure(params)


@st.composite
def valid_param_maps(draw) -> dict[str, str]:
    out: dict[str, str] = {"q": draw(valid_q())}
    for key, strat in (("camera", st.text(max_size=12)), ("cls", st.sampled_from(CLASSES)),
                       ("limit", valid_limit)):
        if draw(st.booleans()):
            out[key] = draw(strat)
    times = sorted(draw(st.lists(valid_time(), min_size=2, max_size=2)), key=_model_time)
    if draw(st.booleans()):
        out["start"] = times[0]
    if draw(st.booleans()):
        out["end"] = times[1]
    return out


@given(valid_param_maps())
def test_all_valid_maps_give_search_request(params):
    assert isinstance(_check_pure(params), dict)


@given(st.text(alphabet=WHITESPACE, max_size=30), param_maps())
def test_blank_q_rejected_pure(blank, others):
    params = {**others, "q": blank}
    got = validate_search_params(params, CFG)
    assert isinstance(got, ParamError) and got.param == "q"


def test_q_length_is_raw_length():
    assert isinstance(validate_search_params({"q": "a" * MAX_LEN}, CFG), SearchRequest)
    assert validate_search_params({"q": "a" * (MAX_LEN + 1)}, CFG).param == "q"
    padded = "  red  " + " " * (MAX_LEN - 7)  # exactly MAX_LEN raw chars
    assert isinstance(validate_search_params({"q": padded}, CFG), SearchRequest)
    assert validate_search_params({"q": padded + " "}, CFG).param == "q"


# ---------------------------------------------------------------------------------------------
# HTTP: GET /api/search
# ---------------------------------------------------------------------------------------------


class _SpyEngine:
    """Wraps the real SearchEngine and records each search call."""

    def __init__(self, inner: SearchEngine) -> None:
        self.inner = inner
        self.calls: list[tuple[Any, ...]] = []

    def search(self, q, flt, limit):
        self.calls.append((q, flt, limit))
        return self.inner.search(q, flt, limit)


@pytest.fixture(scope="module")
def api(tmp_path_factory):
    root = tmp_path_factory.mktemp("api_params")
    cfg = Config(root=root)
    cfg.thumbs_dir.mkdir(parents=True)
    cfg.playback_dir.mkdir(parents=True)
    db = MetadataStore(":memory:")
    db.upsert_camera("CAM01", "Reception")
    index = VectorIndex()
    enc = FakeEncoder(colour_mode=True)
    spy = _SpyEngine(SearchEngine(cfg, db, index, enc))
    services = Services(cfg, db, index, enc, False, spy, QueryLog(cfg.query_log_path))  # type: ignore[arg-type]
    client = TestClient(create_app(services, web_dir=root / "noweb"))
    yield client, spy, enc
    db.close()


def _assert_422(resp, param: str) -> None:
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert set(body) == {"error"}  # no FastAPI "detail" key
    err = body["error"]
    assert set(err) == {"code", "param", "message"}
    assert err["code"] == "invalid_parameter"
    assert err["param"] == param
    assert isinstance(err["message"], str) and err["message"]


@given(params=param_maps())
def test_api_search_validation(api, params):
    client, spy, enc = api
    expected = model(params)
    calls_before, text_before = len(spy.calls), len(enc.text_calls)
    resp = client.get("/api/search", params=params)
    if isinstance(expected, str):
        _assert_422(resp, expected)
        assert len(spy.calls) == calls_before  # no search
        assert len(enc.text_calls) == text_before  # no query encoding
    else:
        assert resp.status_code == 200, (params, resp.text)
        assert isinstance(resp.json(), list)
        assert len(spy.calls) == calls_before + 1


@given(blank=st.text(alphabet=WHITESPACE, max_size=30), others=param_maps())
def test_api_blank_q_422_without_encoding(api, blank, others):
    """Property 21 (API half): whitespace-only q -> 422 param q, encoder never called."""
    client, spy, enc = api
    calls_before, text_before = len(spy.calls), len(enc.text_calls)
    resp = client.get("/api/search", params={**others, "q": blank})
    _assert_422(resp, "q")
    assert len(spy.calls) == calls_before
    assert len(enc.text_calls) == text_before


def test_api_missing_q(api):
    client, spy, enc = api
    calls_before = len(spy.calls)
    _assert_422(client.get("/api/search"), "q")
    _assert_422(client.get("/api/search", params={"limit": "0", "cls": "dog"}), "q")
    assert len(spy.calls) == calls_before
