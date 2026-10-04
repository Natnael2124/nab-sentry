"""Property 42: Query log is append-only, one line per request (task 14.9).

**Validates: Requirements 10.10**

A random sequence of requests (valid searches, invalid search parameters, a 503 search with the
engine unloaded, and non-search endpoints) is sent to the app. After every request:

* the previous Query_Log bytes are an unchanged prefix of the current file;
* the line count equals the number of search requests answered with 200 or 422 so far;
* each new line is JSON with the right ``status`` and raw ``q``, ``result_count`` equal to the
  response length for 200 (0 for 422), raw parameter strings in ``filters`` for 422,
  ``latency_ms >= 0`` and an ISO ``ts`` carrying a UTC offset.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nab_sentry.api.app import Services, create_app
from nab_sentry.config import Config
from nab_sentry.querylog import QueryLog
from nab_sentry.search.engine import SearchEngine
from nab_sentry.store.db import MetadataStore
from nab_sentry.store.vector_index import VectorIndex
from tests.fakes import FakeEncoder
from tests.test_api_sanity import _populate

_FILTER_KEYS = ("camera", "start", "end", "cls", "limit")

# -- request strategies: each yields (kind, path, params) -------------------------------------

_words = st.sampled_from(["red", "blue", "square", "truck", "person", "Gate", "näive", "x"])
_valid_q = st.lists(_words, min_size=1, max_size=4).map(" ".join)

_valid_search = st.fixed_dictionaries(
    {"q": _valid_q},
    optional={
        "cls": st.sampled_from(["person", "car", "truck", "bus", "motorcycle", "bicycle"]),
        "limit": st.integers(1, 100).map(str),
        "camera": st.sampled_from(["CAM01", "CAM99"]),
        "start": st.just("2025-01-01T07:00:00+01:00"),
    },
).map(lambda p: ("search", "/api/search", p))

_bad_q = st.sampled_from(["", " ", "\t  ", "a" * 257])
_invalid_search = st.one_of(
    st.fixed_dictionaries({"q": _bad_q}, optional={"cls": st.just("dog")}),
    st.fixed_dictionaries({}, optional={"limit": st.just("5")}),  # missing q
    st.fixed_dictionaries({"q": _valid_q, "start": st.sampled_from(["yesterday", "2025-13-01"])}),
    st.fixed_dictionaries({"q": _valid_q, "cls": st.sampled_from(["dog", "Person", ""])}),
    st.fixed_dictionaries({"q": _valid_q, "limit": st.sampled_from(["0", "101", "-1", "5.0", "x"])}),
    st.fixed_dictionaries(
        {"q": _valid_q, "start": st.just("2025-01-02T00:00:00+00:00"),
         "end": st.just("2025-01-01T00:00:00+00:00")}
    ),
).map(lambda p: ("search", "/api/search", p))

_unavailable = _valid_q.map(lambda q: ("unavailable", "/api/search", {"q": q}))

_other = st.sampled_from(
    [
        ("other", "/api/health", {}),
        ("other", "/api/cameras", {}),
        ("other", "/media/video/1", {}),
        ("other", "/media/video/999", {}),
        ("other", "/media/thumbs/nope.jpg", {}),
        ("other", "/api/nope", {"q": "red"}),
    ]
)

_request = st.one_of(_valid_search, _invalid_search, _unavailable, _other)


def _services(root: Path) -> Services:
    cfg = Config(root=root)
    cfg.thumbs_dir.mkdir(parents=True)
    cfg.playback_dir.mkdir(parents=True)
    db = MetadataStore(":memory:")
    index = VectorIndex()
    _populate(cfg, db, index)
    enc = FakeEncoder(colour_mode=True)
    engine = SearchEngine(cfg, db, index, enc)
    return Services(cfg, db, index, enc, False, engine, QueryLog(cfg.query_log_path))


def _read(path: Path) -> bytes:
    return path.read_bytes() if path.exists() else b""


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(requests=st.lists(_request, min_size=1, max_size=12))
def test_query_log_append_only_one_line_per_search(requests):
    with tempfile.TemporaryDirectory() as tmp:
        svc = _services(Path(tmp))
        try:
            client = TestClient(create_app(svc, web_dir=Path(tmp) / "noweb"))
            log_path = svc.cfg.query_log_path
            expected_lines = 0
            prev = _read(log_path)

            for kind, path, params in requests:
                if kind == "unavailable":
                    engine, svc.engine = svc.engine, None
                    try:
                        resp = client.get(path, params=params)
                    finally:
                        svc.engine = engine
                    assert resp.status_code == 503
                else:
                    resp = client.get(path, params=params)

                logged = kind == "search" and resp.status_code in (200, 422)
                if kind == "search":
                    assert resp.status_code in (200, 422), (params, resp.status_code, resp.text)

                cur = _read(log_path)
                assert cur.startswith(prev), "earlier Query_Log bytes changed"
                if logged:
                    expected_lines += 1
                lines = cur.decode("utf-8").splitlines()
                assert len(lines) == expected_lines
                assert cur == b"" or cur.endswith(b"\n")

                if not logged:
                    assert cur == prev
                else:
                    new = cur[len(prev):].decode("utf-8")
                    assert new.count("\n") == 1
                    rec = json.loads(new)
                    assert set(rec) == {"ts", "status", "q", "filters", "result_count",
                                        "latency_ms"}
                    assert rec["status"] == resp.status_code
                    assert rec["q"] == params.get("q")
                    assert isinstance(rec["latency_ms"], (int, float)) and rec["latency_ms"] >= 0
                    ts = datetime.fromisoformat(rec["ts"])
                    assert ts.tzinfo is not None and ts.utcoffset() is not None
                    assert set(rec["filters"]) == set(_FILTER_KEYS)
                    if resp.status_code == 200:
                        assert rec["result_count"] == len(resp.json())
                        assert rec["filters"]["limit"] == int(params.get("limit", "20"))
                        assert rec["filters"]["camera"] == params.get("camera")
                        assert rec["filters"]["cls"] == params.get("cls")
                    else:
                        assert rec["result_count"] == 0
                        assert rec["filters"] == {k: params.get(k) for k in _FILTER_KEYS}
                prev = cur
        finally:
            svc.db.close()
