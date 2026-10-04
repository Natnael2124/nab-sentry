"""Property 51: request values never alter SQL (Requirement 18.4).

``MetadataStore._conn`` is wrapped in a recording proxy that captures every ``(sql, params)``
pair passed to ``execute``/``executemany``/``executescript``. (The sqlite3 trace callback
can't be used: on CPython 3.12 it receives the *expanded* SQL with bound values inlined.)

For hostile request values we assert:
  (a) every executed SQL text is one of the templates seen for benign requests, and no
      hostile value ever appears inside SQL text (only, possibly, inside bound params);
  (b) a hash of the schema and of every row of every table is unchanged;
  (c) every response is 200, 404 or 422 -- never 500.
"""

from __future__ import annotations

import hashlib
import sqlite3
import tempfile
from pathlib import Path
from urllib.parse import quote

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

SEARCH_PARAMS = ("q", "camera", "start", "end", "cls", "limit")
ALLOWED_STATUS = {200, 404, 422}


class RecordingConnection:
    """Delegating proxy around a ``sqlite3.Connection`` that records every statement."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._real = conn
        self.calls: list[tuple[str, object]] = []

    def execute(self, sql, params=()):
        self.calls.append((sql, params))
        return self._real.execute(sql, params)

    def executemany(self, sql, seq):
        seq = list(seq)
        self.calls.append((sql, seq))
        return self._real.executemany(sql, seq)

    def executescript(self, script):
        self.calls.append((script, None))
        return self._real.executescript(script)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _db_fingerprint(conn: sqlite3.Connection) -> str:
    """SHA-256 over sqlite_master plus every row of every table (incl. counts)."""
    h = hashlib.sha256()
    schema = conn.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
    ).fetchall()
    h.update(repr(schema).encode())
    tables = [r[1] for r in schema if r[0] == "table" and not r[1].startswith("sqlite_")]
    for t in ("schema_meta", "cameras", "videos", "frames", "vectors"):
        assert t in tables, f"table {t} missing"
    for t in tables:
        rows = conn.execute(f'SELECT * FROM "{t}" ORDER BY 1').fetchall()  # names from sqlite_master
        h.update(t.encode() + repr(len(rows)).encode() + repr(rows).encode())
    return h.hexdigest()


class Harness:
    def __init__(self, root: Path) -> None:
        cfg = Config(root=root)
        cfg.thumbs_dir.mkdir(parents=True)
        cfg.playback_dir.mkdir(parents=True)
        db = MetadataStore(":memory:")
        index = VectorIndex()
        _populate(cfg, db, index)
        enc = FakeEncoder(colour_mode=True)
        self.real_conn: sqlite3.Connection = db._conn
        self.rec = RecordingConnection(self.real_conn)
        db._conn = self.rec  # type: ignore[assignment]
        self.svc = Services(cfg, db, index, enc, False, SearchEngine(cfg, db, index, enc),
                            QueryLog(cfg.query_log_path))
        self.client = TestClient(create_app(self.svc, web_dir=root / "noweb"),
                                 raise_server_exceptions=False)

    def close(self) -> None:
        self.svc.db._conn = self.real_conn  # type: ignore[assignment]
        self.svc.db.close()

    def requests(self, values: dict[str, str], path_value: str) -> list[int]:
        c = self.client
        seg = quote(path_value, safe="")
        return [
            c.get("/api/search", params=values).status_code,
            c.get("/api/cameras").status_code,
            c.get(f"/media/video/{seg}").status_code,
            c.get(f"/media/thumbs/{seg}").status_code,
        ]


BENIGN_REQUESTS: list[tuple[dict[str, str], str]] = [
    ({"q": "red square"}, "1"),
    ({"q": "red", "camera": "CAM01"}, "999"),
    ({"q": "red", "camera": "NOPE", "cls": "person", "limit": "5"}, "v1_f0000100.jpg"),
    ({"q": "blue", "start": "2025-01-01T08:00:00+01:00", "end": "2025-01-01T09:00:00+01:00"}, "x.jpg"),
    ({"q": "blue", "camera": "CAM01", "start": "2025-01-01T08:00:00", "cls": "car", "limit": "100"}, "2"),
    ({"q": "   "}, "abc"),
    ({"q": "red", "start": "bogus"}, "abc"),
]


def _benign_templates() -> set[str]:
    with tempfile.TemporaryDirectory() as d:
        h = Harness(Path(d))
        try:
            h.rec.calls.clear()
            for values, pv in BENIGN_REQUESTS:
                h.requests(values, pv)
            return {sql for sql, _ in h.rec.calls}
        finally:
            h.close()


_TEMPLATES: set[str] | None = None


def benign_templates() -> set[str]:
    global _TEMPLATES
    if _TEMPLATES is None:
        _TEMPLATES = _benign_templates()
        assert any("vectors" in s for s in _TEMPLATES)  # baseline actually hit the DB
    return _TEMPLATES


PAYLOADS = [
    "'", '"', "`", ";", "--", "/*", "*/", "' OR 1=1 --", "\" OR \"1\"=\"1",
    "'; DROP TABLE videos; --", "1; DROP TABLE frames", "CAM01' --", "') OR ('a'='a",
    "1 UNION SELECT vector_id FROM vectors", "' UNION SELECT name, sql FROM sqlite_master --",
    "/**/OR/**/1=1", "%' OR '%'='", "\\'; DELETE FROM cameras; --", "1)) OR 1=1 --",
    "2025-01-01T08:00:00' OR 1=1 --", "person' OR '1'='1", "20; DROP TABLE vectors",
    "ATTACH DATABASE 'x.db' AS x; --", "PRAGMA writable_schema=1;", "'||(SELECT 1)||'",
    "Ünïcødé ' 🎥 ; --", "\u2019 OR 1=1 --", "x\x00'; DROP TABLE cameras; --",
]

# Pieces glued together so random payloads still contain SQL metacharacters.
_pieces = st.sampled_from(PAYLOADS + ["CAM01", "person", "car", "5", "2025-01-01T08:00:00", "red", "1"])
_noise = st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=8)
hostile = st.one_of(
    st.sampled_from(PAYLOADS),
    st.builds(lambda a, n, b: a + n + b, _pieces, _noise, _pieces),
)

params_strategy = st.fixed_dictionaries(
    {},
    optional={p: hostile for p in SEARCH_PARAMS},
)


def _hostile_values(values: dict[str, str], path_value: str) -> list[str]:
    """Values distinctive enough that their presence inside SQL text would mean injection."""
    out = [v for v in [*values.values(), path_value] if any(ch in v for ch in "'\";-/*=()")]
    return [v for v in out if len(v.strip()) >= 3]


@settings(max_examples=60, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
@given(values=params_strategy, path_value=hostile, q_sometimes_benign=st.booleans())
def test_request_values_never_alter_sql(values, path_value, q_sometimes_benign):
    """**Validates: Requirements 18.4**"""
    templates = benign_templates()
    if q_sometimes_benign:
        values = {**values, "q": "red square"}  # let hostile filters reach the DB query path
    with tempfile.TemporaryDirectory() as d:
        h = Harness(Path(d))
        try:
            before = _db_fingerprint(h.real_conn)
            h.rec.calls.clear()

            statuses = h.requests(values, path_value)

            # (c) never 500 (or any other unexpected status)
            assert set(statuses) <= ALLOWED_STATUS, (statuses, values, path_value)

            # (a) SQL text is always a known fixed template; hostile values only in params
            executed = [sql for sql, _ in h.rec.calls]
            unknown = set(executed) - templates
            assert not unknown, f"unexpected SQL executed: {unknown!r}"
            for v in _hostile_values(values, path_value):
                for sql in executed:
                    assert v not in sql, f"request value {v!r} appeared in SQL {sql!r}"

            # (b) schema and every row unchanged
            assert _db_fingerprint(h.real_conn) == before
        finally:
            h.close()


def test_known_payloads_each_param():
    """Every curated payload in every search param and both path params, one at a time."""
    templates = benign_templates()
    with tempfile.TemporaryDirectory() as d:
        h = Harness(Path(d))
        try:
            before = _db_fingerprint(h.real_conn)
            for payload in PAYLOADS:
                for p in SEARCH_PARAMS:
                    values = {"q": "red square", p: payload}
                    h.rec.calls.clear()
                    statuses = h.requests(values, payload)
                    assert set(statuses) <= ALLOWED_STATUS, (p, payload, statuses)
                    executed = {sql for sql, _ in h.rec.calls}
                    assert executed <= templates, executed - templates
                    if len(payload.strip()) >= 3:
                        assert all(payload not in sql for sql in executed)
            assert _db_fingerprint(h.real_conn) == before
            r = h.client.get("/api/search", params={"q": "red square"})
            assert r.status_code == 200 and len(r.json()) >= 1  # still fully functional
        finally:
            h.close()
