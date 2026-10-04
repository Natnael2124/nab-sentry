"""Sanity checks that the real Console files in nab_sentry/web are served (task 16.1).

Detailed static checks of the markup (CSP, labels, controls) are task 16.3.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nab_sentry.api.app import WEB_DIR, Services, create_app
from nab_sentry.config import Config
from nab_sentry.querylog import QueryLog
from nab_sentry.store.db import MetadataStore
from nab_sentry.store.vector_index import VectorIndex


@pytest.fixture
def client(tmp_path: Path):
    cfg = Config(root=tmp_path)
    db = MetadataStore(":memory:")
    svc = Services(cfg, db, VectorIndex(), None, False, None, QueryLog(cfg.query_log_path))
    try:
        yield TestClient(create_app(svc))  # default web_dir = the shipped Console
    finally:
        db.close()


def test_default_web_dir_has_console_files():
    assert (WEB_DIR / "index.html").is_file()
    assert (WEB_DIR / "styles.css").is_file()


def test_root_serves_shipped_index(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    body = r.text
    assert "default-src 'self'; media-src 'self'; img-src 'self'" in body
    assert 'href="/static/styles.css"' in body
    assert 'src="/static/app.js"' in body
    for el_id in ("search-form", "q", "q-error", "camera", "start", "end", "cls", "search-btn",
                  "status", "results", "player", "video", "timeline", "timeline-span",
                  "replay", "prev", "next", "player-error"):
        assert f'id="{el_id}"' in body, el_id


def test_static_styles_served(client):
    r = client.get("/static/styles.css")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/css")
    assert ":focus-visible { outline: 3px solid #1a73e8; outline-offset: 2px; }" in r.text
