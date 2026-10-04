"""Example-based unit tests for the API endpoints (task 14.12).

Requirements 10.2, 10.3, 10.11-10.14, 11.1, 11.6-11.8, 18.6, 18.7.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pytest
from fastapi.testclient import TestClient

from nab_sentry.api.app import Services, create_app
from nab_sentry.config import Config
from nab_sentry.querylog import QueryLog
from nab_sentry.search.engine import SearchEngine
from nab_sentry.store.db import MetadataStore, NewFrame, NewVector, NewVideo
from nab_sentry.store.vector_index import VectorIndex
from tests.fakes import BLUE_DIRECTION, RED_DIRECTION, FakeEncoder

T0 = datetime(2025, 1, 1, 8, 0, tzinfo=timezone(timedelta(hours=1)))
VIDEO_BYTES = bytes(range(256)) * 4  # 1024 bytes
INDEX_HTML = "<!doctype html><title>NAB Sentry Console</title>"
SECRET = b"TOP-SECRET-DECOY-BYTES-0123456789"
INTERNAL_ERROR = {"error": {"code": "internal_error", "message": "Internal server error"}}


def _populate(cfg: Config, db: MetadataStore, index: VectorIndex) -> tuple[int, list[str]]:
    db.upsert_camera("CAM01", "Reception")
    db.upsert_camera("CAM02", "Loading Bay")
    vid = db.insert_video(NewVideo("CAM01", "a.mp4", "h1", T0, 10.0, 16, 12, 60.0))
    ids, vecs, thumbs = [], [], []
    for off, vec in [(10.0, RED_DIRECTION), (11.0, RED_DIRECTION), (40.0, BLUE_DIRECTION)]:
        name = f"v{vid}_f{int(off * 10):07d}.jpg"
        fid = db.insert_frame(NewFrame(vid, int(off * 10), off, T0 + timedelta(seconds=off),
                                       "motion", 0.5, name))
        (cfg.thumbs_dir / name).write_bytes(b"\xff\xd8jpeg" + name.encode())
        ids.append(db.insert_vector(NewVector(fid, "frame")))
        vecs.append(vec)
        thumbs.append(name)
    db.finalize_video(vid, sampled=60, passed=3, duration_s=60.0, playback_path=f"v{vid}.mp4")
    (cfg.playback_dir / f"v{vid}.mp4").write_bytes(VIDEO_BYTES)
    index.add(np.array(ids, dtype=np.int64), np.stack(vecs).astype(np.float32))
    return vid, thumbs


def _make_services(tmp_path: Path, *, populate: bool = True, encoder: bool = True,
                   query_log_path: Path | None = None) -> Services:
    cfg = Config(root=tmp_path)
    cfg.thumbs_dir.mkdir(parents=True)
    cfg.playback_dir.mkdir(parents=True)
    db = MetadataStore(":memory:")
    index = VectorIndex()
    if populate:
        _populate(cfg, db, index)
    enc = FakeEncoder(colour_mode=True) if encoder else None
    engine = SearchEngine(cfg, db, index, enc) if encoder else None
    qlog = QueryLog(query_log_path if query_log_path is not None else cfg.query_log_path)
    return Services(cfg, db, index, enc, False, engine, qlog)


def _web_dir(tmp_path: Path) -> Path:
    web = tmp_path / "web"
    web.mkdir(exist_ok=True)
    (web / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    return web


@pytest.fixture
def svc(tmp_path):
    s = _make_services(tmp_path)
    yield s
    s.db.close()


@pytest.fixture
def client(svc, tmp_path):
    return TestClient(create_app(svc, web_dir=_web_dir(tmp_path)))


# --- /api/health, /api/cameras, / (10.2, 10.3, 10.14) ----------------------------------------


def test_health_shape(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    assert r.json() == {
        "status": "ok",
        "videos": 1,
        "vectors": 3,
        "models_loaded": {"detector": False, "embedder": True},
    }


def test_health_shape_without_encoder(tmp_path):
    s = _make_services(tmp_path, populate=False, encoder=False)
    try:
        body = TestClient(create_app(s, web_dir=_web_dir(tmp_path))).get("/api/health").json()
        assert body == {"status": "ok", "videos": 0, "vectors": 0,
                        "models_loaded": {"detector": False, "embedder": False}}
    finally:
        s.db.close()


def test_cameras_empty(tmp_path):
    s = _make_services(tmp_path, populate=False)
    try:
        r = TestClient(create_app(s, web_dir=_web_dir(tmp_path))).get("/api/cameras")
        assert r.status_code == 200 and r.json() == []
    finally:
        s.db.close()


def test_cameras_populated(client):
    r = client.get("/api/cameras")
    assert r.status_code == 200
    assert sorted(r.json(), key=lambda c: c["camera_id"]) == [
        {"camera_id": "CAM01", "label": "Reception"},
        {"camera_id": "CAM02", "label": "Loading Bay"},
    ]


def test_root_serves_console_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert r.text == INDEX_HTML


# --- /api/search (10.11, 10.12, 10.13) --------------------------------------------------------


def test_search_unknown_camera_returns_empty_list(client):
    r = client.get("/api/search", params={"q": "red square", "camera": "NO_SUCH_CAMERA"})
    assert r.status_code == 200 and r.json() == []


def test_search_503_when_encoder_none(tmp_path):
    s = _make_services(tmp_path, encoder=False)
    try:
        r = TestClient(create_app(s, web_dir=_web_dir(tmp_path))).get(
            "/api/search", params={"q": "red square"})
        assert r.status_code == 503
        body = r.json()
        assert body["error"]["code"] == "search_unavailable"
        assert set(body) == {"error"} and isinstance(body["error"]["message"], str)
    finally:
        s.db.close()


def test_query_log_write_failure_still_returns_results(tmp_path, caplog):
    bad_log = tmp_path / "log_is_a_directory"
    bad_log.mkdir()
    s = _make_services(tmp_path, query_log_path=bad_log)
    try:
        c = TestClient(create_app(s, web_dir=_web_dir(tmp_path)))
        with caplog.at_level(logging.WARNING, logger="nab_sentry"):
            r = c.get("/api/search", params={"q": "red square"})
        assert r.status_code == 200
        events = r.json()
        assert len(events) >= 1 and events[0]["camera_id"] == "CAM01"
        assert bad_log.is_dir()  # nothing was written in its place
        assert any("Query_Log" in rec.getMessage() for rec in caplog.records)
    finally:
        s.db.close()


def test_query_log_success_writes_line(client, svc):
    assert client.get("/api/search", params={"q": "red square"}).status_code == 200
    lines = svc.cfg.query_log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["status"] == 200


# --- /media/video (11.1, 11.6) ----------------------------------------------------------------


def test_video_full_body_headers(client):
    r = client.get("/media/video/1")
    assert r.status_code == 200
    assert r.content == VIDEO_BYTES
    assert r.headers["content-type"] == "video/mp4"
    assert r.headers["accept-ranges"] == "bytes"
    assert r.headers["content-length"] == str(len(VIDEO_BYTES))
    assert "content-range" not in r.headers


def test_video_unknown_id_404(client):
    for vid in ("999", "0", "abc", "-1", "1.0", "9" * 30):
        r = client.get(f"/media/video/{vid}")
        assert r.status_code == 404, vid
        assert r.json() == {"error": {"code": "not_found", "message": "Not found"}}


def test_video_missing_playback_file_404(client, svc):
    (svc.cfg.playback_dir / "v1.mp4").unlink()
    r = client.get("/media/video/1")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


# --- /media/thumbs (11.7, 11.8) ---------------------------------------------------------------


def test_thumbnail_200_jpeg(client):
    name = "v1_f0000100.jpg"
    r = client.get(f"/media/thumbs/{name}")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.content == b"\xff\xd8jpeg" + name.encode()


def test_thumbnail_url_from_search_resolves(client):
    events = client.get("/api/search", params={"q": "red square"}).json()
    r = client.get(events[0]["thumbnail_url"])
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"


def test_thumbnail_hostile_names_404_without_file_bytes(client, svc, tmp_path):
    # Decoy secrets next to and above thumbs_dir.
    (svc.cfg.thumbs_dir.parent / "secret.jpg").write_bytes(SECRET)
    (tmp_path / "secret.jpg").write_bytes(SECRET)
    abs_secret = str((tmp_path / "secret.jpg").resolve())
    drive = abs_secret[:2] if abs_secret[1:2] == ":" else "C:"
    hostile = [
        "..%2Fsecret.jpg",
        "..%2F..%2Fsecret.jpg",
        "%2e%2e%2fsecret.jpg",
        "..\\secret.jpg",
        "..%5Csecret.jpg",
        "..%5C..%5Csecret.jpg",
        quote(abs_secret, safe=""),
        quote(abs_secret.replace("\\", "/"), safe=""),
        "%2Fsecret.jpg",
        f"{drive}secret.jpg",
        quote(f"{drive}\\secret.jpg", safe=""),
        "secret.jpg%00.jpg",
        "v1_f0000100.jpg%00",
        "%00",
        "nope.jpg",
    ]
    for name in hostile:
        r = client.get(f"/media/thumbs/{name}")
        assert r.status_code == 404, name
        assert SECRET not in r.content, name
        assert r.json()["error"]["code"] == "not_found", name


# --- unhandled errors (18.6, 18.7) ------------------------------------------------------------


def test_unhandled_exception_sanitised_500_then_recovers(client, svc, caplog):
    secret_path = r"C:\Users\secret\nab\metadata.db"
    sql = "SELECT camera_id, label FROM cameras WHERE 1=1"

    def boom():
        raise RuntimeError(f"failed at {secret_path} running {sql}")

    svc.db.cameras = boom  # type: ignore[method-assign]
    with caplog.at_level(logging.ERROR, logger="nab_sentry"):
        r = client.get("/api/cameras")

    assert r.status_code == 500
    assert r.json() == INTERNAL_ERROR
    text = r.text
    for leak in ("Traceback", secret_path, "C:\\", "SELECT", "cameras WHERE", "RuntimeError"):
        assert leak not in text, leak

    # The traceback goes to the server log only.
    logged = [rec for rec in caplog.records if rec.name.startswith("nab_sentry")
              and rec.levelno >= logging.ERROR]
    assert logged and any(rec.exc_info and rec.exc_info[0] is RuntimeError for rec in logged)

    # The server keeps serving.
    h = client.get("/api/health")
    assert h.status_code == 200 and h.json()["status"] == "ok"
