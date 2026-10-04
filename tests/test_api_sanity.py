"""Fast sanity checks for the API app (spec tests live in tasks 14.7-14.12)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from nab_sentry.api.app import ParamError, SearchRequest, Services, create_app, validate_search_params
from nab_sentry.config import Config
from nab_sentry.querylog import QueryLog
from nab_sentry.search.engine import SearchEngine
from nab_sentry.store.db import MetadataStore, NewFrame, NewVector, NewVideo
from nab_sentry.store.vector_index import VectorIndex
from tests.fakes import BLUE_DIRECTION, RED_DIRECTION, FakeEncoder

T0 = datetime(2025, 1, 1, 8, 0, tzinfo=timezone(timedelta(hours=1)))
VIDEO_BYTES = bytes(range(256)) * 4  # 1024 bytes


def _populate(cfg: Config, db: MetadataStore, index: VectorIndex) -> int:
    db.upsert_camera("CAM01", "Reception")
    vid = db.insert_video(NewVideo("CAM01", "a.mp4", "h1", T0, 10.0, 16, 12, 60.0))
    ids, vecs = [], []
    for off, vec in [(10.0, RED_DIRECTION), (11.0, RED_DIRECTION), (40.0, BLUE_DIRECTION)]:
        name = f"v{vid}_f{int(off * 10):07d}.jpg"
        fid = db.insert_frame(NewFrame(vid, int(off * 10), off, T0 + timedelta(seconds=off),
                                       "motion", 0.5, name))
        (cfg.thumbs_dir / name).write_bytes(b"\xff\xd8jpeg" + name.encode())
        ids.append(db.insert_vector(NewVector(fid, "frame")))
        vecs.append(vec)
    db.finalize_video(vid, sampled=60, passed=3, duration_s=60.0, playback_path=f"v{vid}.mp4")
    (cfg.playback_dir / f"v{vid}.mp4").write_bytes(VIDEO_BYTES)
    index.add(np.array(ids, dtype=np.int64), np.stack(vecs).astype(np.float32))
    return vid


def _services(tmp_path: Path, *, encoder=True) -> Services:
    cfg = Config(root=tmp_path)
    cfg.thumbs_dir.mkdir(parents=True)
    cfg.playback_dir.mkdir(parents=True)
    db = MetadataStore(":memory:")
    index = VectorIndex()
    _populate(cfg, db, index)
    enc = FakeEncoder(colour_mode=True) if encoder else None
    engine = SearchEngine(cfg, db, index, enc) if encoder else None
    return Services(cfg, db, index, enc, False, engine, QueryLog(cfg.query_log_path))


@pytest.fixture
def svc(tmp_path):
    s = _services(tmp_path)
    yield s
    s.db.close()


@pytest.fixture
def client(svc, tmp_path):
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<!doctype html><title>NAB Sentry</title>", encoding="utf-8")
    return TestClient(create_app(svc, web_dir=web))


def _log_lines(svc: Services) -> list[dict]:
    p = svc.cfg.query_log_path
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []


def test_validate_order_and_defaults():
    cfg = Config(root=Path("."))
    ok = validate_search_params({"q": "red"}, cfg)
    assert isinstance(ok, SearchRequest) and ok.limit == 20 and ok.filter.is_empty()
    bad = validate_search_params({"q": " ", "start": "x", "limit": "0"}, cfg)
    assert bad == ParamError("q", bad.message)
    assert validate_search_params({"q": "a", "start": "x", "cls": "dog"}, cfg).param == "start"
    r = validate_search_params(
        {"q": "a", "start": "2025-01-01T09:00:00+00:00", "end": "2025-01-01T08:00:00+00:00"}, cfg)
    assert r.param == "end"
    assert validate_search_params({"q": "a", "cls": "Person"}, cfg).param == "cls"
    for lim in ("0", "101", "-1", "5.0", " 5", ""):
        assert validate_search_params({"q": "a", "limit": lim}, cfg).param == "limit"
    naive = validate_search_params({"q": "a", "start": "2025-01-01T08:00:00"}, cfg)
    assert naive.filter.start.tzinfo is not None


def test_health_and_cameras(client):
    h = client.get("/api/health")
    assert h.status_code == 200
    assert h.json() == {"status": "ok", "videos": 1, "vectors": 3,
                        "models_loaded": {"detector": False, "embedder": True}}
    assert client.get("/api/cameras").json() == [{"camera_id": "CAM01", "label": "Reception"}]


def test_search_200_and_query_log(client, svc):
    r = client.get("/api/search", params={"q": "red square", "limit": "5"})
    assert r.status_code == 200
    events = r.json()
    assert len(events) >= 1
    top = events[0]
    assert top["camera_label"] == "Reception"
    start = datetime.fromisoformat(top["start_time"])
    assert abs((start - (T0 + timedelta(seconds=top["start_offset_s"]))).total_seconds()) < 1e-3
    assert top["start_offset_s"] <= top["end_offset_s"] and top["start_offset_s"] <= 10.0
    assert client.get(top["thumbnail_url"]).status_code == 200
    lines = _log_lines(svc)
    assert len(lines) == 1 and lines[0]["status"] == 200 and lines[0]["result_count"] == len(events)
    assert lines[0]["latency_ms"] >= 0
    # unknown camera -> []
    assert client.get("/api/search", params={"q": "red", "camera": "x' OR 1=1 --"}).json() == []


def test_search_422_shape_and_log(client, svc):
    r = client.get("/api/search", params={"q": "   "})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_parameter" and r.json()["error"]["param"] == "q"
    assert client.get("/api/search").json()["error"]["param"] == "q"
    lines = _log_lines(svc)
    assert [x["status"] for x in lines] == [422, 422] and all(x["result_count"] == 0 for x in lines)
    assert "detail" not in r.json()


def test_search_503_without_encoder(tmp_path):
    s = _services(tmp_path, encoder=False)
    c = TestClient(create_app(s, web_dir=tmp_path / "noweb"))
    r = c.get("/api/search", params={"q": "red"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "search_unavailable"
    assert c.get("/").status_code == 404  # missing index.html handled gracefully
    s.db.close()


def test_video_range_responses(client):
    full = client.get("/media/video/1")
    assert full.status_code == 200 and full.content == VIDEO_BYTES
    assert full.headers["accept-ranges"] == "bytes"
    assert full.headers["content-length"] == str(len(VIDEO_BYTES))
    assert full.headers["content-type"] == "video/mp4"

    part = client.get("/media/video/1", headers={"Range": "bytes=10-19"})
    assert part.status_code == 206 and part.content == VIDEO_BYTES[10:20]
    assert part.headers["content-range"] == f"bytes 10-19/{len(VIDEO_BYTES)}"
    assert part.headers["content-length"] == "10"

    bad = client.get("/media/video/1", headers={"Range": "bytes=5000-"})
    assert bad.status_code == 416 and bad.content == b""
    assert bad.headers["content-range"] == f"bytes */{len(VIDEO_BYTES)}"

    assert client.get("/media/video/999").status_code == 404
    assert client.get("/media/video/abc").json()["error"]["code"] == "not_found"


def test_thumbnail_traversal_404(client, tmp_path):
    (tmp_path / "secret.jpg").write_bytes(b"SECRET")
    for name in ("..%2Fsecret.jpg", "..\\secret.jpg", "C:secret.jpg", "nope.jpg", "%2e%2e%2fsecret.jpg"):
        r = client.get(f"/media/thumbs/{name}")
        assert r.status_code == 404 and b"SECRET" not in r.content


def test_console_and_unhandled_500(client, svc):
    assert client.get("/").status_code == 200

    def boom():
        raise RuntimeError(r"C:\secret\path SELECT * FROM videos")

    svc.db.cameras = boom  # type: ignore[method-assign]
    r = client.get("/api/cameras")
    assert r.status_code == 500
    assert r.json() == {"error": {"code": "internal_error", "message": "Internal server error"}}
    assert client.get("/api/health").status_code == 200
