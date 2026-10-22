"""GET /api/activities (Step 3, Requirement 20.4)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from nab_sentry.api.app import Services, create_app
from nab_sentry.querylog import QueryLog
from tests.test_pipeline_activities import ingest
from tests.test_pipeline_tracks import frames, name


def client(tmp_path: Path):
    cfg, db, pipe = ingest(tmp_path, {name(0): frames(0, 25)})
    svc = Services(cfg, db, pipe.index, None, False, None, QueryLog(cfg.query_log_path))
    return TestClient(create_app(svc)), db


def test_activities_of_a_day(tmp_path: Path):
    c, db = client(tmp_path)
    try:
        r = c.get("/api/activities", params={"date": "2026-01-01"})
        assert r.status_code == 200
        [item] = r.json()
        assert item["camera_id"] == "CAM-A" and item["camera_label"] == dict(db.cameras())["CAM-A"]
        assert "on the lawn" in item["summary"] and item["activity_id"] >= 1
        assert item["video_url"].startswith("/media/video/") and item["thumbnail_url"].startswith("/media/thumbs/")
        assert item["start_offset_s"] >= 0 and item["end_offset_s"] > item["start_offset_s"]
        assert c.get(item["thumbnail_url"]).status_code == 200
        assert c.get("/api/activities", params={"date": "2026-01-02"}).json() == []
        assert c.get("/api/activities", params={"date": "2026-01-01", "camera": "CAM-B"}).json() == []
    finally:
        db.close()


def test_bad_parameters(tmp_path: Path):
    c, db = client(tmp_path)
    try:
        for params in ({"date": "yesterday"}, {"date": "2026-02-30"}, {"camera": "bad id!"}):
            r = c.get("/api/activities", params=params)
            assert r.status_code == 422, params
    finally:
        db.close()
