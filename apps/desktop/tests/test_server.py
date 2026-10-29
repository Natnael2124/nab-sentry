"""The local API with fake models: token, cases, videos, range, processing, search, AI check, media paths."""

from __future__ import annotations

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import FakeClip
from nab_find import processor as proc_mod
from nab_find import server as server_mod
from nab_find.server import AppState, create_app
from test_store_search import FakeModels, fake_reader_factory


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(proc_mod, "FrameReader", fake_reader_factory((100, 130)))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("NAB_FIND_GEMINI_KEY", raising=False)
    state = AppState(tmp_path / "data", tmp_path / "models", load_models_now=False)
    state.set_models(FakeModels())
    client = TestClient(create_app(state, tmp_path / "web"))
    client.headers["X-Token"] = state.token
    return state, client


def add_fake_video(state: AppState, cid: str) -> int:
    st = state.case(cid)
    vid = st.add_video("C:/x/Front_20261028T140000.mp4", "Front", 1_800_000_000.0, False, 600, 25, 640, 360, 1.0)
    st.set_meta(status="ready")
    return vid


def wait_status(client, cid, want, timeout=20):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = client.get(f"/api/cases/{cid}/progress").json()
        if s["status"] == want:
            return s
        time.sleep(0.05)
    raise AssertionError(f"status never became {want}: {s['status']}")


def test_token_required(env):
    state, client = env
    assert TestClient(client.app).get("/api/cases").status_code == 403
    assert TestClient(client.app).get("/api/cases?t=wrong").status_code == 403
    assert client.get("/api/cases").status_code == 200
    assert TestClient(client.app).get(f"/api/cases?t={state.token}").status_code == 200


def test_case_lifecycle_process_search(env):
    state, client = env
    c = client.post("/api/cases", json={"name": "Shop theft"}).json()
    assert c["status"] == "new" and c["name"] == "Shop theft"
    cid = c["id"]
    assert client.post(f"/api/cases/{cid}/run/start").status_code == 400  # no videos yet
    add_fake_video(state, cid)
    assert client.put(f"/api/cases/{cid}/range", json={"start_ts": 5, "end_ts": 1}).status_code == 400
    assert client.post(f"/api/cases/{cid}/run/start").status_code == 200
    s = wait_status(client, cid, "done")
    assert s["progress"]["percent"] == 100 and s["counts"]["frames"] > 0
    assert [e["event"] for e in s["events"]] == ["quick_done", "done"]
    detail = client.get(f"/api/cases/{cid}").json()
    assert detail["quick_done"] and detail["videos_list"][0]["camera"] == "Front"
    r = client.get(f"/api/cases/{cid}/search", params={"q": "person"}).json()
    assert r["moments"] and "thumb" in r["moments"][0]
    assert client.get(f"/api/cases/{cid}/search", params={"q": " "}).status_code == 400
    v = client.get(f"/api/cases/{cid}/visits", params={"sort": "longest"}).json()["visits"]
    assert v and v[0]["cls"] == "person"
    thumb = r["moments"][0]["thumb"]
    assert client.get(f"/api/cases/{cid}/thumb/{thumb}").status_code == 200
    assert client.get(f"/api/cases/{cid}/thumb/..%2Fcase.db").status_code == 404
    speeds = state.settings.get("speeds")
    assert set(speeds) == {"quick", "detail"}  # measured speeds kept for the next estimate
    # changing the range forgets the work
    client.put(f"/api/cases/{cid}/range", json={"start_ts": 1_800_000_000.0, "end_ts": 1_800_000_100.0})
    assert client.get(f"/api/cases/{cid}").json()["counts"]["frames"] == 0
    assert client.delete(f"/api/cases/{cid}").json() == {"ok": True}
    assert client.get("/api/cases").json() == []


def test_changes_refused_while_processing(env, monkeypatch):
    state, client = env
    cid = client.post("/api/cases", json={}).json()["id"]
    add_fake_video(state, cid)
    gate = __import__("threading").Event()
    orig = proc_mod.Processor.quick_pass

    def slow(self, *a):
        gate.wait(5)
        return orig(self, *a)

    monkeypatch.setattr(proc_mod.Processor, "quick_pass", slow)
    client.post(f"/api/cases/{cid}/run/start")
    wait_status(client, cid, "processing")
    assert client.put(f"/api/cases/{cid}/range", json={}).status_code == 409
    assert client.post(f"/api/cases/{cid}/videos", json={"paths": []}).status_code == 409
    client.post(f"/api/cases/{cid}/run/pause")
    assert client.get(f"/api/cases/{cid}/progress").json()["status"] == "paused"
    client.post(f"/api/cases/{cid}/run/resume")
    client.post(f"/api/cases/{cid}/run/cancel")
    gate.set()
    wait_status(client, cid, "stopped")


def test_video_time_fix_shifts_siblings(env):
    state, client = env
    cid = client.post("/api/cases", json={}).json()["id"]
    st = state.case(cid)
    a = st.add_video("a.mp4", "Gate", 1000.0, True, 600, 25, 640, 360, 1.0)
    st.add_video("b.mp4", "Gate", 1600.0, True, 600, 25, 640, 360, 1.0)
    st.add_video("c.mp4", "Door", 1000.0, True, 600, 25, 640, 360, 1.0)
    r = client.patch(f"/api/cases/{cid}/videos/{a}", json={"start_ts": 1100.0, "shift_siblings": True}).json()
    starts = {v["name"]: (v["start_ts"], v["start_estimated"]) for v in r["videos"]}
    assert starts == {"a.mp4": (1100.0, False), "b.mp4": (1700.0, False), "c.mp4": (1000.0, True)}


def test_ai_check_needs_key_and_estimates(env):
    state, client = env
    cid = client.post("/api/cases", json={}).json()["id"]
    m = {"video_id": 1, "camera": "Front", "start_ts": 1.0, "end_ts": 5.0, "best_ts": 3.0, "offset_s": 3.0,
         "score": 0.3, "hits": 1, "thumb": "t.jpg", "visit_id": None, "extra": "ignored"}
    est = client.post(f"/api/cases/{cid}/aicheck/estimate", json={"moments": [m] * 30}).json()
    assert est["moments"] == 20 and est["key_set"] is False  # capped by the ai_moments setting
    r = client.post(f"/api/cases/{cid}/aicheck", json={"q": "x", "moments": [m]})
    assert r.status_code == 400 and "Settings" in r.json()["error"]


def test_settings_never_return_key(env):
    state, client = env
    r = client.put("/api/settings", json={"gemini_key": "AIza-secret", "gemini_model": "gemini-3.8-flash",
                                          "evil": 1}).json()
    assert r["gemini_key_set"] and r["gemini_model"] == "gemini-3.8-flash" and "evil" not in r
    assert "AIza-secret" not in client.get("/api/settings").text
    assert "AIza-secret" not in client.get("/api/state").text


def test_unknown_case_and_bad_ids(env):
    _, client = env
    assert client.get("/api/cases/nope").status_code == 404
    assert client.get("/api/cases/..%5C..%5Cwindows").status_code == 404
