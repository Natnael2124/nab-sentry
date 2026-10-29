"""AI check request/response handling (no network) and the settings key storage."""

from __future__ import annotations

import base64
import json

import httpx
import pytest

from nab_find import aicheck
from nab_find.aicheck import (AICheckError, GeminiClient, Job, MomentImages, build_request, estimate,
                              parse_response, rank_key, run_job, CheckResult)
from nab_find.search import Moment
from nab_find.settings import Settings, protect, unprotect
from nab_find.store import CaseStore


def moment(i: int, visit: int | None = None) -> Moment:
    return Moment(1, "Front", 1e9 + i, 1e9 + i + 4, 1e9 + i + 2, float(i), 0.3, 1, "t.jpg", visit)


def test_estimate_counts_images_and_cost():
    e = estimate([moment(0), moment(1, visit=5)], "gemini-3.5-flash-lite")
    assert e["images"] == 3 + 6 and e["moments"] == 2
    assert e["tokens_in"] == 9 * 280 + 2 * 40 + 350
    assert 0 < e["usd"] < 0.01


def test_build_request_shape():
    body = build_request("  red jacket ", [MomentImages(1, "Moment 1 (Front, 28 Oct 14:00:00)", [b"abc", b"de"])])
    parts = body["contents"][0]["parts"]
    assert '"red jacket"' in parts[0]["text"]
    assert parts[1] == {"text": "Moment 1 (Front, 28 Oct 14:00:00):"}
    assert base64.b64decode(parts[2]["inline_data"]["data"]) == b"abc"
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert body["generationConfig"]["mediaResolution"] == "MEDIA_RESOLUTION_LOW"


def reply(results, usage=(1000, 50)) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": json.dumps({"results": results})}]}}],
            "usageMetadata": {"promptTokenCount": usage[0], "candidatesTokenCount": usage[1]}}


def test_parse_response_filters_and_clamps():
    body = reply([{"moment": 1, "match": "YES", "confidence": 1.7, "description": "a man in red"},
                  {"moment": 9, "match": "yes", "confidence": 0.9, "description": "not asked"},
                  {"moment": 2, "match": "maybe", "confidence": -1, "description": ""}])
    res, usage = parse_response(body, {1, 2})
    assert [(r.index, r.match, r.confidence) for r in res] == [(1, "yes", 1.0), (2, "unsure", 0.0)]
    assert usage.tokens_in == 1000 and usage.tokens_out == 50


def test_parse_response_errors():
    with pytest.raises(AICheckError, match="no answer"):
        parse_response({"promptFeedback": {"blockReason": "SAFETY"}}, {1})
    with pytest.raises(AICheckError, match="unexpected format"):
        parse_response({"candidates": [{"content": {"parts": [{"text": "not json"}]}}]}, {1})


def test_rank_yes_first_then_unchecked_then_no():
    rs = {1: CheckResult(1, "no", 0.9, ""), 2: CheckResult(2, "yes", 0.5, ""), 3: None,
          4: CheckResult(4, "yes", 0.9, ""), 5: CheckResult(5, "unsure", 0.2, "")}
    assert sorted(rs, key=lambda i: rank_key(rs[i], i)) == [4, 2, 3, 5, 1]


@pytest.mark.parametrize("status,body,msg", [
    (400, {"error": {"message": "API key not valid. Please pass a valid API key."}}, "not accepted"),
    (403, {"error": {"message": "denied"}}, "not accepted"),
    (404, {"error": {"message": "model not found"}}, "not available"),
    (429, {"error": {"message": "quota"}}, "busy or your quota"),
])
def test_http_errors_are_plain(status, body, msg, monkeypatch):
    monkeypatch.setattr(aicheck.time, "sleep", lambda s: None)
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(req)
        return httpx.Response(status, json=body)

    c = GeminiClient("k3y", "gemini-3.5-flash-lite", transport=httpx.MockTransport(handler))
    with pytest.raises(AICheckError, match=msg):
        c.generate({})
    assert calls[0].headers["x-goog-api-key"] == "k3y" and "key=" not in str(calls[0].url)
    assert len(calls) == (2 if status == 429 else 1)  # one retry on 429 only


def test_network_error_is_plain(monkeypatch):
    monkeypatch.setattr(aicheck.time, "sleep", lambda s: None)

    def handler(req):
        raise httpx.ConnectError("offline")

    c = GeminiClient("k", "m", transport=httpx.MockTransport(handler))
    with pytest.raises(AICheckError, match="internet"):
        c.generate({})


def test_run_job_batches_and_reranks(tmp_path, monkeypatch):
    st = CaseStore.create(tmp_path, "c")
    st.add_video("v.mp4", "Front", 1e9, False, 600, 25, 640, 360, 1.0)
    monkeypatch.setattr(aicheck, "grab_jpeg", lambda path, off: b"jpeg")
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        labels = [p["text"] for p in body["contents"][0]["parts"][1:] if "text" in p]
        idx = [int(s.split()[1]) for s in labels]
        seen.append(idx)
        return httpx.Response(200, json=reply([{"moment": i, "match": "yes" if i == 7 else "no",
                                                "confidence": 0.8, "description": f"m{i}"} for i in idx]))

    job = Job("q", [moment(i) for i in range(12)], "gemini-3.5-flash-lite")
    client = GeminiClient("k", job.model, transport=httpx.MockTransport(handler))
    run_job(job, st, client)
    assert sorted(len(s) for s in seen) == [2, 5, 5] and job.done and not job.errors
    d = job.to_dict()
    assert d["items"][0]["check"]["description"] == "m7" and d["progress"] == [3, 3]
    assert st.query("SELECT COUNT(*) FROM ai_checks")[0][0] == 1


def test_dpapi_roundtrip_and_settings_hide_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("NAB_FIND_GEMINI_KEY", raising=False)
    assert unprotect(protect("secret-ä")) == "secret-ä"
    s = Settings(tmp_path)
    s.set_gemini_key("AIza-test")
    raw = (tmp_path / "settings.json").read_text(encoding="utf-8")
    assert "AIza-test" not in raw
    assert Settings(tmp_path).gemini_key() == "AIza-test"
    pub = s.public()
    assert pub["gemini_key_set"] is True and "gemini_key_enc" not in pub and "AIza" not in json.dumps(pub)
    s.set_gemini_key(None)
    assert Settings(tmp_path).gemini_key() is None
