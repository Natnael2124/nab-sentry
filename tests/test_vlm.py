"""On-demand VLM (Step 3, Requirements 20.3, 21, tasks 19.1-19.4)."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from nab_sentry.describe.vlm import (
    OllamaVLM,
    VLMError,
    parse_yes_no,
    storyboard,
    track_storyboard,
)
from nab_sentry.notify.outbox import Outbox
from nab_sentry.notify.telegram import TelegramClient
from nab_sentry.store.db import MetadataStore, NewAlert
from tests.fakes import FakeTelegramTransport
from tests.test_notify_state import T0, fake_clip, setup_store


def test_parse_yes_no():
    assert parse_yes_no("Yes, someone climbs the fence.") is True
    assert parse_yes_no("  no. The person walks past.") is False
    assert parse_yes_no("**Yes**") is True
    assert parse_yes_no("Maybe") is None and parse_yes_no("") is None
    assert parse_yes_no("Nobody is there") is None  # 'no' must be a whole word


def test_storyboard_layout():
    crops = [np.full((50, 20, 3), 60 * i, np.uint8) for i in range(1, 4)]
    grid = storyboard(crops)
    assert grid.shape == (2 * 288, 2 * 384, 3)
    assert grid[288:, 384:].max() == 0  # the 4th panel is empty


def test_ollama_client_is_loopback_only_and_sends_think_false():
    import httpx

    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "No. It is a cat."}})
    vlm = OllamaVLM(client=httpx.Client(transport=httpx.MockTransport(handler)))
    answer, words = vlm.yes_no(np.zeros((20, 20, 3), np.uint8), "Is a person climbing the fence?")
    assert answer is False and words == "No. It is a cat."
    body = sent[0]
    assert body["think"] is False and body["options"]["temperature"] == 0 and body["messages"][0]["images"]
    with pytest.raises(ValueError):
        OllamaVLM(host="http://192.168.1.5:11434")

    def down(request):
        raise httpx.ConnectError("refused")
    with pytest.raises(VLMError):
        OllamaVLM(client=httpx.Client(transport=httpx.MockTransport(down))).describe(np.zeros((5, 5, 3), np.uint8))


def test_track_storyboard_needs_stored_footage(tmp_path: Path):
    db = MetadataStore(tmp_path / "x.db")
    try:
        with pytest.raises(VLMError, match="no stored observations"):
            track_storyboard(db, 1)
    finally:
        db.close()


# -- verified rules in the outbox ----------------------------------------------------------------


def verify_store(tmp: Path):
    cfg, db, ids = setup_store(tmp, n_rule=0, health=False)
    snap = "a_v1_fence_1.jpg"
    (cfg.alerts_dir / snap).write_bytes(b"jpeg")
    vid = db.insert_alert(NewAlert("rule", round((T0 + 5) * 1000), rule_id="fence", rule_name="Fence",
                                   severity="verify", camera_id="CAM01", det_class="person", snapshot_path=snap))
    now = db.insert_alert(NewAlert("rule", round((T0 + 6) * 1000), rule_id="yard", rule_name="Yard",
                                   severity="immediate", camera_id="CAM01", det_class="person", snapshot_path=snap))
    return cfg, db, vid, now


@pytest.mark.parametrize("outcome,state,note", [
    ((True, "Yes, climbing."), "sent", "(verified: Yes, climbing.)"),
    ((False, "No, just walking."), "suppressed", "dismissed by the VLM: No, just walking."),
    ((None, "Hard to say."), "sent", '(not verified: unclear answer "Hard to say.")'),
    (RuntimeError("ollama down"), "sent", "(not verified: ollama down)"),
])
def test_verification_outcomes(outcome, state, note):
    calls = []

    def verifier(snapshot, question):
        calls.append(question)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, vid, _ = verify_store(Path(tmp))
        tr = FakeTelegramTransport()
        ob = Outbox(cfg, db, TelegramClient(tr), [111], clip_maker=fake_clip,
                    questions={"fence": "Is a person climbing the fence?"}, verifier=verifier)
        for _ in range(3):
            ob.tick(T0 + 60)
        a = db.get_alert(vid)
        assert a["state"] == state and a["note"].startswith(note[:30])
        assert calls == ["Is a person climbing the fence?"]  # asked once
        photos_for_fence = [d for _, d in tr.sent("sendPhoto") if d["caption"].startswith("🚨 Fence")]
        assert bool(photos_for_fence) == (state == "sent")
        db.close()


def test_vlm_off_sends_unverified_and_immediate_alerts_never_wait():
    order = []

    def slow_verifier(snapshot, question):
        order.append("verify")
        return True, "Yes."
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, vid, now = verify_store(Path(tmp))
        tr = FakeTelegramTransport()

        class Spy(TelegramClient):
            def send_photo(self, chat, path, caption, **kw):
                order.append(caption.splitlines()[0])
                return super().send_photo(chat, path, caption, **kw)
        ob = Outbox(cfg, db, Spy(tr), [111], clip_maker=fake_clip,
                    questions={"fence": "Is a person climbing the fence?"}, verifier=slow_verifier)
        ob.tick(T0 + 60)
        assert order[0] == "🚨 Yard"  # the immediate alert went out before the verification
        off = Outbox(cfg, db, TelegramClient(tr), [111], clip_maker=fake_clip,
                     questions={"fence": "q?"}, verifier=None)
        db.update_alert(vid, state="pending", note=None)
        off.tick(T0 + 70)
        assert db.get_alert(vid)["note"] == "(not verified: VLM off)"
        db.close()


@pytest.mark.slow
def test_real_vlm_answers_about_a_real_frame():
    """Only when Ollama with qwen3-vl:2b-instruct is running locally."""
    import cv2
    import httpx

    try:
        tags = httpx.get("http://127.0.0.1:11434/api/tags", timeout=3).json()
    except Exception:  # noqa: BLE001
        pytest.skip("Ollama is not running")
    if not any(m.get("name", "").startswith("qwen3-vl:2b-instruct") for m in tags.get("models", [])):
        pytest.skip("qwen3-vl:2b-instruct is not installed")
    video = Path(__file__).resolve().parents[1] / "data" / "videos" / "CAM-SEC01_20261007T230010.mp4"
    if not video.is_file():
        pytest.skip("Compilation_Video not present")
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, 200_000)  # the man mowing the lawn
    ok, img = cap.read()
    cap.release()
    vlm = OllamaVLM(timeout_s=300)
    answer, words = vlm.yes_no(img, "Is a person mowing a lawn?")
    assert answer is True, words
    answer2, words2 = vlm.yes_no(img, "Is a car on fire?")
    assert answer2 is False, words2
