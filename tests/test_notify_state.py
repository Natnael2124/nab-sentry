"""Alert delivery state machine (V2 Requirements 13.3, 14.2-14.4, task 15.3): Property 20."""

from __future__ import annotations

import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nab_sentry.config import Config
from nab_sentry.notify.outbox import Outbox, backoff_s, caption
from nab_sentry.notify.telegram import TelegramClient, TelegramError
from nab_sentry.store.db import MetadataStore, NewAlert, NewVideo
from tests.fakes import FakeTelegramTransport

CHATS = [111, 222]
T0 = 1_800_000_000.0  # epoch seconds of the first Segment


def setup_store(root: Path, *, n_rule=2, health=True, footage=True):
    cfg = Config(root=root, alert_max_attempts=4)
    cfg.alerts_dir.mkdir(parents=True, exist_ok=True)
    db = MetadataStore(cfg.db_path)
    db.upsert_camera("CAM01", "Front gate")
    if footage:
        for i in range(3):  # three contiguous 10 s Segments
            src = root / f"CAM01_{i}.mkv"
            src.write_bytes(b"x")
            vid = db.insert_video(NewVideo("CAM01", str(src), f"h{i}",
                                           datetime.fromtimestamp(T0 + 10 * i, timezone.utc), 10, 64, 48, 10))
            db.finalize_video(vid, sampled=10, passed=1, duration_s=10.0, playback_path=f"v{vid}.mp4")
    ids = []
    for k in range(n_rule):
        snap = f"a_v1_yard_{k}.jpg"
        (cfg.alerts_dir / snap).write_bytes(b"jpeg")
        ids.append(db.insert_alert(NewAlert("rule", round((T0 + 5 + 7 * k) * 1000), rule_id="yard",
                                            rule_name="Person in the yard", camera_id="CAM01",
                                            det_class="person", snapshot_path=snap)))
    if health:
        ids.append(db.insert_alert(NewAlert("camera_offline", round((T0 + 30) * 1000), camera_id="CAM01")))
    db.insert_alert(NewAlert("rule", round(T0 * 1000), state="suppressed", rule_id="yard", camera_id="CAM01"))
    return cfg, db, ids


def fake_clip(parts, out, width):
    Path(out).write_bytes(b"mp4")
    return Path(out)


def outbox(cfg, db, transport, clip_maker=fake_clip):
    return Outbox(cfg, db, TelegramClient(transport), CHATS, camera_labels={"CAM01": "Front gate"},
                  clip_maker=clip_maker)


def test_happy_path_photo_then_clip_then_text():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, ids = setup_store(Path(tmp))
        tr = FakeTelegramTransport()
        ob = outbox(cfg, db, tr)
        now = T0 + 60  # all footage stored
        for _ in range(3):
            ob.tick(now)
        states = {a: db.get_alert(a)["state"] for a in ids}
        assert set(states.values()) == {"sent"}
        assert len(tr.sent("sendPhoto")) == 4 and len(tr.sent("sendVideo")) == 4  # 2 alerts x 2 chats
        assert len(tr.sent("sendMessage")) == 2  # the health alert
        photo_ids = {(a, c): m for a in ids[:2] for (c, step), m in db.sent_steps(a).items() if step == "photo"}
        for chat, data in tr.sent("sendVideo"):  # clips reply to their photo
            assert data["reply_to_message_id"] in photo_ids.values()
        assert db.get_alert(ids[0])["clip_path"].endswith(".mp4")
        assert all(a["state"] == "suppressed" for a in db.alerts_between(0, 2**62) if a["state"] not in ("sent",))
        db.close()


def test_clip_waits_for_the_next_segment_then_gives_up():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, ids = setup_store(Path(tmp), n_rule=1, health=False, footage=False)
        tr = FakeTelegramTransport()
        ob = outbox(cfg, db, tr)
        ob.tick(T0 + 6)  # photo goes out at once
        a = db.get_alert(ids[0])
        assert a["state"] == "photo_sent" and len(tr.sent("sendPhoto")) == 2
        ob.tick(T0 + 12)  # clip window ends at T0 + 10; no footage yet: wait
        assert db.get_alert(ids[0])["state"] == "photo_sent"
        ob.tick(T0 + 60)  # waited > segment_s + 15 s past the window: give up on the clip
        a = db.get_alert(ids[0])
        assert a["state"] == "sent" and "no video clip" in a["note"] and tr.sent("sendVideo") == []
        db.close()


def test_failed_clip_still_delivers_with_a_note():
    def broken(parts, out, width):
        raise RuntimeError("ffmpeg exploded")
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, ids = setup_store(Path(tmp), n_rule=1, health=False)
        ob = outbox(cfg, db, FakeTelegramTransport(), clip_maker=broken)
        ob.tick(T0 + 60)
        a = db.get_alert(ids[0])
        assert a["state"] == "sent" and "could not cut" in a["note"]
        db.close()


def test_retry_after_is_honoured_without_counting_an_attempt():
    first = {"done": False}

    def fail(method, i):
        if method == "sendPhoto" and not first["done"]:
            first["done"] = True
            return TelegramError(429, "Too Many Requests", retry_after=17)
        return None
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, ids = setup_store(Path(tmp), n_rule=1, health=False)
        ob = outbox(cfg, db, FakeTelegramTransport(fail=fail))
        ob.tick(T0 + 60)
        a = db.get_alert(ids[0])
        assert a["attempts"] == 0 and a["next_attempt_epoch_ms"] == round((T0 + 77) * 1000)
        ob.tick(T0 + 70)  # too early: not due
        assert db.get_alert(ids[0])["state"] == "pending"
        ob.tick(T0 + 78)
        assert db.get_alert(ids[0])["state"] in ("photo_sent", "sent")
        db.close()


def test_persistent_failure_ends_in_failed():
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, ids = setup_store(Path(tmp), n_rule=1, health=False)
        ob = outbox(cfg, db, FakeTelegramTransport(fail=lambda m, i: TelegramError(500, "down")))
        now = T0 + 60
        for _ in range(10):
            ob.tick(now)
            now += backoff_s(10)
        a = db.get_alert(ids[0])
        assert a["state"] == "failed" and a["attempts"] == cfg.alert_max_attempts
        assert "delivery failed" in a["note"]
        db.close()


# -- Property 20 ---------------------------------------------------------------------------------


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(st.sampled_from(["ok", "ok", "ok", "err", "429"]), min_size=1, max_size=60),
       st.sets(st.integers(0, 40), max_size=6))
def test_property_20_at_most_once_and_always_terminal(outcomes, restarts):
    """Random transport failures and Notifier restarts: no (alert, chat, step) is accepted twice,
    and every alert ends sent or failed."""
    def fail(method, i):
        o = outcomes[i % len(outcomes)] if i < 3 * len(outcomes) else "ok"
        if o == "err":
            return TelegramError(502, "Bad Gateway")
        if o == "429":
            return TelegramError(429, "Too Many Requests", retry_after=1)
        return None
    with tempfile.TemporaryDirectory() as tmp:
        cfg, db, ids = setup_store(Path(tmp))
        tr = FakeTelegramTransport(fail=fail)
        ob = outbox(cfg, db, tr)
        now = T0 + 60
        for tick in range(120):
            if tick in restarts:
                ob = outbox(cfg, db, tr)  # a restarted Notifier: only the store survives
            ob.tick(now)
            now += 60
        counts = Counter()
        for method, chat, data in tr.accepted:
            cap = data.get("caption") or data.get("text")
            counts[(method, chat, cap)] += 1
        assert all(c == 1 for c in counts.values()), counts
        for a in ids:
            assert db.get_alert(a)["state"] in ("sent", "failed")
            for (chat, step), _mid in db.sent_steps(a).items():
                assert chat in CHATS
        db.close()


def test_caption_shapes():
    rule = {"kind": "rule", "rule_id": "yard", "rule_name": "Person in the yard", "camera_id": "CAM01",
            "trigger_epoch_ms": 0, "det_class": "person", "n_tracks": 3, "note": None, "feedback": "false"}
    text = caption(rule, "Front gate")
    assert text.startswith("🚨 Person in the yard") and "Front gate" in text and "3 × person" in text
    assert "false alarm" in text
    assert caption({"kind": "disk_low", "trigger_epoch_ms": 0, "camera_id": None, "note": "4.2 GB free"}, None).startswith("⚠️ Disk space low")
