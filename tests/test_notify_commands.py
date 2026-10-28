"""Bot commands, callbacks and isolation (V2 Requirements 14.1, 14.5-14.8, tasks 15.4-15.7)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nab_sentry.config import Config
from nab_sentry.notify.commands import MUTE_MAX_MIN, NAMES, Commands, parse_command
from nab_sentry.notify.outbox import Outbox
from nab_sentry.notify.service import OFFSET_KEY, Notifier
from nab_sentry.notify.telegram import API_HOST, HttpxTransport, TelegramClient, TelegramError
from nab_sentry.rules.config import Rule
from nab_sentry.store.db import MetadataStore, NewAlert
from tests.fakes import FakeTelegramTransport

ALLOWED = 111
NOW = 1_800_000_000.0

# -- Property 21: grammar ------------------------------------------------------------------------

ids = st.from_regex(r"[A-Za-z0-9_-]{1,20}", fullmatch=True)


@given(st.sampled_from(["start", "help", "status", "alerts", "rules"]), st.sampled_from(["", "@NabBot"]),
       st.booleans())
def test_property_21_no_arg_commands(name, suffix, upper):
    text = f"/{name.upper() if upper else name}{suffix}"
    assert parse_command(text) == parse_command(text + "  ")
    cmd = parse_command(text)
    assert cmd.name == name and cmd.error is None and cmd.args == ()
    assert parse_command(text + " extra").error is not None


@given(ids, st.one_of(st.none(), st.integers(1, MUTE_MAX_MIN)))
def test_property_21_mute(rule, minutes):
    text = f"/mute {rule}" + ("" if minutes is None else f" {minutes}")
    cmd = parse_command(text)
    assert cmd.error is None and cmd.args == (rule, str(minutes or 60))


@given(ids, st.one_of(st.integers(-5, 0), st.integers(MUTE_MAX_MIN + 1, 10**6)))
def test_property_21_mute_rejects_bad_minutes(rule, minutes):
    assert parse_command(f"/mute {rule} {minutes}").error is not None


@given(st.text(max_size=40))
def test_property_21_anything_parses_to_a_known_command_or_none(text):
    cmd = parse_command(text)
    if not text.strip().startswith("/") or len(text.strip().split()[0]) < 2:
        assert cmd is None
    else:
        assert cmd.name in NAMES


def test_parse_examples():
    assert parse_command("hello") is None
    assert parse_command("/snapshot CAM01") .args == ("CAM01",)
    assert parse_command("/snapshot a b").error is not None
    assert parse_command("/unmute").error is not None
    assert parse_command("/launch").error == "unknown command /launch"
    assert parse_command("/mute yard 1h").error is not None


# -- handlers and service ------------------------------------------------------------------------


@pytest.fixture
def env(tmp_path: Path):
    cfg = Config(root=tmp_path)
    db = MetadataStore(cfg.db_path)
    db.upsert_camera("CAM01", "Front gate")
    rules = [Rule("yard", "Person in the yard", "CAM01", frozenset({"person"}), "enter")]
    grabbed: list[Path] = []

    def grab(video: Path, out: Path) -> Path:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"jpg")
        grabbed.append(video)
        return out
    cmds = Commands(db, {"CAM01": "Front gate"}, rules, live_dir=cfg.live_dir,
                    offline_after_s=lambda cam: 30.0, frame_grabber=grab)
    tr = FakeTelegramTransport()
    client = TelegramClient(tr)
    notifier = Notifier(db, client, [ALLOWED], cmds, Outbox(cfg, db, client, [ALLOWED]),
                        camera_labels={"CAM01": "Front gate"}, clock=lambda: NOW)
    yield cfg, db, tr, notifier, grabbed
    db.close()


def msg(update_id, text, chat=ALLOWED):
    return {"update_id": update_id, "message": {"message_id": update_id, "chat": {"id": chat}, "text": text}}


def test_commands_reply_and_offset_persists(env):
    cfg, db, tr, notifier, _ = env
    tr.updates = [msg(5, "/help"), msg(6, "/mute yard 30"), msg(7, "/rules"), msg(8, "just chatting")]
    assert notifier.poll_updates() == 4
    texts = [d["text"] for _, d in tr.sent("sendMessage")]
    assert texts[0].startswith("NAB Sentry") and texts[1].startswith("Muted yard until")
    assert "muted until" in texts[2] and len(texts) == 3  # plain text gets no reply
    assert db.active_mutes(round(NOW * 1000)) and db.kv_get(OFFSET_KEY) == "9"
    assert notifier.poll_updates() == 0  # nothing handled twice


def test_unknown_chats_are_ignored(env, caplog):
    _, _, tr, notifier, _ = env
    tr.updates = [msg(1, "/status", chat=999), msg(2, "/status", chat=999)]
    with caplog.at_level("WARNING"):
        notifier.poll_updates()
    assert tr.sent("sendMessage") == []
    assert sum("ignoring Telegram chat 999" in r.message for r in caplog.records) == 1


def test_status_alerts_and_snapshot(env, tmp_path):
    cfg, db, tr, notifier, grabbed = env
    from datetime import datetime, timezone

    from nab_sentry.store.db import NewVideo
    src = tmp_path / "CAM01_x.mkv"
    src.write_bytes(b"x")
    vid = db.insert_video(NewVideo("CAM01", str(src), "h", datetime.fromtimestamp(NOW - 100, timezone.utc), 10, 64, 48, 10))
    db.finalize_video(vid, sampled=1, passed=1, duration_s=10.0, playback_path="v.mp4")
    db.insert_alert(NewAlert("rule", round((NOW - 50) * 1000), rule_id="yard", rule_name="Person in the yard",
                             camera_id="CAM01", state="sent"))
    tr.updates = [msg(1, "/status"), msg(2, "/alerts"), msg(3, "/snapshot"), msg(4, "/snapshot CAM09")]
    notifier.poll_updates()
    texts = [d["text"] for _, d in tr.sent("sendMessage")]
    assert "Front gate (CAM01): OFFLINE, last footage 1 min ago" in texts[0]
    assert "Person in the yard" in texts[1]
    assert texts[2].startswith("Unknown camera CAM09")
    [(chat, photo)] = tr.sent("sendPhoto")
    assert grabbed == [src] and "Front gate" in photo["caption"]


def test_feedback_button_updates_alert_and_caption(env):
    cfg, db, tr, notifier, _ = env
    aid = db.insert_alert(NewAlert("rule", round(NOW * 1000), rule_id="yard", rule_name="Person in the yard",
                                   camera_id="CAM01", state="sent"))
    cb = {"update_id": 1, "callback_query": {"id": "cb1", "data": f"fb:{aid}:false",
                                             "message": {"message_id": 77, "chat": {"id": ALLOWED}, "photo": [{}]}}}
    bad = {"update_id": 2, "callback_query": {"id": "cb2", "data": "fb:999:real",
                                              "message": {"message_id": 78, "chat": {"id": ALLOWED}}}}
    tr.updates = [cb, bad]
    notifier.poll_updates()
    assert db.get_alert(aid)["feedback"] == "false"
    answers = [d["text"] for _, d in tr.sent("answerCallbackQuery")]
    assert answers == ["Thanks - noted as a false alarm", "Unknown alert"]
    [(_, edit)] = tr.sent("editMessageCaption")
    assert "marked false alarm" in edit["caption"] and edit["message_id"] == 77


def test_a_failing_reply_does_not_block_the_next_update(env):
    _, db, tr, notifier, _ = env
    tr.fail = lambda m, i: TelegramError(400, "chat not found") if m == "sendMessage" and i == 1 else None
    tr.updates = [msg(1, "/help"), msg(2, "/help")]
    notifier.poll_updates()
    assert db.kv_get(OFFSET_KEY) == "3" and len(tr.sent("sendMessage")) == 1


# -- isolation (Requirement 14.8, Property for 15.7) --------------------------------------------


def test_real_transport_only_talks_to_telegram_and_masks_the_token():
    import httpx

    token = "123456:SECRET-token-value"
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/sendMessage"):
            return httpx.Response(200, json={"ok": True, "result": {"message_id": 5}})
        return httpx.Response(429, json={"ok": False, "error_code": 429, "description": "Too Many Requests",
                                         "parameters": {"retry_after": 3}})
    tr = HttpxTransport(token, client=httpx.Client(transport=httpx.MockTransport(handler)))
    client = TelegramClient(tr)
    assert client.send_message(1, "hi") == 5
    with pytest.raises(TelegramError) as exc:
        client.get_updates(None, 1)
    assert exc.value.retry_after == 3
    assert {r.url.host for r in seen} == {API_HOST} and all(r.url.scheme == "https" for r in seen)

    def boom(request):
        raise httpx.ConnectError(f"cannot reach {request.url}")
    tr2 = HttpxTransport(token, client=httpx.Client(transport=httpx.MockTransport(boom)))
    with pytest.raises(TelegramError) as exc:
        TelegramClient(tr2).send_message(1, "hi")
    assert token not in str(exc.value) and "***" in str(exc.value)
