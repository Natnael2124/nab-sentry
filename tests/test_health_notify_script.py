"""Health alerts (V2 Requirements 15.1-15.2, task 16.1) and scripts/notify.py (task 15.6)."""

from __future__ import annotations

import importlib.util
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from nab_sentry.config import Config
from nab_sentry.live.recorder import Heartbeat, heartbeat_path
from nab_sentry.notify.health import DISK_REPEAT_S, HealthMonitor
from nab_sentry.store.db import MetadataStore, NewAlert, NewVideo
from tests.fakes import FakeTelegramTransport

T0 = 1_800_000_000.0
CAMS = {"CAM01": ("Front gate", 10.0), "CAM02": ("Back door", 10.0)}


def kinds(db):
    return [(a["kind"], a["camera_id"]) for a in db.alerts_between(0, 2**62)]


def add_video(db, cam, start_s, dur=10.0, i=[0]):
    i[0] += 1
    vid = db.insert_video(NewVideo(cam, f"x{i[0]}.mkv", f"h{i[0]}", datetime.fromtimestamp(start_s, timezone.utc),
                                   10, 64, 48, dur))
    db.finalize_video(vid, sampled=1, passed=1, duration_s=dur, playback_path=f"v{vid}.mp4")


def test_offline_then_recovered_once_each_and_survives_restart(tmp_path: Path):
    cfg = Config(root=tmp_path)
    db = MetadataStore(cfg.db_path)
    plenty = lambda p: 500e9  # noqa: E731
    hm = HealthMonitor(cfg, db, CAMS, started_s=T0, disk_free=plenty)
    add_video(db, "CAM01", T0 - 10)
    add_video(db, "CAM02", T0 - 10)
    assert hm.check(T0 + 25) == []  # footage ended at T0: 25 s silent < 3 x 10 s
    assert hm.check(T0 + 35) == ["camera_offline", "camera_offline"]  # 35 s > 30 s
    assert hm.check(T0 + 60) == []  # raised once
    hm2 = HealthMonitor(cfg, db, CAMS, started_s=T0 + 61, disk_free=plenty)  # restart
    assert hm2.check(T0 + 70) == []
    # CAM01 footage resumes (heartbeat of the recorder) -> recovered once
    Heartbeat("CAM01", "running", 0, 1, "CAM01_x.mkv", T0 + 80, T0 + 80).write(heartbeat_path(cfg.live_dir, "CAM01"))
    assert hm2.check(T0 + 85) == ["camera_recovered"]
    assert hm2.check(T0 + 90) == []
    assert kinds(db) == [("camera_offline", "CAM01"), ("camera_offline", "CAM02"), ("camera_recovered", "CAM01")]
    db.close()


def test_camera_that_never_recorded_counts_from_start(tmp_path: Path):
    cfg = Config(root=tmp_path)
    db = MetadataStore(cfg.db_path)
    hm = HealthMonitor(cfg, db, {"CAM03": ("New", 10.0)}, started_s=T0, disk_free=lambda p: 500e9)
    assert hm.check(T0 + 20) == []
    assert hm.check(T0 + 31) == ["camera_offline"]
    assert db.alerts_between(0, 2**62)[0]["note"] == "no footage received yet"
    db.close()


def test_disk_low_at_most_every_six_hours(tmp_path: Path):
    cfg = Config(root=tmp_path, disk_low_gb=10.0)
    db = MetadataStore(cfg.db_path)
    hm = HealthMonitor(cfg, db, {}, started_s=T0, disk_free=lambda p: 4.2e9)
    assert hm.check(T0) == ["disk_low"]
    assert hm.check(T0 + 3600) == []
    assert hm.check(T0 + DISK_REPEAT_S) == ["disk_low"]
    assert "4.2 GB free" in db.alerts_between(0, 2**62)[0]["note"]
    db.close()


# -- scripts/notify.py -----------------------------------------------------------------------------

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "notify.py"


def _load():
    spec = importlib.util.spec_from_file_location("notify_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ns = _load()


def test_parse_chats():
    assert ns.parse_chats("123, -456") == [123, -456]
    for bad in (None, "", "abc", "12,x"):
        with pytest.raises(ValueError):
            ns.parse_chats(bad)


def test_refuses_to_start_without_token_or_chats(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("NAB_TELEGRAM_TOKEN", raising=False)
    monkeypatch.setenv("NAB_TELEGRAM_CHATS", "123")
    assert ns.main(["--set", f"root={tmp_path}"]) == 2
    assert "NAB_TELEGRAM_TOKEN is not set" in capsys.readouterr().err
    monkeypatch.setenv("NAB_TELEGRAM_TOKEN", "1:secret")
    monkeypatch.delenv("NAB_TELEGRAM_CHATS", raising=False)
    assert ns.main(["--set", f"root={tmp_path}"]) == 2
    err = capsys.readouterr().err
    assert "NAB_TELEGRAM_CHATS" in err and "1:secret" not in err


def test_runs_and_delivers_a_pending_alert(monkeypatch, tmp_path):
    monkeypatch.setenv("NAB_TELEGRAM_CHATS", "123")
    cfg = Config(root=tmp_path)
    db = MetadataStore(cfg.db_path)
    db.insert_alert(NewAlert("disk_low", 1000, note="test"))
    db.close()
    tr = FakeTelegramTransport()
    stop = threading.Event()
    threading.Timer(2.0, stop.set).start()
    assert ns.main(["--set", f"root={tmp_path}", "--cameras", str(tmp_path / "none.yaml"),
                    "--rules", str(tmp_path / "none.yaml")], transport=tr, stop=stop) == 0
    assert [(chat, d["text"].splitlines()[0]) for chat, d in tr.sent("sendMessage")][:1] == [(123, "⚠️ Disk space low")]
    db = MetadataStore(cfg.db_path)
    assert db.alerts_between(0, 2**62)[0]["state"] == "sent"
    db.close()
