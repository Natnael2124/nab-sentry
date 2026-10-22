"""/today, /ask, /describe and the /ask search (Step 3, Requirement 20, task 18.4)."""

from __future__ import annotations

from pathlib import Path

from nab_sentry.notify.commands import Commands, parse_command
from nab_sentry.notify.search import TodaySearch
from tests.fakes import FakeEncoder
from tests.test_pipeline_activities import ingest
from tests.test_pipeline_tracks import frames, name


def test_parse_new_commands():
    assert parse_command("/ask person in a red top").args == ("person in a red top",)
    assert parse_command("/ask").error is not None
    assert parse_command("/ask " + "x" * 300).error is not None
    assert parse_command("/describe #12").args == ("12",)
    assert parse_command("/describe twelve").error is not None
    assert parse_command("/today CAM-A").args == ("CAM-A",)
    assert parse_command("/today").args == ()


def _commands(db, cfg, **kw):
    return Commands(db, {"CAM-A": "Back yard"}, [], live_dir=cfg.live_dir, offline_after_s=lambda c: 30.0,
                    thumbs_dir=cfg.thumbs_dir, **kw)


def test_today_lists_activities(tmp_path: Path):
    cfg, db, _ = ingest(tmp_path, {name(0): frames(0, 25)})
    try:
        [a] = db.activities_between(0, 2**62)
        now = a["last_epoch_ms"] / 1000 + 10
        cmds = _commands(db, cfg)
        text = cmds.handle(parse_command("/today"), now).text
        assert text.startswith("Today (1):") and f"#{a['activity_id']} " in text and "Back yard" in text
        assert cmds.handle(parse_command("/today CAM-Z"), now).text.startswith("Unknown camera")
        assert cmds.handle(parse_command("/today"), now + 86400 * 2).text == "Nothing recorded today yet."
    finally:
        db.close()


def test_describe_caches_the_vlm_answer(tmp_path: Path):
    cfg, db, _ = ingest(tmp_path, {name(0): frames(0, 25)})
    calls = []

    def describer(activity):
        calls.append(activity["activity_id"])
        return "A man walks across the lawn from left to right."
    try:
        [a] = db.activities_between(0, 2**62)
        cmds = _commands(db, cfg, describer=describer)
        first = cmds.handle(parse_command(f"/describe {a['activity_id']}"), 0).text
        second = cmds.handle(parse_command(f"/describe {a['activity_id']}"), 0).text
        assert first == second == f"#{a['activity_id']} A man walks across the lawn from left to right."
        assert calls == [a["activity_id"]]  # second answer came from the cache
        assert cmds.handle(parse_command("/describe 999"), 0).text == "No activity #999."
        other = db.activities_between(0, 2**62)[0]["activity_id"]
        db._conn.execute("UPDATE activities SET vlm_text = NULL")  # forget the cached answer
        assert "not available" in _commands(db, cfg).handle(parse_command(f"/describe {other}"), 0).text
    finally:
        db.close()


def test_ask_with_the_real_search_over_today(tmp_path: Path):
    cfg, db, pipe = ingest(tmp_path, {name(0): frames(0, 25)})
    try:
        [a] = db.activities_between(0, 2**62)
        now = a["last_epoch_ms"] / 1000 + 10
        search = TodaySearch(cfg, db, lambda: FakeEncoder())
        hits = search("person", now - 3600, now)
        assert hits and hits[0]["camera_id"] == "CAM-A" and hits[0]["thumb_path"]
        reply = _commands(db, cfg, searcher=search).handle(parse_command("/ask person"), now)
        assert reply.text.startswith('Best match for "person": Back yard at') and reply.photo is not None
        assert f"#{a['activity_id']}" in reply.text
        assert search("person", now + 10, now + 20) == []  # outside the window
        assert _commands(db, cfg).handle(parse_command("/ask x"), now).text == "Search is not available here."
    finally:
        db.close()
