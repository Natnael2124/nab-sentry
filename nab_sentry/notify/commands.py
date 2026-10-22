"""Telegram bot commands (V2 Requirement 14.5-14.7) - deterministic, no language model.

Grammar (``/name@botname`` is accepted, names are case-insensitive)::

    /start | /help
    /status                       cameras, last footage, alerts waiting
    /alerts                       today's alerts
    /rules                        the configured rules and their mutes
    /snapshot [CAMERA_ID]         a recent frame (the newest stored Segment)
    /mute RULE_ID [MINUTES]       default 60, 1..10080 (a week)
    /unmute RULE_ID
    /today [CAMERA_ID]            today's activity log (Step 3)
    /ask WORDS                    best match of a text search over today's footage
    /describe ACTIVITY_ID         a description by the on-demand VLM (slow: about a minute)

:func:`parse_command` is pure; anything else (or bad arguments) gives a :class:`Command` with an
``error`` that the bot answers with a short usage line.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from nab_sentry.ingest.transcode import ffmpeg_exe
from nab_sentry.rules.config import Rule
from nab_sentry.store.db import MetadataStore

NAMES = ("start", "help", "status", "alerts", "rules", "snapshot", "mute", "unmute", "today", "ask",
         "describe")
ASK_MAX_LEN = 200
ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
MUTE_DEFAULT_MIN, MUTE_MAX_MIN = 60, 7 * 24 * 60
USAGE = {
    "status": "/status", "alerts": "/alerts", "rules": "/rules", "snapshot": "/snapshot [CAMERA]",
    "mute": "/mute RULE [MINUTES]", "unmute": "/unmute RULE", "help": "/help", "start": "/start",
    "today": "/today [CAMERA]", "ask": "/ask WORDS", "describe": "/describe ACTIVITY_NUMBER",
}
HELP = ("NAB Sentry\n"
        "/status - cameras and alerts waiting\n"
        "/alerts - today's alerts\n"
        "/rules - watch rules and mutes\n"
        "/snapshot [camera] - latest picture\n"
        "/mute rule [minutes] - pause a rule (default 60)\n"
        "/unmute rule - resume it\n"
        "/today [camera] - today's activity log\n"
        "/ask words - find it in today's footage\n"
        "/describe number - describe an activity in words (about a minute)\n"
        "Alerts have buttons to mark them real or a false alarm.")


@dataclass(frozen=True)
class Command:
    name: str
    args: tuple[str, ...] = ()
    error: str | None = None


def parse_command(text: str) -> Command | None:
    """``None`` for text that is not a command at all."""
    if not isinstance(text, str):
        return None
    parts = text.strip().split()
    if not parts or not parts[0].startswith("/") or len(parts[0]) < 2:
        return None
    name = parts[0][1:].split("@", 1)[0].lower()
    args = tuple(parts[1:])
    if name not in NAMES:
        return Command("help", (), f"unknown command /{name}")
    if name in ("start", "help", "status", "alerts", "rules"):
        return Command(name) if not args else Command(name, (), f"usage: {USAGE[name]}")
    if name == "ask":
        words = " ".join(args).strip()
        if not words or len(words) > ASK_MAX_LEN:
            return Command(name, (), f"usage: {USAGE[name]} (1-{ASK_MAX_LEN} characters)")
        return Command(name, (words,))
    if name == "describe":
        if len(args) != 1 or not args[0].lstrip("#").isdigit():
            return Command(name, (), f"usage: {USAGE[name]}")
        return Command(name, (args[0].lstrip("#"),))
    if name in ("snapshot", "today"):
        if len(args) > 1 or (args and not ID_RE.fullmatch(args[0])):
            return Command(name, (), f"usage: {USAGE[name]}")
        return Command(name, args)
    if name == "unmute":
        if len(args) != 1 or not ID_RE.fullmatch(args[0]):
            return Command(name, (), f"usage: {USAGE[name]}")
        return Command(name, args)
    # mute
    if not 1 <= len(args) <= 2 or not ID_RE.fullmatch(args[0]):
        return Command(name, (), f"usage: {USAGE[name]}")
    minutes = MUTE_DEFAULT_MIN
    if len(args) == 2:
        if not args[1].isdigit() or not 1 <= int(args[1]) <= MUTE_MAX_MIN:
            return Command(name, (), f"minutes must be 1 to {MUTE_MAX_MIN}")
        minutes = int(args[1])
    return Command(name, (args[0], str(minutes)))


@dataclass(frozen=True)
class Reply:
    text: str
    photo: Path | None = None


def grab_last_frame(video: Path, out: Path) -> Path:
    """A frame from the last second of ``video`` as JPEG (``/snapshot``)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                           "-sseof", "-1", "-i", str(video), "-frames:v", "1", "-q:v", "3", str(out)],
                          capture_output=True, timeout=60)
    if proc.returncode != 0 or not out.is_file():
        raise RuntimeError(proc.stderr.decode("utf-8", errors="replace").strip()[-200:] or "no frame")
    return out


class Commands:
    def __init__(self, db: MetadataStore, cameras: dict[str, str], rules: Sequence[Rule], *,
                 live_dir: Path, offline_after_s: Callable[[str], float],
                 frame_grabber: Callable[[Path, Path], Path] = grab_last_frame,
                 thumbs_dir: Path | None = None,
                 searcher: Callable[[str, float, float], list[dict]] | None = None,
                 describer: Callable[[dict], str] | None = None) -> None:
        """``searcher(words, from_s, to_s)`` -> hits ``{camera_id, start_s, thumb_path}`` best
        first; ``describer(activity)`` -> text (the on-demand VLM). Both optional."""
        self.db = db
        self.cameras = cameras  # camera_id -> label
        self.rules = {r.id: r for r in rules}
        self.live_dir = live_dir
        self.offline_after_s = offline_after_s
        self.grab = frame_grabber
        self.thumbs_dir = thumbs_dir
        self.searcher = searcher
        self.describer = describer

    def handle(self, cmd: Command, now_s: float) -> Reply:
        if cmd.error is not None:
            return Reply(f"{cmd.error}\n\n{HELP}" if cmd.name == "help" else cmd.error)
        return getattr(self, f"_{cmd.name}")(cmd.args, now_s)

    def _start(self, args, now_s) -> Reply:
        return Reply(HELP)

    _help = _start

    def _status(self, args, now_s) -> Reply:
        lines = ["Cameras:"]
        for cam, label in sorted(self.cameras.items()):
            v = self.db.latest_video(cam)
            if v is None:
                lines.append(f"• {label} ({cam}): no footage yet")
                continue
            age = now_s - (v[1] / 1000 + v[2])
            state = "OFFLINE" if age > self.offline_after_s(cam) else "ok"
            lines.append(f"• {label} ({cam}): {state}, last footage {_ago(age)} ago")
        waiting = len(self.db.due_alerts(2**62, 1000))
        failed = sum(1 for a in self._alerts_today(now_s) if a["state"] == "failed")
        lines.append(f"Alerts waiting to be sent: {waiting}" + (f"; failed today: {failed}" if failed else ""))
        return Reply("\n".join(lines))

    def _alerts_today(self, now_s: float) -> list[dict]:
        midnight = datetime.fromtimestamp(now_s).replace(hour=0, minute=0, second=0, microsecond=0)
        return self.db.alerts_between(round(midnight.timestamp() * 1000), round(now_s * 1000) + 1)

    def _alerts(self, args, now_s) -> Reply:
        rows = self._alerts_today(now_s)
        if not rows:
            return Reply("No alerts today.")
        lines = [f"Alerts today ({len(rows)}):"]
        for a in rows[-30:]:
            when = datetime.fromtimestamp(a["trigger_epoch_ms"] / 1000).strftime("%H:%M:%S")
            what = a.get("rule_name") or a["kind"].replace("_", " ")
            cam = self.cameras.get(a.get("camera_id") or "", a.get("camera_id") or "")
            flag = {"suppressed": " (muted)", "failed": " (not delivered)"}.get(a["state"], "")
            fb = {"real": " ✅", "false": " ❌"}.get(a.get("feedback") or "", "")
            lines.append(f"{when} · {cam} · {what}{flag}{fb}")
        if len(rows) > 30:
            lines.insert(1, f"(last 30 of {len(rows)})")
        return Reply("\n".join(lines))

    def _rules(self, args, now_s) -> Reply:
        if not self.rules:
            return Reply("No watch rules are configured (config/rules.yaml).")
        mutes = self.db.active_mutes(round(now_s * 1000))
        lines = ["Watch rules:"]
        for rid, r in sorted(self.rules.items()):
            until = mutes.get(rid)
            m = f" - muted until {datetime.fromtimestamp(until / 1000):%H:%M}" if until else ""
            lines.append(f"• {rid}: {r.name} ({r.camera}){m}")
        return Reply("\n".join(lines))

    def _snapshot(self, args, now_s) -> Reply:
        cams = sorted(self.cameras)
        if not cams:
            return Reply("No cameras are configured.")
        cam = args[0] if args else cams[0]
        if cam not in self.cameras:
            return Reply(f"Unknown camera {cam}. Cameras: {', '.join(cams)}")
        v = self.db.latest_video(cam)
        if v is None or not Path(v[0]).is_file():
            return Reply(f"No footage stored for {cam} yet.")
        try:
            out = self.grab(Path(v[0]), self.live_dir / f"{cam}_telegram.jpg")
        except Exception as exc:  # noqa: BLE001
            return Reply(f"Could not take a picture from {cam}: {exc}")
        age = now_s - (v[1] / 1000 + v[2])
        return Reply(f"{self.cameras[cam]} ({cam}), {_ago(age)} ago", photo=out)

    def _mute(self, args, now_s) -> Reply:
        rid, minutes = args[0], int(args[1])
        if rid not in self.rules:
            return Reply(f"Unknown rule {rid}. Rules: {', '.join(sorted(self.rules)) or 'none'}")
        until = datetime.fromtimestamp(now_s) + timedelta(minutes=minutes)
        self.db.set_mute(rid, round(until.timestamp() * 1000))
        return Reply(f"Muted {rid} until {until:%H:%M} ({minutes} min). Alerts are still recorded.")

    def _midnight(self, now_s: float) -> float:
        return datetime.fromtimestamp(now_s).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

    def _today(self, args, now_s) -> Reply:
        cam = args[0] if args else None
        if cam is not None and cam not in self.cameras:
            return Reply(f"Unknown camera {cam}. Cameras: {', '.join(sorted(self.cameras))}")
        rows = self.db.activities_between(round(self._midnight(now_s) * 1000), round(now_s * 1000) + 1, cam)
        if not rows:
            return Reply("Nothing recorded today yet.")
        lines = [f"Today ({len(rows)}):"]
        for a in rows[-30:]:
            lines.append(f"#{a['activity_id']} {a['summary']}")
        if len(rows) > 30:
            lines.insert(1, f"(last 30 of {len(rows)})")
        lines.append("/describe NUMBER for a description in words")
        return Reply("\n".join(lines))

    def _ask(self, args, now_s) -> Reply:
        if self.searcher is None:
            return Reply("Search is not available here.")
        hits = self.searcher(args[0], self._midnight(now_s), now_s)
        if not hits:
            return Reply(f"Nothing like \"{args[0]}\" today.")
        best = hits[0]
        label = self.cameras.get(best["camera_id"], best["camera_id"])
        when = datetime.fromtimestamp(best["start_s"]).strftime("%H:%M:%S")
        acts = [a for a in self.db.activities_between(round((best["start_s"] - 60) * 1000),
                                                     round((best["start_s"] + 60) * 1000), best["camera_id"])]
        near = min(acts, key=lambda a: abs(a["first_epoch_ms"] / 1000 - best["start_s"])) if acts else None
        text = f"Best match for \"{args[0]}\": {label} at {when}"
        if near is not None:
            text += f"\n#{near['activity_id']} {near['summary']}"
        if len(hits) > 1:
            more = ", ".join(datetime.fromtimestamp(h["start_s"]).strftime("%H:%M") for h in hits[1:4])
            text += f"\nAlso: {more}"
        photo = self.thumbs_dir / best["thumb_path"] if self.thumbs_dir and best.get("thumb_path") else None
        return Reply(text, photo=photo if photo is not None and photo.is_file() else None)

    def _describe(self, args, now_s) -> Reply:
        a = self.db.get_activity(int(args[0]))
        if a is None:
            return Reply(f"No activity #{args[0]}.")
        if a.get("vlm_text"):
            return Reply(f"#{a['activity_id']} {a['vlm_text']}")
        if self.describer is None:
            return Reply("Descriptions are not available here (the on-demand VLM is off).")
        try:
            text = self.describer(a).strip()
        except Exception as exc:  # noqa: BLE001
            return Reply(f"Could not describe #{a['activity_id']}: {exc}")
        self.db.set_activity_vlm(a["activity_id"], text)
        return Reply(f"#{a['activity_id']} {text}")

    def _unmute(self, args, now_s) -> Reply:
        rid = args[0]
        if rid not in self.rules:
            return Reply(f"Unknown rule {rid}.")
        self.db.set_mute(rid, None)
        return Reply(f"{rid} is active again.")


def _ago(seconds: float) -> str:
    s = max(0, int(seconds))
    if s < 90:
        return f"{s} s"
    if s < 90 * 60:
        return f"{s // 60} min"
    return f"{s // 3600} h {s % 3600 // 60} min"
