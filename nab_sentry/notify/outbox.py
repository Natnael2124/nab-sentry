"""Alert delivery state machine (V2 Requirements 13.3, 14.2-14.4; design "Notifier").

::

    pending --photo accepted by every chat--> photo_sent --clip sent / not possible--> sent
       |                                          |
       +--- error: attempts += 1, back off -------+--> failed (attempts >= max_attempts)
    suppressed (muted at creation; never sent)

* Every Telegram message is recorded in ``alert_messages`` as ``(alert, chat, step)`` the moment
  Telegram accepts it, before anything else happens, so a crash or restart never sends the same
  step to the same chat twice (Property 20). The only window for a duplicate is a crash between
  Telegram accepting a message and the record being written.
* A rule alert sends its snapshot photo first (seconds after the Segment is ingested), then the
  clip as a reply to it. The clip waits for the next Segment when its window runs past the
  stored footage (:func:`nab_sentry.alerts.clips.clip_status`); if it cannot be made the alert
  is still ``sent`` with the reason in ``note`` (Requirement 13.3).
* Health alerts (``camera_offline``, ``camera_recovered``, ``disk_low``) are one text message.
* ``retry_after`` from Telegram is honoured without counting an attempt; other errors count
  and back off ``min(300, 5 * 2^(attempts-1))`` seconds.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

from nab_sentry.alerts.clips import Seg, clip_status, covering, make_clip
from nab_sentry.config import Config
from nab_sentry.logging_setup import get_logger
from nab_sentry.notify.telegram import TelegramClient, TelegramError
from nab_sentry.store.db import MetadataStore

log = get_logger("notify")

BACKOFF_BASE_S, BACKOFF_MAX_S = 5.0, 300.0
CLIP_POLL_S = 3.0  # re-check for the next Segment this often while a clip waits
HEALTH_TEXT = {
    "camera_offline": "⚠️ Camera offline",
    "camera_recovered": "✅ Camera back online",
    "disk_low": "⚠️ Disk space low",
}


def backoff_s(attempts: int) -> float:
    return min(BACKOFF_MAX_S, BACKOFF_BASE_S * 2 ** max(0, attempts - 1))


def feedback_buttons(alert_id: int) -> list[list[tuple[str, str]]]:
    return [[("✅ Real", f"fb:{alert_id}:real"), ("❌ False alarm", f"fb:{alert_id}:false")]]


def caption(alert: dict, camera_label: str | None) -> str:
    when = datetime.fromtimestamp(alert["trigger_epoch_ms"] / 1000).strftime("%a %d %b %H:%M:%S")
    cam = camera_label or alert.get("camera_id") or ""
    if alert["kind"] != "rule":
        head = HEALTH_TEXT.get(alert["kind"], alert["kind"])
        lines = [head, f"{cam} · {when}" if cam else when]
    else:
        what = alert.get("det_class") or "object"
        if (alert.get("n_tracks") or 1) > 1:
            what = f"{alert['n_tracks']} × {what}"
        lines = [f"🚨 {alert.get('rule_name') or alert['rule_id']}", f"{cam} · {when}", what]
    if alert.get("feedback"):
        lines.append("✅ marked real" if alert["feedback"] == "real" else "❌ marked false alarm")
    if alert.get("note"):
        lines.append(alert["note"])
    return "\n".join(lines)


class Outbox:
    def __init__(self, cfg: Config, db: MetadataStore, client: TelegramClient, chats: Sequence[int], *,
                 camera_labels: dict[str, str] | None = None,
                 segment_s: Callable[[str], float] = lambda cam: 10.0,
                 clip_maker: Callable[..., Path] = make_clip) -> None:
        self.cfg = cfg
        self.db = db
        self.client = client
        self.chats = list(chats)
        self.labels = camera_labels or {}
        self.segment_s = segment_s
        self.clip_maker = clip_maker

    def tick(self, now_s: float, limit: int = 20) -> int:
        """Advance every due alert by one step; returns how many were worked on."""
        due = self.db.due_alerts(round(now_s * 1000), limit)
        for alert in due:
            try:
                self._advance(alert, now_s)
            except TelegramError as exc:
                self._failed_try(alert, now_s, exc, exc.retry_after)
            except OSError as exc:
                self._failed_try(alert, now_s, exc, None)
        return len(due)

    # -- steps -----------------------------------------------------------------------------------

    def _advance(self, alert: dict, now_s: float) -> None:
        aid = alert["alert_id"]
        sent = self.db.sent_steps(aid)
        label = self.labels.get(alert.get("camera_id") or "")
        if alert["kind"] != "rule":
            for chat in self.chats:
                if (chat, "text") not in sent:
                    self._record(aid, chat, "text", self.client.send_message(chat, caption(alert, label)))
            self.db.update_alert(aid, state="sent", attempts=0)
            return
        if alert["state"] == "pending":
            snap = self.cfg.alerts_dir / alert["snapshot_path"] if alert.get("snapshot_path") else None
            for chat in self.chats:
                if (chat, "photo") in sent:
                    continue
                if snap is not None and snap.is_file():
                    mid = self.client.send_photo(chat, snap, caption(alert, label), buttons=feedback_buttons(aid))
                else:
                    mid = self.client.send_message(chat, caption(alert, label), buttons=feedback_buttons(aid))
                self._record(aid, chat, "photo", mid)
            self.db.update_alert(aid, state="photo_sent", attempts=0, next_attempt_epoch_ms=round(now_s * 1000))
            alert = self.db.get_alert(aid)  # type: ignore[assignment]
            sent = self.db.sent_steps(aid)
        # photo_sent: the clip
        if not alert.get("clip_path") and not alert.get("note"):
            if not self._prepare_clip(alert, now_s):
                return  # waiting for the next Segment
            alert = self.db.get_alert(aid)  # type: ignore[assignment]
        clip = self.cfg.alerts_dir / alert["clip_path"] if alert.get("clip_path") else None
        if clip is not None and clip.is_file():
            for chat in self.chats:
                if (chat, "clip") not in sent:
                    mid = self.client.send_video(chat, clip, caption(alert, label),
                                                 reply_to=sent.get((chat, "photo")))
                    self._record(aid, chat, "clip", mid)
        self.db.update_alert(aid, state="sent", attempts=0)

    def _prepare_clip(self, alert: dict, now_s: float) -> bool:
        """Cut the clip if its footage is ready; False while waiting for the next Segment. A clip
        that cannot be made leaves a note instead (the alert is still delivered)."""
        trig = alert["trigger_epoch_ms"] / 1000
        start, end = trig - self.cfg.alert_clip_pre_s, trig + self.cfg.alert_clip_post_s
        cam = alert.get("camera_id") or ""
        segs = [Seg(Path(p), ms / 1000, d) for p, ms, d in
                self.db.clip_segments(cam, round(start * 1000), round(end * 1000)) if Path(p).is_file()]
        status = clip_status(segs, end, now_s, self.segment_s(cam) + 15.0)
        if status == "wait":
            self.db.update_alert(alert["alert_id"], next_attempt_epoch_ms=round((now_s + CLIP_POLL_S) * 1000))
            return False
        if status == "none":
            self.db.update_alert(alert["alert_id"], note="(no video clip: footage not available)")
            return True
        name = Path(alert["snapshot_path"] or f"a_v0_{alert['rule_id']}_{alert['trigger_epoch_ms']}.jpg").with_suffix(".mp4").name
        try:
            self.clip_maker(covering(segs, start, end), self.cfg.alerts_dir / name, self.cfg.alert_clip_width)
        except Exception as exc:  # noqa: BLE001 - the alert still goes out with its photo
            log.warning("clip for alert %s failed: %s", alert["alert_id"], exc)
            self.db.update_alert(alert["alert_id"], note="(no video clip: could not cut it)")
            return True
        self.db.update_alert(alert["alert_id"], clip_path=name)
        return True

    def _record(self, alert_id: int, chat: int, step: str, message_id: int) -> None:
        self.db.record_message(alert_id, chat, step, message_id)

    def _failed_try(self, alert: dict, now_s: float, exc: Exception, retry_after: float | None) -> None:
        aid = alert["alert_id"]
        current = self.db.get_alert(aid) or alert
        if retry_after:
            self.db.update_alert(aid, next_attempt_epoch_ms=round((now_s + float(retry_after)) * 1000))
            log.warning("alert %s: Telegram asks to wait %.0f s", aid, retry_after)
            return
        attempts = int(current["attempts"]) + 1
        if attempts >= self.cfg.alert_max_attempts:
            self.db.update_alert(aid, state="failed", attempts=attempts, note=f"delivery failed: {exc}")
            log.error("alert %s failed after %d attempts: %s", aid, attempts, exc)
            return
        self.db.update_alert(aid, attempts=attempts,
                             next_attempt_epoch_ms=round((now_s + backoff_s(attempts)) * 1000))
        log.warning("alert %s attempt %d failed: %s", aid, attempts, exc)
