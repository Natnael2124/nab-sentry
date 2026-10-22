"""The Notifier loop (V2 Requirement 14): outbox, bot commands, feedback buttons, health checks.

One thread, one tick at a time: deliver due alerts, then poll Telegram for messages and button
presses with a short long-poll timeout (so alerts never wait behind a poll), then run the
health checks (when given). The ``getUpdates`` offset lives in the store (``kv``), so a restart
never handles a message twice.

Only Allowed_Chats are served; anything from another chat is ignored and logged with its chat
ID once (which is also how an Integrator finds the ID to put into ``NAB_TELEGRAM_CHATS``).
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable, Sequence

from nab_sentry.logging_setup import get_logger
from nab_sentry.notify.commands import Commands, parse_command
from nab_sentry.notify.outbox import Outbox, caption, feedback_buttons
from nab_sentry.notify.telegram import TelegramClient, TelegramError
from nab_sentry.store.db import MetadataStore

log = get_logger("notify")

OFFSET_KEY = "telegram_offset"
FEEDBACK_RE = re.compile(r"^fb:(\d+):(real|false)$")
POLL_TIMEOUT_S = 2


class Notifier:
    def __init__(self, db: MetadataStore, client: TelegramClient, chats: Sequence[int],
                 commands: Commands, outbox: Outbox, *, camera_labels: dict[str, str] | None = None,
                 health: Callable[[float], None] | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.db = db
        self.client = client
        self.chats = set(int(c) for c in chats)
        self.commands = commands
        self.outbox = outbox
        self.labels = camera_labels or {}
        self.health = health
        self.clock = clock
        self._warned_chats: set[int] = set()

    # -- updates ---------------------------------------------------------------------------------

    def _allowed(self, chat_id: int | None) -> bool:
        if chat_id in self.chats:
            return True
        if chat_id is not None and chat_id not in self._warned_chats:
            self._warned_chats.add(chat_id)
            log.warning("ignoring Telegram chat %s (not in NAB_TELEGRAM_CHATS)", chat_id)
        return False

    def handle_update(self, upd: dict) -> None:
        if "callback_query" in upd:
            self._callback(upd["callback_query"])
            return
        msg = upd.get("message") or {}
        chat = (msg.get("chat") or {}).get("id")
        if not self._allowed(chat):
            return
        cmd = parse_command(msg.get("text") or "")
        if cmd is None:
            return  # ordinary text: not for the bot
        reply = self.commands.handle(cmd, self.clock())
        if reply.photo is not None:
            self.client.send_photo(chat, reply.photo, reply.text)
        else:
            self.client.send_message(chat, reply.text)

    def _callback(self, cb: dict) -> None:
        msg = cb.get("message") or {}
        chat = (msg.get("chat") or {}).get("id")
        if not self._allowed(chat):
            return
        m = FEEDBACK_RE.match(cb.get("data") or "")
        alert = self.db.get_alert(int(m.group(1))) if m else None
        if alert is None:
            self.client.answer_callback(cb["id"], "Unknown alert")
            return
        self.db.update_alert(alert["alert_id"], feedback=m.group(2))
        alert = self.db.get_alert(alert["alert_id"])
        self.client.answer_callback(cb["id"], "Thanks - noted as real" if m.group(2) == "real"
                                    else "Thanks - noted as a false alarm")
        if msg.get("message_id") is not None:
            label = self.labels.get(alert.get("camera_id") or "")
            try:
                if msg.get("photo"):
                    self.client.edit_caption(chat, msg["message_id"], caption(alert, label),
                                             buttons=feedback_buttons(alert["alert_id"]))
            except TelegramError as exc:  # e.g. "message is not modified"
                log.info("caption update skipped: %s", exc)

    def poll_updates(self, timeout_s: int = POLL_TIMEOUT_S) -> int:
        raw = self.db.kv_get(OFFSET_KEY)
        updates = self.client.get_updates(int(raw) if raw else None, timeout_s)
        for upd in updates:
            try:
                self.handle_update(upd)
            except TelegramError as exc:
                log.warning("reply failed: %s", exc)
            finally:
                # Advance past this update even if the reply failed: never handle it twice.
                self.db.kv_set(OFFSET_KEY, str(int(upd["update_id"]) + 1))
        return len(updates)

    # -- loop ------------------------------------------------------------------------------------

    def tick(self) -> None:
        self.outbox.tick(self.clock())
        try:
            self.poll_updates()
        except TelegramError as exc:
            log.warning("getUpdates failed: %s", exc)
            if exc.retry_after:
                time.sleep(min(float(exc.retry_after), 30.0))
        if self.health is not None:
            self.health(self.clock())

    def run(self, stop: threading.Event, idle_s: float = 1.0) -> None:
        while not stop.is_set():
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - keep delivering; the error is in the log
                log.exception("notifier tick failed")
                stop.wait(5.0)
            stop.wait(idle_s)
