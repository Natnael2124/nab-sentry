"""Minimal Telegram Bot API client (V2 Requirement 14).

:class:`TelegramClient` wraps the few methods NAB Sentry uses over a :class:`Transport`, so tests
inject a fake one. :class:`HttpxTransport` is the real one: HTTPS POSTs to ``api.telegram.org``
only. The bot token is part of every request URL, so it is masked in every error message this
module raises (``httpx`` puts the URL into its exceptions).

Errors are raised as :class:`TelegramError` with ``retry_after`` set when Telegram asks for a
pause (HTTP 429 / "Too Many Requests").
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

API_HOST = "api.telegram.org"
API_BASE = f"https://{API_HOST}"
REQUEST_TIMEOUT_S = 30.0

Buttons = Sequence[Sequence[tuple[str, str]]]  # rows of (text, callback_data)


class TelegramError(Exception):
    def __init__(self, status: int, description: str, retry_after: float | None = None) -> None:
        super().__init__(f"Telegram error {status}: {description}"
                         + (f" (retry after {retry_after:.0f} s)" if retry_after else ""))
        self.status = status
        self.description = description
        self.retry_after = retry_after


class Transport(Protocol):
    def call(self, method: str, data: dict[str, Any], files: dict[str, tuple[str, bytes]] | None,
             timeout: float) -> Any:
        """POST one Bot API method; return its ``result`` or raise :class:`TelegramError`."""
        ...


def mask_token(text: str, token: str) -> str:
    return text.replace(token, "***") if token else text


class HttpxTransport:
    """The real transport: HTTPS to ``api.telegram.org`` with ``httpx``."""

    def __init__(self, token: str, client: Any = None) -> None:
        import httpx

        self._token = token
        self._client = client or httpx.Client(timeout=REQUEST_TIMEOUT_S)

    def url(self, method: str) -> str:
        return f"{API_BASE}/bot{self._token}/{method}"

    def call(self, method: str, data: dict[str, Any], files: dict[str, tuple[str, bytes]] | None,
             timeout: float) -> Any:
        import httpx

        form = {k: (json.dumps(v) if isinstance(v, (dict, list)) else str(v)) for k, v in data.items()}
        try:
            resp = self._client.post(self.url(method), data=form, files=files, timeout=timeout)
        except httpx.HTTPError as exc:
            raise TelegramError(0, mask_token(f"{type(exc).__name__}: {exc}", self._token)) from None
        try:
            body = resp.json()
        except ValueError:
            raise TelegramError(resp.status_code, f"non-JSON response ({resp.status_code})") from None
        if not body.get("ok"):
            params = body.get("parameters") or {}
            raise TelegramError(int(body.get("error_code", resp.status_code)),
                                mask_token(str(body.get("description", "")), self._token),
                                params.get("retry_after"))
        return body.get("result")

    def close(self) -> None:
        self._client.close()


def _keyboard(buttons: Buttons | None) -> dict | None:
    if not buttons:
        return None
    return {"inline_keyboard": [[{"text": t, "callback_data": d} for t, d in row] for row in buttons]}


class TelegramClient:
    def __init__(self, transport: Transport) -> None:
        self.transport = transport

    def _send(self, method: str, data: dict[str, Any], files=None, timeout=REQUEST_TIMEOUT_S) -> Any:
        return self.transport.call(method, {k: v for k, v in data.items() if v is not None}, files, timeout)

    def send_message(self, chat_id: int, text: str, *, buttons: Buttons | None = None,
                     reply_to: int | None = None) -> int:
        res = self._send("sendMessage", {"chat_id": chat_id, "text": text,
                                         "reply_markup": _keyboard(buttons),
                                         "reply_to_message_id": reply_to})
        return int(res["message_id"])

    def send_photo(self, chat_id: int, path: Path, caption: str, *, buttons: Buttons | None = None) -> int:
        data = Path(path).read_bytes()
        res = self._send("sendPhoto", {"chat_id": chat_id, "caption": caption,
                                       "reply_markup": _keyboard(buttons)},
                         {"photo": (Path(path).name, data)})
        return int(res["message_id"])

    def send_video(self, chat_id: int, path: Path, caption: str, *, reply_to: int | None = None) -> int:
        data = Path(path).read_bytes()
        res = self._send("sendVideo", {"chat_id": chat_id, "caption": caption, "supports_streaming": True,
                                       "reply_to_message_id": reply_to},
                         {"video": (Path(path).name, data)}, timeout=120.0)
        return int(res["message_id"])

    def edit_caption(self, chat_id: int, message_id: int, caption: str, *,
                     buttons: Buttons | None = None) -> None:
        self._send("editMessageCaption", {"chat_id": chat_id, "message_id": message_id,
                                          "caption": caption, "reply_markup": _keyboard(buttons)})

    def answer_callback(self, callback_id: str, text: str) -> None:
        self._send("answerCallbackQuery", {"callback_query_id": callback_id, "text": text})

    def get_updates(self, offset: int | None, timeout_s: int) -> list[dict]:
        res = self._send("getUpdates", {"offset": offset, "timeout": timeout_s,
                                        "allowed_updates": ["message", "callback_query"]},
                         timeout=timeout_s + 10.0)
        return list(res or [])
