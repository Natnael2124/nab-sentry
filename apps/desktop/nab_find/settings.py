"""App settings in ``settings.json`` under the data folder; the Gemini key encrypted with Windows DPAPI
(readable only by this Windows user on this PC). The key is never logged or returned to the UI."""

from __future__ import annotations

import base64
import ctypes
import json
import os
import threading
from ctypes import wintypes
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "gemini_model": "gemini-3.5-flash-lite",
    "ai_moments": 20,
    "device": "auto",  # auto | CPU | GPU
    "speeds": {},  # measured footage speeds per pass, seeds the time-left estimate
}
# USD per 1M tokens (input, output) from ai.google.dev/gemini-api/docs/pricing, checked 2026-10-29.
# 3.8 Flash is at an introductory price until 2026-12-31 ($1.50 / $7.50 from 2027-01-01).
MODELS = {
    "gemini-3.5-flash-lite": {"label": "3.5 Flash-Lite · cheapest", "price": (0.30, 2.50)},
    "gemini-3.8-flash": {"label": "3.8 Flash · sharper", "price": (0.75, 3.75)},
}


def default_data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "NAB Find"


class _Blob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]


def _blob(data: bytes) -> _Blob:
    buf = ctypes.create_string_buffer(data, len(data))
    return _Blob(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))


def _dpapi(data: bytes, encrypt: bool) -> bytes:
    crypt32, kernel32 = ctypes.windll.crypt32, ctypes.windll.kernel32
    src, out = _blob(data), _Blob()
    fn = crypt32.CryptProtectData if encrypt else crypt32.CryptUnprotectData
    ok = fn(ctypes.byref(src), None, None, None, None, 0x1, ctypes.byref(out))  # CRYPTPROTECT_UI_FORBIDDEN
    if not ok:
        raise OSError("DPAPI failed")
    try:
        return ctypes.string_at(out.pbData, out.cbData)
    finally:
        kernel32.LocalFree(out.pbData)


def protect(text: str) -> str:
    return base64.b64encode(_dpapi(text.encode("utf-8"), True)).decode("ascii")


def unprotect(token: str) -> str:
    return _dpapi(base64.b64decode(token), False).decode("utf-8")


class Settings:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.path = self.data_dir / "settings.json"
        self._lock = threading.Lock()
        self._d = dict(DEFAULTS)
        if self.path.is_file():
            try:
                self._d.update(json.loads(self.path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                pass

    def _save(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._d, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def get(self, key: str) -> Any:
        with self._lock:
            return self._d.get(key, DEFAULTS.get(key))

    def update(self, **kv: Any) -> None:
        with self._lock:
            for k, v in kv.items():
                if k in DEFAULTS:
                    self._d[k] = v
            self._save()

    # -- Gemini key
    def gemini_key(self) -> str | None:
        env = os.environ.get("NAB_FIND_GEMINI_KEY") or os.environ.get("GEMINI_API_KEY")
        if env:
            return env.strip()
        with self._lock:
            tok = self._d.get("gemini_key_enc")
        if not tok:
            return None
        try:
            return unprotect(tok)
        except OSError:
            return None

    def set_gemini_key(self, key: str | None) -> None:
        with self._lock:
            if key:
                self._d["gemini_key_enc"] = protect(key.strip())
            else:
                self._d.pop("gemini_key_enc", None)
            self._save()

    def public(self) -> dict[str, Any]:
        """What the UI may see: everything except the key (only whether one is set)."""
        with self._lock:
            d = {k: v for k, v in self._d.items() if k != "gemini_key_enc"}
        d["gemini_key_set"] = self.gemini_key() is not None
        d["models"] = MODELS
        d["data_dir"] = str(self.data_dir)
        return d
