"""On-demand VLM: describe an activity, verify an alert (Step 3, Requirements 20.3, 21).

Talks to a local Ollama server (loopback only) running ``qwen3-vl:2b-instruct`` by default:
about 40 s per image on the Target_Hardware, so it never runs per event - only for ``/describe``
and for rules with a ``verify`` question. Thinking is switched off (the plain ``qwen3-vl:2b`` tag
thinks regardless and was replaced for that reason), temperature is 0.

* :func:`storyboard` - up to 4 frames spread over a track's time, each cropped around the
  object with padding, in a 2x2 grid labelled ``[1]..[4]`` (the order is the time order).
* :class:`OllamaVLM` - ``describe(image, prompt)`` and ``yes_no(image, question)``.
* :func:`describer_from_config` - the ``/describe`` function for the Notifier.
"""

from __future__ import annotations

import base64
import re
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np

from nab_sentry.config import Config
from nab_sentry.store.db import MetadataStore

PANEL_W, PANEL_H = 384, 288
CROP_PAD = 0.35
DESCRIBE_PROMPT = (
    "These are up to four CCTV frames of the same moment in time order: [1] top-left, [2] top-right, "
    "[3] bottom-left, [4] bottom-right. In at most two sentences, say who or what is there, what they "
    "look like and what they do. Only describe what is visible.")
YES_NO_SUFFIX = " Answer with one word, yes or no, then one short sentence of reason."


class VLMError(RuntimeError):
    pass


def _loopback(host: str) -> bool:
    return bool(re.match(r"^http://(127\.0\.0\.1|localhost)(:\d+)?/?$", host))


def parse_yes_no(text: str) -> bool | None:
    """The first word decides: yes -> True, no -> False, anything else -> None."""
    m = re.match(r"\W*(yes|no)\b", text.strip().lower())
    return None if m is None else m.group(1) == "yes"


def fit_panel(image: np.ndarray, w: int = PANEL_W, h: int = PANEL_H) -> np.ndarray:
    panel = np.zeros((h, w, 3), np.uint8)
    ih, iw = image.shape[:2]
    if ih == 0 or iw == 0:
        return panel
    s = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    panel[(h - nh) // 2:(h - nh) // 2 + nh, (w - nw) // 2:(w - nw) // 2 + nw] = cv2.resize(
        image, (nw, nh), interpolation=cv2.INTER_AREA)
    return panel


def storyboard(crops: Sequence[np.ndarray]) -> np.ndarray:
    """2x2 grid of up to four crops in time order, labelled [1]..[4]."""
    panels = []
    for i in range(4):
        p = fit_panel(crops[i]) if i < len(crops) else np.zeros((PANEL_H, PANEL_W, 3), np.uint8)
        if i < len(crops):
            for color, th in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                cv2.putText(p, f"[{i + 1}]", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, th, cv2.LINE_AA)
        panels.append(p)
    return np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])


def padded(image: np.ndarray, box: Sequence[int], pad: float = CROP_PAD) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
    return image[max(0, int(y1 - dy)):min(h, int(y2 + dy)), max(0, int(x1 - dx)):min(w, int(x2 + dx))]


def track_storyboard(db: MetadataStore, track_id: int, n: int = 4) -> np.ndarray:
    """Storyboard of a stored track: ``n`` observations spread over its time, read back from
    the source videos (or their playback copies)."""
    rows = db.track_frames(track_id)
    if not rows:
        raise VLMError("no stored observations for this activity")
    picks = [rows[round(i * (len(rows) - 1) / max(1, n - 1))] for i in range(min(n, len(rows)))]
    crops = []
    for src, offset_s, box in picks:
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_POS_MSEC, offset_s * 1000)
        ok, img = cap.read()
        cap.release()
        if ok and img is not None:
            crops.append(padded(img, box))
    if not crops:
        raise VLMError("the footage of this activity is no longer stored")
    return storyboard(crops)


class OllamaVLM:
    def __init__(self, host: str = "http://127.0.0.1:11434", model: str = "qwen3-vl:2b-instruct",
                 timeout_s: float = 180.0, client=None) -> None:
        if not _loopback(host):
            raise ValueError("the VLM host must be a loopback address (footage never leaves the machine)")
        import httpx

        self.host = host.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self._client = client or httpx.Client(timeout=timeout_s)

    def ask(self, image: np.ndarray, prompt: str, max_tokens: int = 120) -> str:
        import httpx

        ok, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if not ok:
            raise VLMError("could not encode the image")
        body = {"model": self.model, "stream": False, "think": False, "keep_alive": "10m",
                "messages": [{"role": "user", "content": prompt,
                              "images": [base64.b64encode(jpg.tobytes()).decode("ascii")]}],
                "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": 4096}}
        try:
            r = self._client.post(f"{self.host}/api/chat", json=body, timeout=self.timeout_s)
            r.raise_for_status()
            text = (r.json().get("message") or {}).get("content", "").strip()
        except httpx.HTTPError as exc:
            raise VLMError(f"{type(exc).__name__}: {exc}") from None
        if not text:
            raise VLMError("empty answer")
        return text

    def describe(self, image: np.ndarray, prompt: str = DESCRIBE_PROMPT) -> str:
        return self.ask(image, prompt)

    def yes_no(self, image: np.ndarray, question: str) -> tuple[bool | None, str]:
        text = self.ask(image, question.strip() + YES_NO_SUFFIX, max_tokens=40)
        return parse_yes_no(text), text


def describer_from_config(cfg: Config, db: MetadataStore,
                          vlm: OllamaVLM | None = None) -> Callable[[dict], str]:
    model = vlm or OllamaVLM(cfg.vlm_host, cfg.vlm_model, cfg.vlm_timeout_s)

    def describe(activity: dict) -> str:
        return model.describe(track_storyboard(db, int(activity["track_id"])))

    return describe


def verifier_from_config(cfg: Config, vlm: OllamaVLM | None = None) -> Callable[[Path, str], tuple[bool | None, str]]:
    model = vlm or OllamaVLM(cfg.vlm_host, cfg.vlm_model, cfg.vlm_timeout_s)

    def verify(snapshot: Path, question: str) -> tuple[bool | None, str]:
        img = cv2.imread(str(snapshot))
        if img is None:
            raise VLMError(f"cannot read the snapshot {snapshot.name}")
        return model.yes_no(img, question)

    return verify
