"""AI check: ask Gemini whether each top moment matches the question (Requirement 9, design: aicheck.py).

For each moment a few small images leave the PC - never whole videos: whole frames at about -2 s,
0 s and +2 s (384 px) and, when the moment is a tracked person/vehicle, up to 3 crops of it from the
stored thumbnails. Images are sent at Gemini's low media resolution: measured 2026-10-29 on
gemini-3.5-flash-lite, about 280 input tokens per image (default resolution: about 1,100) with the
same verdicts on the test moments; 20 moments are roughly 30k tokens, under a cent with Flash-Lite.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

import httpx

from nab_find.media import MediaError, grab_jpeg
from nab_find.search import Moment
from nab_find.settings import MODELS
from nab_find.store import CaseStore

log = logging.getLogger("nab_find.aicheck")

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
TOKENS_PER_IMAGE = 280  # measured at MEDIA_RESOLUTION_LOW
MEDIA_RESOLUTION = "MEDIA_RESOLUTION_LOW"
TOKENS_PER_MOMENT_TEXT = 40
TOKENS_PROMPT = 350
OUTPUT_TOKENS_PER_MOMENT = 60
BATCH = 5
PARALLEL = 3
TIMEOUT_S = 90.0
CONTEXT_OFFSETS = (-2.0, 0.0, 2.0)
MAX_CROPS = 3

PROMPT = """You check results of a video search for a small shop or home owner.
The user is looking for: "{query}"
Below are {n} moments from their CCTV recordings. Each moment has whole frames from about 2 seconds
before, at, and 2 seconds after the moment, and sometimes close-ups of the main person or vehicle.
For each moment decide if it shows what the user is looking for.
- match: "yes" if it clearly does, "no" if it clearly does not, "unsure" if the images cannot tell.
- confidence: 0 to 1.
- description: one short plain sentence about what is visible that is relevant to the search
  (who/what, what they do). Do not guess identities, names, age or ethnicity. Say "unclear" for
  anything you cannot see.
Answer only for the moment numbers given."""

SCHEMA = {
    "type": "OBJECT",
    "properties": {"results": {"type": "ARRAY", "items": {
        "type": "OBJECT",
        "properties": {
            "moment": {"type": "INTEGER"},
            "match": {"type": "STRING", "enum": ["yes", "no", "unsure"]},
            "confidence": {"type": "NUMBER"},
            "description": {"type": "STRING"},
        },
        "required": ["moment", "match", "confidence", "description"],
    }}},
    "required": ["results"],
}


class AICheckError(RuntimeError):
    """A plain message for the user (no key, rejected key, no network, quota...)."""


@dataclass
class MomentImages:
    index: int  # 1-based moment number in the request
    label: str
    images: list[bytes]


@dataclass
class CheckResult:
    index: int
    match: str
    confidence: float
    description: str


@dataclass
class Usage:
    tokens_in: int = 0
    tokens_out: int = 0


def images_per_moment(m: Moment) -> int:
    return len(CONTEXT_OFFSETS) + (MAX_CROPS if m.visit_id is not None else 0)


def estimate(moments: Sequence[Moment], model: str) -> dict:
    """Images and rough cost (USD) before asking for consent."""
    n_img = sum(images_per_moment(m) for m in moments)
    batches = max(1, -(-len(moments) // BATCH))
    t_in = n_img * TOKENS_PER_IMAGE + len(moments) * TOKENS_PER_MOMENT_TEXT + batches * TOKENS_PROMPT
    t_out = len(moments) * OUTPUT_TOKENS_PER_MOMENT
    p_in, p_out = MODELS.get(model, {"price": (1.5, 7.5)})["price"]
    return {"moments": len(moments), "images": n_img, "tokens_in": t_in, "tokens_out": t_out,
            "usd": round(t_in / 1e6 * p_in + t_out / 1e6 * p_out, 4)}


def collect_images(store: CaseStore, m: Moment, index: int) -> MomentImages:
    v = store.video(m.video_id)
    imgs: list[bytes] = []
    for d in CONTEXT_OFFSETS:
        off = min(max(0.0, m.offset_s + d), max(0.0, v.duration_s - 0.2))
        try:
            imgs.append(grab_jpeg(v.path, off))
        except MediaError:
            log.warning("frame grab failed for moment %d at %.1f s", index, off)
    if m.visit_id is not None:
        rows = store.query("SELECT thumb, offset_s FROM vectors WHERE visit_id = ? AND thumb IS NOT NULL", (m.visit_id,))
        rows.sort(key=lambda r: abs(r[1] - m.offset_s))
        for thumb, _ in rows[:MAX_CROPS]:
            p = store.thumbs / thumb
            if p.is_file():
                imgs.append(p.read_bytes())
    when = datetime.fromtimestamp(m.best_ts).strftime("%d %b %H:%M:%S")
    return MomentImages(index, f"Moment {index} ({m.camera}, {when})", imgs)


def build_request(query: str, items: Sequence[MomentImages]) -> dict:
    parts: list[dict] = [{"text": PROMPT.format(query=query.strip(), n=len(items))}]
    for it in items:
        parts.append({"text": it.label + ":"})
        parts.extend({"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(b).decode("ascii")}}
                     for b in it.images)
    return {"contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json",
                                 "responseSchema": SCHEMA, "mediaResolution": MEDIA_RESOLUTION}}


def parse_response(body: dict, wanted: set[int]) -> tuple[list[CheckResult], Usage]:
    cands = body.get("candidates") or []
    if not cands:
        reason = (body.get("promptFeedback") or {}).get("blockReason")
        raise AICheckError(f"the AI service returned no answer{f' ({reason})' if reason else ''}")
    text = "".join(p.get("text", "") for p in (cands[0].get("content") or {}).get("parts", []))
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise AICheckError("the AI service answered in an unexpected format") from exc
    out = []
    for r in data.get("results", []) if isinstance(data, dict) else []:
        try:
            idx = int(r["moment"])
        except (KeyError, TypeError, ValueError):
            continue
        if idx not in wanted:
            continue
        match = str(r.get("match", "unsure")).lower()
        out.append(CheckResult(idx, match if match in ("yes", "no", "unsure") else "unsure",
                               min(1.0, max(0.0, float(r.get("confidence") or 0))),
                               str(r.get("description") or "").strip()[:300]))
    um = body.get("usageMetadata") or {}
    usage = Usage(int(um.get("promptTokenCount") or 0),
                  int(um.get("candidatesTokenCount") or 0) + int(um.get("thoughtsTokenCount") or 0))
    return out, usage


def _http_error(resp: httpx.Response) -> AICheckError:
    code = resp.status_code
    try:
        msg = resp.json().get("error", {}).get("message", "")
    except ValueError:
        msg = ""
    if code in (400, 401, 403) and ("API key" in msg or "PERMISSION" in msg.upper() or code in (401, 403)):
        return AICheckError("the AI check key was not accepted - check it in Settings")
    if code == 429:
        return AICheckError("the AI service is busy or your quota is used up - try again in a minute")
    if code == 404:
        return AICheckError("the chosen AI model is not available - pick another one in Settings")
    return AICheckError(f"the AI service returned an error ({code})")


class GeminiClient:
    def __init__(self, key: str, model: str, transport: httpx.BaseTransport | None = None) -> None:
        self.model = model
        self._http = httpx.Client(timeout=TIMEOUT_S, transport=transport,
                                  headers={"x-goog-api-key": key, "Content-Type": "application/json"})

    def generate(self, body: dict) -> dict:
        url = ENDPOINT.format(model=self.model)
        for attempt in range(2):
            try:
                resp = self._http.post(url, json=body)
            except httpx.TransportError as exc:
                if attempt == 0:
                    time.sleep(1.5)
                    continue
                raise AICheckError("could not reach the AI service - check the internet connection") from exc
            if resp.status_code in (429, 500, 502, 503, 504) and attempt == 0:
                time.sleep(2.0)
                continue
            if resp.status_code != 200:
                raise _http_error(resp)
            return resp.json()
        raise AICheckError("the AI service did not answer")

    def close(self) -> None:
        self._http.close()


@dataclass
class Job:
    query: str
    moments: list[Moment]
    model: str
    results: dict[int, CheckResult] = field(default_factory=dict)  # by 1-based moment index
    errors: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    done: bool = False
    batches_done: int = 0
    batches: int = 0

    def to_dict(self) -> dict[str, Any]:
        order = sorted(range(1, len(self.moments) + 1), key=lambda i: rank_key(self.results.get(i), i))
        p_in, p_out = MODELS.get(self.model, {"price": (1.5, 7.5)})["price"]
        return {
            "query": self.query, "done": self.done, "errors": self.errors,
            "progress": [self.batches_done, self.batches],
            "usd": round(self.usage.tokens_in / 1e6 * p_in + self.usage.tokens_out / 1e6 * p_out, 4),
            "items": [{"moment": self.moments[i - 1].to_dict(),
                       "check": self.results[i].__dict__ if i in self.results else None} for i in order],
        }


def rank_key(r: CheckResult | None, i: int) -> tuple:
    """yes (most confident first), then unchecked/unsure in the original order, then no."""
    if r is None:
        return (1, 0.0, i)
    return ({"yes": 0, "unsure": 1, "no": 2}[r.match], -r.confidence if r.match == "yes" else 0.0, i)


def run_job(job: Job, store: CaseStore, client: GeminiClient,
            on_update: Callable[[Job], None] | None = None) -> Job:
    batches = [list(range(i, min(i + BATCH, len(job.moments)))) for i in range(0, len(job.moments), BATCH)]
    job.batches = len(batches)

    def one(batch: list[int]) -> None:
        items = [collect_images(store, job.moments[i], i + 1) for i in batch]
        try:
            res, usage = parse_response(client.generate(build_request(job.query, items)), {i + 1 for i in batch})
        except AICheckError as exc:
            job.errors.append(str(exc))
            return
        finally:
            job.batches_done += 1
        for r in res:
            job.results[r.index] = r
        job.usage.tokens_in += usage.tokens_in
        job.usage.tokens_out += usage.tokens_out
        if on_update:
            on_update(job)

    try:
        with ThreadPoolExecutor(PARALLEL) as ex:
            list(ex.map(one, batches))
    finally:
        job.done = True
        job.errors = sorted(set(job.errors))
        store.add_ai_check(job.query, job.model, job.to_dict(), job.usage.tokens_in, job.usage.tokens_out)
        if on_update:
            on_update(job)
    return job
