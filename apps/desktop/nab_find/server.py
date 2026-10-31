"""Local HTTP server behind the window (design: server.py). Bound to 127.0.0.1 on a random port; every
/api call needs the per-launch token (header ``X-Token`` or query ``t`` for <img>/<video> sources)."""

from __future__ import annotations

import logging
import os
import secrets
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from nab_find import __version__
from nab_find.aicheck import AICheckError, GeminiClient, Job, estimate, run_job
from nab_find.library import add_videos
from nab_find.media import MediaError, cut_clip
from nab_find.models import Models, load_models
from nab_find.processor import Processor, merge, total
from nab_find.search import Filters, Moment, Searcher
from nab_find.settings import Settings
from nab_find.store import CaseStore, list_cases

log = logging.getLogger("nab_find.server")

PREVIEW_BEFORE_S, PREVIEW_AFTER_S = 8.0, 22.0
EXPORT_BEFORE_S, EXPORT_AFTER_S = 10.0, 20.0


class Runner:
    """One processing job at a time, on its own thread; others wait in a queue."""

    def __init__(self, app_state: "AppState") -> None:
        self.s = app_state
        self.queue: list[str] = []
        self.current: str | None = None
        self.proc: Processor | None = None
        self.progress: dict[str, dict] = {}
        self.status: dict[str, str] = {}
        self.events: dict[str, list[dict]] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        threading.Thread(target=self._loop, name="nab-find-worker", daemon=True).start()

    def start(self, cid: str) -> None:
        with self._lock:
            if cid != self.current and cid not in self.queue:
                self.queue.append(cid)
                self.status[cid] = "queued"
        self._wake.set()

    def pause(self, cid: str) -> None:
        if self.current == cid and self.proc:
            self.proc.pause()
            self.status[cid] = "paused"

    def resume(self, cid: str) -> None:
        if self.current == cid and self.proc:
            self.proc.resume()
            self.status[cid] = "processing"

    def cancel(self, cid: str) -> None:
        with self._lock:
            if cid in self.queue:
                self.queue.remove(cid)
                self.status[cid] = "stopped"
        if self.current == cid and self.proc:
            self.proc.cancel()

    def _event(self, cid: str, name: str) -> None:
        self.events.setdefault(cid, []).append({"event": name, "at": time.time()})

    def _loop(self) -> None:
        while True:
            self._wake.wait()
            with self._lock:
                if not self.queue:
                    self._wake.clear()
                    continue
                cid = self.queue.pop(0)
                self.current = cid
            try:
                self._run(cid)
            except Exception:  # noqa: BLE001 - keep the worker alive
                log.exception("processing %s failed", cid)
                self.status[cid] = "error"
            finally:
                self.current, self.proc = None, None

    def _run(self, cid: str) -> None:
        store = self.s.case(cid)
        models = self.s.wait_models()
        self.status[cid] = "processing"
        store.set_meta(status="processing")
        self.proc = Processor(store, models, seed_speeds=self.s.settings.get("speeds"),
                              on_progress=lambda snap: self.progress.__setitem__(cid, snap),
                              on_event=lambda e: self._on_event(cid, store, e))
        result = self.proc.run()
        speeds = dict(self.s.settings.get("speeds") or {})
        speeds.update(self.proc.progress.speeds() if self.proc.progress else {})
        self.s.settings.update(speeds=speeds)
        final = {"done": "done", "cancelled": "stopped"}.get(result, "error")
        self.status[cid] = final
        store.set_meta(status=final)

    def _on_event(self, cid: str, store: CaseStore, e: str) -> None:
        self._event(cid, e)
        if e == "quick_done":
            store.set_meta(quick_done=True)


class AppState:
    def __init__(self, data_dir: Path, model_dir: Path, load_models_now: bool = True) -> None:
        self.data_dir = Path(data_dir)
        self.cases_dir = self.data_dir / "cases"
        self.cases_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir = Path(model_dir)
        self.settings = Settings(self.data_dir)
        self.token = secrets.token_urlsafe(24)
        self._cases: dict[str, CaseStore] = {}
        self._cases_lock = threading.Lock()
        self.models: Models | None = None
        self.models_error: str | None = None
        self._models_ready = threading.Event()
        self.jobs: dict[str, Job] = {}
        self.runner = Runner(self)
        if load_models_now:
            threading.Thread(target=self._load_models, daemon=True).start()

    def _load_models(self) -> None:
        try:
            self.models = load_models(self.model_dir, self.data_dir / "ovcache", prefer=self.settings.get("device"))
        except Exception as exc:  # noqa: BLE001
            log.exception("model load failed")
            self.models_error = f"{type(exc).__name__}: {exc}"
        finally:
            self._models_ready.set()

    def set_models(self, models: Models) -> None:
        self.models = models
        self._models_ready.set()

    def wait_models(self, timeout: float | None = None) -> Models:
        self._models_ready.wait(timeout)
        if self.models is None:
            raise HTTPException(503, self.models_error or "still starting up")
        return self.models

    def case(self, cid: str) -> CaseStore:
        with self._cases_lock:
            st = self._cases.get(cid)
            if st is None:
                folder = self.cases_dir / cid
                if not (folder / "case.db").is_file() or "/" in cid or "\\" in cid or ".." in cid:
                    raise HTTPException(404, "case not found")
                st = self._cases[cid] = CaseStore(folder)
            return st

    def new_case(self, name: str) -> CaseStore:
        st = CaseStore.create(self.cases_dir, name)
        with self._cases_lock:
            self._cases[st.case_id] = st
        return st

    def drop_case(self, cid: str) -> None:
        self.runner.cancel(cid)
        for _ in range(100):
            if self.runner.current != cid:
                break
            time.sleep(0.05)
        with self._cases_lock:
            st = self._cases.pop(cid, None)
        (st or CaseStore(self.cases_dir / cid)).delete()


# ---------------------------------------------------------------- views


def case_summary(state: AppState, st: CaseStore) -> dict[str, Any]:
    meta = st.meta()
    videos = [v for v in st.videos()]
    footage = sum(v.duration_s for v in videos if v.status != "error")
    status = state.runner.status.get(st.case_id) or meta.get("status", "new")
    if status == "processing" and state.runner.proc and state.runner.proc.paused:
        status = "paused"
    if not videos:
        status = "new"
    return {"id": st.case_id, "name": meta.get("name"), "created": meta.get("created"), "status": status,
            "range": meta.get("range"), "videos": len(videos), "footage_s": footage,
            "quick_done": bool(meta.get("quick_done")), "cameras": sorted({v.camera for v in videos}),
            "first_ts": min((v.start_ts for v in videos), default=None),
            "last_ts": max((v.end_ts for v in videos), default=None)}


def video_dict(v) -> dict[str, Any]:
    return {"video_id": v.video_id, "path": v.path, "name": Path(v.path).name, "camera": v.camera,
            "start_ts": v.start_ts, "end_ts": v.end_ts, "start_estimated": v.start_estimated,
            "duration_s": v.duration_s, "width": v.width, "height": v.height, "status": v.status,
            "error": v.error, "exists": Path(v.path).is_file()}


def coverage(st: CaseStore) -> dict[str, Any]:
    """Per video, the spans each pass has finished (for the timeline bar)."""
    out = {}
    for v in st.videos():
        out[v.video_id] = {"quick": merge(st.spans(v.video_id, "quick"), 0.5),
                           "detail": merge(st.spans(v.video_id, "detail"), 0.5)}
    return out


def _filters(q: dict[str, str]) -> Filters:
    def num(k):
        return float(q[k]) if q.get(k) not in (None, "") else None
    return Filters(num("from"), num("to"), q.get("kind") or "any", q.get("camera") or None)


def default_export_dir() -> Path:
    videos = Path(os.environ.get("USERPROFILE") or Path.home()) / "Videos"
    return (videos if videos.is_dir() else Path.home()) / "NAB Find exports"


def create_app(state: AppState, web_dir: Path) -> FastAPI:
    app = FastAPI(title="NAB Find", docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def auth(request: Request, call_next):
        if request.url.path.startswith("/api/"):
            tok = request.headers.get("x-token") or request.query_params.get("t")
            if not tok or not secrets.compare_digest(tok, state.token):
                return JSONResponse({"error": "forbidden"}, status_code=403)
        return await call_next(request)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    # -- app
    @app.get("/api/state")
    def get_state():
        return {"version": __version__, "models_ready": state.models is not None,
                "models_error": state.models_error, "device": state.models.device if state.models else None,
                "settings": state.settings.public(), "export_dir": str(default_export_dir())}

    @app.get("/api/settings")
    def get_settings():
        return state.settings.public()

    @app.put("/api/settings")
    def put_settings(body: dict = Body(...)):
        if "gemini_key" in body:
            state.settings.set_gemini_key((body.pop("gemini_key") or "").strip() or None)
        state.settings.update(**{k: v for k, v in body.items() if k in ("gemini_model", "ai_moments", "device")})
        return state.settings.public()

    # -- cases
    @app.get("/api/cases")
    def get_cases():
        out = []
        for folder in list_cases(state.cases_dir):
            try:
                out.append(case_summary(state, state.case(folder.name)))
            except Exception:  # noqa: BLE001 - one broken case never hides the others
                log.warning("cannot open case %s", folder, exc_info=True)
        return out

    @app.post("/api/cases")
    def post_case(body: dict = Body(...)):
        name = (body.get("name") or "").strip() or datetime.now().strftime("Search %d %b %H:%M")
        return case_summary(state, state.new_case(name[:80]))

    @app.get("/api/cases/{cid}")
    def get_case(cid: str):
        st = state.case(cid)
        return {**case_summary(state, st), "videos_list": [video_dict(v) for v in st.videos()],
                "counts": st.counts(), "progress": state.runner.progress.get(cid),
                "coverage": coverage(st), "events": state.runner.events.get(cid, [])}

    @app.patch("/api/cases/{cid}")
    def patch_case(cid: str, body: dict = Body(...)):
        st = state.case(cid)
        if body.get("name"):
            st.set_meta(name=str(body["name"]).strip()[:80])
        return case_summary(state, st)

    @app.delete("/api/cases/{cid}")
    def delete_case(cid: str):
        state.case(cid)
        state.drop_case(cid)
        return {"ok": True}

    # -- videos and range
    def _not_running(cid: str) -> None:
        if state.runner.current == cid or cid in state.runner.queue:
            raise HTTPException(409, "stop the processing first")

    @app.post("/api/cases/{cid}/videos")
    def post_videos(cid: str, body: dict = Body(...)):
        st = state.case(cid)
        _not_running(cid)
        added = add_videos(st, body.get("paths") or [])
        if any(a.video_id for a in added):
            st.set_meta(status="ready", quick_done=False)
        return {"added": [a.__dict__ for a in added], "videos": [video_dict(v) for v in st.videos()]}

    @app.patch("/api/cases/{cid}/videos/{vid}")
    def patch_video(cid: str, vid: int, body: dict = Body(...)):
        st = state.case(cid)
        _not_running(cid)
        v = st.video(vid)
        if v is None:
            raise HTTPException(404, "video not found")
        changes: dict[str, Any] = {}
        if body.get("camera"):
            changes["camera"] = str(body["camera"]).strip()[:40]
        if body.get("start_ts") is not None:
            new = float(body["start_ts"])
            shift = new - v.start_ts
            changes.update(start_ts=new, start_estimated=False)
            if body.get("shift_siblings") and shift:
                for o in st.videos():
                    if o.video_id != vid and o.camera == v.camera:
                        st.update_video(o.video_id, start_ts=o.start_ts + shift, start_estimated=False)
        st.update_video(vid, **changes)
        if "start_ts" in changes and st.meta().get("range"):
            st.reset_work()  # what is inside the range changed
            st.set_meta(status="ready", quick_done=False)
        return {"videos": [video_dict(x) for x in st.videos()]}

    @app.delete("/api/cases/{cid}/videos/{vid}")
    def delete_video(cid: str, vid: int):
        st = state.case(cid)
        _not_running(cid)
        st.remove_video(vid)
        return {"videos": [video_dict(x) for x in st.videos()]}

    @app.put("/api/cases/{cid}/range")
    def put_range(cid: str, body: dict = Body(...)):
        st = state.case(cid)
        _not_running(cid)
        new = None
        if body.get("start_ts") is not None and body.get("end_ts") is not None:
            a, b = float(body["start_ts"]), float(body["end_ts"])
            if b <= a:
                raise HTTPException(400, "the end must be after the start")
            new = [a, b]
        if new != st.meta().get("range"):
            st.reset_work()
            st.set_meta(range=new, status="ready", quick_done=False)
            state.runner.progress.pop(cid, None)
        return case_summary(state, st)

    # -- processing
    @app.post("/api/cases/{cid}/run/{action}")
    def control(cid: str, action: str):
        st = state.case(cid)
        if action == "start":
            if not [v for v in st.videos() if v.status != "error"]:
                raise HTTPException(400, "add at least one video that can be read")
            state.runner.start(cid)
        elif action == "pause":
            state.runner.pause(cid)
        elif action == "resume":
            state.runner.resume(cid)
        elif action == "cancel":
            state.runner.cancel(cid)
        else:
            raise HTTPException(404, "unknown action")
        return case_summary(state, st)

    @app.get("/api/cases/{cid}/progress")
    def get_progress(cid: str):
        st = state.case(cid)
        return {**case_summary(state, st), "progress": state.runner.progress.get(cid), "counts": st.counts(),
                "coverage": coverage(st), "events": state.runner.events.get(cid, [])}

    # -- search
    @app.get("/api/cases/{cid}/search")
    def search(cid: str, request: Request):
        st = state.case(cid)
        q = dict(request.query_params)
        text = (q.get("q") or "").strip()
        if not text:
            raise HTTPException(400, "type what you are looking for")
        models = state.wait_models(60)
        t0 = time.perf_counter()
        res = Searcher(st, models.clip).search(text, _filters(q), limit=int(q.get("limit") or 60))
        return {"query": text, "ms": round((time.perf_counter() - t0) * 1000), "moments": [m.to_dict() for m in res]}

    @app.get("/api/cases/{cid}/visits")
    def visits(cid: str, request: Request):
        st = state.case(cid)
        q = dict(request.query_params)
        return {"visits": Searcher(st, None).visits(_filters(q), q.get("sort") or "time",
                                                    float(q.get("min_s") or 0))}

    @app.get("/api/cases/{cid}/similar/{visit_id}")
    def similar(cid: str, visit_id: int, request: Request):
        st = state.case(cid)
        return {"moments": [m.to_dict() for m in Searcher(st, None).similar(visit_id, _filters(dict(request.query_params)))]}

    # -- AI check
    def _moments(body: dict) -> list[Moment]:
        ms = body.get("moments") or []
        limit = int(state.settings.get("ai_moments") or 20)
        return [Moment(**{k: m[k] for k in Moment.__dataclass_fields__ if k in m}) for m in ms[:min(limit, 50)]]

    @app.post("/api/cases/{cid}/aicheck/estimate")
    def ai_estimate(cid: str, body: dict = Body(...)):
        state.case(cid)
        model = state.settings.get("gemini_model")
        return {**estimate(_moments(body), model), "model": model,
                "key_set": state.settings.gemini_key() is not None}

    @app.post("/api/cases/{cid}/aicheck")
    def ai_start(cid: str, body: dict = Body(...)):
        st = state.case(cid)
        key = state.settings.gemini_key()
        if not key:
            raise HTTPException(400, "add an AI check key in Settings first")
        q = (body.get("q") or "").strip()
        moments = _moments(body)
        if not q or not moments:
            raise HTTPException(400, "nothing to check")
        job = Job(q, moments, state.settings.get("gemini_model"))
        jid = uuid.uuid4().hex[:12]
        state.jobs[jid] = job

        def work():
            client = GeminiClient(key, job.model)
            try:
                run_job(job, st, client)
            except AICheckError as exc:
                job.errors.append(str(exc))
                job.done = True
            except Exception:  # noqa: BLE001
                log.exception("AI check failed")
                job.errors.append("the AI check failed unexpectedly")
                job.done = True
            finally:
                client.close()

        threading.Thread(target=work, daemon=True).start()
        return {"job": jid}

    @app.get("/api/aicheck/{jid}")
    def ai_status(jid: str):
        job = state.jobs.get(jid)
        if job is None:
            raise HTTPException(404, "unknown check")
        return job.to_dict()

    # -- media
    @app.get("/api/cases/{cid}/thumb/{name}")
    def thumb(cid: str, name: str):
        st = state.case(cid)
        p = (st.thumbs / name).resolve()
        if p.parent != st.thumbs.resolve() or not p.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})

    @app.get("/api/cases/{cid}/preview")
    def preview(cid: str, video: int, at: float):
        st = state.case(cid)
        v = st.video(video)
        if v is None:
            raise HTTPException(404, "video not found")
        if not Path(v.path).is_file():
            raise HTTPException(410, "the video file was moved or deleted")
        start = max(0.0, at - PREVIEW_BEFORE_S)
        dur = min(PREVIEW_BEFORE_S + PREVIEW_AFTER_S, max(0.5, v.duration_s - start))
        dst = st.previews / f"v{video}_{int(start * 10)}_{int(dur * 10)}.mp4"
        if not dst.is_file():
            try:
                cut_clip(v.path, start, dur, dst, max_width=1280)
            except MediaError as exc:
                raise HTTPException(500, str(exc)) from exc
        return FileResponse(dst, media_type="video/mp4", headers={"X-Clip-Start": str(start)})

    @app.post("/api/cases/{cid}/export")
    def export(cid: str, body: dict = Body(...)):
        st = state.case(cid)
        v = st.video(int(body["video"]))
        if v is None:
            raise HTTPException(404, "video not found")
        t = float(body["t"])
        start = max(0.0, t - float(body.get("before", EXPORT_BEFORE_S)))
        end = min(v.duration_s, t + float(body.get("after", EXPORT_AFTER_S)))
        folder = Path(body.get("folder") or default_export_dir())
        when = datetime.fromtimestamp(v.start_ts + start)
        name = f"{v.camera} {when:%Y-%m-%d %H-%M-%S}.mp4".replace("/", "-").replace(":", "-")
        overlay = f"{v.camera}  {when:%d %b %Y %H:%M:%S}"
        try:
            out = cut_clip(v.path, start, end - start, folder / name, overlay=overlay, max_width=1920)
        except MediaError as exc:
            raise HTTPException(500, str(exc)) from exc
        return {"path": str(out), "folder": str(folder)}

    if web_dir.is_dir():
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
    return app
