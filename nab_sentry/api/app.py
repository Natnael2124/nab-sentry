"""FastAPI application: search API, media serving, Console static files.

Requirements 10.2-10.14, 11.1-11.10, 18.4, 18.6, 18.7.

``create_app(services)`` builds the app from already-loaded components so tests can pass fakes.
Parameter validation lives in the pure :func:`validate_search_params`; every error response uses
one JSON shape::

    {"error": {"code": "invalid_parameter", "param": "q", "message": "q must be 1-256 characters"}}

FastAPI's default validation and 404 formats never reach the client. Unhandled exceptions become a
fixed 500 body; the traceback goes only to the server log (18.7).

``serve()`` / ``python -m nab_sentry.api.app`` runs the startup sequence (design "Startup
sequence (API_Server)") and then ``uvicorn.run(app, host="127.0.0.1", port=cfg.port)``.
Exit codes: 2 invalid Config or non-loopback host, 3 port bind failure, 4 manifest check
failed, 5 vector index unavailable or inconsistent, 6 model load failure, 130 interrupted.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from nab_sentry.api.media import (
    iter_file,
    media_response_plan,
    parse_range,
    resolve_playback,
    resolve_thumb,
)
from nab_sentry.embed.clip_encoder import EmptyQueryError
from nab_sentry.logging_setup import get_logger
from nab_sentry.querylog import QueryLog, QueryRecord
from nab_sentry.search.engine import (
    VALID_CLASSES,
    FilterError,
    SearchFilter,
    SearchUnavailable,
)
from nab_sentry.store.db import as_aware, iso_ms

if TYPE_CHECKING:
    from nab_sentry.config import Config
    from nab_sentry.embed.clip_encoder import Encoder
    from nab_sentry.search.clustering import Event
    from nab_sentry.search.engine import SearchEngine
    from nab_sentry.store.db import MetadataStore
    from nab_sentry.store.vector_index import VectorIndex

__all__ = [
    "Services",
    "SearchRequest",
    "ParamError",
    "PARAM_ORDER",
    "WEB_DIR",
    "validate_search_params",
    "error_body",
    "event_out",
    "create_app",
    "serve",
]

log = get_logger("api")

WEB_DIR = Path(__file__).resolve().parents[1] / "web"

# Validation order for 422 responses (Property 41).
PARAM_ORDER = ("q", "start", "end", "cls", "limit")
_FILTER_KEYS = ("camera", "start", "end", "cls", "limit")
_LIMIT_RE = re.compile(r"[0-9]{1,4}")  # ASCII digits only; range-checked afterwards
_VIDEO_ID_RE = re.compile(r"[0-9]{1,18}")  # fits SQLite INTEGER without overflow

_INTERNAL_ERROR = {"error": {"code": "internal_error", "message": "Internal server error"}}


# ---------------------------------------------------------------------------------------------
# Services and request types
# ---------------------------------------------------------------------------------------------


@dataclass
class Services:
    cfg: Config
    db: MetadataStore
    index: VectorIndex | None
    encoder: Encoder | None
    detector_loaded: bool
    engine: SearchEngine | None
    query_log: QueryLog


@dataclass(frozen=True)
class SearchRequest:
    q: str
    filter: SearchFilter
    limit: int

    def log_filters(self) -> dict[str, Any]:
        """Parsed filter values as written to the Query_Log for 200 responses."""
        f = self.filter
        return {
            "camera": f.camera_id,
            "start": None if f.start is None else iso_ms(f.start),
            "end": None if f.end is None else iso_ms(f.end),
            "cls": f.cls,
            "limit": self.limit,
        }


class ParamError(Exception):
    """An invalid search parameter: ``param`` names it, ``message`` explains why (10.7-10.9)."""

    def __init__(self, param: str, message: str) -> None:
        super().__init__(message)
        self.param = param
        self.message = message

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ParamError) and (self.param, self.message) == (
            other.param,
            other.message,
        )

    def __hash__(self) -> int:
        return hash((self.param, self.message))

    def __repr__(self) -> str:
        return f"ParamError(param={self.param!r}, message={self.message!r})"


# ---------------------------------------------------------------------------------------------
# Pure validation
# ---------------------------------------------------------------------------------------------


def _parse_time(param: str, raw: str) -> datetime | ParamError:
    try:
        dt = datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return ParamError(param, f"{param} must be an ISO 8601 date-time")
    try:
        return as_aware(dt)  # naive -> local time zone
    except (OverflowError, OSError, ValueError):
        return ParamError(param, f"{param} is outside the supported date range")


def validate_search_params(params: Mapping[str, str], cfg: Config) -> SearchRequest | ParamError:
    """Return a ``SearchRequest`` or the ``ParamError`` for the first invalid parameter in the
    order ``q, start, end, cls, limit``. Pure: no I/O, no logging."""
    max_len = int(cfg.max_query_len)
    max_limit = int(cfg.max_limit)

    q = params.get("q")
    if not isinstance(q, str) or not q.strip() or len(q) > max_len:
        return ParamError("q", f"q must be 1-{max_len} characters and not only whitespace")

    start: datetime | None = None
    raw_start = params.get("start")
    if raw_start is not None:
        parsed = _parse_time("start", raw_start)
        if isinstance(parsed, ParamError):
            return parsed
        start = parsed

    end: datetime | None = None
    raw_end = params.get("end")
    if raw_end is not None:
        parsed = _parse_time("end", raw_end)
        if isinstance(parsed, ParamError):
            return parsed
        end = parsed
    if start is not None and end is not None and start > end:
        return ParamError("end", "end must not be earlier than start")

    cls = params.get("cls")
    if cls is not None and cls not in VALID_CLASSES:
        return ParamError("cls", f"cls must be one of: {', '.join(sorted(VALID_CLASSES))}")

    limit = int(cfg.default_limit)
    raw_limit = params.get("limit")
    if raw_limit is not None:
        if not isinstance(raw_limit, str) or _LIMIT_RE.fullmatch(raw_limit) is None:
            return ParamError("limit", f"limit must be an integer from 1 to {max_limit}")
        limit = int(raw_limit)
        if not 1 <= limit <= max_limit:
            return ParamError("limit", f"limit must be an integer from 1 to {max_limit}")

    camera = params.get("camera")
    return SearchRequest(
        q=q,
        filter=SearchFilter(camera_id=camera, start=start, end=end, cls=cls),
        limit=limit,
    )


# ---------------------------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------------------------


def error_body(code: str, message: str, param: str | None = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code}
    if param is not None:
        err["param"] = param
    err["message"] = message
    return {"error": err}


def _error(status: int, code: str, message: str, param: str | None = None) -> JSONResponse:
    return JSONResponse(error_body(code, message, param), status_code=status)


def _not_found() -> JSONResponse:
    return _error(404, "not_found", "Not found")


def event_out(ev: Event, video_start: datetime, camera_label: str) -> dict[str, Any]:
    """``EventOut`` JSON for one Event; times are ``video_start + offset`` at ms precision (10.6)."""
    start = as_aware(video_start)
    return {
        "camera_id": ev.camera_id,
        "camera_label": camera_label,
        "video_id": int(ev.video_id),
        "start_offset_s": float(ev.start_s),
        "end_offset_s": float(ev.end_s),
        "start_time": (start + timedelta(seconds=float(ev.start_s))).isoformat(
            timespec="milliseconds"
        ),
        "end_time": (start + timedelta(seconds=float(ev.end_s))).isoformat(timespec="milliseconds"),
        "score": float(ev.score),
        "relative_score": float(ev.relative_score),
        "hit_count": len(ev.hits),
        "thumbnail_url": f"/media/thumbs/{quote(ev.thumb_path, safe='')}",
        "video_url": f"/media/video/{int(ev.video_id)}",
    }


def _raw_filters(params: Mapping[str, str]) -> dict[str, Any]:
    return {k: params.get(k) for k in _FILTER_KEYS}


_FILTER_PART_PARAM = {"time_range": "end", "cls": "cls"}


class _CatchAllMiddleware:
    """Pure ASGI middleware: any unhandled exception -> sanitised 500, traceback to the log only.

    Unlike an ``Exception`` handler on ``ServerErrorMiddleware``, this does not re-raise, so the
    server keeps serving later requests and test clients see the 500 response (18.7).
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = False

        async def send_wrapper(message: Any) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            log.exception("unhandled error while serving %s %s", scope.get("method"), scope.get("path"))
            if started:
                return  # headers already sent; the client sees a truncated body
            await JSONResponse(_INTERNAL_ERROR, status_code=500)(scope, receive, send)


# ---------------------------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------------------------


def create_app(services: Services, *, web_dir: Path | None = None) -> FastAPI:
    web = Path(web_dir) if web_dir is not None else WEB_DIR
    app = FastAPI(
        title="NAB Sentry",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.services = services
    app.add_middleware(_CatchAllMiddleware)

    # -- exception handlers -------------------------------------------------------------------

    @app.exception_handler(ParamError)
    async def _param_error(_req: Request, exc: ParamError) -> JSONResponse:
        return _error(422, "invalid_parameter", exc.message, exc.param)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_req: Request, exc: RequestValidationError) -> JSONResponse:
        param = None
        errors = exc.errors()
        if errors:
            loc = errors[0].get("loc") or ()
            if loc:
                param = str(loc[-1])
        return _error(422, "invalid_parameter", "invalid request parameter", param)

    @app.exception_handler(FilterError)
    async def _filter_error(_req: Request, exc: FilterError) -> JSONResponse:
        return _error(422, "invalid_parameter", exc.message, _FILTER_PART_PARAM.get(exc.part, exc.part))

    @app.exception_handler(SearchUnavailable)
    async def _unavailable(_req: Request, exc: SearchUnavailable) -> JSONResponse:
        log.warning("search unavailable: %s", exc)
        return _error(503, "search_unavailable", "Search is unavailable")

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_req: Request, exc: StarletteHTTPException) -> Response:
        if exc.status_code == 404:
            return _not_found()
        if exc.status_code == 405:
            return _error(405, "method_not_allowed", "Method not allowed")
        return _error(exc.status_code, "http_error", "Request failed")

    # -- API ----------------------------------------------------------------------------------

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        counts = services.db.counts()
        index = services.index
        return {
            "status": "ok",
            "videos": int(counts.videos),
            "vectors": int(index.ntotal) if index is not None else 0,
            "models_loaded": {
                "detector": bool(services.detector_loaded),
                "embedder": services.encoder is not None,
            },
        }

    @app.get("/api/cameras")
    def cameras() -> list[dict[str, str]]:
        return [{"camera_id": cid, "label": label} for cid, label in services.db.cameras()]

    @app.get("/api/search")
    def search(request: Request) -> Response:
        t0 = time.perf_counter()
        params = request.query_params
        raw_q = params.get("q")

        def log_422(param: str, message: str) -> JSONResponse:
            services.query_log.append(
                QueryRecord(
                    status=422,
                    q=raw_q,
                    filters=_raw_filters(params),
                    result_count=0,
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                )
            )
            return _error(422, "invalid_parameter", message, param)

        parsed = validate_search_params(params, services.cfg)
        if isinstance(parsed, ParamError):
            return log_422(parsed.param, parsed.message)

        engine = services.engine
        if engine is None or services.encoder is None:
            raise SearchUnavailable("search unavailable: embedder or index not loaded")
        try:
            events = engine.search(parsed.q, parsed.filter, parsed.limit)
        except EmptyQueryError:
            return log_422("q", "q must not be empty or only whitespace")
        except FilterError as exc:
            return log_422(_FILTER_PART_PARAM.get(exc.part, exc.part), exc.message)

        # Video start and camera label per Event, from one representative hit each.
        rows = services.db.hit_rows([ev.hits[0].vector_id for ev in events if ev.hits])
        out: list[dict[str, Any]] = []
        for ev in events:
            row = rows.get(ev.hits[0].vector_id) if ev.hits else None
            if row is None:  # row vanished between search and lookup: skip the Event
                continue
            out.append(event_out(ev, row.start_ts, row.camera_label))

        services.query_log.append(
            QueryRecord(
                status=200,
                q=parsed.q,
                filters=parsed.log_filters(),
                result_count=len(out),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )
        )
        return JSONResponse(out)

    # -- media --------------------------------------------------------------------------------

    @app.get("/media/video/{video_id}")
    def video(video_id: str, request: Request) -> Response:
        if _VIDEO_ID_RE.fullmatch(video_id) is None:
            return _not_found()
        stored = services.db.playback_for(int(video_id))
        path = resolve_playback(services.cfg.playback_dir, stored)
        if path is None:
            return _not_found()
        try:
            size = path.stat().st_size
        except OSError:
            return _not_found()
        plan = media_response_plan(parse_range(request.headers.get("range"), size), size)
        if plan.start is None or plan.end is None:
            return Response(content=b"", status_code=plan.status, headers=plan.headers,
                            media_type="video/mp4")
        return StreamingResponse(
            iter_file(path, plan.start, plan.end),
            status_code=plan.status,
            headers=plan.headers,
            media_type="video/mp4",
        )

    @app.get("/media/thumbs/{name:path}")
    def thumb(name: str) -> Response:
        path = resolve_thumb(services.cfg.thumbs_dir, name)
        if path is None:
            return _not_found()
        return FileResponse(path, media_type="image/jpeg")

    # -- Console ------------------------------------------------------------------------------

    @app.get("/")
    def index_page() -> Response:
        page = web / "index.html"
        if not page.is_file():
            return _not_found()
        return FileResponse(page, media_type="text/html; charset=utf-8")

    if web.is_dir():  # web/ files arrive in task 16; without it /static/* is a plain 404
        app.mount("/static", StaticFiles(directory=web), name="static")

    return app


# ---------------------------------------------------------------------------------------------
# Startup sequence (design "Startup sequence (API_Server)")
# ---------------------------------------------------------------------------------------------

EXIT_OK = 0
EXIT_CONFIG = 2
EXIT_BIND = 3
EXIT_MODELS = 4
EXIT_INDEX = 5
EXIT_MODEL_LOAD = 6
EXIT_INTERRUPTED = 130

REPAIR_HINT = r"run venv\Scripts\python.exe scripts\ingest.py --repair"
WARMUP_QUERY = "a person walking"


class _PendingEncoder:
    """Placeholder so ``check_ready`` compares ID sets before the (slow) encoder load."""


def _err(msg: str) -> None:
    import sys

    print(f"error: {msg}", file=sys.stderr)


def probe_bind(host: str, port: int) -> None:
    """Bind and close a TCP socket on ``host:port``; raises ``OSError`` if the port is taken.

    No SO_REUSEADDR on Windows (there it would let a second socket share a listening port).
    """
    import socket
    import sys

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if sys.platform != "win32":  # match uvicorn so a TIME_WAIT port is not "in use"
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, int(port)))
    finally:
        sock.close()


def _build_serve_parser() -> Any:
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m nab_sentry.api.app",
        description="Run the NAB Sentry API_Server and Console on 127.0.0.1.",
    )
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="NAME=VALUE",
                   help="override a Config value (repeatable)")
    return p


def serve(
    argv: list[str] | None = None,
    *,
    encoder_factory: Any = None,
    detector_factory: Any = None,
    run: Any = None,
    web_dir: Path | None = None,
) -> int:
    """Startup sequence, then ``run(app, host="127.0.0.1", port=cfg.port)``; returns an exit code.

    ``encoder_factory`` / ``detector_factory`` take ``(cfg, manifest_files)``; ``run`` defaults
    to ``uvicorn.run``. All three are injectable so tests can check exit codes without models.
    """
    try:
        args = _build_serve_parser().parse_args(argv)
    except SystemExit as exc:  # argparse usage error (2) or --help (0)
        return int(exc.code) if isinstance(exc.code, int) else EXIT_CONFIG
    try:
        return _serve(args, encoder_factory, detector_factory, run, web_dir)
    except KeyboardInterrupt:
        _err("interrupted")
        return EXIT_INTERRUPTED


def _serve(args: Any, encoder_factory: Any, detector_factory: Any, run: Any,
           web_dir: Path | None) -> int:
    from nab_sentry import model_files
    from nab_sentry.config import ConfigError, load_config, parse_set_args
    from nab_sentry.errors import ModelMissingError
    from nab_sentry.logging_setup import setup_logging
    from nab_sentry.search.engine import SearchEngine
    from nab_sentry.startup import (
        LOOPBACK_HOST,
        StartupError,
        enable_offline_mode,
        ensure_loopback,
        log_security_warning,
        require_models,
    )
    from nab_sentry.store.db import MetadataStore
    from nab_sentry.store.vector_index import DEFAULT_DIM, IndexUnavailable, VectorIndex

    enable_offline_mode()  # idempotent; already done by ``import nab_sentry`` (13.3)

    # 1. Config (exit 2)
    try:
        cfg = load_config(parse_set_args(args.overrides))
        cfg.require_valid()
    except ConfigError as exc:
        _err(str(exc))
        return EXIT_CONFIG

    # 2. Loopback only (exit 2) -- before any socket is created (18.1, 18.5)
    try:
        ensure_loopback(cfg.host)
    except StartupError as exc:
        _err(str(exc))
        return EXIT_CONFIG

    setup_logging(cfg.logs_dir)

    # 3. Probe-bind (exit 3)
    try:
        probe_bind(LOOPBACK_HOST, cfg.port)
    except OSError as exc:
        log.error("cannot bind %s:%s (%s); is another server running on that port?",
                  LOOPBACK_HOST, cfg.port, exc)
        return EXIT_BIND

    # 4. Model manifest (exit 4; require_models logs each failing file + fetch command)
    try:
        files = require_models(cfg.models_dir, logger=log)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else EXIT_MODELS

    # 5. Security posture (18.2) -- logged before the server accepts any request
    log_security_warning(log)

    # 6. Metadata_Store + Vector_Index (exit 5); never create an empty store or index here
    for path, what in ((cfg.db_path, "metadata store"), (cfg.index_path, "vector index")):
        if not Path(path).is_file():
            log.error("search unavailable: %s file not found: %s; %s", what, path, REPAIR_HINT)
            return EXIT_INDEX

    db = MetadataStore(cfg.db_path)
    try:
        try:
            index = VectorIndex.load(cfg.index_path, DEFAULT_DIM,
                                     force_postfilter=cfg.force_postfilter)
        except IndexUnavailable as exc:
            log.error("%s; %s", exc, REPAIR_HINT)
            return EXIT_INDEX

        engine = SearchEngine(cfg, db, index, _PendingEncoder())  # type: ignore[arg-type]
        try:
            engine.check_ready()  # ID sets of FAISS and ``vectors`` must match (8.15)
        except SearchUnavailable as exc:
            log.error("%s; %s", exc, REPAIR_HINT)
            return EXIT_INDEX

        # 7. Encoder + Detector, warm-up encode_text (exit 6, naming the file)
        enc_factory = encoder_factory or model_files.default_encoder_factory
        det_factory = detector_factory or model_files.default_detector_factory
        detector = None
        stage_path = model_files.embedder_path(cfg, files)
        try:
            if cfg.enable_detector:
                stage_path = model_files.detector_path(cfg, files)
                detector = det_factory(cfg, files)
            stage_path = model_files.embedder_path(cfg, files)
            encoder = enc_factory(cfg, files)
            encoder.encode_text(WARMUP_QUERY)
        except ModelMissingError as exc:
            log.error("model load failed: %s", exc)
            return EXIT_MODEL_LOAD
        except Exception as exc:  # noqa: BLE001  any load error names the failing file
            log.error("model load failed: %s (%s: %s)", stage_path, type(exc).__name__, exc)
            return EXIT_MODEL_LOAD
        engine.encoder = encoder

        services = Services(
            cfg=cfg,
            db=db,
            index=index,
            encoder=encoder,
            detector_loaded=detector is not None,
            engine=engine,
            query_log=QueryLog(cfg.query_log_path),
        )
        app = create_app(services, web_dir=web_dir)
        app.state.detector = detector

        # 8. Serve on loopback only; the host is a literal, never taken from Config.
        if run is None:
            import uvicorn

            run = uvicorn.run
        log.info("serving NAB Sentry on http://%s:%s/", LOOPBACK_HOST, cfg.port)
        run(app, host=LOOPBACK_HOST, port=cfg.port)
        return EXIT_OK
    finally:
        db.close()


if __name__ == "__main__":
    import sys

    sys.exit(serve())
