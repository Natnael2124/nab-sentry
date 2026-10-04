"""Network isolation and write locations (Requirements 13.6, 13.7, 13.9).

Both tests run the same flow in a throwaway workspace root:

1. write the Synthetic_Video into ``data/videos/``;
2. ingest it with ``scripts/ingest.py`` (startup checks, FileSource, real ffmpeg Playback_File);
3. run the API_Server startup sequence via ``serve()`` and, inside the injected ``run``, send
   ``/api/health``, ``/api/search?q=red square``, ``/media/video/<id>`` (full and ranged) and
   ``/media/thumbs/<name>`` through a ``TestClient``.

While the flow runs:

- the ``network_guard`` fixture blocks and records every non-loopback connect / resolve (13.6);
- a write audit (``sys.addaudithook``) records every path the process writes, creates, renames,
  removes, or opens as a SQLite database, plus every absolute path handed to an ffmpeg child
  process; each must resolve under ``data/`` or ``models/`` (13.7).

The fast test uses FakeEncoder (colour mode) with the detector off and a stub manifest, so the
write-location property is exercised on any machine with the bundled ffmpeg. The slow test uses
the real OpenCLIP and YOLO11n models from the real ``models/`` and is skipped when the model
manifest check fails.

Two whitelists are applied, neither of which stores NAB Sentry data:

- CPython's bytecode cache (``__pycache__/`` and its ``*.pyc``), written by the interpreter's
  import system on a module's first import (disable with ``PYTHONDONTWRITEBYTECODE=1``);
- the null device (``os.devnull``), opened for writing by ``subprocess.DEVNULL``.

The OS temp directory is deliberately not whitelisted: Requirement 13.7 allows footage and log
files only under ``data/`` and ``models/``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import socket
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest

from nab_sentry.config import Config
from nab_sentry.synthetic import SyntheticObject, SyntheticSpec, write_synthetic

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ingest.py"

# --------------------------------------------------------------------------- write audit

_WRITE_FLAGS = (
    os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
    | getattr(os, "O_EXCL", 0)
)
_PATH_EVENTS = {
    "os.mkdir": (0,),
    "os.remove": (0,),
    "os.rmdir": (0,),
    "os.rename": (0, 1),  # os.rename and os.replace
    "os.truncate": (0,),
    "os.symlink": (1,),
    "os.link": (1,),
    "shutil.copyfile": (1,),
    "shutil.rmtree": (0,),
    "sqlite3.connect": (0,),
}

_lock = threading.Lock()
_active: list[tuple[str, str]] | None = None  # (event, path) while a recording is running


def _as_path(p: Any) -> str | None:
    if isinstance(p, int) or p is None:
        return None  # file descriptors / dir_fd
    try:
        p = os.fspath(p)
    except TypeError:
        return None
    if isinstance(p, bytes):
        p = os.fsdecode(p)
    return p


def _is_ffmpeg(args: Any) -> bool:
    try:
        exe = os.fspath(args[0])
    except (TypeError, IndexError):
        return False
    return "ffmpeg" in os.path.basename(str(exe)).lower()


_SOURCE_DIRS = tuple(
    os.path.normcase(str(Path(__file__).resolve().parents[1] / d)) + os.sep
    for d in ("nab_sentry", "scripts")
)


def _origin() -> str:
    """Innermost NAB Sentry source line on the stack (for failure messages), or ''."""
    f = sys._getframe(2)
    while f is not None:
        name = os.path.normcase(f.f_code.co_filename)
        if name.startswith(_SOURCE_DIRS):
            return f"{os.path.basename(name)}:{f.f_lineno}"
        f = f.f_back
    return ""


def _audit(event: str, args: tuple[Any, ...]) -> None:
    rec = _active
    if rec is None:
        return
    try:
        found: list[tuple[str, str]] = []
        if event == "open":
            path, mode, flags = (tuple(args) + (None, None, None))[:3]
            writes = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and flags & _WRITE_FLAGS
            )
            p = _as_path(path)
            if writes and p is not None:
                found.append((event, p))
        elif event in _PATH_EVENTS:
            for i in _PATH_EVENTS[event]:
                p = _as_path(args[i]) if i < len(args) else None
                if p is not None and p != ":memory:":
                    found.append((event, p))
        elif event == "subprocess.Popen":
            argv = args[1] if len(args) > 1 else None
            if isinstance(argv, (list, tuple)) and _is_ffmpeg(argv):
                # Every absolute path given to ffmpeg (inputs and the output file) is checked;
                # the child's own writes are not visible to this process's audit hook.
                for a in argv[1:]:
                    p = _as_path(a)
                    if p is not None and os.path.isabs(p):
                        found.append(("ffmpeg-arg", p))
        if found:
            origin = _origin()
            if origin:
                found = [(f"{e} <- {origin}", p) for e, p in found]
            with _lock:
                rec.extend(found)
    except Exception:  # an audit hook must never break the code under test
        pass


sys.addaudithook(_audit)  # cannot be removed; it records only while _active is set


class WriteAudit:
    """Context manager collecting ``(event, path)`` for every write-like operation."""

    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []

    def __enter__(self) -> WriteAudit:
        global _active
        _active = self.records
        return self

    def __exit__(self, *exc: object) -> None:
        global _active
        _active = None


def _norm(p: str | Path) -> Path:
    return Path(os.path.normcase(os.path.realpath(os.path.abspath(p))))


def _is_bytecode_cache(p: Path) -> bool:
    # importlib creates "__pycache__/", writes "<name>.pyc.<id>", then os.replace()s it
    return p.name == "__pycache__" or (p.parent.name == "__pycache__" and ".pyc" in p.name)


def _is_null_device(raw: str) -> bool:
    # subprocess.DEVNULL opens os.devnull ("nul" on Windows) for writing; it stores nothing.
    return os.path.normcase(raw) == os.path.normcase(os.devnull)


def outside_allowed(records: list[tuple[str, str]], allowed: list[Path]) -> list[tuple[str, str]]:
    roots = [_norm(r) for r in allowed]
    bad = []
    for event, raw in records:
        if _is_null_device(raw):
            continue
        p = _norm(raw)
        if any(p == r or p.is_relative_to(r) for r in roots) or _is_bytecode_cache(p):
            continue
        bad.append((event, raw))
    return bad


# --------------------------------------------------------------------------- flow helpers


@pytest.fixture(autouse=True)
def _restore_logging():
    """ingest/serve call setup_logging(); drop the handlers they add (and close log files)."""
    logger = logging.getLogger("nab_sentry")
    level, handlers = logger.level, list(logger.handlers)
    yield
    for h in list(logger.handlers):
        if h not in handlers:
            logger.removeHandler(h)
            h.close()
    logger.setLevel(level)


def _load_ingest_script():
    spec = importlib.util.spec_from_file_location("ingest_script_offline", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class FlowResult:
    ingest_code: int
    serve_code: int
    responses: dict[str, Any] = field(default_factory=dict)


def run_flow(
    root: Path,
    models_dir: Path,
    spec: SyntheticSpec,
    *,
    overrides: tuple[str, ...] = (),
    encoder_factory: Callable[..., Any] | None = None,
    detector_factory: Callable[..., Any] | None = None,
) -> FlowResult:
    """Synthetic video -> ingest script -> serve() startup -> media/search requests."""
    from fastapi.testclient import TestClient

    from nab_sentry.api.app import serve

    sets: list[str] = []
    for item in (f"root={root}", f"models_dir={models_dir}", *overrides):
        sets += ["--set", item]

    cfg = Config(root=root, models_dir=models_dir)
    write_synthetic(spec, cfg.videos_dir)

    ingest = _load_ingest_script()
    kw: dict[str, Any] = {}
    if encoder_factory is not None:
        kw["encoder_factory"] = encoder_factory
    if detector_factory is not None:
        kw["detector_factory"] = detector_factory
    ingest_code = ingest.main(sets, **kw)
    result = FlowResult(ingest_code=ingest_code, serve_code=-1)
    if ingest_code != 0:
        return result

    def fake_run(app: Any, **_kw: Any) -> None:
        # serve() closes the Metadata_Store when run() returns, so requests happen here.
        r = result.responses
        with TestClient(app) as client:
            r["health"] = client.get("/api/health")
            r["search"] = client.get("/api/search", params={"q": "red square"})
            events = r["search"].json() if r["search"].status_code == 200 else []
            if events:
                r["video"] = client.get(events[0]["video_url"])
                r["video_range"] = client.get(events[0]["video_url"],
                                              headers={"Range": "bytes=0-99"})
                r["thumb"] = client.get(events[0]["thumbnail_url"])

    result.serve_code = serve([*sets, "--set", f"port={_free_port()}"], run=fake_run, **kw)
    return result


def assert_flow_ok(res: FlowResult) -> None:
    assert res.ingest_code == 0, "ingest failed"
    assert res.serve_code == 0, "serve startup failed"
    r = res.responses
    assert r["health"].status_code == 200
    assert r["health"].json()["videos"] == 1
    assert r["search"].status_code == 200
    assert r["search"].json(), "search returned no Events"
    assert r["video"].status_code == 200
    assert r["video"].headers["content-type"] == "video/mp4"
    assert len(r["video"].content) > 0
    assert r["video_range"].status_code == 206
    assert len(r["video_range"].content) == 100
    assert r["thumb"].status_code == 200
    assert r["thumb"].headers["content-type"] == "image/jpeg"


def assert_writes_inside(audit: WriteAudit, cfg: Config) -> None:
    bad = outside_allowed(audit.records, [cfg.data_dir, cfg.models_dir])
    assert bad == [], (
        "writes outside data/ and models/ (Requirement 13.7):\n"
        + "\n".join(f"  {event}: {path}" for event, path in bad)
    )
    # The audit actually saw the ingest/API outputs (guards against a silently dead hook).
    seen = [_norm(p) for _, p in audit.records]
    for d in (cfg.thumbs_dir, cfg.playback_dir):
        assert any(p.is_relative_to(_norm(d)) for p in seen), f"no recorded write under {d}"
    assert _norm(cfg.query_log_path) in seen
    assert _norm(cfg.db_path) in seen


def _stub_manifest(models: Path) -> None:
    models.mkdir(parents=True, exist_ok=True)
    stub = models / "stub.bin"
    stub.write_bytes(b"stub weights")
    digest = hashlib.sha256(stub.read_bytes()).hexdigest()
    (models / "manifest.json").write_text(
        json.dumps({"manifest_version": 1,
                    "files": [{"role": "embedder", "path": "stub.bin", "sha256": digest}]}),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- unit checks


def test_audit_records_write_opens_and_ignores_reads(tmp_path: Path) -> None:
    target = tmp_path / "a.txt"
    with WriteAudit() as audit:
        target.write_text("x", encoding="utf-8")
        target.read_text(encoding="utf-8")
        os.replace(target, tmp_path / "b.txt")
    paths = [(e, _norm(p)) for e, p in audit.records]
    assert ("open", _norm(target)) in paths
    assert ("os.rename", _norm(tmp_path / "b.txt")) in paths
    assert sum(1 for e, p in paths if e == "open" and p == _norm(target)) == 1  # read not recorded
    assert outside_allowed(audit.records, [tmp_path]) == []
    assert outside_allowed(audit.records, [tmp_path / "data"]) != []


# --------------------------------------------------------------------------- flows


def test_fake_model_flow_writes_only_under_data_and_models(tmp_path: Path, network_guard) -> None:
    """Startup, ingest, search, media with FakeEncoder: no network, writes only in data/models."""
    from tests.fakes import FakeEncoder

    root = tmp_path / "ws"
    models = root / "models"
    _stub_manifest(models)
    spec = SyntheticSpec(
        duration_s=12.0,
        objects=(SyntheticObject("red square", "square", "red", 2.0, 8.0),),
    )
    cfg = Config(root=root, models_dir=models)

    with WriteAudit() as audit:
        res = run_flow(
            root, models, spec,
            overrides=("enable_detector=false",),
            encoder_factory=lambda cfg, files: FakeEncoder(colour_mode=True),
        )

    assert_flow_ok(res)
    network_guard.assert_no_attempts()
    assert_writes_inside(audit, cfg)


@pytest.mark.slow
def test_real_model_flow_is_offline_and_writes_only_under_data_and_models(
    tmp_path: Path, network_guard
) -> None:
    """Requirements 13.6, 13.7, 13.9 with the real OpenCLIP and YOLO11n models."""
    from nab_sentry.startup import verify_manifest

    models = Config().models_dir
    check = verify_manifest(models)
    if not check.ok:
        pytest.skip("real models not provisioned: "
                    + "; ".join(str(f) for f in check.failures))

    root = tmp_path / "ws"
    cfg = Config(root=root, models_dir=models)
    with WriteAudit() as audit:
        res = run_flow(root, models, SyntheticSpec())

    assert_flow_ok(res)
    network_guard.assert_no_attempts()
    assert_writes_inside(audit, cfg)
