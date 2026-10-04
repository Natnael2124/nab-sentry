"""Server startup and binding (task 14.14).

Property 52 (``serve()`` half): a non-loopback host exits with code 2 before any socket exists.
Example tests: the security warning is logged before the server starts; missing model / index
files exit with the documented code and name the file. The slow test runs ``serve()`` with the
real uvicorn in a subprocess and checks it is reachable only on 127.0.0.1.

**Validates: Requirements 10.1, 14.7, 18.1, 18.2, 18.5**
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from nab_sentry.api.app import REPAIR_HINT, serve
from nab_sentry.model_files import EMBEDDER_REL
from nab_sentry.startup import LOOPBACK_HOST, SECURITY_WARNING
from nab_sentry.store.db import MetadataStore
from nab_sentry.store.vector_index import VectorIndex
from tests.fakes import FakeDetector, FakeEncoder

REPO_ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(autouse=True)
def _restore_logging():
    """serve() calls setup_logging(); undo its level/handlers so later tests are unaffected."""
    logger = logging.getLogger("nab_sentry")
    level, handlers = logger.level, list(logger.handlers)
    yield
    for h in list(logger.handlers):
        if h not in handlers:
            logger.removeHandler(h)
            h.close()
    logger.setLevel(level)


def _make_workspace(root: Path) -> Path:
    """Stub manifest (one verified file), empty metadata store and empty vector index."""
    models = root / "models"
    models.mkdir(parents=True)
    stub = models / "stub.bin"
    stub.write_bytes(b"stub weights")
    _write_manifest(models, [("embedder", "stub.bin", hashlib.sha256(b"stub weights").hexdigest())])
    data = root / "data"
    MetadataStore(data / "nab_sentry.db").close()
    VectorIndex().save(data / "vectors.faiss")
    return root


def _write_manifest(models: Path, entries: list[tuple[str, str, str]]) -> None:
    (models / "manifest.json").write_text(
        json.dumps({"manifest_version": 1,
                    "files": [{"role": r, "path": p, "sha256": d} for r, p, d in entries]}),
        encoding="utf-8",
    )


@pytest.fixture()
def ws(tmp_path: Path) -> Path:
    return _make_workspace(tmp_path)


class FakeRun:
    """Stands in for ``uvicorn.run``; records calls and the log messages seen at call time."""

    def __init__(self, seen: list[str] | None = None) -> None:
        self.calls: list[tuple[Any, dict[str, Any]]] = []
        self._seen = seen
        self.messages_at_call: list[str] = []

    def __call__(self, app: Any, **kw: Any) -> None:
        self.calls.append((app, kw))
        if self._seen is not None:
            self.messages_at_call = list(self._seen)


class _Recorder(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)
        self.messages.append(record.getMessage())


@contextlib.contextmanager
def _recording():
    logger = logging.getLogger("nab_sentry")
    rec = _Recorder()
    logger.addHandler(rec)
    try:
        yield rec
    finally:
        logger.removeHandler(rec)


def _serve(ws: Path, *extra: str, run: FakeRun | None = None, **kw: Any) -> int:
    argv = ["--set", f"root={ws}", "--set", f"port={_free_port()}"]
    for item in extra:
        argv += ["--set", item]
    kw.setdefault("encoder_factory", lambda cfg, files: FakeEncoder())
    kw.setdefault("detector_factory", lambda cfg, files: FakeDetector())
    return serve(argv, run=run or FakeRun(), **kw)


# ---------------------------------------------------------------------------------------------
# Property 52 (serve half)
# ---------------------------------------------------------------------------------------------

NEAR_MISSES = [
    "0.0.0.0", "::", "::1", "[::1]", "localhost", "LOCALHOST", "127.0.0.2", "127.1",
    "127.000.000.001", "0177.0.0.1", "2130706433", "192.168.1.10", "10.0.0.5",
    "127.0.0.1:8000", "127.0.0.1\x00",
]
_ipv4 = st.tuples(*[st.integers(0, 255)] * 4).map(lambda t: ".".join(map(str, t)))

# ``--set`` values are whitespace-stripped by the Config parser, so " 127.0.0.1 " on the
# command line *is* loopback; exclude those from the non-loopback space.
non_loopback_hosts = st.one_of(
    st.sampled_from(NEAR_MISSES),
    _ipv4,
    st.ip_addresses(v=6).map(str),
    st.text(),
).filter(lambda h: h.strip() != LOOPBACK_HOST)


class _NoSocket(AssertionError):
    pass


def _forbidden(*_a: Any, **_kw: Any) -> Any:
    raise _NoSocket("a socket was created for a non-loopback host")


# Feature: nab-sentry, Property 52: Only loopback binding is allowed
# **Validates: Requirements 10.1, 18.1, 18.5**
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(host=non_loopback_hosts)
def test_non_loopback_host_exits_before_any_socket(ws: Path, host: str) -> None:
    run = FakeRun()
    err = io.StringIO()
    with mock.patch("socket.socket", _forbidden), \
            mock.patch("socket.create_server", _forbidden), \
            mock.patch("socket.socketpair", _forbidden), \
            contextlib.redirect_stderr(err):
        code = serve(["--set", f"root={ws}", "--set", "port=8765", "--set", f"host={host}"],
                     encoder_factory=_forbidden, detector_factory=_forbidden, run=run)
    assert code == 2
    assert run.calls == []
    if host.strip():  # an empty host is a Config error; any other value is the loopback refusal
        assert "only loopback binding" in err.getvalue()


def test_loopback_host_passes_to_run_as_literal(ws: Path) -> None:
    run = FakeRun()
    assert _serve(ws, f"host={LOOPBACK_HOST}", run=run) == 0
    assert [kw["host"] for _app, kw in run.calls] == ["127.0.0.1"]


# ---------------------------------------------------------------------------------------------
# Security warning (18.2)
# ---------------------------------------------------------------------------------------------


def test_security_warning_logged_before_server_runs(ws: Path) -> None:
    with _recording() as rec:
        run = FakeRun(rec.messages)
        assert _serve(ws, run=run) == 0
    assert len(run.calls) == 1
    assert SECURITY_WARNING in run.messages_at_call
    warn = [r for r in rec.records if r.getMessage() == SECURITY_WARNING]
    assert len(warn) == 1 and warn[0].levelno == logging.WARNING
    # The warning also reaches the log file configured by setup_logging.
    log_file = ws / "data" / "logs" / "nab_sentry.log"
    assert SECURITY_WARNING in log_file.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# Missing files: documented exit codes naming the file (14.7)
# ---------------------------------------------------------------------------------------------


def test_missing_model_file_exits_4_naming_it(ws: Path) -> None:
    models = ws / "models"
    digest = hashlib.sha256(b"stub weights").hexdigest()
    _write_manifest(models, [("embedder", "stub.bin", digest),
                             ("detector", "absent_model.onnx", "0" * 64)])
    run = FakeRun()
    with _recording() as rec:
        assert _serve(ws, run=run) == 4
    assert run.calls == []
    assert any("absent_model.onnx" in m for m in rec.messages)
    assert SECURITY_WARNING not in rec.messages  # exits before the server would start


def test_missing_index_exits_5_naming_it(ws: Path) -> None:
    index_path = ws / "data" / "vectors.faiss"
    index_path.unlink()
    run = FakeRun()
    with _recording() as rec:
        assert _serve(ws, run=run) == 5
    assert run.calls == []
    assert not index_path.exists()
    errors = [m for r, m in zip(rec.records, rec.messages) if r.levelno >= logging.ERROR]
    assert any(str(index_path) in m and REPAIR_HINT in m for m in errors)


def test_missing_db_exits_5_naming_it(ws: Path) -> None:
    db_path = ws / "data" / "nab_sentry.db"
    db_path.unlink()
    with _recording() as rec:
        assert _serve(ws) == 5
    assert not db_path.exists()
    assert any(str(db_path) in m for m in rec.messages)


@pytest.mark.parametrize("stage", ["encoder", "detector"])
def test_model_load_error_exits_6_naming_file(ws: Path, stage: str) -> None:
    def boom(cfg: Any, files: Any) -> Any:
        raise RuntimeError("corrupt weights")

    kw = {"encoder_factory": boom} if stage == "encoder" else {"detector_factory": boom}
    expected = EMBEDDER_REL.rsplit("/", 1)[-1] if stage == "encoder" else "yolo11n.onnx"
    run = FakeRun()
    with _recording() as rec:
        assert _serve(ws, run=run, **kw) == 6
    assert run.calls == []
    assert any(expected in m for m in rec.messages)


# ---------------------------------------------------------------------------------------------
# Slow integration: real uvicorn in a subprocess
# ---------------------------------------------------------------------------------------------

_CHILD = (
    "import sys\n"
    "from nab_sentry.api.app import serve\n"
    "from tests.fakes import FakeDetector, FakeEncoder\n"
    "sys.exit(serve(sys.argv[1:], encoder_factory=lambda c, f: FakeEncoder(),\n"
    "               detector_factory=lambda c, f: FakeDetector()))\n"
)


def _non_loopback_ipv4() -> list[str]:
    found: set[str] = set()
    with contextlib.suppress(OSError):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # UDP connect sends nothing; picks the route
            found.add(s.getsockname()[0])
    with contextlib.suppress(OSError):
        found.update(socket.gethostbyname_ex(socket.gethostname())[2])
    return sorted(ip for ip in found if not ip.startswith("127.") and ip != "0.0.0.0")


@pytest.mark.slow
def test_subprocess_listens_only_on_loopback(tmp_path: Path) -> None:
    ws = _make_workspace(tmp_path / "ws")
    port = _free_port()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO_ROOT), env.get("PYTHONPATH")) if p)
    proc = subprocess.Popen(
        [sys.executable, "-c", _CHILD, "--set", f"root={ws}", "--set", f"port={port}"],
        cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    try:
        url = f"http://127.0.0.1:{port}/api/health"
        deadline = time.monotonic() + 60
        body = None
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                out = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
                pytest.fail(f"serve() exited early with {proc.returncode}:\n{out}")
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    body = json.loads(resp.read())
                    break
            except OSError:
                time.sleep(0.2)
        assert body is not None, "server did not become healthy on 127.0.0.1"
        assert body["status"] == "ok"

        lan = _non_loopback_ipv4()
        if not lan:
            pytest.skip("no non-loopback IPv4 address on this machine")
        for ip in lan:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(3)
                with pytest.raises(OSError):  # refused (or timed out): not listening there
                    s.connect((ip, port))
        # ...while loopback still accepts connections.
        with socket.create_connection((LOOPBACK_HOST, port), timeout=3):
            pass
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)
        if proc.stdout:
            proc.stdout.close()
