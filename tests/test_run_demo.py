"""Tests for run_demo.ps1 (Requirement 17.1, 17.2, 17.6, 17.7).

The script is copied into a temporary workspace. The venv there is a copy of the real venv
launcher (``python.exe`` + ``pyvenv.cfg``), and ``scripts/ingest.py`` and
``nab_sentry/api/app.py`` are small stand-ins, so the script's control flow (venv check,
ingest gating, exit-code propagation, health polling, Console URL) runs without models.
"""

from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run_demo.ps1"
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")

pytestmark = pytest.mark.skipif(
    POWERSHELL is None or sys.platform != "win32", reason="Windows PowerShell not available"
)

URL_PREFIX = "NAB Sentry console: http://127.0.0.1:"


def _run(workspace: Path, *args: str, timeout: float = 90) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(workspace / "run_demo.ps1"), *args],
        cwd=str(workspace), capture_output=True, text=True, timeout=timeout,
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _workspace(tmp_path: Path, *, ingest: str, app: str, indexed: bool) -> Path:
    ws = tmp_path / "ws"
    (ws / "venv" / "Scripts").mkdir(parents=True)
    shutil.copy2(SCRIPT, ws / "run_demo.ps1")
    venv = ROOT / "venv"
    shutil.copy2(venv / "Scripts" / "python.exe", ws / "venv" / "Scripts" / "python.exe")
    shutil.copy2(venv / "pyvenv.cfg", ws / "venv" / "pyvenv.cfg")
    (ws / "scripts").mkdir()
    (ws / "scripts" / "ingest.py").write_text(textwrap.dedent(ingest), encoding="utf-8")
    pkg = ws / "nab_sentry" / "api"
    pkg.mkdir(parents=True)
    (ws / "nab_sentry" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "app.py").write_text(textwrap.dedent(app), encoding="utf-8")
    (ws / "data" / "videos").mkdir(parents=True)
    if indexed:
        (ws / "data" / "nab_sentry.db").write_bytes(b"")
        (ws / "data" / "vectors.faiss").write_bytes(b"")
    return ws


# Stand-in server: records that it started; behaviour chosen by the test.
APP_MARKER = """
import pathlib, sys
pathlib.Path("data", "server_started").write_text("1")
"""

APP_EXIT_4 = APP_MARKER + """
sys.stderr.write("ERROR manifest check failed: models/yolov8n.onnx sha256 mismatch\\n")
sys.exit(4)
"""

APP_HEALTHY = APP_MARKER + """
import json, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

port = int(sys.argv[sys.argv.index("--set") + 1].split("=", 1)[1])

class H(BaseHTTPRequestHandler):
    calls = 0
    def do_GET(self):
        H.calls += 1
        loaded = H.calls >= 2  # first poll: models not loaded yet
        body = json.dumps({"status": "ok", "videos": 1, "vectors": 3,
                           "models_loaded": {"detector": loaded, "embedder": loaded}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        if loaded:
            threading.Timer(1.0, srv.shutdown).start()
    def log_message(self, *a):
        pass

srv = HTTPServer(("127.0.0.1", port), H)
srv.serve_forever()
sys.exit(0)
"""

INGEST_OK = """
import pathlib, sys
pathlib.Path("data", "nab_sentry.db").write_bytes(b"")
pathlib.Path("data", "vectors.faiss").write_bytes(b"")
print("ingested=2 failed=1 already_indexed=0 vectors=40")
sys.exit(0)
"""

INGEST_EMPTY = """
import sys
print("no video files found in data/videos", file=sys.stderr)
print("ingested=0 failed=0 already_indexed=0 vectors=0")
sys.exit(1)
"""

INGEST_NOT_CALLED = """
import pathlib
pathlib.Path("data", "ingest_called").write_text("1")
"""


def test_script_parses_without_errors() -> None:
    cmd = (
        "$t=$null; $e=$null; "
        f"[void][System.Management.Automation.Language.Parser]::ParseFile('{SCRIPT}', [ref]$t, [ref]$e); "
        "if ($e.Count -gt 0) { $e | ForEach-Object { $_.ToString() }; exit 1 } else { exit 0 }"
    )
    r = subprocess.run([POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", cmd],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr


def test_missing_venv_exits_1_with_corrective_step(tmp_path: Path) -> None:
    shutil.copy2(SCRIPT, tmp_path / "run_demo.ps1")
    r = _run(tmp_path)
    out = r.stdout + r.stderr
    assert r.returncode == 1
    assert "venv/ not found" in out
    assert "py -3.12 -m venv venv" in out
    assert URL_PREFIX not in out


def test_ingest_failure_names_videos_dir_and_does_not_start_server(tmp_path: Path) -> None:
    ws = _workspace(tmp_path, ingest=INGEST_EMPTY, app=APP_HEALTHY, indexed=False)
    r = _run(ws, "-Port", str(_free_port()))
    out = r.stdout + r.stderr
    assert r.returncode == 1, out
    assert "data\\videos\\" in out
    assert "0 video(s) ingested, 0 failed" in out
    assert not (ws / "data" / "server_started").exists()
    assert URL_PREFIX not in out


def test_server_startup_failure_propagates_exit_code(tmp_path: Path) -> None:
    ws = _workspace(tmp_path, ingest=INGEST_NOT_CALLED, app=APP_EXIT_4, indexed=True)
    r = _run(ws, "-Port", str(_free_port()))
    out = r.stdout + r.stderr
    assert r.returncode == 4, out
    assert "manifest check failed: models/yolov8n.onnx" in out  # server's own error
    assert "fetch_models.py" in out  # corrective step
    assert not (ws / "data" / "ingest_called").exists()  # both files present: no ingest
    assert URL_PREFIX not in out


def test_ingests_then_prints_console_url_when_models_loaded(tmp_path: Path) -> None:
    port = _free_port()
    ws = _workspace(tmp_path, ingest=INGEST_OK, app=APP_HEALTHY, indexed=False)
    r = _run(ws, "-Port", str(port))
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "2 video(s) ingested, 1 failed" in out
    assert f"{URL_PREFIX}{port}/" in out
    assert out.index("video(s) ingested") < out.index(URL_PREFIX)
    assert (ws / "data" / "server_started").exists()
