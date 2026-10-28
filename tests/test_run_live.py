"""run_live.ps1 (V2 Requirement 16, task 16.4): parses, chooses services, restarts on exit."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run_live.ps1"
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")

pytestmark = pytest.mark.skipif(POWERSHELL is None or sys.platform != "win32",
                                reason="Windows PowerShell not available")


def _workspace(tmp_path: Path, cameras: bool) -> Path:
    ws = tmp_path / "ws"
    (ws / "venv" / "Scripts").mkdir(parents=True)
    shutil.copy2(SCRIPT, ws / "run_live.ps1")
    shutil.copy2(ROOT / "venv" / "Scripts" / "python.exe", ws / "venv" / "Scripts" / "python.exe")
    shutil.copy2(ROOT / "venv" / "pyvenv.cfg", ws / "venv" / "pyvenv.cfg")
    if cameras:
        (ws / "config").mkdir()
        (ws / "config" / "cameras.yaml").write_text("cameras: []\n", encoding="utf-8")
    return ws


def _run(ws: Path, *args: str, env: dict | None = None, timeout: float = 60):
    e = {k: v for k, v in os.environ.items() if not k.startswith("NAB_TELEGRAM")}
    e.update(env or {})
    return subprocess.run([POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                           "-File", str(ws / "run_live.ps1"), *args],
                          cwd=str(ws), capture_output=True, text=True, timeout=timeout, env=e)


def test_script_parses_without_errors() -> None:
    cmd = ("$t=$null; $e=$null; "
           f"[void][System.Management.Automation.Language.Parser]::ParseFile('{SCRIPT}', [ref]$t, [ref]$e); "
           "if ($e.Count -gt 0) { $e | ForEach-Object { $_.ToString() }; exit 1 } else { exit 0 }")
    r = subprocess.run([POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", cmd],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr


def test_dry_run_without_cameras_or_telegram(tmp_path: Path) -> None:
    r = _run(_workspace(tmp_path, cameras=False), "-DryRun")
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "recorder: skipped (no config\\cameras.yaml" in out
    assert "notifier: skipped (set NAB_TELEGRAM_TOKEN" in out
    started = [line.split(":")[0].replace("would start ", "") for line in out.splitlines() if line.startswith("would start")]
    assert started == ["watcher", "api"]


def test_dry_run_with_everything(tmp_path: Path) -> None:
    r = _run(_workspace(tmp_path, cameras=True), "-DryRun", "-Port", "8800",
             env={"NAB_TELEGRAM_TOKEN": "1:x", "NAB_TELEGRAM_CHATS": "5"})
    out = r.stdout
    assert [l.split(":")[0].replace("would start ", "") for l in out.splitlines() if l.startswith("would start")] \
        == ["recorder", "watcher", "notifier", "api"]
    assert "port=8800" in out and "1:x" not in out


def test_missing_venv_exits_1(tmp_path: Path) -> None:
    shutil.copy2(SCRIPT, tmp_path / "run_live.ps1")
    r = subprocess.run([POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                        "-File", str(tmp_path / "run_live.ps1"), "-DryRun"],
                       cwd=str(tmp_path), capture_output=True, text=True, timeout=60)
    assert r.returncode == 1 and "venv/ not found" in r.stdout + r.stderr


def test_restarts_a_service_that_exits(tmp_path: Path) -> None:
    """The watcher stand-in exits at once; the supervisor must log the exit and restart it."""
    ws = _workspace(tmp_path, cameras=False)
    (ws / "scripts").mkdir()
    (ws / "scripts" / "ingest.py").write_text(
        "import pathlib, sys\np = pathlib.Path('data/starts.txt')\n"
        "p.write_text(p.read_text() + 'x' if p.exists() else 'x')\nsys.exit(3)\n", encoding="utf-8")
    proc = subprocess.Popen([POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                             "-File", str(ws / "run_live.ps1"), "-NoServer", "-NoNotifier"],
                            cwd=str(ws), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.time() + 40
        starts = ws / "data" / "starts.txt"
        while time.time() < deadline and not (starts.exists() and len(starts.read_text()) >= 2):
            time.sleep(0.5)
        assert starts.exists() and len(starts.read_text()) >= 2, "watcher was not restarted"
    finally:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    log = (ws / "data" / "logs" / "run_live.log").read_text(encoding="utf-8-sig")
    assert "watcher: exited with code 3" in log and "restarting in 2 s" in log


def test_vlm_switch_passes_vlm_to_the_notifier(tmp_path: Path) -> None:
    env = {"NAB_TELEGRAM_TOKEN": "1:x", "NAB_TELEGRAM_CHATS": "5"}
    ws = _workspace(tmp_path, cameras=False)
    with_vlm = [l for l in _run(ws, "-DryRun", "-Vlm", env=env).stdout.splitlines() if "notifier" in l]
    without = [l for l in _run(ws, "-DryRun", env=env).stdout.splitlines() if "notifier" in l]
    assert with_vlm[0].endswith("notify.py --vlm") and without[0].endswith("notify.py")
