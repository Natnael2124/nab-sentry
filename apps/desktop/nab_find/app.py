"""NAB Find entry point: one window (pywebview / WebView2) on top of the local server (design: app.py).

    venv-desktop\\Scripts\\python.exe -m nab_find.app [--dev] [--data DIR]

``--dev`` keeps data in the repository's ``data/desktop`` and reads models from ``models/desktop``.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import logging.handlers
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

TITLE = "NAB Find"
MUTEX = "Local\\NABFind-SingleInstance"
VIDEO_TYPES = ("Video files (*.mp4;*.mkv;*.avi;*.mov;*.asf;*.wmv;*.ts;*.dav;*.264;*.265;*.h264;*.h265;*.mpg;"
               "*.m4v;*.flv;*.3gp;*.webm)", "All files (*.*)")


def bundle_root() -> Path:
    """Folder with web/ and models/: the PyInstaller bundle, or the source tree."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def single_instance() -> bool:
    """True if this is the only copy; otherwise bring the other window forward and return False."""
    try:
        ctypes.windll.kernel32.CreateMutexW(None, False, MUTEX)
        if ctypes.windll.kernel32.GetLastError() != 183:  # ERROR_ALREADY_EXISTS
            return True
        hwnd = ctypes.windll.user32.FindWindowW(None, TITLE)
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            ctypes.windll.user32.SetForegroundWindow(hwnd)
        return False
    except (AttributeError, OSError):
        return True


def setup_logging(data_dir: Path) -> None:
    logs = data_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    h = logging.handlers.RotatingFileHandler(logs / "nab_find.log", maxBytes=2_000_000, backupCount=3,
                                             encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(h)
    root.setLevel(logging.INFO)
    for noisy in ("uvicorn.access", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def lower_priority() -> None:
    try:
        import psutil

        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:  # noqa: BLE001 - nice to have
        pass


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Api:
    """Native helpers the page calls through ``window.pywebview.api``."""

    def __init__(self) -> None:
        self._window = None  # private: pywebview must not expose or walk it

    def pick_files(self) -> list[str]:
        import webview

        res = self._window.create_file_dialog(webview.FileDialog.OPEN, allow_multiple=True, file_types=VIDEO_TYPES)
        return list(res or [])

    def pick_folder(self) -> list[str]:
        import webview

        res = self._window.create_file_dialog(webview.FileDialog.FOLDER)
        return list(res or [])

    def open_folder(self, path: str) -> bool:
        p = Path(path)
        if p.is_file():
            subprocess.Popen(["explorer", "/select,", str(p)])
        elif p.is_dir():
            os.startfile(str(p))  # noqa: S606 - opens Explorer on a folder we wrote
        else:
            return False
        return True


def serve(state, web_dir: Path, port: int):
    import uvicorn

    from nab_find.server import create_app

    config = uvicorn.Config(create_app(state, web_dir), host="127.0.0.1", port=port, log_level="warning",
                            access_log=False)
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, name="nab-find-server", daemon=True).start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    return server


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="NAB Find")
    p.add_argument("--dev", action="store_true", help="use the repository's data/desktop and models/desktop")
    p.add_argument("--data", type=Path)
    p.add_argument("--debug", action="store_true", help="enable the web inspector")
    args = p.parse_args(argv)
    if str(Path(__file__).resolve().parents[1]) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from nab_find.settings import default_data_dir

    if args.dev:
        data_dir, model_dir = repo_root() / "data" / "desktop", repo_root() / "models" / "desktop"
    else:
        data_dir, model_dir = args.data or default_data_dir(), bundle_root() / "models"
    if args.data:
        data_dir = args.data
    data_dir.mkdir(parents=True, exist_ok=True)
    if not single_instance():
        return 0
    setup_logging(data_dir)
    lower_priority()
    log = logging.getLogger("nab_find.app")
    log.info("starting %s, data %s, models %s", TITLE, data_dir, model_dir)

    import webview
    from webview.dom import DOMEventHandler

    from nab_find.server import AppState

    state = AppState(data_dir, model_dir)
    port = free_port()
    serve(state, bundle_root() / "web", port)
    api = Api()
    window = webview.create_window(TITLE, f"http://127.0.0.1:{port}/index.html#t={state.token}", js_api=api,
                                   width=1320, height=860, min_size=(1024, 680), background_color="#0f1115")
    api._window = window

    def on_drop(e) -> None:
        paths = [f.get("pywebviewFullPath") for f in e.get("dataTransfer", {}).get("files", [])]
        paths = [x for x in paths if x]
        if paths:
            window.evaluate_js(f"window.NF && window.NF.onDrop({json.dumps(paths)})")

    def on_loaded() -> None:
        window.dom.document.events.drop += DOMEventHandler(on_drop, prevent_default=True, stop_propagation=True)

    window.events.loaded += on_loaded
    webview.start(debug=args.debug, private_mode=False, storage_path=str(data_dir / "webview"))
    log.info("window closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
