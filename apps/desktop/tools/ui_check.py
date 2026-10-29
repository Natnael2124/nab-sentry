"""Drive the real UI in Edge (Playwright) against the real server and models, and save screenshots
(task 4.4). Development tool; needs ``pip install playwright`` in venv-desktop and Microsoft Edge.

Usage::

    venv-desktop\\Scripts\\python.exe apps\\desktop\\tools\\ui_check.py VIDEO [--out DIR] [--width 1366 --height 768] [--dark]

Flow: home -> new search -> add VIDEO (the same call a drag-and-drop makes) -> start -> watch
progress -> search -> open a result -> visits tab -> settings. Uses a throw-away data folder.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "apps" / "desktop"))

from nab_find.app import free_port, serve  # noqa: E402
from nab_find.server import AppState  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("video", type=Path)
    p.add_argument("--out", type=Path, default=_ROOT / "data" / "desktop" / "ui_check")
    p.add_argument("--width", type=int, default=1366)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--dark", action="store_true")
    p.add_argument("--query", default="woman pushing a baby stroller")
    args = p.parse_args(argv)
    from playwright.sync_api import sync_playwright

    data = _ROOT / "data" / "desktop" / "ui_check_data"
    shutil.rmtree(data, ignore_errors=True)
    args.out.mkdir(parents=True, exist_ok=True)
    state = AppState(data, _ROOT / "models" / "desktop")
    port = free_port()
    serve(state, _ROOT / "apps" / "desktop" / "web", port)
    shots = []
    errors: list[str] = []

    def shot(page, name):
        path = args.out / f"{len(shots) + 1:02d}-{name}{'-dark' if args.dark else ''}.png"
        page.screenshot(path=str(path))
        shots.append(path)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": args.width, "height": args.height},
                                color_scheme="dark" if args.dark else "light")
        page.on("pageerror", lambda e: errors.append(f"page error: {e}"))
        page.on("console", lambda m: errors.append(f"console {m.type}: {m.text}") if m.type == "error" else None)
        page.goto(f"http://127.0.0.1:{port}/index.html#t={state.token}")
        page.wait_for_selector("text=Find the moment in your recordings")
        shot(page, "home-empty")
        page.click("text=New search")
        page.wait_for_selector("text=Add your recordings")
        shot(page, "setup-empty")
        page.evaluate(f"window.NF.onDrop({[str(args.video.resolve())]!r})".replace("'", '"'))
        page.wait_for_selector("table.vlist", timeout=60000)
        page.wait_for_timeout(500)
        shot(page, "setup-with-video")
        page.click("text=Only a time window")
        page.wait_for_timeout(200)
        shot(page, "setup-range")
        page.click("text=Go through everything")
        page.click("button:has-text('Start')")
        page.wait_for_selector("#progress", timeout=30000)
        t0 = time.time()
        took_progress = False
        while time.time() - t0 < 900:
            txt = page.inner_text("#progress")
            if not took_progress and "%" in txt and "ready" in txt.lower():
                page.wait_for_timeout(1500)
                shot(page, "progress")
                took_progress = True
            if "Ready to search" in txt:
                break
            page.wait_for_timeout(500)
        shot(page, "done")
        page.fill("input[type=search]", args.query)
        page.keyboard.press("Enter")
        page.wait_for_selector(".result", timeout=60000)
        page.wait_for_timeout(800)
        shot(page, "results")
        page.click(".result >> nth=0")
        page.wait_for_selector("video", timeout=60000)
        page.wait_for_timeout(2500)
        shot(page, "player")
        page.keyboard.press("Escape")
        page.click("text=People & vehicles")
        page.wait_for_selector(".result", timeout=30000)
        page.wait_for_timeout(800)
        shot(page, "visits")
        page.click("#open-settings")
        page.wait_for_selector("dialog[open]")
        shot(page, "settings")
        page.keyboard.press("Escape")
        page.click("#go-home")
        page.wait_for_selector(".case-card")
        shot(page, "home-with-case")
        browser.close()
    print("\n".join(str(s) for s in shots))
    print("browser errors:", errors or "none")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
