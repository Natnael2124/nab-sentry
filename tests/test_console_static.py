"""Static checks for the Operator Console files under ``nab_sentry/web`` (task 16.3).

Validates: Requirements 12.1, 12.2, 12.12, 13.10
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from nab_sentry.api.app import WEB_DIR
from nab_sentry.ingest.detector import TARGET_CLASSES

INDEX = WEB_DIR / "index.html"
APP_JS = WEB_DIR / "app.js"
STYLES = WEB_DIR / "styles.css"

EXPECTED_CSP = "default-src 'self'; media-src 'self'; img-src 'self'"
IGNORED_INPUT_TYPES = {"hidden", "submit", "button", "reset", "image"}


class _ConsoleParser(HTMLParser):
    """Collects the bits of index.html the checks need."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.controls: list[dict[str, str | None]] = []
        self.label_fors: set[str] = set()
        self.metas: list[dict[str, str | None]] = []
        self.by_id: dict[str, dict[str, str | None]] = {}
        self.inline_scripts: list[str] = []
        self.handler_attrs: list[tuple[str, str]] = []
        self.style_attrs: list[str] = []
        self.has_style_element = False
        # #cls option collection
        self.cls_options: list[tuple[str | None, str]] = []
        self._in_cls = False
        self._cur_option: list | None = None
        self._script_has_src = False
        self._script_body: list[str] | None = None

    def handle_starttag(self, tag: str, attrs_list):  # noqa: C901 - flat checks
        attrs = dict(attrs_list)
        for name, _ in attrs_list:
            if name.startswith("on"):
                self.handler_attrs.append((tag, name))
            if name == "style":
                self.style_attrs.append(tag)
        if attrs.get("id"):
            self.by_id[attrs["id"]] = {"tag": tag, **attrs}
        if tag == "meta":
            self.metas.append(attrs)
        elif tag == "label" and attrs.get("for"):
            self.label_fors.add(attrs["for"])
        elif tag in ("select", "textarea") or (
            tag == "input" and (attrs.get("type") or "text").lower() not in IGNORED_INPUT_TYPES
        ):
            self.controls.append({"tag": tag, **attrs})
        if tag == "select" and attrs.get("id") == "cls":
            self._in_cls = True
        elif tag == "option" and self._in_cls:
            self._cur_option = [attrs.get("value"), ""]
        elif tag == "script":
            self._script_has_src = "src" in attrs
            self._script_body = []
        elif tag == "style":
            self.has_style_element = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "option" and self._cur_option is not None:
            self.cls_options.append((self._cur_option[0], self._cur_option[1].strip()))
            self._cur_option = None
        elif tag == "select":
            self._in_cls = False
        elif tag == "script" and self._script_body is not None:
            body = "".join(self._script_body).strip()
            if body or not self._script_has_src:
                self.inline_scripts.append(body)
            self._script_body = None

    def handle_data(self, data: str) -> None:
        if self._cur_option is not None:
            self._cur_option[1] += data
        if self._script_body is not None:
            self._script_body.append(data)


@pytest.fixture(scope="module")
def parsed() -> _ConsoleParser:
    p = _ConsoleParser()
    p.feed(INDEX.read_text(encoding="utf-8"))
    p.close()
    return p


def test_console_files_exist() -> None:
    for f in (INDEX, APP_JS, STYLES):
        assert f.is_file(), f"missing console file: {f}"


def test_no_external_urls_in_web_dir() -> None:
    """13.10: the Console makes no external requests — no http(s):// anywhere under web/."""
    url_re = re.compile(r"https?://", re.IGNORECASE)
    offenders = []
    for path in sorted(WEB_DIR.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if url_re.search(line):
                offenders.append(f"{path.relative_to(WEB_DIR)}:{lineno}: {line.strip()}")
    assert not offenders, "external URLs found:\n" + "\n".join(offenders)


def test_csp_meta_present(parsed: _ConsoleParser) -> None:
    csp = [
        m for m in parsed.metas
        if (m.get("http-equiv") or "").lower() == "content-security-policy"
    ]
    assert len(csp) == 1, "expected exactly one CSP meta tag"
    assert csp[0].get("content") == EXPECTED_CSP


def test_every_form_control_has_label(parsed: _ConsoleParser) -> None:
    """12.12: every visible input/select/textarea has an id with a matching <label for>."""
    assert parsed.controls, "no form controls found"
    missing = []
    for c in parsed.controls:
        cid = c.get("id")
        if not cid or cid not in parsed.label_fors:
            missing.append(f"<{c['tag']} id={cid!r} name={c.get('name')!r}>")
    assert not missing, "controls without <label for>: " + ", ".join(missing)
    # Every label points at an existing element.
    dangling = sorted(f for f in parsed.label_fors if f not in parsed.by_id)
    assert not dangling, f"labels pointing at missing ids: {dangling}"


def test_expected_controls_present(parsed: _ConsoleParser) -> None:
    ids = {c.get("id") for c in parsed.controls}
    assert {"q", "camera", "start", "end", "cls"} <= ids
    assert parsed.by_id["q"].get("maxlength") == "256"


def test_status_region_is_live(parsed: _ConsoleParser) -> None:
    status = parsed.by_id.get("status")
    assert status is not None, "#status element missing"
    assert status.get("role") == "status"
    assert status.get("aria-live") == "polite"


def test_class_select_lists_target_classes(parsed: _ConsoleParser) -> None:
    """12.2: #cls = "All classes" plus exactly the six Target_Classes."""
    opts = parsed.cls_options
    values = [v for v, _ in opts]
    assert len(values) == len(set(values)), f"duplicate #cls options: {values}"
    assert ("", "All classes") in opts, f"missing 'All classes' option: {opts}"
    class_values = {v for v in values if v != ""}
    assert class_values == set(TARGET_CLASSES.values())
    assert len(class_values) == 6
    assert len(opts) == 7


def test_no_inline_script_handlers_or_styles(parsed: _ConsoleParser) -> None:
    """The CSP blocks inline code, so index.html must not rely on it."""
    assert not parsed.inline_scripts, f"inline <script> bodies: {parsed.inline_scripts}"
    assert not parsed.handler_attrs, f"inline on* handlers: {parsed.handler_attrs}"
    assert not parsed.style_attrs, f"style= attributes on: {parsed.style_attrs}"
    assert not parsed.has_style_element, "inline <style> element present"


def test_app_js_sets_alt_on_result_images() -> None:
    """12.12: result thumbnails get alt text."""
    js = APP_JS.read_text(encoding="utf-8")
    assert re.search(r"createElement\(\s*[\"']img[\"']\s*\)", js), "app.js creates no <img>"
    assert re.search(r"\.alt\s*=|setAttribute\(\s*[\"']alt[\"']", js), "app.js never sets img alt"


def test_app_js_avoids_html_injection_sinks() -> None:
    js = APP_JS.read_text(encoding="utf-8")
    sinks = re.findall(
        r"\.(?:innerHTML|outerHTML)\s*[+]?=|insertAdjacentHTML\s*\(|document\.write\s*\(|\beval\s*\(",
        js,
    )
    assert not sinks, f"HTML/code injection sinks in app.js: {sinks}"
