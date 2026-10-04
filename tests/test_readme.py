"""Static checks on README.md at the workspace root.

_Requirements: 17.4, 18.3_
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parent.parent / "README.md"

# Section heading -> the exact PowerShell command it must give in a code block.
SECTION_COMMANDS = {
    "Setup": "venv\\Scripts\\python.exe -m pip install -r requirements.txt",
    "Model provisioning": "venv\\Scripts\\python.exe scripts\\fetch_models.py",
    "Ingest": "venv\\Scripts\\python.exe scripts\\ingest.py data\\videos",
    "Demo": ".\\run_demo.ps1",
    "Evaluation": "venv\\Scripts\\python.exe scripts\\evaluate.py",
}

SECURITY = "Security limitations"


def _sections(text: str) -> dict[str, str]:
    """Split the README into ``## `` level sections keyed by heading text."""
    parts = re.split(r"^## +(.+?)\s*$", text, flags=re.MULTILINE)
    return {parts[i].strip(): parts[i + 1] for i in range(1, len(parts) - 1, 2)}


def _code_blocks(body: str) -> list[str]:
    return re.findall(r"```[^\n]*\n(.*?)```", body, flags=re.DOTALL)


@pytest.fixture(scope="module")
def sections() -> dict[str, str]:
    assert README.is_file(), "README.md missing at workspace root"
    return _sections(README.read_text(encoding="utf-8"))


@pytest.mark.parametrize("heading", [*SECTION_COMMANDS, SECURITY])
def test_section_exists(sections, heading):
    assert heading in sections, f"README lacks '## {heading}'"


@pytest.mark.parametrize("heading,command", list(SECTION_COMMANDS.items()))
def test_section_gives_command_in_code_block(sections, heading, command):
    blocks = _code_blocks(sections[heading])
    lines = [line.strip() for block in blocks for line in block.splitlines()]
    assert command in lines, f"'## {heading}' has no code block with: {command}"


def test_powershell_code_blocks_are_tagged(sections):
    for heading in SECTION_COMMANDS:
        assert "```powershell" in sections[heading], heading


@pytest.mark.parametrize(
    "pattern",
    [
        r"no authentication",
        r"no access control",
        r"no audit logging",
        r"binds only to 127\.0\.0\.1",
        r"Query_Log\b.*\bnot tamper-evident",
        r"access control and a tamper-evident audit log are planned after the MVP",
    ],
)
def test_security_section_states_limitation(sections, pattern):
    body = sections[SECURITY]
    assert re.search(pattern, body, flags=re.IGNORECASE), (
        f"security section lacks statement matching: {pattern}"
    )
