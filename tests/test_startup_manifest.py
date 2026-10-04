"""Property 46: Manifest verification reports exactly the failing files.

**Validates: Requirements 13.4, 13.5**
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nab_sentry.startup import (
    EXIT_MODELS,
    FILE_MISSING,
    MANIFEST_MISSING,
    MANIFEST_NAME,
    SHA256_MISMATCH,
    require_models,
    verify_manifest,
)

KEEP, DELETE, MODIFY = "keep", "delete", "modify"

# Lowercase-only names (Windows is case-insensitive) with an "m_" prefix so no
# segment can be a reserved device name such as CON or NUL.
_segment = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789", min_size=1, max_size=8).map(
    lambda s: "m_" + s
)
_rel_path = st.builds(
    lambda dirs, name: "/".join([*dirs, name + ".bin"]),
    st.lists(_segment, max_size=2),
    _segment,
)


@st.composite
def model_sets(draw):
    """Unique model files, each with content and an action (keep / delete / modify)."""
    # File names end in ".bin" and directory segments never do, so no file path can
    # collide with a directory path.
    paths = draw(st.lists(_rel_path, min_size=1, max_size=6, unique=True))
    entries = []
    for i, rel in enumerate(paths):
        content = draw(st.binary(min_size=0, max_size=256))
        action = draw(st.sampled_from([KEEP, DELETE, MODIFY]))
        entries.append({"role": f"role{i}", "path": rel, "content": content, "action": action})
    return entries


def _modified(content: bytes) -> bytes:
    """Bytes guaranteed to differ from ``content`` (flip a byte, or append one if empty)."""
    if not content:
        return b"\x00"
    return bytes([content[0] ^ 0xFF]) + content[1:]


def _write_models(models: Path, entries, mapping_format: bool) -> None:
    for e in entries:
        target = models / e["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(e["content"])
    shas = {e["path"]: hashlib.sha256(e["content"]).hexdigest() for e in entries}
    if mapping_format:
        manifest = {"manifest_version": 1, "files": shas}
    else:
        manifest = {
            "manifest_version": 1,
            "files": [{"role": e["role"], "path": e["path"], "sha256": shas[e["path"]]} for e in entries],
        }
    (models / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")


def _apply_actions(models: Path, entries) -> None:
    for e in entries:
        target = models / e["path"]
        if e["action"] == DELETE:
            target.unlink()
        elif e["action"] == MODIFY:
            target.write_bytes(_modified(e["content"]))


@given(entries=model_sets(), mapping_format=st.booleans())
def test_manifest_reports_exactly_failing_files(entries, mapping_format):
    with tempfile.TemporaryDirectory() as tmp:
        models = Path(tmp) / "models"
        models.mkdir()
        _write_models(models, entries, mapping_format)
        _apply_actions(models, entries)

        result = verify_manifest(models)

        expected = {
            e["path"]: FILE_MISSING if e["action"] == DELETE else SHA256_MISMATCH
            for e in entries
            if e["action"] != KEEP
        }
        reported = {f.path: f.reason for f in result.failures}
        assert len(result.failures) == len(reported)  # one failure per failing file
        assert reported == expected

        kept = {e["path"] for e in entries if e["action"] == KEEP}
        assert set(result.files) == kept
        for rel in kept:
            assert result.files[rel] == (models / rel).resolve()
        if not mapping_format:
            assert set(result.roles) == {e["role"] for e in entries if e["action"] == KEEP}
        assert result.ok == (not expected)

        # Startup exits non-zero (4) whenever the failure list is non-empty.
        if expected:
            with pytest.raises(SystemExit) as exc:
                require_models(models)
            assert exc.value.code == EXIT_MODELS
        else:
            assert set(require_models(models)) == kept


@given(entries=model_sets())
def test_missing_manifest_is_reported(entries):
    with tempfile.TemporaryDirectory() as tmp:
        models = Path(tmp) / "models"
        models.mkdir()
        _write_models(models, entries, mapping_format=False)
        (models / MANIFEST_NAME).unlink()

        result = verify_manifest(models)

        assert [(f.path, f.reason) for f in result.failures] == [(MANIFEST_NAME, MANIFEST_MISSING)]
        assert not result.ok and result.files == {}
        with pytest.raises(SystemExit) as exc:
            require_models(models)
        assert exc.value.code == EXIT_MODELS


def test_missing_models_dir_reports_manifest_missing(tmp_path):
    result = verify_manifest(tmp_path / "does-not-exist")
    assert [(f.path, f.reason) for f in result.failures] == [(MANIFEST_NAME, MANIFEST_MISSING)]


def test_failure_message_names_file_and_fetch_hint(models_dir, capsys):
    entries = [{"role": "detector", "path": "m_yolo.bin", "content": b"abc", "action": DELETE}]
    _write_models(models_dir, entries, mapping_format=False)
    _apply_actions(models_dir, entries)
    with pytest.raises(SystemExit):
        require_models(models_dir)
    err = capsys.readouterr().err
    assert "m_yolo.bin: file missing" in err
    assert "fetch_models.py" in err
