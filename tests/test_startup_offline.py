"""Importing ``nab_sentry`` forces offline mode before any model library loads.

Runs in a fresh subprocess so the parent test process (which may already have
imported torch/open_clip or nab_sentry) cannot mask the result.

**Validates: Requirements 13.3**
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]

_CHILD = r"""
import json, os, sys
import nab_sentry
print(json.dumps({
    "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
    "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
    "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE"),
    "HF_HOME": os.environ.get("HF_HOME"),
    "TORCH_HOME": os.environ.get("TORCH_HOME"),
    "loaded": sorted(m for m in ("torch", "open_clip", "huggingface_hub", "transformers")
                     if m in sys.modules),
}))
"""


def test_import_sets_offline_env_before_model_libraries():
    env = dict(os.environ)
    # Inherited values that would re-enable network access must be overwritten.
    env["HF_HUB_OFFLINE"] = "0"
    env["TRANSFORMERS_OFFLINE"] = "0"
    env["HF_DATASETS_OFFLINE"] = "0"
    env["HF_HOME"] = str(WORKSPACE_ROOT / "elsewhere")

    proc = subprocess.run(
        [sys.executable, "-c", _CHILD],
        cwd=str(WORKSPACE_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])

    assert result["HF_HUB_OFFLINE"] == "1"
    assert result["TRANSFORMERS_OFFLINE"] == "1"
    assert result["HF_DATASETS_OFFLINE"] == "1"

    models = (WORKSPACE_ROOT / "models").resolve()
    assert Path(result["HF_HOME"]).resolve().is_relative_to(models)
    assert Path(result["TORCH_HOME"]).resolve().is_relative_to(models)

    assert "torch" not in result["loaded"]
    assert "open_clip" not in result["loaded"]
    assert result["loaded"] == []
