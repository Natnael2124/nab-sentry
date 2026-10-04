"""Shared pytest configuration: Hypothesis profiles and temporary workspace fixtures."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest
from hypothesis import settings

# `default`: the minimum of 100 examples per property; no deadline because CPU timing varies.
settings.register_profile("default", max_examples=100, deadline=None, derandomize=False)
# `thorough`: CI-style runs. Select with HYPOTHESIS_PROFILE=thorough or `--hypothesis-profile thorough`.
settings.register_profile("thorough", max_examples=500, deadline=None, derandomize=False)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))


@dataclass(frozen=True)
class Workspace:
    """A throwaway workspace root laid out like the real one (data/ and models/)."""

    root: Path
    data: Path
    models: Path
    thumbs: Path
    playback: Path
    logs: Path


def _make_workspace(root: Path) -> Workspace:
    data = root / "data"
    models = root / "models"
    ws = Workspace(
        root=root,
        data=data,
        models=models,
        thumbs=data / "thumbs",
        playback=data / "playback",
        logs=data / "logs",
    )
    for d in (ws.data, ws.models, ws.thumbs, ws.playback, ws.logs):
        d.mkdir(parents=True, exist_ok=True)
    return ws


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    """Fresh workspace under tmp_path with data/{thumbs,playback,logs} and models/."""
    return _make_workspace(tmp_path / "ws")


@pytest.fixture
def data_dir(workspace: Workspace) -> Path:
    return workspace.data


@pytest.fixture
def models_dir(workspace: Workspace) -> Path:
    return workspace.models
