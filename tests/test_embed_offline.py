"""Offline loading of the OpenCLIP encoder (Requirements 5.7, 5.8, 13.6).

Every test runs under the ``network_guard`` fixture (tests/conftest.py), which raises on
and records any connect/resolve to a non-loopback address.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import numpy as np
import pytest

from nab_sentry.config import Config
from nab_sentry.embed.clip_encoder import EMBED_DIM, OpenClipEncoder
from nab_sentry.errors import ModelMissingError
from nab_sentry.startup import FETCH_HINT
from tests.conftest import NetworkAttempt, NetworkGuard

REAL_WEIGHTS = (
    Config().models_dir / "open_clip" / "ViT-B-32-laion2b_s34b_b79k" / "open_clip_pytorch_model.bin"
)


def _assert_model_missing(exc: ModelMissingError, path: Path) -> None:
    msg = str(exc)
    assert str(path.resolve()) in msg
    assert FETCH_HINT in msg


# ------------------------------------------------------------------ guard self-check


def test_network_guard_blocks_non_loopback_and_allows_loopback(network_guard: NetworkGuard) -> None:
    # TEST-NET-3 address: must be blocked before any packet is sent.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        with pytest.raises(NetworkAttempt):
            s.connect(("203.0.113.1", 443))
    with pytest.raises(NetworkAttempt):
        socket.create_connection(("example.com", 443), timeout=1)
    assert [api for api, _ in network_guard.attempts] == ["socket.connect", "socket.create_connection"]

    # Loopback keeps working.
    network_guard.attempts.clear()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        with socket.create_connection(srv.getsockname(), timeout=2):
            pass
    network_guard.assert_no_attempts()


# ------------------------------------------------------------------ missing weights (fast)


@pytest.mark.parametrize("kind", ["missing_file", "directory"])
def test_missing_weights_raise_without_network_or_model_imports(
    kind: str, tmp_path: Path, network_guard: NetworkGuard, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "models" / "open_clip" / "open_clip_pytorch_model.bin"
    if kind == "directory":
        path.mkdir(parents=True)  # present but not a readable file
    # Any import of torch/open_clip would now raise ImportError, so getting
    # ModelMissingError proves the check runs before the model stack is touched (5.8).
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "open_clip", None)

    with pytest.raises(ModelMissingError) as info:
        OpenClipEncoder(path, batch_size=8)

    _assert_model_missing(info.value, path)
    network_guard.assert_no_attempts()


# ------------------------------------------------------------------ slow: real model stack


@pytest.mark.slow
def test_corrupt_weights_raise_model_missing_without_network(
    tmp_path: Path, network_guard: NetworkGuard
) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("open_clip")
    path = tmp_path / "open_clip_pytorch_model.bin"
    path.write_bytes(b"not a checkpoint \x00\xff" * 64)

    with pytest.raises(ModelMissingError) as info:
        OpenClipEncoder(path, batch_size=4)

    _assert_model_missing(info.value, path)
    assert "unreadable" in str(info.value)
    network_guard.assert_no_attempts()


@pytest.mark.slow
def test_real_encoder_loads_and_encodes_offline(
    network_guard: NetworkGuard, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not REAL_WEIGHTS.is_file():
        pytest.skip(f"OpenCLIP weights not present at {REAL_WEIGHTS}; run {FETCH_HINT}")
    pytest.importorskip("torch")
    pytest.importorskip("open_clip")
    # Mirror production (startup.enable_offline_mode) without leaking env changes.
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.setenv(var, "1")

    enc = OpenClipEncoder(REAL_WEIGHTS, batch_size=2, threads=2)
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 256, size=(120, 160, 3), dtype=np.uint8) for _ in range(3)]
    img_vecs = enc.encode_images(images)
    txt_vec = enc.encode_text("person in blue jacket")

    assert img_vecs.shape == (3, EMBED_DIM) and img_vecs.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(img_vecs, axis=1), 1.0, atol=1e-4)
    assert txt_vec.shape == (EMBED_DIM,) and txt_vec.dtype == np.float32
    assert abs(float(np.linalg.norm(txt_vec)) - 1.0) < 1e-4
    network_guard.assert_no_attempts()
