"""Shared pytest configuration: Hypothesis profiles and temporary workspace fixtures."""

from __future__ import annotations

import ipaddress
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

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


# ------------------------------------------------------------------ network guard (13.6)


class NetworkAttempt(OSError):
    """Raised by the network guard when code tries to reach a non-loopback address."""


_LOOPBACK_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"}


def _is_loopback_host(host: object) -> bool:
    if host is None:
        return True  # getaddrinfo(None, ...) resolves to the local wildcard/loopback
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if not isinstance(host, str):
        return False
    h = host.strip().lower().strip("[]").split("%", 1)[0]
    if h in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def _is_loopback_address(address: object) -> bool:
    # AF_UNIX paths (str/bytes not shaped like a host) are local by definition.
    if isinstance(address, (str, bytes)):
        return True
    if isinstance(address, tuple) and address:
        return _is_loopback_host(address[0])
    return False


@dataclass
class NetworkGuard:
    """Records every blocked non-loopback attempt as ``(api, target)``."""

    attempts: list[tuple[str, object]]

    def assert_no_attempts(self) -> None:
        assert self.attempts == [], f"non-loopback network attempts: {self.attempts!r}"


@pytest.fixture
def network_guard(monkeypatch: pytest.MonkeyPatch) -> NetworkGuard:
    """Fail (and record) any socket connect / resolve to a non-loopback address.

    Patches ``socket.socket.connect``/``connect_ex`` (inherited by ``ssl.SSLSocket``),
    ``socket.create_connection`` and ``socket.getaddrinfo``. Loopback traffic is allowed,
    so local servers and test clients keep working. Tests should still call
    ``guard.assert_no_attempts()`` because callers may swallow the raised error.
    """
    guard = NetworkGuard(attempts=[])
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_create_connection = socket.create_connection
    real_getaddrinfo = socket.getaddrinfo

    def _block(api: str, target: object) -> NoReturn:
        guard.attempts.append((api, target))
        raise NetworkAttempt(f"network guard blocked {api} to {target!r}")

    def connect(self: socket.socket, address: object) -> None:
        if not _is_loopback_address(address):
            _block("socket.connect", address)
        return real_connect(self, address)

    def connect_ex(self: socket.socket, address: object) -> int:
        if not _is_loopback_address(address):
            _block("socket.connect_ex", address)
        return real_connect_ex(self, address)

    def create_connection(address: object, *args: object, **kwargs: object) -> socket.socket:
        if not _is_loopback_address(address):
            _block("socket.create_connection", address)
        return real_create_connection(address, *args, **kwargs)

    def getaddrinfo(host: object, *args: object, **kwargs: object) -> list:
        if not _is_loopback_host(host):
            _block("socket.getaddrinfo", host)
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "create_connection", create_connection)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    return guard
