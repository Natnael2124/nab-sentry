"""Fast sanity checks for ``serve()`` exit codes (task 14.13). Spec tests live in task 14.14."""

from __future__ import annotations

import hashlib
import json
import logging
import socket
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nab_sentry.api.app import serve
from nab_sentry.startup import SECURITY_WARNING
from nab_sentry.store.db import MetadataStore
from nab_sentry.store.vector_index import VectorIndex
from tests.fakes import FakeDetector, FakeEncoder


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(autouse=True)
def _restore_logging():
    """serve() calls setup_logging(); undo its level/handlers so later tests are unaffected."""
    logger = logging.getLogger("nab_sentry")
    level, handlers = logger.level, list(logger.handlers)
    yield
    for h in list(logger.handlers):
        if h not in handlers:
            logger.removeHandler(h)
            h.close()
    logger.setLevel(level)


@pytest.fixture()
def ws(tmp_path: Path) -> Path:
    models = tmp_path / "models"
    models.mkdir()
    stub = models / "stub.bin"
    stub.write_bytes(b"stub weights")
    digest = hashlib.sha256(stub.read_bytes()).hexdigest()
    (models / "manifest.json").write_text(
        json.dumps({"manifest_version": 1,
                    "files": [{"role": "embedder", "path": "stub.bin", "sha256": digest}]}),
        encoding="utf-8",
    )
    data = tmp_path / "data"
    MetadataStore(data / "nab_sentry.db").close()
    VectorIndex().save(data / "vectors.faiss")
    return tmp_path


class FakeRun:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def __call__(self, app: Any, **kw: Any) -> None:
        self.calls.append((app, kw))


def _serve(ws: Path, *extra: str, run: FakeRun | None = None, **kw: Any) -> int:
    argv = ["--set", f"root={ws}", "--set", f"port={_free_port()}"]
    for item in extra:
        argv += ["--set", item]
    kw.setdefault("encoder_factory", lambda cfg, files: FakeEncoder())
    kw.setdefault("detector_factory", lambda cfg, files: FakeDetector())
    return serve(argv, run=run or FakeRun(), **kw)


def test_success_serves_on_loopback(ws: Path, caplog: pytest.LogCaptureFixture) -> None:
    run = FakeRun()
    with caplog.at_level(logging.WARNING, logger="nab_sentry"):
        assert _serve(ws, run=run) == 0
    assert len(run.calls) == 1
    app, kw = run.calls[0]
    assert kw["host"] == "127.0.0.1"
    assert app.state.services.detector_loaded is True
    assert app.state.services.encoder is not None
    assert any(SECURITY_WARNING in r.getMessage() for r in caplog.records)


def test_invalid_config_exits_2(ws: Path) -> None:
    run = FakeRun()
    assert _serve(ws, "sample_rate=99", run=run) == 2
    assert serve(["--set", "no_such_param=1"], run=run) == 2
    assert run.calls == []


def test_non_loopback_host_exits_2(ws: Path) -> None:
    run = FakeRun()
    assert _serve(ws, "host=0.0.0.0", run=run) == 2
    assert run.calls == []


def test_port_in_use_exits_3(ws: Path) -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
        assert _serve(ws, f"port={port}") == 3


def test_missing_manifest_exits_4(ws: Path) -> None:
    (ws / "models" / "manifest.json").unlink()
    assert _serve(ws) == 4


@pytest.mark.parametrize("name", ["vectors.faiss", "nab_sentry.db"])
def test_missing_store_or_index_exits_5(ws: Path, name: str) -> None:
    (ws / "data" / name).unlink()
    assert _serve(ws) == 5
    assert not (ws / "data" / name).exists()  # nothing created silently


def test_inconsistent_index_exits_5(ws: Path) -> None:
    index = VectorIndex()
    index.add(np.array([7], dtype=np.int64), np.eye(1, 512, dtype=np.float32))
    index.save(ws / "data" / "vectors.faiss")
    assert _serve(ws) == 5


def test_model_load_failure_exits_6(ws: Path, caplog: pytest.LogCaptureFixture) -> None:
    def boom(cfg: Any, files: Any) -> Any:
        raise RuntimeError("corrupt weights")

    with caplog.at_level(logging.ERROR, logger="nab_sentry"):
        assert _serve(ws, encoder_factory=boom) == 6
    assert any("open_clip_pytorch_model.bin" in r.getMessage() for r in caplog.records)
