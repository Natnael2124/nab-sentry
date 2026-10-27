"""One embedder per store and scripts/reembed.py (V2 Step 4)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from nab_sentry.embed.compat import EmbedderMismatch, check_embedder
from nab_sentry.embed.reembed import MissingSources, reembed
from nab_sentry.ingest.pipeline import IngestPipeline
from nab_sentry.model_files import EMBEDDER_ID
from nab_sentry.search.engine import SearchEngine, SearchUnavailable
from nab_sentry.store.db import MetadataStore
from nab_sentry.store.vector_index import VectorIndex
from tests.fakes import FakeEncoder, FakeSource
from tests.test_pipeline_activities import FPS, H, W, ingest
from tests.test_pipeline_tracks import frames, name


class Named(FakeEncoder):
    """A FakeEncoder that says which embedder it is; ``salt`` gives a different vector space."""

    def __init__(self, model_id: str, salt: bytes = b"") -> None:
        super().__init__()
        self.model_id = model_id
        self.salt = salt

    def encode_images(self, images):
        vecs = np.asarray(super().encode_images(images), dtype=np.float32)
        if self.salt:
            vecs = np.roll(vecs, len(self.salt), axis=1)  # same norms, other directions
        return vecs


def stored(index: VectorIndex) -> dict[int, np.ndarray]:
    fi = index.faiss_index
    return {int(i): fi.reconstruct(int(i)) for i in index.ids()}


def setup(tmp_path: Path):
    videos = {name(0): frames(0, 25), name(30): frames(30, 20)}
    cfg, db, pipe = ingest(tmp_path, videos)
    pipe.index.save(cfg.index_path)
    opener = lambda p: FakeSource(videos[Path(p).name], fps=FPS, width=W, height=H, path=p)  # noqa: E731
    return cfg, db, pipe, opener


def test_new_store_records_the_configured_embedder(tmp_path: Path):
    db = MetadataStore(tmp_path / "x.db")
    try:
        assert db.embed_model() == EMBEDDER_ID
        check_embedder(db, Named("other/model"))  # empty store: takes the encoder's embedder
        assert db.embed_model() == "other/model"
        check_embedder(db, FakeEncoder())  # encoders without model_id are not checked
    finally:
        db.close()


def test_mismatch_blocks_ingest_and_search(tmp_path: Path):
    cfg, db, pipe, opener = setup(tmp_path)
    try:
        db.set_embed_model("old/model", 512)
        new = Named("new/model")
        with pytest.raises(EmbedderMismatch, match="reembed.py"):
            check_embedder(db, new)
        with pytest.raises(EmbedderMismatch):
            IngestPipeline(cfg, db, VectorIndex(), new, None, open_source=opener)
        with pytest.raises(SearchUnavailable, match="old/model"):
            SearchEngine(cfg, db, pipe.index, new).check_ready()
        assert db.embed_model() == "old/model"  # nothing adopted while vectors exist
    finally:
        db.close()


def test_reembed_with_the_same_encoder_reproduces_the_index(tmp_path: Path):
    """Decoding again at the stored frame index and box gives exactly the ingested images."""
    cfg, db, pipe, opener = setup(tmp_path)
    try:
        before = stored(pipe.index)
        rep = reembed(cfg, db, Named("same/model"), open_source=opener)
        after = stored(VectorIndex.load(cfg.index_path, 512))
        assert rep.vectors == len(before) == len(after) and rep.videos == 2
        assert all(np.allclose(before[i], after[i], atol=1e-6) for i in before)
        assert db.embed_model() == "same/model"
    finally:
        db.close()


def test_reembed_switches_space_and_unblocks_search(tmp_path: Path):
    cfg, db, pipe, opener = setup(tmp_path)
    try:
        before = stored(pipe.index)
        db.set_embed_model("old/model", 512)
        new = Named("new/model", salt=b"xyz")
        reembed(cfg, db, new, open_source=opener)
        index = VectorIndex.load(cfg.index_path, 512)
        after = stored(index)
        assert set(after) == set(before) and not np.allclose(before[min(before)], after[min(after)])
        check_embedder(db, new)
        SearchEngine(cfg, db, index, new).check_ready()
    finally:
        db.close()


def test_missing_source_refuses_then_drops(tmp_path: Path):
    cfg, db, pipe, opener = setup(tmp_path)
    try:
        (cfg.videos_dir / name(30)).unlink()
        index_bytes = cfg.index_path.read_bytes()
        n_before = db.vector_ids().size
        with pytest.raises(MissingSources, match="--drop-missing"):
            reembed(cfg, db, Named("new/model"), open_source=opener)
        assert cfg.index_path.read_bytes() == index_bytes and db.vector_ids().size == n_before
        assert db.embed_model() == EMBEDDER_ID
        rep = reembed(cfg, db, Named("new/model"), drop_missing=True, open_source=opener)
        assert rep.videos == 1 and len(rep.dropped_videos) == 1 and rep.dropped_vectors > 0
        index = VectorIndex.load(cfg.index_path, 512)
        assert np.array_equal(np.sort(index.ids()), np.sort(db.vector_ids()))
        assert rep.vectors == n_before - rep.dropped_vectors
    finally:
        db.close()


def test_failed_encoder_changes_nothing(tmp_path: Path):
    cfg, db, pipe, opener = setup(tmp_path)

    class Broken(Named):
        def encode_images(self, images):
            raise RuntimeError("out of memory")
    try:
        index_bytes = cfg.index_path.read_bytes()
        with pytest.raises(RuntimeError, match="out of memory"):
            reembed(cfg, db, Broken("new/model"), open_source=opener)
        assert cfg.index_path.read_bytes() == index_bytes and db.embed_model() == EMBEDDER_ID
    finally:
        db.close()


# -- scripts/reembed.py ----------------------------------------------------------------------------

_spec = importlib.util.spec_from_file_location("reembed_script", Path(__file__).resolve().parents[1] / "scripts" / "reembed.py")
cli = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cli
_spec.loader.exec_module(cli)


def test_cli_asks_first_and_reports_missing_sources(tmp_path: Path, monkeypatch, capsys):
    cfg, db, pipe, _ = setup(tmp_path)
    db.close()
    args = ["--set", f"root={tmp_path}"]
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    assert cli.main(args, encoder=Named("new/model")) == 1
    assert "not confirmed" in capsys.readouterr().out
    (cfg.videos_dir / name(30)).unlink()
    assert cli.main([*args, "--yes"], encoder=Named("new/model")) == 1
    assert "--drop-missing" in capsys.readouterr().err
    with MetadataStore(cfg.db_path) as db2:
        assert db2.embed_model() == EMBEDDER_ID  # nothing changed


def test_cli_nothing_to_do_on_the_same_embedder(tmp_path: Path, capsys):
    cfg, db, pipe, _ = setup(tmp_path)
    db.close()
    assert cli.main(["--set", f"root={tmp_path}"], encoder=Named(EMBEDDER_ID)) == 0
    assert "nothing to do" in capsys.readouterr().out


def test_engine_checks_the_embedder_after_a_placeholder_passed(tmp_path: Path):
    """The API checks ID sets with a placeholder before the real encoder is loaded."""
    cfg, db, pipe, _ = setup(tmp_path)
    try:
        db.set_embed_model("old/model", 512)
        engine = SearchEngine(cfg, db, pipe.index, object())  # placeholder: no model_id
        engine.check_ready()
        engine.encoder = Named("new/model")
        with pytest.raises(SearchUnavailable, match="reembed"):
            engine.check_ready()
    finally:
        db.close()


def test_api_search_answers_503_with_the_reembed_message(tmp_path: Path):
    from fastapi.testclient import TestClient

    from nab_sentry.api.app import Services, create_app
    from nab_sentry.querylog import QueryLog

    cfg, db, pipe, _ = setup(tmp_path)
    try:
        db.set_embed_model("old/model", 512)
        enc = Named("new/model")
        svc = Services(cfg, db, pipe.index, enc, False, SearchEngine(cfg, db, pipe.index, enc),
                       QueryLog(cfg.query_log_path))
        r = TestClient(create_app(svc)).get("/api/search", params={"q": "person"})
        assert r.status_code == 503 and "reembed.py" in r.text
    finally:
        db.close()
