"""Regression: building the real encoder must not add root log handlers (every nab_sentry log
line used to be printed twice once the OpenCLIP tokenizer had configured the root logger)."""

from __future__ import annotations

import logging

import pytest


@pytest.mark.slow
def test_real_encoder_leaves_root_logger_alone():
    from nab_sentry.config import Config
    from nab_sentry.model_files import default_encoder_factory
    from nab_sentry.startup import verify_manifest

    cfg = Config()
    manifest = verify_manifest(cfg.models_dir)
    if manifest.failures:
        pytest.skip("models not provisioned")
    before = list(logging.root.handlers), logging.root.level
    enc = default_encoder_factory(cfg, manifest.files)
    enc.encode_text("a person")
    assert (list(logging.root.handlers), logging.root.level) == before
