"""Tests for nab_sentry.config."""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pytest

from nab_sentry.config import Config, ConfigError, load_config, parse_set_args


def test_defaults_are_valid() -> None:
    cfg = Config()
    assert cfg.validate() == []
    cfg.require_valid()


def test_derived_paths(tmp_path: Path) -> None:
    cfg = Config(root=tmp_path)
    assert cfg.data_dir == tmp_path / "data"
    assert cfg.models_dir == tmp_path / "models"
    assert cfg.db_path == tmp_path / "data" / "nab_sentry.db"
    assert cfg.index_path == tmp_path / "data" / "vectors.faiss"
    assert cfg.thumbs_dir == tmp_path / "data" / "thumbs"
    assert cfg.playback_dir == tmp_path / "data" / "playback"
    assert cfg.logs_dir == tmp_path / "data" / "logs"
    assert cfg.videos_dir == tmp_path / "data" / "videos"
    assert cfg.query_log_path == tmp_path / "data" / "logs" / "query_log.jsonl"


def test_replace_recomputes_derived_paths(tmp_path: Path) -> None:
    cfg = dataclasses.replace(Config(), data_dir=tmp_path / "d")
    assert cfg.db_path == tmp_path / "d" / "nab_sentry.db"


def test_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        Config().sample_rate = 2.0  # type: ignore[misc]


def test_asdict_contains_every_16_5_parameter() -> None:
    d = dataclasses.asdict(Config())
    for name in ("sample_rate", "motion_threshold", "keyframe_interval_s", "gate_width",
                 "det_conf", "det_max_per_frame", "batch_size", "top_k", "merge_gap_s",
                 "event_padding_s", "label_boost"):
        assert name in d


@pytest.mark.parametrize(
    ("name", "value", "reason"),
    [
        ("sample_rate", 0.05, "out of range"),
        ("sample_rate", 31, "out of range"),
        ("sample_rate", math.nan, "NaN"),
        ("sample_rate", "fast", "non-numeric"),
        ("sample_rate", None, "missing"),
        ("motion_threshold", 1.5, "out of range"),
        ("keyframe_interval_s", 0.5, "out of range"),
        ("gate_width", 63, "out of range"),
        ("gate_width", 320.5, "not an integer"),
        ("det_conf", -0.1, "out of range"),
        ("det_max_per_frame", 101, "out of range"),
        ("batch_size", 0, "out of range"),
        ("batch_size", True, "non-numeric"),
        ("merge_gap_s", math.inf, "not finite"),
        ("motion_method", "knn", "not an allowed value"),
    ],
)
def test_single_invalid_field(name: str, value: object, reason: str) -> None:
    cfg = Config(**{name: value})
    issues = cfg.validate()
    assert [i.name for i in issues] == [name]
    assert issues[0].reason == reason


def test_require_valid_names_each_parameter_and_range() -> None:
    cfg = Config(sample_rate=50, det_conf=2.0)
    with pytest.raises(ConfigError) as exc:
        cfg.require_valid()
    msg = str(exc.value)
    assert "sample_rate=50" in msg and "0.1 to 30" in msg
    assert "det_conf=2.0" in msg and "0.0 to 1.0" in msg
    assert {i.name for i in exc.value.issues} == {"sample_rate", "det_conf"}


def test_load_config_parses_overrides(tmp_path: Path) -> None:
    cfg = load_config({"sample_rate": "2.5", "batch_size": "16", "enable_detector": "false",
                       "motion_method": "mog2", "root": str(tmp_path)})
    assert cfg.sample_rate == 2.5
    assert cfg.batch_size == 16
    assert cfg.enable_detector is False
    assert cfg.motion_method == "mog2"
    assert cfg.db_path == tmp_path / "data" / "nab_sentry.db"
    assert cfg.validate() == []


def test_load_config_keeps_bad_values_for_validate() -> None:
    cfg = load_config({"sample_rate": "abc", "det_conf": "nan", "batch_size": "", "gate_width": "3.5"})
    reasons = {i.name: i.reason for i in cfg.validate()}
    assert reasons == {"sample_rate": "non-numeric", "det_conf": "NaN",
                       "batch_size": "missing", "gate_width": "not an integer"}


def test_load_config_unknown_name() -> None:
    with pytest.raises(ConfigError, match="no_such"):
        load_config({"no_such": "1"})


def test_parse_set_args() -> None:
    assert parse_set_args(["sample_rate=2", "top_k=50"]) == {"sample_rate": "2", "top_k": "50"}
    with pytest.raises(ConfigError):
        parse_set_args(["sample_rate"])


# ------------------------------------------------------------- Property 11 (validate half)

from hypothesis import given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

# name -> (lo, hi, integer) for the parameters named in Property 11.
_P11_RANGES: dict[str, tuple[float, float, bool]] = {
    "sample_rate": (0.1, 30, False),
    "motion_threshold": (0.0, 1.0, False),
    "keyframe_interval_s": (1, 600, False),
    "gate_width": (64, 1920, True),
    "det_conf": (0.0, 1.0, False),
    "det_max_per_frame": (1, 100, True),
    "batch_size": (1, 64, True),
}


def _valid_value(name: str) -> st.SearchStrategy[object]:
    lo, hi, integer = _P11_RANGES[name]
    if integer:
        return st.integers(min_value=int(lo), max_value=int(hi))
    return st.floats(min_value=lo, max_value=hi, allow_nan=False, allow_infinity=False)


def _invalid_value(name: str) -> st.SearchStrategy[object]:
    lo, hi, integer = _P11_RANGES[name]
    if integer:
        below = st.integers(max_value=int(lo) - 1)
        above = st.integers(min_value=int(hi) + 1)
    else:
        below = st.floats(max_value=lo, exclude_max=True, allow_nan=False, allow_infinity=False)
        above = st.floats(min_value=hi, exclude_min=True, allow_nan=False, allow_infinity=False)
    non_numeric = st.text(max_size=12)  # any str is non-numeric to validate(), even "1.0"
    return st.one_of(below, above, st.just(math.nan), non_numeric)


@st.composite
def _configs_with_bad_subset(draw: st.DrawFn) -> tuple[Config, set[str]]:
    names = sorted(_P11_RANGES)
    bad = draw(st.sets(st.sampled_from(names)))
    kwargs = {
        name: draw(_invalid_value(name) if name in bad else _valid_value(name)) for name in names
    }
    return Config(**kwargs), bad


@given(_configs_with_bad_subset())
def test_property_11_validate_names_exactly_the_bad_subset(case: tuple[Config, set[str]]) -> None:
    """Feature: nab-sentry, Property 11: Config validation names every out-of-range parameter.

    **Validates: Requirements 2.8, 3.8, 4.9**
    """
    cfg, bad = case
    issues = cfg.validate()
    names = [i.name for i in issues]
    assert sorted(names) == sorted(bad)  # exactly the subset, each named once
    for issue in issues:
        assert issue.allowed  # every issue carries its allowed range
        assert issue.value is getattr(cfg, issue.name) or (
            isinstance(issue.value, float) and math.isnan(issue.value)
        )
    if bad:
        with pytest.raises(ConfigError) as exc:
            cfg.require_valid()
        assert {i.name for i in exc.value.issues} == bad
        for name in bad:
            assert name in str(exc.value)
    else:
        cfg.require_valid()
