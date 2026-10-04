"""Central configuration for NAB Sentry.

Every tunable parameter (Requirement 16.5) is defined exactly once on :class:`Config`.
Ingest, search, the detector, the API and the evaluator all read from a ``Config``
instance; scripts accept ``--set name=value`` overrides that go through
:func:`load_config`.

``Config`` deliberately does not type-check on construction: an invalid value
(out of range, NaN, a non-numeric string, or ``None`` for "missing") is stored as
given, and :meth:`Config.validate` reports one :class:`ConfigIssue` per bad field.
:meth:`Config.require_valid` turns those issues into a :class:`ConfigError`, which
ingest (Requirements 2.8, 3.8), the Detector constructor (4.9) and API startup call
before doing any work.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ConfigIssue:
    """One invalid Config parameter: its name, the offending value, and the allowed range."""

    name: str
    value: Any
    allowed: str
    reason: str = "out of range"

    def __str__(self) -> str:
        return f"{self.name}={self.value!r} ({self.reason}; allowed: {self.allowed})"


class ConfigError(ValueError):
    """Raised when a Config has one or more invalid parameters."""

    def __init__(self, issues: list[ConfigIssue]) -> None:
        self.issues = list(issues)
        lines = "\n".join(f"  - {issue}" for issue in self.issues)
        super().__init__(f"Invalid configuration ({len(self.issues)} issue(s)):\n{lines}")


@dataclass(frozen=True)
class Config:
    # paths (all under the workspace); data_dir/models_dir default to root/data, root/models
    root: Path = _WORKSPACE_ROOT
    data_dir: Path | None = None
    models_dir: Path | None = None
    # server
    host: str = "127.0.0.1"
    port: int = 8765
    # sampling / gating
    sample_rate: float = 1.0  # 0.1..30
    motion_method: str = "diff"  # "diff" | "mog2"
    motion_threshold: float = 0.02  # 0.0..1.0
    motion_pixel_delta: int = 25  # 1..255, grey-level change counted as "changed"
    keyframe_interval_s: float = 10.0  # 1..600
    gate_width: int = 320  # 64..1920
    # detection
    enable_detector: bool = True
    det_conf: float = 0.35  # 0.0..1.0
    det_iou: float = 0.45  # 0.0..1.0
    det_max_per_frame: int = 10  # 1..100
    det_input_size: int = 640
    # embedding
    batch_size: int = 8  # 1..64
    num_threads: int = 0  # 0 = physical cores
    # search
    top_k: int = 300  # 1..10000
    force_postfilter: bool = False  # test/diagnostic hook: skip IDSelectorBatch path
    merge_gap_s: float = 8.0  # >= 0
    event_padding_s: float = 3.0  # >= 0
    label_boost: float = 0.02  # >= 0
    default_limit: int = 20
    max_limit: int = 100
    max_query_len: int = 256
    # media
    thumb_width: int = 320
    playback_max_width: int = 1280

    # derived paths (computed in __post_init__, not constructor arguments)
    db_path: Path = field(init=False)
    index_path: Path = field(init=False)
    thumbs_dir: Path = field(init=False)
    playback_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    videos_dir: Path = field(init=False)
    query_log_path: Path = field(init=False)

    def __post_init__(self) -> None:
        set_ = object.__setattr__  # frozen dataclass
        root = Path(self.root)
        set_(self, "root", root)
        data = Path(self.data_dir) if self.data_dir is not None else root / "data"
        models = Path(self.models_dir) if self.models_dir is not None else root / "models"
        set_(self, "data_dir", data)
        set_(self, "models_dir", models)
        set_(self, "db_path", data / "nab_sentry.db")
        set_(self, "index_path", data / "vectors.faiss")
        set_(self, "thumbs_dir", data / "thumbs")
        set_(self, "playback_dir", data / "playback")
        set_(self, "logs_dir", data / "logs")
        set_(self, "videos_dir", data / "videos")
        set_(self, "query_log_path", data / "logs" / "query_log.jsonl")

    def validate(self) -> list[ConfigIssue]:
        """Return one ConfigIssue per out-of-range, NaN, non-numeric, or missing field."""
        issues: list[ConfigIssue] = []
        for name, check in _CHECKS.items():
            issue = check(name, getattr(self, name), self)
            if issue is not None:
                issues.append(issue)
        return issues

    def require_valid(self) -> None:
        """Raise ConfigError naming every invalid parameter and its allowed range."""
        issues = self.validate()
        if issues:
            raise ConfigError(issues)


# --------------------------------------------------------------------------- checks

_Check = Callable[[str, Any, Config], "ConfigIssue | None"]


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _num(lo: float | None, hi: float | None, *, integer: bool) -> _Check:
    """Numeric range check, inclusive on both ends; ``None`` means unbounded."""
    if lo is not None and hi is not None:
        allowed = f"{lo} to {hi} inclusive"
    elif lo is not None:
        allowed = f">= {lo}"
    else:
        allowed = f"<= {hi}"
    if integer:
        allowed = f"integer {allowed}"

    def check(name: str, v: Any, _cfg: Config) -> ConfigIssue | None:
        if v is None:
            return ConfigIssue(name, v, allowed, "missing")
        if not _is_number(v):
            return ConfigIssue(name, v, allowed, "non-numeric")
        if isinstance(v, float) and math.isnan(v):
            return ConfigIssue(name, v, allowed, "NaN")
        if integer and not isinstance(v, int):
            return ConfigIssue(name, v, allowed, "not an integer")
        if not math.isfinite(v):
            return ConfigIssue(name, v, allowed, "not finite")
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            return ConfigIssue(name, v, allowed, "out of range")
        return None

    return check


def _bool(name: str, v: Any, _cfg: Config) -> ConfigIssue | None:
    if v is None:
        return ConfigIssue(name, v, "true or false", "missing")
    if not isinstance(v, bool):
        return ConfigIssue(name, v, "true or false", "not a boolean")
    return None


def _choice(*options: str) -> _Check:
    allowed = " | ".join(repr(o) for o in options)

    def check(name: str, v: Any, _cfg: Config) -> ConfigIssue | None:
        if v is None:
            return ConfigIssue(name, v, allowed, "missing")
        if v not in options:
            return ConfigIssue(name, v, allowed, "not an allowed value")
        return None

    return check


def _nonempty_str(name: str, v: Any, _cfg: Config) -> ConfigIssue | None:
    if v is None:
        return ConfigIssue(name, v, "non-empty string", "missing")
    if not isinstance(v, str) or not v.strip():
        return ConfigIssue(name, v, "non-empty string", "invalid")
    return None


def _default_limit(name: str, v: Any, cfg: Config) -> ConfigIssue | None:
    hi = cfg.max_limit if _is_number(cfg.max_limit) and isinstance(cfg.max_limit, int) else 100
    return _num(1, max(1, min(hi, 100)), integer=True)(name, v, cfg)


_CHECKS: dict[str, _Check] = {
    "host": _nonempty_str,
    "port": _num(1, 65535, integer=True),
    "sample_rate": _num(0.1, 30, integer=False),
    "motion_method": _choice("diff", "mog2"),
    "motion_threshold": _num(0.0, 1.0, integer=False),
    "motion_pixel_delta": _num(1, 255, integer=True),
    "keyframe_interval_s": _num(1, 600, integer=False),
    "gate_width": _num(64, 1920, integer=True),
    "enable_detector": _bool,
    "det_conf": _num(0.0, 1.0, integer=False),
    "det_iou": _num(0.0, 1.0, integer=False),
    "det_max_per_frame": _num(1, 100, integer=True),
    "det_input_size": _num(32, 1920, integer=True),
    "batch_size": _num(1, 64, integer=True),
    "num_threads": _num(0, 256, integer=True),
    "top_k": _num(1, 10000, integer=True),
    "force_postfilter": _bool,
    "merge_gap_s": _num(0.0, None, integer=False),
    "event_padding_s": _num(0.0, None, integer=False),
    "label_boost": _num(0.0, None, integer=False),
    "max_limit": _num(1, 100, integer=True),
    "default_limit": _default_limit,
    "max_query_len": _num(1, 4096, integer=True),
    "thumb_width": _num(16, 1920, integer=True),
    "playback_max_width": _num(16, 7680, integer=True),
}

# ------------------------------------------------------------------- load_config

_PATH_FIELDS = {"root", "data_dir", "models_dir"}
_TRUE = {"true", "1", "yes", "on"}
_FALSE = {"false", "0", "no", "off"}


def _field_kinds() -> dict[str, str]:
    """Map each overridable field name to 'path', 'bool', 'int', 'float', or 'str'."""
    kinds: dict[str, str] = {}
    defaults = Config()
    for f in dataclasses.fields(Config):
        if not f.init:
            continue
        if f.name in _PATH_FIELDS:
            kinds[f.name] = "path"
            continue
        d = getattr(defaults, f.name)
        if isinstance(d, bool):
            kinds[f.name] = "bool"
        elif isinstance(d, int):
            kinds[f.name] = "int"
        elif isinstance(d, float):
            kinds[f.name] = "float"
        else:
            kinds[f.name] = "str"
    return kinds


def _parse(kind: str, raw: str) -> Any:
    """Parse an override string. Unparseable values are returned as-is so validate() reports them."""
    s = raw.strip()
    if s == "":
        return None  # treated as missing
    if kind == "path":
        return Path(s)
    if kind == "bool":
        low = s.lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        return s
    if kind == "int":
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)  # e.g. "8.5": validate() reports "not an integer"
            except ValueError:
                return s
    if kind == "float":
        try:
            return float(s)  # "nan" parses to NaN, which validate() reports
        except ValueError:
            return s
    return s


def load_config(overrides: Mapping[str, str] | None = None) -> Config:
    """Build a Config from defaults plus ``name=value`` string overrides.

    Values are parsed into the field's type; values that cannot be parsed are kept
    as strings so that :meth:`Config.validate` reports them. Unknown names raise
    :class:`ConfigError` immediately. The returned Config is not validated; callers
    call :meth:`Config.require_valid`.
    """
    if not overrides:
        return Config()
    kinds = _field_kinds()
    unknown = [
        ConfigIssue(name, value, f"one of: {', '.join(sorted(kinds))}", "unknown parameter")
        for name, value in overrides.items()
        if name not in kinds
    ]
    if unknown:
        raise ConfigError(unknown)
    kwargs = {name: _parse(kinds[name], value) for name, value in overrides.items()}
    return Config(**kwargs)


def parse_set_args(items: list[str] | None) -> dict[str, str]:
    """Turn ``["name=value", ...]`` (from ``--set``) into a mapping for :func:`load_config`."""
    out: dict[str, str] = {}
    for item in items or []:
        name, sep, value = item.partition("=")
        if not sep or not name.strip():
            raise ConfigError([ConfigIssue(item, item, "name=value", "malformed --set override")])
        out[name.strip()] = value
    return out
