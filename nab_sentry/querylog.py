"""Append-only Query_Log in JSON-lines format (Requirements 10.10, 10.14).

Each completed search request (HTTP 200 or 422) is written as exactly one JSON
object on its own line of ``data/logs/query_log.jsonl``::

    {"ts": "2025-01-01T08:05:00.123+01:00", "status": 200, "q": "white pickup truck",
     "filters": {"camera": "GATE1", "start": null, "end": null, "cls": "truck", "limit": 20},
     "result_count": 4, "latency_ms": 212.4}

The file is only ever opened in append mode, so earlier lines are never modified.
Write failures are logged as warnings and reported by ``append`` returning
``False``; they never raise, so the API keeps serving search responses (10.14).
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from nab_sentry.logging_setup import get_logger

_log = get_logger("querylog")


def now_iso() -> str:
    """Current local time, ISO 8601 with millisecond precision and UTC offset."""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class QueryRecord:
    """One Query_Log line.

    ``filters`` holds the parsed filter values for 200 responses and the raw
    parameter strings for 422 responses; ``result_count`` is 0 for 422.
    """

    status: int
    q: str | None
    filters: dict[str, Any]
    result_count: int
    latency_ms: float
    ts: str = field(default_factory=now_iso)

    def to_json(self) -> str:
        """Serialise as a single JSON line (no embedded newlines)."""
        d = asdict(self)
        ordered = {k: d[k] for k in ("ts", "status", "q", "filters", "result_count", "latency_ms")}
        # ensure_ascii=False keeps query text readable; json.dumps escapes any
        # newline characters inside strings, so the output is always one line.
        return json.dumps(ordered, ensure_ascii=False, default=str)


class QueryLog:
    """Thread-safe, append-only JSON-lines writer."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, record: QueryRecord) -> bool:
        """Append one line for ``record``. Returns ``False`` (and logs a warning) on failure."""
        try:
            line = record.to_json() + "\n"
        except (TypeError, ValueError) as exc:
            _log.warning("Query_Log record could not be serialised: %s", exc)
            return False
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a", encoding="utf-8", newline="\n") as fh:
                    fh.write(line)
                    fh.flush()
            except OSError as exc:
                _log.warning("Query_Log write to %s failed: %s", self.path, exc)
                return False
        return True
