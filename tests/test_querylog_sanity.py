"""Sanity checks for nab_sentry.querylog (spec unit tests live in test_querylog.py)."""

from __future__ import annotations

import json
import threading
from datetime import datetime

from nab_sentry.querylog import QueryLog, QueryRecord


def _rec(i: int = 0, q: str = "white truck") -> QueryRecord:
    return QueryRecord(status=200, q=q, filters={"camera": None, "limit": 20},
                       result_count=i, latency_ms=1.5)


def test_record_line_has_fields_and_offset_timestamp(tmp_path):
    log = QueryLog(tmp_path / "logs" / "query_log.jsonl")
    assert log.append(_rec(3, q="line\nbreak ünïcode")) is True
    lines = log.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert list(obj) == ["ts", "status", "q", "filters", "result_count", "latency_ms"]
    assert obj["q"] == "line\nbreak ünïcode" and obj["result_count"] == 3
    assert datetime.fromisoformat(obj["ts"]).utcoffset() is not None


def test_concurrent_appends_give_one_line_each(tmp_path):
    log = QueryLog(tmp_path / "query_log.jsonl")
    threads = [threading.Thread(target=lambda i=i: [log.append(_rec(i)) for _ in range(20)])
               for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lines = log.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 160
    assert all(json.loads(line)["status"] == 200 for line in lines)


def test_directory_as_path_returns_false(tmp_path):
    assert QueryLog(tmp_path).append(_rec()) is False
