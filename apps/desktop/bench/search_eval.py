"""Search quality on a processed case with the V2 evaluation queries (hit@1/3/5, MRR).

Usage::

    venv-desktop\\Scripts\\python.exe apps\\desktop\\bench\\search_eval.py CASE_FOLDER [--queries eval\\queries_compilation.yaml]

A result counts as right when its moment (padded clock span) overlaps the query's window, the rule
V2's scripts/evaluate.py uses. Only for cases built from the videos the queries were written for.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "apps" / "desktop"))

from nab_find.embed import OnnxClip  # noqa: E402
from nab_find.search import Searcher  # noqa: E402
from nab_find.store import CaseStore  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("case", type=Path)
    p.add_argument("--queries", type=Path, default=_ROOT / "eval" / "queries_compilation.yaml")
    args = p.parse_args(argv)
    qs = yaml.safe_load(args.queries.read_text(encoding="utf-8"))["queries"]
    store = CaseStore(args.case)
    s = Searcher(store, OnnxClip(_ROOT / "models" / "desktop"))
    ranks = []
    for q in qs:
        a = datetime.fromisoformat(q["start"]).timestamp()
        b = datetime.fromisoformat(q["end"]).timestamp()
        res = s.search(q["text"], limit=10)
        rank = next((i + 1 for i, m in enumerate(res) if m.start_ts <= b and m.end_ts >= a), None)
        ranks.append(rank)
        print(f"{rank or '-':>3}  {q['text']}")
    n = len(ranks)
    hit = lambda k: sum(1 for r in ranks if r and r <= k) / n  # noqa: E731
    mrr = sum(1 / r for r in ranks if r) / n
    print(f"hit@1 {hit(1):.2f}  hit@3 {hit(3):.2f}  hit@5 {hit(5):.2f}  MRR {mrr:.3f}  ({n} queries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
