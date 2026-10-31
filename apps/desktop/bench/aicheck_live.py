"""Live AI check against Gemini on a processed case (task 3.3; needs a key in Settings or GEMINI_API_KEY).

Usage::

    venv-desktop\\Scripts\\python.exe apps\\desktop\\bench\\aicheck_live.py CASE_FOLDER "query" [--top 10] [--model M]

Searches the case, sends the top moments to Gemini, and prints each moment's local rank, the AI's
verdict and description, whether the moment is inside the query's labelled window (when the query
is in eval/queries_compilation.yaml), the tokens used and the cost.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "apps" / "desktop"))

from nab_find.aicheck import GeminiClient, Job, estimate, run_job  # noqa: E402
from nab_find.embed import OnnxClip  # noqa: E402
from nab_find.search import Searcher  # noqa: E402
from nab_find.settings import Settings  # noqa: E402
from nab_find.store import CaseStore  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("case", type=Path)
    p.add_argument("query")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--model", default=None)
    args = p.parse_args(argv)
    settings = Settings(_ROOT / "data" / "desktop")
    key = settings.gemini_key()
    if not key:
        print("no Gemini key: add one in Settings or set GEMINI_API_KEY")
        return 2
    model = args.model or settings.get("gemini_model")
    store = CaseStore(args.case)
    moments = Searcher(store, OnnxClip(_ROOT / "models" / "desktop")).search(args.query, limit=args.top)
    window = None
    for q in yaml.safe_load((_ROOT / "eval" / "queries_compilation.yaml").read_text(encoding="utf-8"))["queries"]:
        if q["text"] == args.query:
            window = (datetime.fromisoformat(q["start"]).timestamp(), datetime.fromisoformat(q["end"]).timestamp())
    print(f"model {model}; estimate {estimate(moments, model)}")
    t0 = time.perf_counter()
    job = run_job(Job(args.query, moments, model), store, GeminiClient(key, model))
    wall = time.perf_counter() - t0
    local_rank = {id(m): i + 1 for i, m in enumerate(moments)}
    d = job.to_dict()
    for item in d["items"]:
        m, c = item["moment"], item["check"]
        rank = next(i + 1 for i, mm in enumerate(moments) if mm.best_ts == m["best_ts"] and mm.video_id == m["video_id"])
        inside = "" if window is None else ("RIGHT" if m["start_ts"] <= window[1] and m["end_ts"] >= window[0] else "wrong")
        verdict = f"{c['match']:>6} {c['confidence']:.2f}  {c['description']}" if c else "(no answer)"
        print(f"local #{rank:<2} {inside:5} {verdict}")
    print(f"errors: {job.errors or 'none'}; tokens in {job.usage.tokens_in}, out {job.usage.tokens_out}; "
          f"cost ${d['usd']}; {wall:.1f} s")
    del local_rank
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
