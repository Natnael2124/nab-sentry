"""Pure parts of scripts/embed_eval.py (V2 Step 4)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("embed_eval_script", ROOT / "scripts" / "embed_eval.py")
ee = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ee
spec.loader.exec_module(ee)


def test_first_relevant_rank():
    assert ee.first_relevant_rank([False, False, True, True]) == 3
    assert ee.first_relevant_rank([True]) == 1
    assert ee.first_relevant_rank([False]) is None and ee.first_relevant_rank([]) is None


def test_summarize():
    s = ee.summarize([1, 2, None, 4], [0.2, 0.2, 0.0, 0.2])
    assert s["n"] == 4 and s["hit@1"] == 0.25 and s["hit@3"] == 0.5
    assert s["mrr"] == pytest.approx((1 + 0.5 + 0.25) / 4) and s["p@5"] == pytest.approx(0.15)
    assert ee.summarize([], [])["mrr"] == 0.0


@given(st.lists(st.one_of(st.none(), st.integers(1, 50)), min_size=1, max_size=30))
def test_summary_bounds(ranks):
    s = ee.summarize(ranks, [0.0] * len(ranks))
    assert 0 <= s["hit@1"] <= s["hit@3"] <= 1 and s["hit@1"] <= s["mrr"] <= 1


def test_l2n_rows_have_unit_length():
    x = ee.l2n(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(x[0], [0.6, 0.8]) and np.all(np.isfinite(x))


def test_render_lists_every_query_and_model():
    results = {"a": {"ranks": [1, None], "hit@1": 0.5, "hit@3": 0.5, "mrr": 0.5, "p@5": 0.1, "info": {"dim": 512}},
               "b": {"ranks": [2, 3], "hit@1": 0.0, "hit@3": 1.0, "mrr": 0.42, "p@5": 0.2, "info": {}}}
    text = ee.render(results, [{"text": "red car"}, {"text": "dog"}])
    assert "| a | 512 | 0.50 |" in text and "| red car | 1 | 2 |" in text and "| dog | - | 3 |" in text


def test_unknown_model_is_refused():
    with pytest.raises(SystemExit):
        ee.make_candidate("nope")
