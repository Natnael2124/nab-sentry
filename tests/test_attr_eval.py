"""Pure parts of scripts/attr_eval.py (V2 Step 4)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("attr_eval_script", ROOT / "scripts" / "attr_eval.py")
ae = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ae
spec.loader.exec_module(ae)


def test_matches_handles_equivalents_and_merged_tags():
    assert ae.matches("grey", "silver|grey") and not ae.matches("white", "silver|grey")
    assert ae.matches("SUV/van", "SUV") and ae.matches("SUV/van", "van|SUV") and not ae.matches("SUV/van", "sedan")


def test_tally_by_threshold():
    res = [("red", 0.9, "red"), ("blue", 0.7, "red"), ("red", 0.3, "red"), (None, 0.0, "red")]
    assert ae.tally(res, 0.0) == {"right": 2, "wrong": 1, "none": 1}
    assert ae.tally(res, 0.8) == {"right": 1, "wrong": 0, "none": 3}


def test_isolated_and_spaced():
    assert ae.isolated((0, 0, 10, 10), [(20, 20, 30, 30)]) and not ae.isolated((0, 0, 10, 10), [(5, 5, 30, 30)])
    assert ae.isolated((0, 0, 10, 10), [(10, 0, 20, 10)])  # touching edges do not overlap
    assert ae.spaced([(0.0, 1), (1.0, 2), (3.5, 3), (4.0, 4), (7.0, 5)], 3.0) == [1, 3, 5]
