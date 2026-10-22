"""Ground_Truth_File validation (V2 Requirement 2.3): every single invalid entry is reported
exactly once, by name; valid documents load unchanged."""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from nab_sentry.ground_truth import (
    CLASS_NAMES,
    GroundTruthError,
    load_ground_truth,
    parse_ground_truth,
)

offsets = st.floats(min_value=0, max_value=1000, allow_nan=False)


@st.composite
def incidents(draw):
    start = draw(offsets)
    end = start + draw(st.floats(min_value=0.5, max_value=100))
    classes = draw(st.lists(st.sampled_from(sorted(CLASS_NAMES)), min_size=1, max_size=3, unique=True))
    return {"start": start, "end": end, "classes": classes,
            "description": draw(st.text(max_size=40)), "zone": draw(st.none() | st.just("gate"))}


@st.composite
def documents(draw):
    return {"video": "CAM01_20260101T000000.mp4",
            "scene_cuts": draw(st.lists(offsets, max_size=10)),
            "incidents": draw(st.lists(incidents(), max_size=8))}


@given(documents())
def test_valid_documents_load(doc):
    gt = parse_ground_truth(doc)
    assert len(gt.incidents) == len(doc["incidents"])
    assert list(gt.scene_cuts) == sorted(doc["scene_cuts"])
    for inc, raw in zip(gt.incidents, doc["incidents"]):
        assert (inc.start, inc.end, list(inc.classes)) == (raw["start"], raw["end"], raw["classes"])


BREAKERS = {
    "end before start": (lambda i: {**i, "end": i["start"] - 0.1 if i["start"] > 0.1 else -1}, ""),
    "negative start": (lambda i: {**i, "start": -5.0}, ".start"),
    "unknown class": (lambda i: {**i, "classes": ["dog"]}, ".classes"),
    "empty classes": (lambda i: {**i, "classes": []}, ".classes"),
    "string start": (lambda i: {**i, "start": "ten"}, ".start"),
    "past duration": (lambda i: {**i, "end": 5000.0}, ".end"),
}


@given(documents().filter(lambda d: d["incidents"]), st.data(), st.sampled_from(sorted(BREAKERS)))
def test_one_bad_incident_gives_exactly_one_issue_naming_it(doc, data, kind):
    i = data.draw(st.integers(0, len(doc["incidents"]) - 1))
    doc = {**doc, "duration_s": 2000.0, "scene_cuts": [c for c in doc["scene_cuts"] if c <= 2000]}
    doc["incidents"] = [dict(x) for x in doc["incidents"]]
    breaker, _ = BREAKERS[kind]
    doc["incidents"][i] = breaker(doc["incidents"][i])
    with pytest.raises(GroundTruthError) as exc:
        parse_ground_truth(doc)
    assert len(exc.value.issues) == 1
    assert exc.value.issues[0].startswith(f"incidents[{i}]")


def test_bad_scene_cut_and_top_level():
    with pytest.raises(GroundTruthError) as exc:
        parse_ground_truth({"video": "v.mp4", "scene_cuts": [1.0, -2.0], "incidents": []})
    assert exc.value.issues == ["scene_cuts[1]: must be a number >= 0, got -2.0"]
    with pytest.raises(GroundTruthError):
        parse_ground_truth(["not", "a", "mapping"])
    with pytest.raises(GroundTruthError) as exc:
        parse_ground_truth({"incidents": []})
    assert exc.value.issues == ["video: must be a non-empty file name"]


def test_load_from_file(tmp_path: Path):
    f = tmp_path / "CAM01.yaml"
    f.write_text("video: CAM01_20260101T000000.mp4\nscene_cuts: [0, 12.5]\nincidents:\n"
                 "  - {start: 2, end: 9, classes: [person], description: man at gate, zone: gate}\n",
                 encoding="utf-8")
    gt = load_ground_truth(f)
    assert gt.scene_cuts == (0.0, 12.5)
    assert gt.incidents[0].contains(5) and not gt.incidents[0].contains(10)
    f.write_text("video: [unclosed", encoding="utf-8")
    with pytest.raises(GroundTruthError, match="unparseable"):
        load_ground_truth(f)
