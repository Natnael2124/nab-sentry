"""Camera_Config (V2 Requirement 6): Property 6 (round trip; one issue per invalid field) and
Property 7 (URL masking never leaks the password and is idempotent)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

from nab_sentry.live.camera_config import (
    CameraConfigError,
    CameraSpec,
    dump_cameras,
    load_cameras,
    mask_url,
    parse_cameras,
)
from tests.strategies import camera_ids

unit = st.floats(min_value=0, max_value=1, allow_nan=False)
points = st.tuples(unit, unit)
names = st.from_regex(r"[A-Za-z0-9_-]{1,12}", fullmatch=True)
hosts = st.from_regex(r"[a-z][a-z0-9]{0,10}(\.[a-z0-9]{1,8}){0,2}", fullmatch=True)
secrets = st.from_regex(r"[A-Za-z0-9!$%^*]{6,16}", fullmatch=True)
sources = st.one_of(
    st.builds(lambda h, pw: f"rtsp://admin:{pw}@{h}:554/stream1", hosts, secrets),
    st.builds(lambda h: f"http://{h}:8080/video", hosts),
    st.from_regex(r"[A-Za-z0-9 ()-]{1,30}", fullmatch=True).filter(str.strip).map(lambda d: f"dshow:{d}"),
)


@st.composite
def rois(draw):
    x1, x2 = sorted(draw(st.lists(unit, min_size=2, max_size=2, unique=True)))
    y1, y2 = sorted(draw(st.lists(unit, min_size=2, max_size=2, unique=True)))
    return (x1, y1, x2, y2)


@st.composite
def lines(draw):
    a = draw(points)
    b = draw(points.filter(lambda p: p != a))
    return (a, b)


specs = st.builds(
    CameraSpec,
    camera_id=camera_ids,
    label=st.from_regex(r"[A-Za-z0-9][A-Za-z0-9 ]{0,40}", fullmatch=True).map(str.strip),
    source=sources,
    segment_s=st.integers(5, 300),
    encoder=st.sampled_from(["auto", "copy", "qsv", "x264"]),
    roi=st.none() | rois(),
    zones=st.dictionaries(names, st.lists(points, min_size=3, max_size=6).map(tuple), max_size=3),
    lines=st.dictionaries(names, lines(), max_size=2),
)


@given(st.lists(specs, min_size=1, max_size=4, unique_by=lambda s: s.camera_id))
def test_round_trip(cams):
    assert parse_cameras(yaml.safe_load(dump_cameras(cams))) == cams


BREAKERS = {
    "camera_id": ("camera_id", "bad id!"),
    "label": ("label", ""),
    "source": ("source", "ftp://host/x"),
    "segment_s": ("segment_s", 2),
    "encoder": ("encoder", "h265"),
    "roi": ("roi", [0.5, 0.5, 0.2, 0.9]),
    "zones.z1": ("zones", {"z1": [[0, 0], [1, 1]]}),
    "lines.l1": ("lines", {"l1": [[0.5, 0.5], [0.5, 0.5]]}),
}


@given(st.lists(specs, min_size=1, max_size=3, unique_by=lambda s: s.camera_id), st.data(),
       st.sampled_from(sorted(BREAKERS)))
def test_one_invalid_field_gives_one_issue_naming_it(cams, data, which):
    raw = [c.to_dict() for c in cams]
    i = data.draw(st.integers(0, len(raw) - 1))
    key, bad = BREAKERS[which]
    raw[i][key] = bad
    with pytest.raises(CameraConfigError) as exc:
        parse_cameras({"cameras": raw})
    assert len(exc.value.issues) == 1
    assert exc.value.issues[0].startswith(f"cameras[{i}].{which}")


def test_duplicate_ids_and_empty_file():
    cam = {"camera_id": "CAM01", "label": "a", "source": "dshow:x"}
    with pytest.raises(CameraConfigError) as exc:
        parse_cameras({"cameras": [cam, dict(cam)]})
    assert exc.value.issues == ["cameras[1].camera_id: duplicate 'CAM01'"]
    with pytest.raises(CameraConfigError):
        parse_cameras({"cameras": []})


def test_resolved_encoder_and_kind():
    rtsp = CameraSpec("C1", "a", "rtsp://h/x")
    cam = CameraSpec("C2", "b", "dshow:Integrated Webcam")
    web = CameraSpec("C3", "c", "http://phone:8080/video", encoder="x264")
    assert (rtsp.kind, rtsp.resolved_encoder) == ("rtsp", "copy")
    assert (cam.kind, cam.resolved_encoder) == ("dshow", "qsv")
    assert (web.kind, web.resolved_encoder) == ("http", "x264")


@given(hosts, secrets, st.text(max_size=20).filter(lambda t: "://" not in t))
def test_mask_url_hides_password_and_is_idempotent(host, pw, noise):
    text = f"{noise} rtsp://admin:{pw}@{host}:554/live {noise}"
    masked = mask_url(text)
    assert f":{pw}@" not in masked
    assert f"rtsp://***@{host}:554/live" in masked
    assert mask_url(masked) == masked


def test_mask_url_leaves_urls_without_credentials():
    assert mask_url("see http://example.com/a?b=1") == "see http://example.com/a?b=1"


def test_example_file_loads():
    root = Path(__file__).resolve().parents[1]
    cams = load_cameras(root / "config" / "cameras.example.yaml")
    assert [c.camera_id for c in cams] == ["CAM01", "CAM02"]
