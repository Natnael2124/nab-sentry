"""Real-footage track sanity (V2 task 11.5, slow).

Ingests the first 120 s of the Compilation_Video with the real detector, embedder and tracker,
then checks against the Integrator's Ground_Truth_File:

* no stored track spans a labelled scene cut. Cuts are labelled to about a second either way
  (thumbnails 1 s apart; the cut seen at "104.0" in the Compilation_Video really happens at
  about 104.1 s), so a track spans a cut at ``x`` only if it runs from before ``x - 1`` to
  after ``x + 1``;
* the number of person tracks is at most twice the number of labelled person incidents (a
  coarse fragmentation bound: one person split into many tracks fails it).

Skipped until ``eval/ground_truth/CAM-SEC01.yaml`` exists (task 2.4) and the models and the
video are present.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "data" / "videos" / "CAM-SEC01_20261007T230010.mp4"
LABELS = ROOT / "eval" / "ground_truth" / "CAM-SEC01.yaml"
SECONDS = 120

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def stored(tmp_path_factory):
    if not LABELS.is_file():
        pytest.skip(f"needs the Integrator's labels: {LABELS.relative_to(ROOT)} (task 2.4)")
    if not VIDEO.is_file():
        pytest.skip(f"needs {VIDEO.relative_to(ROOT)}")
    from nab_sentry.config import Config
    from nab_sentry.ingest.pipeline import IngestPipeline
    from nab_sentry.ingest.transcode import ffmpeg_exe
    from nab_sentry.model_files import default_detector_factory, default_encoder_factory
    from nab_sentry.startup import verify_manifest
    from nab_sentry.store.db import MetadataStore
    from nab_sentry.store.vector_index import VectorIndex
    from nab_sentry.track.live import LiveTracks

    models = ROOT / "models"
    manifest = verify_manifest(models)
    if manifest.failures:
        pytest.skip("models not provisioned")
    root = tmp_path_factory.mktemp("ws")
    cfg = Config(root=root, models_dir=models)
    cfg.videos_dir.mkdir(parents=True)
    clip = cfg.videos_dir / VIDEO.name
    subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", str(VIDEO), "-t",
                    str(SECONDS), "-c", "copy", str(clip)], check=True)
    db = MetadataStore(cfg.db_path)
    try:
        pipe = IngestPipeline(cfg, db, VectorIndex(), default_encoder_factory(cfg, manifest.files),
                              default_detector_factory(cfg, manifest.files), tracker=LiveTracks.from_config(cfg))
        [res] = pipe.ingest_paths([cfg.videos_dir]).results
        assert res.status == "ingested"
        start = db._conn.execute("SELECT start_epoch_ms FROM videos").fetchone()[0]
        tracks = [(cls, (f - start) / 1000, (l - start) / 1000) for cls, f, l in db._conn.execute(
            "SELECT det_class, first_epoch_ms, last_epoch_ms FROM tracks")]
    finally:
        db.close()
    from nab_sentry.ground_truth import load_ground_truth

    return tracks, load_ground_truth(LABELS)


def test_no_track_spans_a_labelled_scene_cut(stored):
    tracks, gt = stored
    cuts = [c for c in gt.scene_cuts if 0 < c < SECONDS]
    spanning = [(cls, a, b, x) for cls, a, b in tracks for x in cuts if a < x - 1.0 and b > x + 1.0]
    assert spanning == []


def test_person_tracks_not_badly_fragmented(stored):
    tracks, gt = stored
    incidents = [i for i in gt.incidents if "person" in i.classes and i.start < SECONDS]
    people = [t for t in tracks if t[0] == "person"]
    assert len(people) <= 2 * max(1, len(incidents)), (len(people), len(incidents))
