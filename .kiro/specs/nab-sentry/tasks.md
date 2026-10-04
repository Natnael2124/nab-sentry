# Implementation Plan: NAB Sentry MVP

## Overview

The plan follows the design's 3-day phasing. Phase 0 sets up the venv, pinned dependencies, Config, offline startup checks, model provisioning, and a benchmark spike. Phase 1 builds the frame-only ingest path (metadata, FileSource, Sampler, Motion_Gate, SQLite + FAISS with rollback, OpenCLIP encoder, thumbnails/playback, synthetic video), then clustering, search, and a CLI that finds "red square" and "blue circle" in the synthetic video. Phase 2 adds the ONNX YOLO11n detector and crop vectors, filtered search, the FastAPI app with Range media serving, the Query_Log, and the security startup checks. Phase 3 adds the Console, the Evaluator, `run_demo.ps1`, and the README.

Language: Python 3.12.10 (as in the design). All commands use `venv\Scripts\python.exe` from the workspace root in PowerShell. Fast suite: `venv\Scripts\python.exe -m pytest -m "not slow"`. Slow suite (needs `models/`): `venv\Scripts\python.exe -m pytest -m slow`.

## Tasks

### Phase 0: Scaffold, dependencies, configuration, provisioning

- [x] 1. Set up project scaffold and dependencies
  - [x] 1.1 Create package layout, pinned dependencies, and test harness
    - Create `nab_sentry/` with `__init__.py` and sub-packages `ingest/`, `embed/`, `store/`, `search/`, `api/`, plus empty `web/`, `scripts/`, `eval/`, `tests/` directories, matching the design's package layout
    - Install into `venv/`: `venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu`, then `open_clip_torch onnxruntime opencv-python numpy faiss-cpu imageio-ffmpeg fastapi uvicorn pyyaml huggingface_hub ultralytics psutil pytest hypothesis httpx`; check that only one OpenCV package is installed and that the NumPy version is compatible with the installed faiss-cpu wheel
    - Write `requirements.txt` with exact `==` pins for every installed package (from `pip freeze`), with a header comment giving the PyTorch CPU index install command for `torch`
    - Add `models/` to `.gitignore`
    - Create `pytest.ini` registering the `slow` marker; create `tests/conftest.py` with Hypothesis profiles `default` (`max_examples=100`, `deadline=None`) and `thorough` (500 examples), plus `tmp_path` workspace fixtures that lay out `data/` and `models/` subdirectories
    - Create `tests/fakes.py` (empty module, filled in by later tasks) and `tests/strategies.py` with shared strategies: camera IDs, labels, naive/aware datetimes
    - _Requirements: 13.7, 16.5_

- [x] 2. Implement Config and startup checks
  - [x] 2.1 Implement `nab_sentry/config.py`
    - `Config` frozen dataclass with every field and default from the design, derived paths under `data/` and `models/`, `validate()` returning one `ConfigIssue` per out-of-range, NaN, non-numeric, or missing value, `require_valid()` raising `ConfigError` naming each parameter and its allowed range, and `load_config(overrides)` for `--set name=value` overrides
    - _Requirements: 2.8, 3.8, 4.9, 16.5_

  - [x] 2.2 Write property test for Config validation (Config part)
    - **Property 11: Config validation names every out-of-range parameter** (the `validate()` half; the pipeline half is task 10.6)
    - File: `tests/test_config.py`
    - **Validates: Requirements 2.8, 3.8, 4.9**

  - [x] 2.3 Implement `nab_sentry/startup.py`, `nab_sentry/logging_setup.py`, and offline mode in `nab_sentry/__init__.py`
    - `enable_offline_mode()` sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, `HF_HOME=models/.hf`, `TORCH_HOME=models/.torch`, and is called at the top of `nab_sentry/__init__.py` before any other import
    - `verify_manifest()` (SHA-256 in 1 MiB chunks; rejects absolute paths and `..`), `require_models()` (exit 4 with `FETCH_HINT`), `ensure_loopback()` (raises `StartupError` unless host is exactly `127.0.0.1`), `log_security_warning()`
    - Logging to console and `data/logs/nab_sentry.log`
    - _Requirements: 13.3, 13.4, 13.5, 13.7, 18.2, 18.5_

  - [x] 2.4 Write property test for manifest verification
    - **Property 46: Manifest verification reports exactly the failing files**
    - File: `tests/test_startup_manifest.py`
    - **Validates: Requirements 13.4, 13.5**

  - [x] 2.5 Write property test for loopback check (`ensure_loopback` part)
    - **Property 52: Only loopback binding is allowed** (the `serve()` half is task 14.14)
    - File: `tests/test_startup_loopback.py`
    - **Validates: Requirements 10.1, 18.1, 18.5**

  - [x] 2.6 Write unit test for offline environment
    - Run a subprocess that imports `nab_sentry` and checks `HF_HUB_OFFLINE == "1"` and that `torch`/`open_clip` are not yet in `sys.modules`
    - File: `tests/test_startup_offline.py`
    - _Requirements: 13.3_

- [x] 3. Implement model provisioning and benchmark spike
  - [x] 3.1 Implement `scripts/fetch_models.py`
    - Delete any existing `models/manifest.json` first; download `open_clip_pytorch_model.bin` with `huggingface_hub.hf_hub_download` into `models/open_clip/ViT-B-32-laion2b_s34b_b79k/` and `yolo11n.pt` via Ultralytics, each with 3 attempts and exponential backoff; export ONNX (`imgsz=640, dynamic=False, nms=False`) to `models/yolo11n.onnx`; smoke-load both with local paths; write `manifest.json.tmp` then rename to `manifest.json`
    - Any failure: message naming the file, exit 1, no manifest. Downloader and exporter are injectable so tests can stub them
    - Running this script needs internet once, on a connected machine (see checkpoint 4)
    - _Requirements: 13.1, 13.2, 13.8_

  - [x] 3.2 Write unit tests for the Model_Fetcher
    - Stubbed downloader failing 3 times leaves no manifest and names the file; stubbed export failure likewise; successful stub run writes a manifest with 64-hex SHA-256 values and forward-slash relative paths
    - File: `tests/test_fetch_models.py`
    - _Requirements: 13.2, 13.8_

  - [x] 3.3 Implement `scripts/benchmark.py` spike
    - Call `enable_offline_mode()` and `require_models()`; load OpenCLIP from the local checkpoint path and the ONNX model with `CPUExecutionProvider`; take an input video path; run 5 warm-up + ≥ 50 timed iterations and print median/p95 for decode, embedder ms/image at batch 1 and batch 8, and detector ms/frame
    - Stage failures print the stage and file and lead to exit 1 after reporting completed stages
    - Task 18.1 later switches this script to the real component classes and adds the projection and `--search`
    - _Requirements: 14.1, 14.8_

- [x] 4. Checkpoint: Phase 0
  - Run `venv\Scripts\python.exe -m pytest -m "not slow"` and make sure all tests pass, ask the user if questions arise.
  - Connected-machine step (manual, needs internet once): run `venv\Scripts\python.exe scripts\fetch_models.py` on a connected machine and copy `models\` to the Target_Hardware if they differ.
  - Manual verification: run `venv\Scripts\python.exe scripts\benchmark.py <sample video>` on the Target_Hardware and record the per-stage medians.

### Phase 1: Frame-only ingest, search, and CLI on synthetic video

- [x] 5. Implement metadata resolution and the video source
  - [x] 5.1 Implement `nab_sentry/ingest/metadata.py`
    - `CAMERA_ID_RE`, `FILENAME_RE`, `Sidecar`, `SidecarParse`, `ResolvedMetadata`, `format_filename` (manual zero-padded formatting), `parse_filename` (rejects invalid calendar values), `serialize_sidecar`, `parse_sidecar_text` (collects every invalid field name or `["unparseable"]`), `resolve_metadata` (sidecar first, then filename, else `None`, with the log messages from the design), `to_aware_local`, `src_hash` (SHA-256 over 1 MiB chunks)
    - _Requirements: 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.12, 7.1, 7.2_

  - [x] 5.2 Write property test for filename round trip
    - **Property 1: Filename round trip**
    - File: `tests/test_metadata_filename.py`
    - **Validates: Requirements 1.3, 1.7**

  - [x] 5.3 Write property test for invalid filename timestamps
    - **Property 2: Invalid filename timestamps are rejected**
    - File: `tests/test_metadata_filename_invalid.py`
    - **Validates: Requirements 1.3**

  - [x] 5.4 Write property test for sidecar round trip
    - **Property 3: Sidecar round trip**
    - File: `tests/test_metadata_sidecar_roundtrip.py`
    - **Validates: Requirements 1.8**

  - [x] 5.5 Write property test for sidecar validation
    - **Property 4: Sidecar validation reports exactly the invalid fields**
    - File: `tests/test_metadata_sidecar_validation.py`
    - **Validates: Requirements 1.2, 1.6**

  - [x] 5.6 Write property test for metadata source precedence
    - **Property 5: Metadata source precedence**
    - File: `tests/test_metadata_resolve.py`
    - **Validates: Requirements 1.2, 1.3, 1.4, 1.5, 1.6**

  - [x] 5.7 Write property test for content-addressed source hash
    - **Property 25: Source hash is content-addressed**
    - File: `tests/test_metadata_hash.py`
    - **Validates: Requirements 7.2**

  - [x] 5.8 Implement `nab_sentry/ingest/sources.py` and `FakeSource`
    - `DecodedFrame`, `VideoSource` protocol, `FileSource.open` (`cv2.CAP_FFMPEG`; `UnreadableVideo`, `UnknownFrameRate`), `frames()` with `grab()`/`retrieve()`, offsets `index / fps`, decode failures logged with path and offset and not yielded, `decoded_index_count`, `failed_frames`, `close()`
    - Add `FakeSource` (scripted frames, undecodable indices, fps, empty frames) to `tests/fakes.py`
    - _Requirements: 1.1, 1.11, 2.7, 2.9, 2.10_

  - [x] 5.9 Write unit tests for `FileSource`
    - Garbage-bytes file raises `UnreadableVideo`; a short OpenCV-written clip yields first offset 0.0 and non-decreasing offsets; zero fps raises `UnknownFrameRate` (via a stubbed capture)
    - File: `tests/test_sources.py`
    - _Requirements: 1.1, 1.11, 2.9, 2.10_

- [x] 6. Implement the Sampler and Motion_Gate
  - [x] 6.1 Implement `nab_sentry/ingest/sampler.py`
    - Pure `sample_indices` reference and streaming `Sampler.select` using the shared target-time rule with `EPS = 1e-9`
    - _Requirements: 2.1, 2.2, 2.4, 2.5, 2.6_

  - [x] 6.2 Write property test for the target-time rule
    - **Property 8: Sampler follows the target-time rule**
    - File: `tests/test_sampler_rule.py`
    - **Validates: Requirements 2.1**

  - [x] 6.3 Write property test for selected offsets
    - **Property 9: Selected offsets are exact, strictly increasing, and in range**
    - File: `tests/test_sampler_offsets.py`
    - **Validates: Requirements 1.1, 2.2, 2.4**

  - [x] 6.4 Write property test for sample count bounds
    - **Property 10: Sample count bounds**
    - File: `tests/test_sampler_counts.py`
    - **Validates: Requirements 2.5, 2.6, 2.7**

  - [x] 6.5 Implement `nab_sentry/ingest/motion.py`
    - `GateDecision`, `downscale_gray` (INTER_AREA, no upscale, aspect kept, 5×5 blur), `changed_fraction`, `MotionGate` with `reset()` and `evaluate()` in the design's decision order (empty → first → keyframe → motion → static), frame differencing default and `mog2` option
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.7, 3.9_

  - [x] 6.6 Write property test for changed fraction and downscaling
    - **Property 12: Changed fraction is bounded and downscaling never upscales**
    - File: `tests/test_motion_fraction.py`
    - **Validates: Requirements 3.1**

  - [x] 6.7 Write property test for the gate decision model
    - **Property 13: Motion gate matches the reference decision model**
    - File: `tests/test_motion_model.py`
    - **Validates: Requirements 3.2, 3.3, 3.4, 3.7, 3.9**

  - [x] 6.8 Write property test for keyframes on identical frames
    - **Property 14: Identical frames pass once per keyframe interval**
    - File: `tests/test_motion_keyframes.py`
    - **Validates: Requirements 3.5**

- [x] 7. Implement the Synthetic_Generator
  - [x] 7.1 Implement `nab_sentry/synthetic.py` and `scripts/make_synthetic.py`
    - Settings validation (duration, fps, object intervals) naming the invalid setting; pure `render_frame(i, spec)` with seed-derived blurred noise background and 100 px red square / blue circle moving 120 px/s with bounce; `ground_truth_document(spec)`
    - Writer encodes with bundled ffmpeg (`libx264 -qp 0 -preset ultrafast -pix_fmt yuv420p -threads 1`) into a temp dir inside the output dir, writes the MP4 named by the Filename_Convention, the Sidecar_File, and `ground_truth.json`, then renames all three; any failure (including unwritable output dir) deletes the temp dir and exits non-zero
    - Defaults: 90 s, 10 fps, 640×480, `CAM-SYN01`, `2025-01-01T08:00:00`, output `data/videos/`
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.8, 15.9_

  - [x] 7.2 Write property test for synthetic determinism
    - **Property 47: Synthetic generation is deterministic**
    - File: `tests/test_synthetic_determinism.py`
    - **Validates: Requirements 15.1, 15.4**

  - [x] 7.3 Write property test for invalid synthetic settings
    - **Property 48: Invalid synthetic settings are rejected without output**
    - File: `tests/test_synthetic_invalid.py`
    - **Validates: Requirements 15.8**

  - [x] 7.4 Write unit tests for synthetic output and gating
    - Default spec renders objects only inside [10, 25) and [50, 65); sidecar and `ground_truth.json` contents match; unwritable output dir leaves no files; running `Sampler` + `MotionGate` with default Config over `render_frame` output passes ≥ 1 frame per interval and outside intervals only the first frame, keyframes, and at most one frame within 1 s after each interval end
    - File: `tests/test_synthetic_examples.py`
    - _Requirements: 15.1, 15.2, 15.3, 15.5, 15.6, 15.9_

- [x] 8. Implement the Metadata_Store and Vector_Index
  - [x] 8.1 Implement `nab_sentry/store/db.py` and the `SearchFilter` type
    - DDL from the design (`schema_meta`, `cameras`, `videos`, `frames`, `vectors`, indexes), `PRAGMA foreign_keys=ON`, WAL, `busy_timeout`; `MetadataStore` methods from the design (`transaction`, `upsert_camera` returning `created`/`exists`/`label_conflict`, inserts, `finalize_video`, `delete_video`, `vector_ids`, `allowed_vector_ids` with the fixed parameterised template, `hit_rows` via temp table, `cameras`, `counts`, `playback_for`); `?`/named placeholders only
    - Create `nab_sentry/search/engine.py` containing only the `SearchFilter` dataclass with `is_empty()` (the engine is added in task 11.8)
    - _Requirements: 1.9, 1.10, 6.1, 6.2, 6.3, 6.9, 8.3, 8.8, 8.9, 8.13, 18.4_

  - [x] 8.2 Write property test for camera upserts
    - **Property 6: Camera rows are created once and never relabelled**
    - File: `tests/test_store_cameras.py`
    - **Validates: Requirements 1.9, 1.10**

  - [x] 8.3 Write property test for the Allowed_ID_Set
    - **Property 28: Allowed ID set equals the reference filter**
    - File: `tests/test_store_allowed_ids.py`
    - **Validates: Requirements 8.3, 8.8, 8.9, 8.13**

  - [x] 8.4 Write unit tests for the Metadata_Store
    - Insert referencing a missing parent row is rejected; `videos` row holds camera ID, source path, Src_Hash, start time, and duration at start; `delete_video` cascades and returns file names
    - File: `tests/test_store_examples.py`
    - _Requirements: 6.1, 6.9_

  - [x] 8.5 Implement `nab_sentry/store/vector_index.py`
    - `Hit`, `VectorIndex` over `IndexIDMap2(IndexFlatIP(512))` with `load` (type/dim check → `IndexUnavailable`), atomic `save` (tmp + fsync + `os.replace`), `ntotal`, `ids`, `add` (rejects duplicates and bad shapes), `remove` (`IDSelectorBatch`), `search`, `search_restricted` (`SearchParameters(sel=IDSelectorBatch)`), `search_postfilter`; drop `-1` padding; sort by `(-similarity, vector_id)`; `force_postfilter` hook
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.7, 8.10, 8.11_

  - [x] 8.6 Write property test for restricted vs post-filter search
    - **Property 30: Restricted search and post-filter fallback agree**
    - File: `tests/test_index_restricted.py`
    - **Validates: Requirements 8.4, 8.6**

  - [x] 8.7 Write property test for brute-force equivalence
    - **Property 31: Index search equals brute-force ranking**
    - File: `tests/test_index_bruteforce.py`
    - **Validates: Requirements 8.2, 8.7**

  - [x] 8.8 Write property test for save/load round trip
    - **Property 32: Index save/load round trip**
    - File: `tests/test_index_roundtrip.py`
    - **Validates: Requirements 8.11**

  - [x] 8.9 Write unit tests for the Vector_Index
    - Index type is `IndexIDMap2` wrapping `IndexFlatIP(512)`; monkeypatched restricted search raising falls back without error; missing or wrong-type file raises `IndexUnavailable`
    - File: `tests/test_index_examples.py`
    - _Requirements: 8.1, 8.4, 8.15_

- [x] 9. Implement the Embedder, thumbnails, and playback transcoding
  - [x] 9.1 Implement `nab_sentry/embed/clip_encoder.py` and `FakeEncoder`
    - `PROMPT_TEMPLATES`, `EMBED_DIM`, `Encoder` protocol, `batched`, `l2_normalize` (raises `EmbeddingError` on zero/non-finite), `OpenClipEncoder` loading `ViT-B-32` from the absolute local checkpoint path only (missing/unreadable → `ModelMissingError` with `FETCH_HINT`, no download), batched `encode_images`, `encode_text` with strip, `EmptyQueryError`, two templates, per-template normalise, mean, normalise
    - Add `FakeEncoder` (deterministic unit vectors from a hash of image bytes/text; optional colour mode mapping red/blue images and "red square"/"blue circle" text to fixed directions) to `tests/fakes.py`
    - _Requirements: 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9_

  - [x] 9.2 Write property test for batching
    - **Property 18: Batching shape**
    - File: `tests/test_embed_batched.py`
    - **Validates: Requirements 5.5**

  - [x] 9.3 Write property test for unit embeddings
    - **Property 19: Embeddings are finite unit vectors** (`l2_normalize` part fast; real-encoder part `@pytest.mark.slow`)
    - File: `tests/test_embed_unit_vectors.py`
    - **Validates: Requirements 5.3**

  - [x] 9.4 Write property test for prompt ensembling
    - **Property 20: Query encoding ensembles both prompt templates** (stub text tower and tokenizer spy)
    - File: `tests/test_embed_prompts.py`
    - **Validates: Requirements 5.4**

  - [x] 9.5 Write property test for blank queries (encoder part)
    - **Property 21: Blank queries are rejected** (the API half is covered in task 14.8)
    - File: `tests/test_embed_blank.py`
    - **Validates: Requirements 5.9, 10.7**

  - [x] 9.6 Write property test for batch/single agreement (slow)
    - **Property 22: Batched and single-image embeddings agree** (`@pytest.mark.slow`)
    - File: `tests/test_embed_batch_agreement.py`
    - **Validates: Requirements 5.6**

  - [x] 9.7 Write slow tests for offline encoder loading
    - Missing weights raise `ModelMissingError` without any network attempt; real encoder loads and encodes under a socket guard that fails on any non-loopback `connect` (`@pytest.mark.slow`)
    - File: `tests/test_embed_offline.py`
    - _Requirements: 5.7, 5.8, 13.6_

  - [x] 9.8 Implement `nab_sentry/ingest/transcode.py`
    - `ProbeInfo`, pure `parse_ffmpeg_probe`, `probe_video` (bundled ffmpeg via `imageio_ffmpeg.get_ffmpeg_exe()`), pure `playback_command` (remux for browser-safe H.264 yuv420p profiles, otherwise `libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -movflags +faststart -an` scaled to ≤ 1280 px), `PlaybackJob` (list argv, no shell, writes `.part`), `write_thumbnail` (320 px wide, aspect kept, JPEG q85, `ThumbnailError`)
    - _Requirements: 6.4, 6.5_

  - [x] 9.9 Write property test for thumbnail dimensions
    - **Property 23: Thumbnail dimensions**
    - File: `tests/test_transcode_thumbs.py`
    - **Validates: Requirements 6.4**

  - [x] 9.10 Write slow integration tests for playback files
    - Remux and transcode paths on generated clips; `parse_ffmpeg_probe` reports H.264; top-level MP4 box scan shows `moov` before `mdat`; duration within 0.5 s of source; unit tests for `parse_ffmpeg_probe` and `playback_command` on fixed stderr samples run in the fast suite
    - File: `tests/test_transcode_playback.py`
    - _Requirements: 6.5, 6.6_

- [x] 10. Implement the ingest pipeline with rollback and reconciliation
  - [x] 10.1 Implement the Detector protocol and box clamping in `nab_sentry/ingest/detector.py`
    - `TARGET_CLASSES`, `Detection`, `Detector` protocol, pure `clamp_box` (floor/ceil, clamp, `None` below 1 px); the ONNX implementation follows in task 13.1
    - Add `FakeDetector` (scripted detections per frame or raises) to `tests/fakes.py`
    - _Requirements: 4.3, 5.10_

  - [x] 10.2 Write property test for box clamping
    - **Property 16: Bounding boxes are clamped integer boxes inside the frame** (`clamp_box` part; stored-row bounds are asserted in task 10.5)
    - File: `tests/test_detector_clamp.py`
    - **Validates: Requirements 4.3, 5.10, 6.3**

  - [x] 10.3 Implement `nab_sentry/ingest/pipeline.py`
    - `VideoResult`, `IngestReport`, `ReconcileReport`, `IngestPipeline` with injectable `open_source` and `start_playback`; `ingest_paths` (config `require_valid()` first, sorted, de-duplicated by hash within the run); `ingest_video` steps 1–9 from the design (hash before writes, skip already indexed, resolve metadata, open source, one `BEGIN IMMEDIATE` transaction, camera upsert, `videos` row `status='ingesting'`, concurrent playback job, gate, `frames` row + thumbnail per passed frame, detector (optional; errors → warning, zero crops), batched embedding of frame + crops, staged vectors, finalize counts/duration/playback, FAISS add + atomic save, COMMIT)
    - `_rollback` in the design's order (kill ffmpeg → ROLLBACK → index remove + save if touched → delete recorded files → log path and reason); `KeyboardInterrupt` follows the same path
    - `reconcile()`: remove FAISS IDs not in `vectors`, delete non-complete videos, delete videos whose vectors are missing from FAISS and their files
    - Works with `detector=None` (`Config.enable_detector=False`) for Phase 1
    - Add `FakePlaybackJob` and `FailureInjector` (including "crash after index save") to `tests/fakes.py`
    - _Requirements: 1.5, 1.9, 1.10, 1.11, 2.3, 2.8, 2.9, 2.10, 3.6, 3.7, 3.8, 4.5, 4.7, 4.8, 5.1, 5.2, 5.10, 6.2, 6.3, 6.4, 6.5, 6.7, 6.8, 6.9, 6.10, 7.1, 7.3, 7.6, 7.7, 7.8, 8.10_

  - [x] 10.4 Write property test for absolute timestamps
    - **Property 7: Absolute timestamp equals start plus offset**
    - File: `tests/test_pipeline_timestamps.py`
    - **Validates: Requirements 1.12, 2.3, 6.2**

  - [x] 10.5 Write property test for pipeline record shape
    - **Property 17: Pipeline record shape per video** (also asserts every stored `crop` row lies within its video's width and height, the pipeline half of Property 16)
    - File: `tests/test_pipeline_records.py`
    - **Validates: Requirements 3.6, 3.7, 4.5, 4.7, 4.8, 5.1, 5.2, 6.2, 6.3**

  - [x] 10.6 Write property test for Config refusal in the pipeline
    - **Property 11: Config validation names every out-of-range parameter** (pipeline half: `ingest_paths` raises `ConfigError` before any row, file, or index entry)
    - File: `tests/test_pipeline_config.py`
    - **Validates: Requirements 2.8, 3.8, 4.9**

  - [x] 10.7 Write property test for store/index consistency under failures
    - **Property 24: Metadata store and vector index stay one-to-one under failures**
    - File: `tests/test_pipeline_rollback.py`
    - **Validates: Requirements 6.7, 6.8, 6.10, 7.7, 8.10**

  - [x] 10.8 Write property test for idempotent and incremental ingest
    - **Property 26: Re-ingest is idempotent and incremental ingest matches full ingest**
    - File: `tests/test_pipeline_idempotent.py`
    - **Validates: Requirements 7.3, 7.4, 7.6**

  - [x] 10.9 Write property test for ingest order independence
    - **Property 27: Ingest order does not change the indexed records**
    - File: `tests/test_pipeline_confluence.py`
    - **Validates: Requirements 7.5**

  - [x] 10.10 Write unit tests for pipeline skip paths
    - Unresolved metadata skipped with "unresolved camera metadata" log and no writes; unreadable video and unknown fps skipped; Src_Hash computed before any write (spy store); unhashable file logged and skipped; video with zero passed frames rolled back
    - File: `tests/test_pipeline_examples.py`
    - _Requirements: 1.5, 1.11, 2.9, 2.10, 7.1, 7.8_

  - [x] 10.11 Implement `scripts/ingest.py`
    - Args `[paths...]` (default `data/videos/`), `--repair`, `--set k=v`; runs `enable_offline_mode`, `load_config` + `require_valid`, `require_models`, opens store/index (creates empty if absent), loads `OpenClipEncoder`, `reconcile()`, `ingest_paths`; prints `ingested=N failed=M already_indexed=K vectors=V`; exit 1 when no video files are found or the index has zero vectors afterwards; exit codes per the design table
    - _Requirements: 2.8, 3.8, 7.6, 17.2, 17.7_

- [x] 11. Implement clustering, the search engine, and the search CLI
  - [x] 11.1 Implement `nab_sentry/search/clustering.py`
    - `ClusterHit`, `VideoInfo`, `ClusterParams`, `Event`, `EPS`, `query_words`, `label_words`, `event_score`, `cluster_hits` following the design's six-step algorithm (median, group + total sort, merge-gap walk, padding clamp, score + single Label_Boost, relative score, representative thumbnail, final sort)
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 9.10, 9.11, 9.12, 9.13, 9.14, 9.15_

  - [x] 11.2 Write property test for hit partitioning
    - **Property 34: Clustering partitions hits by video**
    - File: `tests/test_clustering_partition.py`
    - **Validates: Requirements 9.1, 9.4, 9.6**

  - [x] 11.3 Write property test for the merge gap
    - **Property 35: Merge-gap invariant**
    - File: `tests/test_clustering_gap.py`
    - **Validates: Requirements 9.2, 9.3, 9.5**

  - [x] 11.4 Write property test for event bounds
    - **Property 36: Event bounds with padding**
    - File: `tests/test_clustering_bounds.py`
    - **Validates: Requirements 9.7, 9.8**

  - [x] 11.5 Write property test for scoring and thumbnails
    - **Property 37: Event scoring, label boost, and representative thumbnail**
    - File: `tests/test_clustering_scoring.py`
    - **Validates: Requirements 9.9, 9.10, 9.11, 9.14**

  - [x] 11.6 Write property test for event ordering
    - **Property 38: Event ordering**
    - File: `tests/test_clustering_order.py`
    - **Validates: Requirements 9.12**

  - [x] 11.7 Write property test for hit-order independence
    - **Property 39: Clustering is independent of hit order** (also covers the empty-input case of 9.15 as an example)
    - File: `tests/test_clustering_confluence.py`
    - **Validates: Requirements 9.13**

  - [x] 11.8 Implement `SearchEngine` in `nab_sentry/search/engine.py`
    - `validate_filter` (`FilterError` for `time_range` and `cls`), `SearchEngine` with `check_ready` (`SearchUnavailable` when encoder is `None` or FAISS IDs ≠ `vectors` IDs), `search_vector`, and `search` in the design's order (validate → encode → allowed set, empty → `[]` without index call → index search → `hit_rows` → `cluster_hits` → `[:limit]`)
    - _Requirements: 8.2, 8.3, 8.5, 8.12, 8.13, 8.14, 8.15, 9.15_

  - [x] 11.9 Write property test for filtered hits
    - **Property 29: Filtered hits are always allowed**
    - File: `tests/test_engine_filtered.py`
    - **Validates: Requirements 8.3, 8.5**

  - [x] 11.10 Write property test for invalid filters
    - **Property 33: Invalid filters are rejected before searching** (spy index)
    - File: `tests/test_engine_invalid.py`
    - **Validates: Requirements 8.12, 8.14**

  - [x] 11.11 Write unit tests for engine readiness
    - Missing encoder and inconsistent ID sets raise `SearchUnavailable`; unknown camera returns `[]`
    - File: `tests/test_engine_examples.py`
    - _Requirements: 8.13, 8.15_

  - [x] 11.12 Implement `scripts/search_cli.py`
    - `"query" [--camera] [--start] [--end] [--cls] [--limit] [--set k=v]`; startup checks, loads store, index, and encoder, runs `check_ready`, prints a table of camera, label, ISO start/end, offsets, score, relative score, thumbnail
    - _Requirements: 8.15, 9.12_

  - [x] 11.13 Write slow end-to-end synthetic test
    - Generate the default synthetic video twice with the same seed and compare decoded frames; ingest with real `OpenClipEncoder` and `enable_detector=False`; query "red square" and "blue circle"; assert at least one Event and the top Event overlaps the matching ground-truth interval by ≥ 1 s (`@pytest.mark.slow`)
    - File: `tests/test_e2e_synthetic.py`
    - _Requirements: 15.4, 15.7_

- [x] 12. Checkpoint: Phase 1
  - Run `venv\Scripts\python.exe -m pytest -m "not slow"` and make sure all tests pass, ask the user if questions arise.
  - Manual verification: run `venv\Scripts\python.exe scripts\make_synthetic.py`, `venv\Scripts\python.exe scripts\ingest.py --set enable_detector=false`, then `venv\Scripts\python.exe scripts\search_cli.py "red square"` and `"blue circle"`, and confirm the top results fall in the 10–25 s and 50–65 s windows.

### Phase 2: Detector, filtered search, API, media serving

- [x] 13. Implement the ONNX YOLO11n detector
  - [x] 13.1 Implement `OnnxYoloDetector`, `letterbox`, and `postprocess` in `nab_sentry/ingest/detector.py`
    - `LetterboxMeta`, `letterbox` (RGB, /255, CHW, pad 114), `postprocess` (transpose `(1,84,8400)`, best class per anchor, Target_Classes with score ≥ conf, undo letterbox, class-wise `cv2.dnn.NMSBoxes`, `clamp_box`, sort by conf desc with `(x1, y1)` tie-break, take `max_det`), `OnnxYoloDetector` (validates conf/iou/max_det → `ConfigError`; missing file → `ModelMissingError` with path and `FETCH_HINT`; `CPUExecutionProvider` with thread options; inference errors wrapped as `DetectorInferenceError`)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.6, 4.8, 4.9_

  - [x] 13.2 Write property test for detector postprocessing
    - **Property 15: Detector postprocess keeps only top target-class detections**
    - File: `tests/test_detector_post.py`
    - **Validates: Requirements 4.1, 4.2, 4.4**

  - [x] 13.3 Write unit and slow tests for the detector
    - Missing ONNX file error names the path and fetch command; invalid conf/max_det raise at init; real detector returns frame-coordinate boxes on a generated sample image (`@pytest.mark.slow`)
    - File: `tests/test_detector_examples.py`
    - _Requirements: 4.1, 4.6, 4.9_

  - [x] 13.4 Wire the detector into ingest
    - In `scripts/ingest.py`, construct `OnnxYoloDetector` from the manifest path when `Config.enable_detector` is true (default) and pass it to `IngestPipeline`, so crop vectors are written with class, confidence, and box; the detector loads before the first video is touched
    - _Requirements: 4.5, 4.6, 5.2_

- [x] 14. Implement the API_Server, media serving, and Query_Log
  - [x] 14.1 Implement `nab_sentry/querylog.py`
    - `QueryRecord` and `QueryLog.append` (append mode, one JSON line, flush, lock, `OSError` → warning and `False`)
    - _Requirements: 10.10, 10.14_

  - [x] 14.2 Write unit tests for the Query_Log
    - Append preserves earlier lines; unwritable path returns `False` without raising
    - File: `tests/test_querylog.py`
    - _Requirements: 10.10, 10.14_

  - [x] 14.3 Implement `nab_sentry/api/media.py`
    - `FullBody`, `PartialBody`, `Unsatisfiable`, `RANGE_RE`, pure `parse_range` per the design table, `resolve_thumb` (`^[A-Za-z0-9_-]{1,128}\.jpg$` + resolved-parent containment + regular file), `resolve_playback` (`^v\d+\.mp4$` + containment), `iter_file` (256 KiB chunks)
    - _Requirements: 11.1, 11.2, 11.3, 11.5, 11.8, 11.9, 11.10, 18.6_

  - [x] 14.4 Write property test for the Range model (parser part)
    - **Property 43: Range header model** (`parse_range` vs reference; HTTP header/body part is task 14.10)
    - File: `tests/test_media_range.py`
    - **Validates: Requirements 11.1, 11.2, 11.3, 11.5, 11.9, 11.10**

  - [x] 14.5 Write property test for path containment (resolver part)
    - **Property 45: Media paths cannot escape their directories** (resolvers; HTTP 404 part is task 14.12)
    - File: `tests/test_media_paths.py`
    - **Validates: Requirements 11.8, 18.6**

  - [x] 14.6 Implement `create_app` in `nab_sentry/api/app.py`
    - `Services`, `SearchRequest`, `ParamError`, pure `validate_search_params` (order `q, start, end, cls, limit`; naive datetimes → local zone), `create_app` with `GET /api/health`, `GET /api/cameras`, `GET /api/search` (validate → engine → `EventOut` with ISO times `video_start + offset` at millisecond precision → Query_Log for 200/422 with latency), `GET /media/video/{video_id}` (200/206/416/404 with `Accept-Ranges`, `Content-Range`, `Content-Length`, streaming `iter_file`), `GET /media/thumbs/{name}`, `GET /` and `/static` from `web/`
    - Exception handlers: `ParamError`/`RequestValidationError`/`FilterError` → 422 in the shared error shape; `SearchUnavailable` → 503; any other exception → sanitised 500 with traceback only in the server log
    - _Requirements: 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 10.10, 10.11, 10.12, 10.13, 10.14, 11.1, 11.2, 11.3, 11.5, 11.6, 11.7, 11.8, 11.9, 11.10, 18.4, 18.6, 18.7_

  - [x] 14.7 Write property test for search API results
    - **Property 40: Search API results are bounded, ordered, and time-consistent**
    - File: `tests/test_api_search_results.py`
    - **Validates: Requirements 10.4, 10.5, 10.6**

  - [x] 14.8 Write property test for search parameter validation
    - **Property 41: Search parameter validation** (includes the API half of Property 21 for blank `q`)
    - File: `tests/test_api_params.py`
    - **Validates: Requirements 10.7, 10.8, 10.9**

  - [x] 14.9 Write property test for the Query_Log via the API
    - **Property 42: Query log is append-only, one line per request**
    - File: `tests/test_api_querylog.py`
    - **Validates: Requirements 10.10**

  - [x] 14.10 Write property test for Range reassembly
    - **Property 44: Range reassembly round trip** (also checks the HTTP part of Property 43: status, `Content-Range`, `Content-Length`, body bytes)
    - File: `tests/test_api_media_range.py`
    - **Validates: Requirements 11.1, 11.2, 11.3, 11.4, 11.5, 11.9, 11.10**

  - [x] 14.11 Write property test for SQL injection resistance
    - **Property 51: Request values never alter SQL**
    - File: `tests/test_api_sql_safety.py`
    - **Validates: Requirements 18.4**

  - [x] 14.12 Write unit tests for API endpoints
    - `/api/health` shape; `/api/cameras` empty and populated; `/` serves the Console HTML; unknown camera → `[]`; 503 when encoder is `None`; Query_Log write failure still returns results; full-body media headers; unknown video and missing playback file → 404; thumbnail 200 `image/jpeg`; hostile names → 404 with no file bytes; crafted route exception → 500 body without stack trace, absolute path, or SQL text, and the next request succeeds
    - File: `tests/test_api_examples.py`
    - _Requirements: 10.2, 10.3, 10.11, 10.12, 10.13, 10.14, 11.1, 11.6, 11.7, 11.8, 18.6, 18.7_

  - [x] 14.13 Implement `serve()` and `python -m nab_sentry.api.app`
    - Startup sequence from the design: load + validate Config (exit 2), `ensure_loopback` (exit 2), probe-bind 127.0.0.1:port (exit 3), `require_models` (exit 4), `log_security_warning`, open store and load index, `check_ready` (exit 5 with `scripts\ingest.py --repair` hint), load encoder and detector and warm up `encode_text` (exit 6 naming the file), then `uvicorn.run(app, host="127.0.0.1", port=cfg.port)`
    - _Requirements: 10.1, 13.3, 13.4, 13.5, 14.6, 14.7, 18.1, 18.2, 18.5_

  - [x] 14.14 Write tests for server startup and binding
    - **Property 52: Only loopback binding is allowed** (`serve()` half: non-loopback hosts exit before any listening socket is created)
    - Example tests: security warning is logged before the first request; missing index or model file exits with the documented code and file name
    - Slow integration: `serve()` in a subprocess listens only on `127.0.0.1:<port>`, and a connection to a non-loopback host address is refused (`@pytest.mark.slow`)
    - File: `tests/test_api_serve.py`
    - **Validates: Requirements 10.1, 14.7, 18.1, 18.2, 18.5**

  - [x] 14.15 Write slow tests for network isolation and write locations
    - Under a socket guard, run startup, ingest of the synthetic video, a search, and media requests with real models, asserting zero non-loopback connections; with a `sys.addaudithook` write-mode `open` audit, assert every written path is under `data/` or `models/` (`@pytest.mark.slow`)
    - File: `tests/test_offline_isolation.py`
    - _Requirements: 13.6, 13.7, 13.9_

- [x] 15. Checkpoint: Phase 2
  - Run `venv\Scripts\python.exe -m pytest -m "not slow"` and make sure all tests pass, ask the user if questions arise.
  - Manual verification: start `venv\Scripts\python.exe -m nab_sentry.api.app`, open `http://127.0.0.1:8765/media/video/<id>` in a browser, and confirm the video seeks (Range requests return 206).

### Phase 3: Console, evaluation, demo packaging

- [x] 16. Implement the Operator Console
  - [x] 16.1 Implement `nab_sentry/web/index.html` and `nab_sentry/web/styles.css`
    - CSP meta `default-src 'self'; media-src 'self'; img-src 'self'`; `<form role="search">` with labelled `q` (`maxlength="256"`), camera select ("All cameras" default), two `datetime-local` inputs, class select ("All classes" + six Target_Classes), submit button; `role="status" aria-live="polite"` region; results `<ol>`; player panel with `<video controls preload="metadata">`, timeline span overlay, Replay/Previous/Next buttons; validation message element next to `q`
    - System font stack, no external assets, `:focus-visible { outline: 3px solid #1a73e8; outline-offset: 2px; }`, active-card style
    - _Requirements: 12.1, 12.2, 12.8, 12.11, 12.12, 13.10_

  - [x] 16.2 Implement `nab_sentry/web/app.js`
    - `fetchWithTimeout` (30 s, `AbortController`); camera list load with failure fallback to "All cameras" and an error message; `buildSearchUrl` (trimmed `q`, only set filters); client-side 1–256 validation; loading indicator and disabled search control; result cards as `<button>` with thumbnail `alt="{label}, {start}"`, label, `formatTs` times, score to 3 decimals; "No matching incidents found"; error display keeping inputs; card activation (`loadedmetadata` → `currentTime = start_offset_s`, timeline span, exactly one `aria-current="true"`); Replay/Previous/Next with enable/disable rules; video `error` message in the player area keeping the list and active card
    - _Requirements: 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 12.10, 12.12, 12.13, 12.14, 12.15_

  - [x] 16.3 Write static checks for the Console
    - No `http://` or `https://` URLs in `web/`; CSP meta present; every form control has a `<label for>`; status region has `role="status"`; `app.js` sets `alt` on result images; class select lists exactly the six Target_Classes plus "All classes"
    - File: `tests/test_console_static.py`
    - _Requirements: 12.1, 12.2, 12.12, 13.10_

- [x] 17. Implement the Evaluator
  - [x] 17.1 Implement `nab_sentry/evaluation.py`
    - `EvalQuery`, `QueryError`, `QueryResult`, `Aggregate`, `load_queries` (YAML schema, count 15–20, per-query errors for missing text/fields or end ≤ start), `is_relevant`, `precision_at_5`, `hit_at_1`, `aggregate` (means over non-error queries)
    - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.7, 16.8_

  - [x] 17.2 Write property test for relevance
    - **Property 49: Relevance is camera match plus interval overlap**
    - File: `tests/test_evaluation_relevance.py`
    - **Validates: Requirements 16.2**

  - [x] 17.3 Write property test for metric bounds
    - **Property 50: Metric bounds**
    - File: `tests/test_evaluation_metrics.py`
    - **Validates: Requirements 16.3, 16.4**

  - [x] 17.4 Implement `scripts/evaluate.py` and the `eval/queries.yaml` template
    - `[--queries eval/queries.yaml] [--set k=v]`; startup checks; in-process `SearchEngine` with no filter and Config `top_k`; unknown camera IDs recorded as query errors; per-query precision@5, hit@1, latency; aggregates; report JSON with run time, every Requirement 16.5 parameter, aggregates, and per-query rows written to `data/eval/report-YYYYmmdd-HHMMSS.json`; exit non-zero with a cause message for a missing/unparseable file, out-of-range count, or all queries in error
    - `eval/queries.yaml`: 15 template entries in the design's schema (two for the synthetic video, the rest placeholders with comments telling the Integrator to fill in real camera IDs and ground-truth times)
    - _Requirements: 16.1, 16.2, 16.3, 16.5, 16.6, 16.7, 16.8, 16.9_

  - [x] 17.5 Write unit tests for the Evaluator
    - YAML count bounds and parse errors exit non-zero; per-query error rows excluded from aggregates; report contains every Requirement 16.5 parameter; two runs with `FakeEncoder` give identical metrics
    - File: `tests/test_evaluation_examples.py`
    - _Requirements: 16.1, 16.5, 16.6, 16.7, 16.8, 16.9_

- [x] 18. Implement demo packaging
  - [x] 18.1 Finalise `scripts/benchmark.py`
    - Use `FileSource`, `MotionGate`, `OnnxYoloDetector`, `OpenClipEncoder`, and `write_thumbnail`; measure decode, Motion_Gate fps, detector ms/frame, embedder ms/image at batch 1 and 8, thumbnail ms; measure the pass fraction and mean detections per passed frame on the input video; print projected ingest minutes per footage hour with the design's formula, stating Sample_Rate and pass fraction; `--search N` builds a random N-vector index plus matching store rows and times ≥ 100 sequential filtered and unfiltered searches (p95, excluding the first); sample peak RSS with `psutil`
    - _Requirements: 14.1, 14.2, 14.3, 14.5, 14.8_

  - [x] 18.2 Implement `run_demo.ps1`
    - Steps 1–5 from the design: check `venv\Scripts\python.exe` (error + exit 1), activate venv when allowed, run `scripts\ingest.py data\videos` when `data\nab_sentry.db` or `data\vectors.faiss` is missing and print ingested/failed counts (non-zero exit → error naming `data\videos\`, exit 1, no server), start `-m nab_sentry.api.app`, poll `/api/health` every 500 ms up to 30 s from script start, print the server's error and exit with its code if it stops, print `NAB Sentry console: http://127.0.0.1:<port>/` only when both models report loaded
    - _Requirements: 17.1, 17.2, 17.6, 17.7_

  - [x] 18.3 Write `README.md`
    - Sections: model provisioning (`venv\Scripts\python.exe scripts\fetch_models.py` on a connected machine, then copy `models\`), ingest (`venv\Scripts\python.exe scripts\ingest.py data\videos`), demo (`.\run_demo.ps1`), evaluation (`venv\Scripts\python.exe scripts\evaluate.py`), and security limitations (no authentication or access control, no audit logging, binds only to 127.0.0.1, Query_Log is a local non-tamper-evident file, access control and tamper-evident audit log planned after the MVP); include venv setup, test commands, and a place to record benchmark results
    - _Requirements: 17.4, 18.3_

  - [x] 18.4 Write static test for the README
    - Asserts the five sections exist, the first four each contain their PowerShell command, and the security section contains each required statement
    - File: `tests/test_readme.py`
    - _Requirements: 17.4, 18.3_

- [x] 19. Final checkpoint: Phase 3
  - Run `venv\Scripts\python.exe -m pytest -m "not slow"` and `venv\Scripts\python.exe -m pytest -m slow` (requires `models\`) and make sure all tests pass, ask the user if questions arise.
  - Manual verification (Target_Hardware): run `scripts\benchmark.py` on a sample video and `scripts\benchmark.py --search 200000`; time a 1-hour 1080p ingest; record results in the README (Requirements 14.1–14.6).
  - Manual verification (browser): keyboard-only walkthrough of the Console (Tab order, Enter/Space, visible focus ring), NVDA pass for labels, alt text, and status announcements, seek accuracy via `video.currentTime`, Previous/Next/Replay enablement, 30 s timeout with a paused server, and video load failure with a deleted Playback_File (Requirement 12). Full WCAG conformance needs manual testing with assistive technologies and expert review.
  - Manual verification (offline rehearsal): disable all network adapters, run `.\run_demo.ps1`, submit every `eval/queries.yaml` query in the Console, and play each top result from its start (Requirements 13.9, 17.3, 17.5).

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP; core implementation tasks are never optional.
- Each property test carries the tag comment `# Feature: nab-sentry, Property N: <title>` and runs at least 100 Hypothesis examples. Real-model and full-length ffmpeg tests are `@pytest.mark.slow`.
- The design groups tests by component module (`test_metadata.py`, ...). This plan keeps the component prefix but gives each property its own file (`test_metadata_filename.py`, ...) so property tasks can run in parallel without editing the same file. Shared fakes in `tests/fakes.py` are added by the implementation task that introduces the matching protocol.
- `scripts/fetch_models.py` needs internet once, on a connected machine. Everything else runs offline.
- Manual verification steps (benchmarks on the Target_Hardware, browser checks, network-disabled rehearsal) are listed in the checkpoints only and are not coding tasks.
- Requirements 14.3–14.6 (latency, ingest time, memory, startup time) are measured with `benchmark.py` and `run_demo.ps1` on the Target_Hardware rather than asserted in automated tests.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["2.1"] },
    { "id": 2, "tasks": ["2.2", "2.3"] },
    { "id": 3, "tasks": ["2.4", "2.5", "2.6", "3.1", "3.3", "5.1", "8.5"] },
    { "id": 4, "tasks": ["3.2", "5.2", "5.3", "5.4", "5.5", "5.6", "5.7", "5.8", "8.1", "8.6", "8.7", "8.8", "8.9", "9.8", "11.1"] },
    { "id": 5, "tasks": ["5.9", "6.1", "6.5", "7.1", "8.2", "8.3", "8.4", "9.1", "9.9", "9.10", "11.2", "11.3", "11.4", "11.5", "11.6", "11.7"] },
    { "id": 6, "tasks": ["6.2", "6.3", "6.4", "6.6", "6.7", "6.8", "7.2", "7.3", "7.4", "9.2", "9.3", "9.4", "9.5", "9.6", "9.7", "10.1"] },
    { "id": 7, "tasks": ["10.2", "10.3"] },
    { "id": 8, "tasks": ["10.4", "10.5", "10.6", "10.7", "10.8", "10.9", "10.10", "10.11", "11.8"] },
    { "id": 9, "tasks": ["11.9", "11.10", "11.11", "11.12"] },
    { "id": 10, "tasks": ["11.13", "13.1", "14.1", "14.3"] },
    { "id": 11, "tasks": ["13.2", "13.3", "13.4", "14.2", "14.4", "14.5", "14.6"] },
    { "id": 12, "tasks": ["14.7", "14.8", "14.9", "14.10", "14.11", "14.12", "14.13"] },
    { "id": 13, "tasks": ["14.14", "14.15", "16.1", "17.1"] },
    { "id": 14, "tasks": ["16.2", "17.2", "17.3", "17.4", "18.1"] },
    { "id": 15, "tasks": ["16.3", "17.5", "18.2", "18.3"] },
    { "id": 16, "tasks": ["18.4"] }
  ]
}
```
