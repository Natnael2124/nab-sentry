# Implementation Plan: NAB Sentry V2 (Steps 1-2)

## Overview

Phase A (Step 1) measures the system on the Target_Hardware and records decisions. Phase B (Step 2) adds live recording, watch-mode ingest, tracking, zones, rules with an alert outbox, Telegram delivery, health and retention, then acceptance runs.

Commands use `venv\Scripts\python.exe` from the workspace root. Fast suite: `venv\Scripts\python.exe -m pytest -m "not slow"`. Slow suite: `-m slow`. Every task keeps the full V1 suite passing.

Footage rule: the Compilation_Video is used only for throughput, gate recall and peak-load tests. Gate efficiency, zones, tracking continuity, dwell rules and normal-load latency need Continuous_Footage (task 5.4).

## Tasks

### Phase A: Measure (Step 1)

- [ ] 1. Real-hardware benchmark
  - [x] 1.1 Run `scripts/benchmark.py data\videos\CAM-SEC01_20261007T230010.mp4 --search 200000` on the Target_Hardware
    - Done 2026-10-18 (AC power, Balanced plan): detector 103.6 ms, embedder b8 126.7 ms/image, b1 181.9 ms, decode 3.2 ms, gate 157 frames/s, thumbnail 13.4 ms, search 134.9 / 191.1 ms, projected 26.4 min/footage hour, peak RSS 1,804 MiB
    - _Requirements: 1.1_
  - [x] 1.2 Record the results in the README table with CPU, power plan and date; mark the projection as worst case (continuous activity)
    - _Requirements: 1.1, 1.2_
  - [ ] 1.3 Re-run once with the Windows "Best performance" power mode to bound the power-plan effect; record both
    - _Requirements: 1.1_

- [ ] 2. Ground truth
  - [x] 2.1 Implement `scripts/contact_sheet.py VIDEO` (1 thumbnail/s stamped with offset, sheets of 60 under `data/bench/contact/`, suggested scene cuts with red borders, Ground_Truth_File template)
    - _Requirements: 2.1_
  - [x] 2.2 Implement `nab_sentry/ground_truth.py` loader + validation
    - _Requirements: 2.2, 2.3_
  - [x] 2.3 Property test: every single invalid entry is reported exactly once by name
    - File: `tests/test_ground_truth.py`
    - _Requirements: 2.3_
  - [ ] 2.4 **(Integrator)** Label the Compilation_Video: scene cuts + Incidents into `eval/ground_truth/CAM-SEC01.yaml` (start from `data/bench/contact/CAM-SEC01_20261007T230010/CAM-SEC01.yaml`; 45 suggested cuts, some false where the camera is bumped)
    - _Requirements: 2.2_
  - [ ] 2.5 **(Integrator)** Replace the TODO entries in `eval/queries.yaml` with real queries; run `scripts/evaluate.py` and record recall@k / MRR as the V1 search baseline
    - _Requirements: 2.4_

- [ ] 3. Motion gate ROI and sweep
  - [ ] 3.1 Add optional ROI (normalized rectangle) to `MotionGate` and `Config.motion_roi`
    - _Requirements: 3.1_
  - [ ] 3.2 Property tests: ROI = full frame equals V1 decisions (Property 3); changes outside ROI never pass (Property 4)
    - File: `tests/test_motion_roi.py`
    - _Requirements: 3.1_
  - [ ] 3.3 Implement `scripts/gate_sweep.py` with pure scoring `missed_incidents`, `idle_pass_fraction`
    - _Requirements: 3.2, 3.3, 3.4_
  - [ ] 3.4 Property test for scoring (Property 5)
    - File: `tests/test_gate_sweep.py`
  - [ ] 3.5 Run the recall sweep on the Compilation_Video (needs 2.4): diff vs mog2, thresholds 0.005-0.05, no ROI; record the strictest setting with 0 missed Incidents
    - _Requirements: 3.2, 3.3_
  - [ ] 3.6 Run the efficiency sweep on Continuous_Footage (needs 5.4 / 8.x recorder footage + labels)
    - _Requirements: 3.4_

- [ ] 4. VLM latency
  - [x] 4.1 Implement `scripts/vlm_bench.py` (frame / 2x2 grid / 2x2 person-crop storyboard; cold + warm; Ollama timings; Ollama RSS; think off by default) with pure-helper tests in `tests/test_vlm_bench.py`
    - _Requirements: 4.1, 4.2, 4.3_
  - [x] 4.0 Gemma 4 E2B dropped (2026-10-18): 6.7 GB loaded, 16-23 s per new image, and its image input did not work (see design findings). Model removed from Ollama; `qwen3-vl:2b-instruct` is the first on-demand candidate
  - [x] 4.1a Add the vision sanity check (red image, `HELLO 42` text) and per-call cache busting to `vlm_bench.py`; default model `qwen3-vl:2b-instruct`
    - _Requirements: 4.1, 4.3_
  - [x] 4.1b Idle run of `qwen3-vl:2b-instruct` (2026-10-18): sanity passes; ~1,100 image tokens at any input size, 35-48 s image encoding, ~13 tok/s, ~40 s per call with a one-sentence answer, process RSS peak 4.7 GB. Recorded in README. Follow-up if on-demand latency matters: llama.cpp server with an image-token cap, or a smaller VLM (SmolVLM2 / Moondream) with the same sanity check
  - [ ] 4.2 Run idle (`--label idle`) and loaded (`--label loaded`, while `scripts/benchmark.py` runs) on the Target_Hardware; record both summaries and 3 sample outputs per input in the README
    - _Requirements: 4.4_
  - [x] 4.3 Detector cost at tracking rates (`scripts/det_bench.py`, interleaved timing; results in README): **384x640 rectangular input, 2 threads: 36.7 ms median, recall 0.98 vs 640 (0.93 on small boxes), 11% busy at 3 fps / 18% at 5 fps**. Square 416/480 lose too many small/medium boxes; 4 threads give only 5-10%. The embedder (~37% busy at continuous activity) is now the largest per-camera cost
    - _Requirements: 5.1_
  - [x] 4.4 Spike: OpenCLIP zero-shot attribute tags on 24 person + 12 vehicle crops (results in design findings: vehicles ~80%, person top colour ~60%, clothing/carrying/headwear unreliable; ~25% of person crops are detector false positives)
    - _Requirements: 4A.1_
  - [ ] 4.5 Turn the spike into `nab_sentry/describe/attributes.py` (pure scoring with abstain threshold, prompt groups in config) with property tests; re-measure per Track (best crop + vote across crops) once task 10 exists, on a labelled sample, and record correct / wrong / abstained per group
    - _Requirements: 4A.1, 4A.2, 4A.3_

- [ ] 5. Decisions
  - [ ] 5.1 Write the decision note in the README: gate settings, Track_Rate + detector input size, Segment_Length, cameras per laptop without VLM, local vs cloud VLM for Step 3 — each citing its measurement
    - _Requirements: 5.1_
  - [ ] 5.2 Update Config defaults in this spec's design to the decided values
  - [ ] 5.3 Checkpoint: review decisions with the product owner
  - [ ] 5.4 **(Integrator)** Record ≥ 1 h of Continuous_Footage of one fixed view (phone or IP camera, mostly idle, ideally including a lighting change) and label it; this unblocks 3.6, 17.3

### Phase B1: Live input

- [ ] 6. Camera configuration
  - [x] 6.1 Implement `nab_sentry/live/camera_config.py` (`CameraSpec`, `load_cameras`, `CameraConfigError`, `mask_url`) and `config/cameras.example.yaml`
    - _Requirements: 6.1, 6.2, 6.4_
  - [x] 6.2 Property tests: round trip and one issue per invalid field (Property 6); URL masking (Property 7)
    - File: `tests/test_camera_config.py`
  - [x] 6.3 Implement `scripts/snapshot.py CAMERA_ID` (one frame via ffmpeg, normalized grid overlay, `data/live/`)
    - Not yet run against a real camera (needs `config/cameras.yaml` and the Integrator's OK to switch a camera on)
    - _Requirements: 6.3_

- [ ] 7. Recorder
  - [x] 7.1 Verify the bundled imageio-ffmpeg supports `-f dshow` and RTSP over TCP on this machine; if not, add a pinned ffmpeg to `models/bin/` via `fetch_models.py`
    - ffmpeg 7.1 (gyan essentials): dshow, rtsp demuxer, segment + matroska muxers, libx264 and **h264_qsv** (Intel Quick Sync works on the i5-1145G7: about half the file size of x264 veryfast). The laptop webcam and the phone (Windows virtual camera) show up as dshow devices. Segment names have 1 s resolution, so sources must run in real time
  - [x] 7.2 Implement `nab_sentry/live/recorder.py`: pure `recorder_command` (copy for RTSP, x264 + forced keyframes for dshow), `backoff(n)`, heartbeat writer; `scripts/record.py [CAMERA_ID ...]`
    - _Requirements: 7.1, 7.2, 7.3, 7.4_
  - [x] 7.3 Unit tests for `recorder_command` and `backoff`; slow test: record 25 s from a looped synthetic RTSP-free source (`-re` file input) and assert ≥ 2 readable `.mkv` Segments with valid names
    - File: `tests/test_recorder.py`
  - [x] 7.4 Implement `scripts/replay_camera.py VIDEO CAMERA_ID` (same naming, `-re`, start = now)
    - _Requirements: 17.1_

- [ ] 8. Watch mode
  - [x] 8.1 Implement pure `closed_segments` in `nab_sentry/ingest/watch.py`
    - _Requirements: 8.2_
  - [x] 8.2 Property test (Property 8)
    - File: `tests/test_watch_closed.py`
  - [x] 8.3 Add `--watch` to `scripts/ingest.py`: reconcile on start, 1 s poll, ordered per camera, ≤ 2 retries per failed Segment, clean Ctrl+C
    - _Requirements: 8.1, 8.3, 8.4, 8.5_
  - [x] 8.4 Set `busy_timeout = 5000` on every SQLite connection; test two processes writing concurrently
    - Already set by V1 `MetadataStore`; the cross-process test moves to task 15 (first second writer is the Notifier)
  - [ ] 8.5 Checkpoint: phone (DroidCam/dshow) or RTSP camera → Recorder → Watcher → searchable in the console
    - Replay E2E done 2026-10-18 (40 s of the Compilation_Video as camera CAMT, isolated data dir): 4 Segments, all ingested; 2.8-4.7 s from a Segment closing to it being ingested; the last Segment closes by idleness ~16 s after its last write. Still needs a real camera run
    - Known V1 issue seen here: every log line is printed twice (two handlers); fix before long-running use

### Phase B2: Tracking

- [ ] 9. Detector and Track_Rate path
  - [x] 9.0 Rectangular detector input: generalise `letterbox` to `(h, w)`, add `det_input_h` / `det_input_w` to Config (default 384 x 640; `det_input_size` kept as the square shorthand), export `yolo11n_384x640.onnx` in `fetch_models.py` and add it to the manifest, default detector threads 2; update detector tests (letterbox round trip for rectangles)
    - Done 2026-10-18. Config: `det_input_size` stays the width, new `det_input_h` (default 384) and `det_threads` (default 2). `OnnxYoloDetector` accepts `(h, w)` and refuses an ONNX file exported for another shape. `fetch_models.py` exports `yolo11n.onnx` + `yolo11n_384x640.onnx` and has `--export-only` (offline re-export from `yolo11n.pt`). Local models re-exported: the new 640 export gives identical boxes to the original on 40 frames; the 384x640 default agrees on 94.9% of boxes at 33.7 ms vs 48-51 ms
  - [x] 9.1 `Detection.high`; detector keeps ≥ `det_low_conf`; crop path uses only high; Config fields `det_low_conf`, `track_rate` with cross-field checks
    - Done 2026-10-18. High and low detections each get their own `det_max_per_frame` budget, so the high set is exactly V1's (property test on random YOLO outputs; real model: identical on 60/60 frames, plus ~2 low boxes per frame). No cross-field errors (they broke V1 Property 11: a V1 user setting `det_conf=0.05` would get an error about `det_low_conf`); instead `effective_low_conf = min(det_low_conf, det_conf)` and `effective_track_rate = max(track_rate, sample_rate)`. `track_rate` allowed 1..30
    - _Requirements: 9.1, 9.4_
  - [x] 9.2 Update detector post-processing tests and Config tests for the new fields (`tests/test_detector_low_conf.py`)
  - [ ] 9.3 Pipeline: Sampler at Track_Rate; gate bypass while a Track is active; frames/thumbnails/embeddings only on the Sample_Rate cadence
    - _Requirements: 9.2, 9.3_
  - [ ] 9.4 Property test: V1 rows/vectors unchanged with tracking on vs off (Property 23)
    - File: `tests/test_pipeline_track_rate.py`
    - _Requirements: 9.3_

- [ ] 10. Tracker
  - [ ] 10.1 Implement `nab_sentry/track/bytetrack.py` (two-pass greedy IoU per class, constant-velocity prediction, min hits, buffer expiry, `end_all`, snapshot/restore)
    - _Requirements: 10.1, 10.2, 10.4_
  - [ ] 10.2 Property tests: assignment invariant (9), linear-motion stability (10), expiry (11), snapshot/restore (12)
    - File: `tests/test_bytetrack.py`
  - [ ] 10.3 Scene_Cut handling in the pipeline (`scene_cut_frac`) and per-camera live state carried across Segments with `segment_gap_reset_s`
    - _Requirements: 10.4, 10.5_

- [ ] 11. Schema v2 and track storage
  - [ ] 11.1 Migration v1 → v2 in `MetadataStore.open` (tables `tracks`, `track_obs`, `alerts`, `alert_messages`, `rule_mutes`, `kv`; `vectors.track_id`)
    - _Requirements: 11.1_
  - [ ] 11.2 Property test: migration preserves v1 rows, idempotent (Property 19)
    - File: `tests/test_store_migration.py`
  - [ ] 11.3 Store confirmed Tracks, back-filled observations, and `vectors.track_id`; extend compensation to restore live state on rollback
    - _Requirements: 10.6, 10.7_
  - [ ] 11.4 Extend the V1 rollback tests: failure at every checkpoint leaves no track rows and restores live state
    - File: `tests/test_pipeline_rollback.py`
  - [ ] 11.5 Slow sanity: real YOLO + tracker on the Compilation_Video; no track spans a labelled scene cut; ID count within 2x of labelled people
  - [ ] 11.6 Checkpoint: tracks visible for replayed footage; full V1 suite green

### Phase B3: Zones, rules, outbox

- [ ] 12. Geometry
  - [ ] 12.1 Implement `nab_sentry/rules/geometry.py` (`anchor`, `point_in_polygon`, `segment_cross`)
    - _Requirements: 12.1, 12.2_
  - [ ] 12.2 Property tests (Properties 13, 14)
    - File: `tests/test_geometry.py`

- [ ] 13. Rules
  - [ ] 13.1 Implement `rules.yaml` loader with validation against Camera_Config
    - _Requirements: 12.6_
  - [ ] 13.2 Implement `in_schedule` and pure `evaluate` (enter, dwell, cross, count; debounce; once per track; cooldown)
    - _Requirements: 12.3, 12.4_
  - [ ] 13.3 Property tests (Properties 15, 16, 17)
    - File: `tests/test_rules.py`
  - [ ] 13.4 Pipeline: evaluate rules per Track_Rate frame; write `alerts` rows (`pending` / `suppressed` if muted) and snapshot JPEGs inside the Segment transaction; cooldown from `alerts`
    - _Requirements: 12.5, 14.7_
  - [ ] 13.5 Property test: outbox atomicity under failure at every checkpoint (Property 18)
    - File: `tests/test_pipeline_alerts.py`
  - [ ] 13.6 `reconcile()` removes unreferenced alert media
    - _Requirements: 11.2_

### Phase B4: Delivery

- [ ] 14. Alert media
  - [ ] 14.1 Implement `nab_sentry/alerts/media.py` (`draw_snapshot`, pure `clip_command` with concat over covering Segments)
    - _Requirements: 13.1, 13.2_
  - [ ] 14.2 Tests: clip window spanning 1 and 2 Segments; slow test produces a playable H.264 ≤ 640 px clip
    - File: `tests/test_alert_media.py`

- [ ] 15. Notifier
  - [ ] 15.1 Implement `nab_sentry/notify/telegram.py` (`TelegramClient` with injectable transport, 429 handling) and `FakeTelegramTransport` in `tests/fakes.py`
    - _Requirements: 14.3_
  - [ ] 15.2 Implement the alert state machine with per-(alert, chat, step) message IDs and backoff
    - _Requirements: 14.2, 14.3, 14.4, 13.3_
  - [ ] 15.3 Property test: at-most-once delivery across scripted failures and restarts (Property 20)
    - File: `tests/test_notify_state.py`
  - [ ] 15.4 Implement `parse_command` and handlers: `/status`, `/snapshot`, `/alerts`, `/mute`, `/unmute`; feedback buttons; Allowed_Chat check; `getUpdates` offset in `kv`
    - _Requirements: 14.5, 14.6, 14.7_
  - [ ] 15.5 Property test (Property 21)
    - File: `tests/test_notify_commands.py`
  - [ ] 15.6 `scripts/notify.py`: env vars (`NAB_TELEGRAM_TOKEN`, `NAB_TELEGRAM_CHATS`), refuse to start without them, token never logged
    - _Requirements: 14.1_
  - [ ] 15.7 Isolation test: Notifier contacts only `api.telegram.org`, writes only under `data/`; V1 isolation test extended to the Watcher
    - File: `tests/test_notifier_isolation.py`, `tests/test_offline_isolation.py`
    - _Requirements: 14.8, 18.1, 18.2, 18.3_

- [ ] 16. Health, retention, supervisor
  - [ ] 16.1 Health checks in the Notifier: camera offline/recovered from heartbeats, disk low (6 h throttle)
    - _Requirements: 15.1, 15.2_
  - [ ] 16.2 Retention in the Watcher (hourly): delete old Segments with rows, media and vectors; alert media on its own schedule
    - _Requirements: 15.3_
  - [ ] 16.3 Property test: after retention, `reconcile()` reports no changes (Property 22)
    - File: `tests/test_retention.py`
  - [ ] 16.4 `run_live.ps1`: start Recorders, Watcher, Notifier, API_Server; restart on exit; logs under `data/logs/`
    - _Requirements: 16.1_
  - [ ] 16.5 Checkpoint: real Telegram test chat receives snapshot + clip from a phone-camera rule; `/status`, `/mute` work

### Phase B5: Acceptance

- [ ] 17. Acceptance runs
  - [ ] 17.1 Add `scripts/acceptance.py`: replays a video, runs the chain with a fake or real Telegram, matches Alerts to Ground_Truth Incidents, reports precision, recall, Alert_Latency (median/p95), CPU and peak RAM
    - _Requirements: 17.1, 17.2, 17.3_
  - [ ] 17.2 Peak-load run on the Compilation_Video (scene cuts labelled; rules on full frame)
    - _Requirements: 17.2_
  - [ ] 17.3 Normal-load run on Continuous_Footage with 2-3 Rules (enter, dwell, cross); compare with targets
    - _Requirements: 17.3_
  - [ ] 17.4 Record results in the README; go/no-go for a site pilot

## Notes

- Tasks marked **(Integrator)** need a human: labelling, recording footage, Telegram bot creation (`@BotFather`) and chat IDs.
- Live state reset on Watcher restart is by design (see design decision 4).
- The AGPL licence of YOLO11n must be resolved before the first paid deployment; it is not a Step 2 task.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.2", "1.3", "2.1", "2.2", "3.1", "4.2", "4.3", "6.1", "7.1", "8.1"] },
    { "id": 1, "tasks": ["2.3", "2.4", "3.2", "3.3", "6.2", "6.3", "7.2", "8.2", "9.1", "12.1"] },
    { "id": 2, "tasks": ["2.5", "3.4", "3.5", "7.3", "7.4", "8.3", "8.4", "9.2", "10.1", "11.1", "12.2"] },
    { "id": 3, "tasks": ["5.1", "8.5", "9.3", "10.2", "11.2", "13.1", "13.2", "14.1"] },
    { "id": 4, "tasks": ["5.2", "5.3", "5.4", "9.4", "10.3", "13.3", "14.2", "15.1"] },
    { "id": 5, "tasks": ["11.3", "15.2", "15.4", "15.6"] },
    { "id": 6, "tasks": ["11.4", "11.5", "13.4", "15.3", "15.5", "16.1"] },
    { "id": 7, "tasks": ["11.6", "13.5", "13.6", "15.7", "16.2"] },
    { "id": 8, "tasks": ["16.3", "16.4", "3.6"] },
    { "id": 9, "tasks": ["16.5", "17.1"] },
    { "id": 10, "tasks": ["17.2", "17.3"] },
    { "id": 11, "tasks": ["17.4"] }
  ]
}
```
