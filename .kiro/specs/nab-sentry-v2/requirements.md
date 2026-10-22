# Requirements Document

## Introduction

NAB Sentry V2 turns the V1 offline search engine into a live monitoring system for one site with 1 to 4 cameras, running on the same Target_Hardware (Dell Latitude 5420: Intel Core i5-1145G7, 4 cores / 8 threads, 16 GB RAM, no discrete GPU, Windows 11, Python 3.12.10 in `venv/`).

This spec covers roadmap Steps 1 and 2:

- **Step 1 (Measure):** real-hardware benchmarks, labelled ground truth, motion gate recall and efficiency, small-VLM latency and quality on the Target_Hardware, and a test of OpenCLIP zero-shot attribute tags. Its results decide parameters for Step 2 and the shape of Step 3.

**Step 3 direction (decided 2026-10-18):** Gemma 4 E2B is dropped (about 6.7 GB loaded, about 20 s per event, and its image input did not work on the Target_Hardware). The activity log is built **without a VLM** by default, from Tracks, Zones, dwell times and OpenCLIP zero-shot attribute tags on the crop embeddings that ingest already computes. A small local VLM (first candidate Qwen3-VL 2B Instruct via Ollama; the plain `qwen3-vl:2b` tag is the thinking variant and ignores `think: false`) is used **only on demand**: `/ask` about a specific clip, Rules with `severity: verify`, and optional overnight summaries. A cloud VLM is an opt-in per site, off by default, because it sends images off the premises.
- **Step 2 (Live monitoring):** a segmented recorder for live cameras, watch-mode ingest, multi-object tracking, zones, deterministic watch rules, and Telegram alerts with a snapshot and a short clip, plus camera health alerts and retention.

V1 behaviour is preserved: the `frames` / `vectors` / FAISS one-to-one invariant, per-video transactional ingest with compensation and `reconcile()`, OpenCLIP ViT-B/32 search, and the loopback-only API_Server.

Out of scope (later steps): VLM event descriptions and alert verification (Step 3), any embedder change (Step 4), live frame tap for sub-segment alert latency, multi-segment track persistence across process restarts, web-console zone editor, authentication, replacing the AGPL detector.

### Test footage constraint

The only footage available at the start, `data/videos/CAM-SEC01_20261007T230010.mp4` (6.4 min, 1376x776, 29.66 fps), is a **curated compilation** of short incidents from several scenes joined by hard cuts. It is valid for detector/embedder throughput, gate recall, and peak-load tests. It is **not** valid for gate efficiency (idle pass fraction), ROI/zones, tracking continuity, dwell rules, or normal-load alert latency; those require Continuous_Footage.

## Glossary

V1 glossary terms (Ingest_Pipeline, Sampler, Motion_Gate, Detector, Embedder, Metadata_Store, Vector_Index, Filename_Convention, Absolute_Timestamp, Config, reconcile, Target_Hardware, ...) keep their V1 meaning.

- **Compilation_Video**: The curated test video described above.
- **Continuous_Footage**: An uninterrupted recording from one fixed camera view, at least 1 hour long, mostly without activity.
- **Ground_Truth_File**: A YAML file listing, for one video, its scene cuts and its labelled Incidents.
- **Incident**: A labelled time interval in which a person or vehicle is visibly active, with class, description and (for Continuous_Footage) zone.
- **Camera_Config**: The file `config/cameras.yaml` describing each live camera: ID, label, source, Segment_Length, motion ROI, and Zones.
- **Segment**: One file written by the Recorder, covering Segment_Length seconds of one camera, named by the Filename_Convention with extension `.mkv`.
- **Segment_Length**: Configured seconds per Segment (default 10, allowed 5 to 300).
- **Recorder**: The process that captures a live source into Segments without re-encoding.
- **Watcher**: `scripts/ingest.py --watch`, which ingests closed Segments as they appear.
- **Closed_Segment**: A Segment that the Recorder has finished writing.
- **Track_Rate**: Frames per second at which the Detector and Tracker run while there is activity (default 3, allowed 1 to 30, at least Sample_Rate).
- **Tracker**: The component that links detections across frames into Tracks (ByteTrack-style).
- **Track**: A sequence of detections of one object of one class on one camera, with a camera-unique ID.
- **Scene_Cut**: A frame whose changed-pixel fraction against the previous Track_Rate frame is at or above Scene_Cut_Fraction (default 0.6).
- **Zone**: A named polygon in normalized image coordinates (0..1) on one camera.
- **Anchor_Point**: The bottom-centre point of a detection box, used for Zone membership.
- **Rule**: A watch rule in `config/rules.yaml`: camera, Zone, classes, Schedule, Trigger, debounce, cooldown, severity.
- **Schedule**: Days of week plus a local-time window; a window whose end is earlier than its start wraps past midnight.
- **Trigger**: One of `enter`, `dwell` (seconds), `cross` (line + direction), `count` (minimum simultaneous Tracks).
- **Alert**: A row in the `alerts` table produced by a Rule firing or by a health check.
- **Outbox**: The pattern in which Alerts are written in the same SQLite transaction as the Segment that produced them, and delivered later by a separate process.
- **Notifier**: `scripts/notify.py`, the only process allowed to reach the network, which delivers Alerts to Telegram and answers bot commands.
- **Allowed_Chat**: A Telegram chat ID listed in `NAB_TELEGRAM_CHATS`.
- **Alert_Latency**: Wall-clock time from the Absolute_Timestamp of the triggering frame to Telegram's acceptance of the snapshot message.

## Requirements

### Requirement 1: Real-hardware benchmark

**User Story:** As an Integrator, I want per-stage numbers measured on the Target_Hardware with real footage, so that capacity decisions use facts.

#### Acceptance Criteria

1. THE Integrator SHALL run `scripts/benchmark.py` on the Compilation_Video with `--search 200000` on the Target_Hardware and record every stage median and p95, the pass fraction, peak RSS, CPU model and date in the README benchmark table.
2. THE README SHALL state that the projected ingest minutes per footage hour from the Compilation_Video are a worst case (continuous activity).

### Requirement 2: Ground truth labels

**User Story:** As an Integrator, I want labelled Incidents and scene cuts, so that gate recall, rules and search can be scored.

#### Acceptance Criteria

1. THE NAB_Sentry SHALL provide `scripts/contact_sheet.py VIDEO` that writes JPEG sheets under `data/bench/` with one thumbnail per second of video, each stamped with its offset in seconds.
2. THE Ground_Truth_File `eval/ground_truth/<CAMERA_ID>.yaml` SHALL list scene cut offsets and Incidents with `start`, `end` (seconds), `classes`, `description`, and optional `zone`.
3. THE NAB_Sentry SHALL provide a loader that rejects a Ground_Truth_File with `end <= start`, negative offsets, unknown classes, or Incidents past the video duration, naming each bad entry.
4. THE `eval/queries.yaml` placeholders SHALL be replaced with real queries on the Compilation_Video.

### Requirement 3: Motion gate recall and efficiency

**User Story:** As an Integrator, I want to choose gate settings that miss no Incident and pass as few idle frames as possible.

#### Acceptance Criteria

1. THE Motion_Gate SHALL accept an optional ROI rectangle in normalized coordinates and compute the changed-pixel fraction only inside it; with no ROI, behaviour SHALL equal V1.
2. THE NAB_Sentry SHALL provide `scripts/gate_sweep.py` that, for each combination of motion method, Motion_Threshold and ROI given on the command line, reports sampled frames, passed frames, pass fraction, and the number of Ground_Truth Incidents with no passed frame inside them (missed Incidents).
3. WHEN run on the Compilation_Video, THE sweep report SHALL be used only for missed Incidents and SHALL label pass fraction as not representative.
4. WHEN Continuous_Footage with a Ground_Truth_File exists, THE sweep SHALL report the idle pass fraction (passed frames outside all Incidents / sampled frames outside all Incidents).

### Requirement 4: VLM latency and quality on the Target_Hardware

**User Story:** As an Integrator, I want measured latency, memory and basic correctness of a small local VLM, so that on-demand VLM features in Step 3 can be sized.

#### Acceptance Criteria

1. BEFORE timing a model, THE vlm_bench SHALL run a vision sanity check (a solid red image must be called red; an image showing `HELLO 42` must be read back containing `42`) and SHALL refuse to time a model that fails it unless `--skip-sanity` is given.
2. THE NAB_Sentry SHALL provide `scripts/vlm_bench.py VIDEO` (default model `qwen3-vl:2b-instruct`) that builds, from person-bearing windows of the video, a single-frame image, a 2x2 full-frame storyboard and a 2x2 person-crop storyboard, each panel stamped `[i] t=..s`.
3. THE vlm_bench SHALL send a slightly different image on every call so that the server's prompt cache never reuses an image encoding.
4. THE vlm_bench SHALL call the local Ollama `/api/chat` endpoint on a loopback address only, with thinking off by default, temperature 0 and a JSON output schema, for each configured context size, measuring a cold call after unloading the model and warm repeated calls.
5. FOR each call THE vlm_bench SHALL record wall time, model load time, prompt token count and duration, generated token count and rate, and peak resident memory of all Ollama processes, plus the loaded model size reported by Ollama, and SHALL write all results, images and outputs under `data/bench/vlm/`.
6. THE vlm_bench SHALL be run once with the machine idle and once while an ingest or benchmark runs, and both summaries SHALL be recorded in the README.

### Requirement 4A: Zero-shot attribute tags

**User Story:** As an Integrator, I want to know which appearance attributes OpenCLIP can tag reliably from existing crop embeddings, so that the activity log can describe people and vehicles without a VLM.

#### Acceptance Criteria

1. THE NAB_Sentry SHALL score crop embeddings against attribute groups (each a list of text prompts) with softmax over the group at CLIP's logit scale, and SHALL report a tag only when its probability is at or above a configured minimum, otherwise "unknown".
2. THE attribute test SHALL report, per attribute group and object class, the share of correct, wrong and abstained tags on a hand-checked sample of at least 24 person and 12 vehicle crops.
3. Attribute groups below 70% correct among non-abstained tags SHALL NOT be used in the activity log.

### Requirement 5: Step 1 decision record

**User Story:** As the product owner, I want the Step 1 numbers turned into decisions.

#### Acceptance Criteria

1. THE README SHALL contain a decision note giving: chosen gate settings, Track_Rate, Segment_Length, estimated cameras per Target_Hardware, the attribute groups used in the activity log, and the on-demand VLM model with its per-call latency and memory, each with the measurement it is based on.

### Requirement 6: Camera configuration

**User Story:** As an Integrator, I want each live camera, its ROI and its Zones described in one file.

#### Acceptance Criteria

1. THE NAB_Sentry SHALL load `config/cameras.yaml` with, per camera: `camera_id` (Filename_Convention-valid), `label`, `source` (RTSP/HTTP URL or `dshow:<device name>`), `segment_s`, optional `roi`, and `zones` (name → polygon with at least 3 points, each coordinate in 0..1) and optional `lines` (name → two points).
2. IF the Camera_Config is invalid, THEN loading SHALL fail with one message per invalid field and no process SHALL start.
3. THE NAB_Sentry SHALL provide `scripts/snapshot.py CAMERA_ID` that saves one current frame from the camera's source under `data/live/` with a normalized-coordinate grid drawn on it, for drawing Zones.
4. THE source URL SHALL never be logged with credentials; user info in URLs SHALL be masked in all log lines.

### Requirement 7: Segment recorder

**User Story:** As an Integrator, I want live video persisted as playable files before any AI touches it.

#### Acceptance Criteria

1. THE Recorder SHALL capture each configured source with the bundled ffmpeg using the segment muxer, stream copy (no re-encode), and no audio, writing Segments of Segment_Length seconds named `<CAMERA_ID>_<YYYYMMDDTHHMMSS>.mkv` into `data/videos/`.
2. WHEN the source fails or ffmpeg exits, THE Recorder SHALL restart it with exponential backoff (1 s doubling to 60 s), and SHALL log each restart.
3. THE Recorder SHALL write a heartbeat file `data/live/<CAMERA_ID>.json` at least every Segment_Length seconds containing the last Segment name, its time, and the ffmpeg state.
4. THE Recorder SHALL write files only under `data/`.

### Requirement 8: Watch-mode ingest

**User Story:** As an Integrator, I want new Segments ingested automatically, in order, through the V1 transactional pipeline.

#### Acceptance Criteria

1. WHEN started with `--watch`, THE Watcher SHALL poll `data/videos/` and ingest each Closed_Segment exactly once, in start-time order per camera.
2. THE Watcher SHALL treat a Segment as closed when a later Segment of the same camera exists, or when its size has not changed for Segment_Length + 5 seconds.
3. THE Watcher SHALL ingest each Segment with the V1 per-video transaction, compensation and Src_Hash de-duplication unchanged.
4. WHEN the Watcher starts, THE Watcher SHALL run `reconcile()` first.
5. WHEN a Segment fails to ingest, THE Watcher SHALL log it, continue with the next Segment, and retry the failed Segment at most twice.

### Requirement 9: Detection at Track_Rate

**User Story:** As an Integrator, I want detection frequent enough for tracking without changing V1 search.

#### Acceptance Criteria

1. THE Detector SHALL return detections at or above `det_low_conf` (default 0.1), each flagged high when at or above `det_conf`.
2. THE Ingest_Pipeline SHALL run the Detector and Tracker on frames at Track_Rate whenever the Motion_Gate passes the frame or any Track is active.
3. THE Ingest_Pipeline SHALL create `frames` rows, Thumbnails, frame Embeddings and crop Embeddings only for frames at the Sample_Rate cadence, and crop Embeddings only for high detections, so V1 search results are unchanged for the same footage.
4. Config validation SHALL accept every combination of in-range values (V1 Property 11); THE Ingest_Pipeline SHALL use `max(track_rate, sample_rate)` as the tracking rate and `min(det_low_conf, det_conf)` as the low-detection floor.

### Requirement 10: Multi-object tracking

**User Story:** As an Integrator, I want each person and vehicle followed across frames and Segments, so rules can reason about entry, dwell and direction.

#### Acceptance Criteria

1. THE Tracker SHALL associate detections to Tracks per class, matching high detections first and low detections second against predicted boxes, using IoU at or above `track_match_iou`.
2. THE Tracker SHALL confirm a new Track only after `track_min_hits` matched frames, and SHALL end a Track not matched for `track_buffer_s` seconds of video time.
3. FOR ALL frames, no Track SHALL be assigned more than one detection and no detection SHALL be assigned to more than one Track (invariant).
4. WHEN a Scene_Cut occurs, THE Tracker SHALL end all Tracks of that camera.
5. THE Tracker state SHALL carry over between consecutive Segments of the same camera WHEN the next Segment starts within `segment_gap_reset_s` of the previous Segment's end; otherwise it SHALL be reset.
6. IF a Segment's ingest is rolled back, THEN the Tracker, Zone and Rule state SHALL be restored to the state before that Segment.
7. THE Ingest_Pipeline SHALL store each confirmed Track and its observations, and SHALL set `track_id` on crop vectors whose detection belongs to a Track.

### Requirement 11: Schema migration

**User Story:** As an Integrator, I want an existing V1 database upgraded in place.

#### Acceptance Criteria

1. WHEN the Metadata_Store opens a database with schema version 1, THE Metadata_Store SHALL migrate it to version 2 in one transaction, adding the `tracks` and `track_obs` tables and the nullable `vectors.track_id` column, and leave all V1 rows and the Vector_Index unchanged. The alert tables are added by a later migration (version 3) with the rule engine. A database with a newer schema version SHALL be refused, not downgraded.
2. `reconcile()` SHALL also delete alert media files that no `alerts` row references.

### Requirement 12: Zones, rules and the alert outbox

**User Story:** As a business owner, I want precise, explainable alerts such as "person in the yard at night".

#### Acceptance Criteria

1. A Track SHALL be inside a Zone when its Anchor_Point is inside the Zone polygon (points on the boundary count as inside).
2. A `cross` Trigger SHALL fire when consecutive Anchor_Points of a Track lie on opposite sides of the line segment, the movement intersects the segment, and the direction matches the Rule's direction.
3. A Rule SHALL fire only for confirmed Tracks of its classes, inside its Schedule (local time, DST-safe, wrap past midnight), after at least `debounce` consecutive qualifying observations, and not within `cooldown_s` of its previous Alert on the same camera.
4. A Rule SHALL fire at most once per Track.
5. WHEN a Rule fires, THE Ingest_Pipeline SHALL write an `alerts` row with state `pending` and a snapshot JPEG (frame with box and Zone drawn) within the Segment's transaction; on rollback both SHALL be removed.
6. IF `config/rules.yaml` is invalid or references an unknown camera or Zone, THEN loading SHALL fail naming each bad entry.

### Requirement 13: Alert media

**User Story:** As a business owner, I want a short clip of what happened.

#### Acceptance Criteria

1. THE Notifier SHALL cut a clip from `alert_clip_pre_s` (default 3) before to `alert_clip_post_s` (default 5) after the trigger, encoded H.264, at most 640 px wide, no audio, `faststart`, under `data/alerts/`.
2. WHEN the clip window extends into the next Segment, THE Notifier SHALL join the Segments; IF the next Segment does not exist within Segment_Length + 15 seconds, THEN the clip SHALL end at the last available Segment.
3. IF clip creation fails, THEN the Alert SHALL still be delivered with its snapshot and the failure SHALL be noted in the caption.

### Requirement 14: Telegram delivery and commands

**User Story:** As a business owner, I want alerts on my phone and simple commands, with nobody else able to use the bot.

#### Acceptance Criteria

1. THE Notifier SHALL read the bot token from `NAB_TELEGRAM_TOKEN` and Allowed_Chats from `NAB_TELEGRAM_CHATS`, and SHALL refuse to start if either is missing; the token SHALL never be logged.
2. FOR each `pending` Alert, THE Notifier SHALL send the snapshot with a caption (camera label, rule name, local time, class) to every Allowed_Chat, set state `photo_sent`, then send the clip and set state `sent`.
3. WHEN Telegram returns HTTP 429, THE Notifier SHALL wait `retry_after` seconds; on other failures it SHALL retry with backoff, and after `alert_max_attempts` failures set state `failed`.
4. THE Notifier SHALL deliver each Alert at most once per chat across restarts, by persisting the Telegram message ID per Alert and chat before moving to the next step.
5. THE Notifier SHALL answer, only for Allowed_Chats and without any language model: `/status` (cameras, last Segment age, pending Alerts), `/snapshot <camera>`, `/alerts` (today's Alerts), `/mute <rule> <minutes>`, `/unmute <rule>`; messages from other chats SHALL be ignored and logged by chat ID.
6. Each Alert message SHALL carry "Real" / "False alarm" buttons; a press SHALL store the feedback on the Alert.
7. WHILE a Rule is muted, Alerts it produces SHALL be stored with state `suppressed` and not sent.
8. THE Notifier SHALL be the only NAB_Sentry process that opens non-loopback connections, and only to `api.telegram.org` over HTTPS.

### Requirement 15: Health, disk and retention

**User Story:** As a business owner, I want to know when a camera stops working, and I want the disk never to fill up.

#### Acceptance Criteria

1. WHEN no new Segment of a camera has appeared for `camera_offline_factor` × Segment_Length seconds, THE Notifier SHALL create a `camera_offline` Alert once, and a `camera_recovered` Alert when Segments resume.
2. WHEN free space on the data volume is below `disk_low_gb`, THE Notifier SHALL create a `disk_low` Alert at most once per 6 hours.
3. THE Watcher SHALL delete Segments older than `retention_days` together with their rows, thumbnails, playback files and vectors, keeping the Metadata_Store and Vector_Index consistent, and SHALL keep alert media for `alert_retention_days`.

### Requirement 16: Supervisor

**User Story:** As an Integrator, I want one command that starts and keeps the live system running.

#### Acceptance Criteria

1. `run_live.ps1` SHALL start one Recorder per camera, the Watcher, the Notifier and the API_Server, and restart any that exits, logging each restart under `data/logs/`.

### Requirement 17: Acceptance tests

**User Story:** As the product owner, I want Step 2 judged on measured precision, recall and latency.

#### Acceptance Criteria

1. THE NAB_Sentry SHALL provide `scripts/replay_camera.py VIDEO CAMERA_ID` that feeds a file through ffmpeg at real time (`-re`) into Segments as if it were a live camera.
2. WHEN the Compilation_Video is replayed through the full chain, THE run SHALL complete without errors and report Alert_Latency (median, p95), CPU and peak RAM at peak load.
3. WHEN Continuous_Footage with a Ground_Truth_File is replayed with 2 to 3 Rules, THE run SHALL report Rule precision and recall against the labelled Incidents and Alert_Latency; the targets are recall 1.0 for `enter` rules, median Alert_Latency under 25 seconds with 10-second Segments, and at most one false Alert per Rule per hour of idle footage.

### Requirement 18: Isolation and write locations

**User Story:** As an Integrator, I want V1's offline guarantees kept everywhere except the Notifier.

#### Acceptance Criteria

1. The V1 offline isolation and write-location tests SHALL continue to pass for ingest (including watch mode) and the API_Server.
2. A test SHALL show that the Notifier, using a fake transport, contacts only `api.telegram.org` and writes only under `data/`.
3. All new processes SHALL write only under `data/`, `models/` and read `config/`.
