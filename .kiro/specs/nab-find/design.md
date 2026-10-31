# NAB Find - Design

## Overview

```
┌──────────────────────── NAB Find.exe (one process) ────────────────────────┐
│                                                                            │
│  Window (pywebview → Edge WebView2)        Local server (FastAPI/uvicorn)  │
│  ┌──────────────────────────────┐  HTTP   ┌─────────────────────────────┐  │
│  │ web/ index.html app.js css   │ ──────► │ server.py  127.0.0.1:random │  │
│  │ Home · New search · Case     │ token   │ /api/...  media streaming   │  │
│  └──────────────────────────────┘         └──────┬──────────────────────┘  │
│        ▲ native file dialogs, drag-drop paths    │                         │
│        └─ js_api bridge (pywebview)              ▼                         │
│                                   ┌───────────────────────────────┐        │
│                                   │ Library: cases, settings      │        │
│                                   │ Worker thread: Processor      │        │
│                                   │   media.FrameReader ◄─ ffmpeg │        │
│                                   │   models (OpenVINO/ONNX RT)   │        │
│                                   │ Search · Visits · Find again  │        │
│                                   │ AI check ──► Gemini API (opt) │        │
│                                   └───────────────────────────────┘        │
│                           Case folder: case.db (SQLite, vectors included), │
│                           thumbs/, previews/                               │
└────────────────────────────────────────────────────────────────────────────┘
```

One Windows process. The UI is HTML/JS in a native window (WebView2 ships with Windows 10/11),
served by a local FastAPI app bound to 127.0.0.1 on a random port with a per-launch token. A
single worker thread processes cases; searches run on the server's threads against the same case
store.

### Why this shape

| Decision | Chosen | Why / alternatives |
|---|---|---|
| UI | HTML/JS in pywebview | Modern look with plain CSS, no build step, video playback with seeking via HTTP Range. Electron/Tauri add a second runtime; Qt is heavier and harder to make look friendly. |
| Backend | Python, same language as V2 | Reuses the V2 detector post-processing, tracker, vector index and tagger. The time goes to models and decoding (native code), so Python glue is not the bottleneck. |
| Decoding | FFmpeg subprocess, scaled frames on a pipe | Measured 42x real time decoding every frame vs 8x for OpenCV's capture as V2 uses it; keyframe-only reads 254x (bench/decode_bench.py). Also reads DVR formats OpenCV cannot. |
| Models | OpenVINO IR (fp16-stored weights) on the Intel GPU when it passes a self-test, else OpenVINO on the CPU | CLIP image tower 13 ms/image on the Iris Xe GPU vs ~30-75 ms on CPU (ONNX RT) vs ~127 ms in V2 (PyTorch); YOLO 384x640 7.8 ms GPU vs 18-31 ms CPU (bench/embed_bench.py). No PyTorch in the app. On the CPU, OpenVINO with the LATENCY hint measured 27 ms/image vs 38 for ONNX Runtime, and the fp16-stored IR halves the model download (590 -> 299 MB); ONNX Runtime was dropped from the build. |
| int8 CLIP | Rejected | Cosine to fp32 only 0.95; not worth the accuracy. |
| Embedder | Same as V2: OpenCLIP ViT-B/32 DataComp-XL, exported (tools/export_clip.py; parity cosine 1.0000 images and text) | Best speed/quality found in V2 Step 4. |
| Store | Per-case SQLite with the embeddings in it; exact NumPy dot-product search over an in-memory copy | A case is a folder: easy delete. With the vectors in SQLite a crash cannot leave an index out of step, and a case's tens of thousands of vectors search in milliseconds without FAISS. |
| Cloud VLM | Gemini API over REST (httpx), default `gemini-3.5-flash-lite` | Cheap per image, JSON output, no extra SDK. Only small frames of top moments are sent, after consent. |
| Video files | Referenced in place, never copied | Footage can be tens of GB. Missing files are reported per video. |

## Isolation from V2 (Requirement 12)

- Code: `apps/desktop/nab_find/` (package), `apps/desktop/web/` (UI), `apps/desktop/tests/` (own
  `pytest.ini`; the root `pytest.ini` collects only `tests/`), `apps/desktop/tools/`,
  `apps/desktop/bench/`, `apps/desktop/packaging/`.
- Imports from V2, read-only: `nab_sentry.ingest.detector` (`letterbox`, `postprocess`,
  `Detection`), `nab_sentry.ingest.motion` (`downscale_gray`, `changed_fraction`),
  `nab_sentry.track.bytetrack.ByteTracker`, `nab_sentry.describe.attributes` (`AttributeTagger`,
  `phrase`). If V2 ever needs to change one
  of these, the change is made on the V2 branch with V2's suites, never from here.
- `apps/desktop/tools/check_isolation.py` fails if the branch changes any V2 path relative to
  its merge base with `v2-step1-measure`; run before every push with the V2 fast suite.
- Environment `venv-desktop` (pinned in `apps/desktop/requirements.txt`); models in
  `models/desktop/` (gitignored); data in `%LOCALAPPDATA%\NAB Find` (development:
  `data/desktop/`).

## Components

### `media.py` - reading footage
- `probe(path) -> VideoInfo` (duration, fps, size, codec, keyframe spacing estimate from the first
  60 s) via FFmpeg.
- `FrameReader(path, start_s, end_s, mode, width)` yields `(offset_s, bgr)`: `mode="keyframes"`
  (`-skip_frame nokey`), or `mode="rate"` with `fps=N` filter; frames scaled to `width`
  (default min(source, 1280)) in FFmpeg; `-ss` input seeking for ranges. Pipe of raw BGR; the
  reader kills FFmpeg on close.
- `cut_clip(path, start_s, dur_s, dst, overlay_text)` H.264 MP4 for previews and exports;
  previews cached per moment.

### `cliptime.py` - clock times
- File-name patterns (first match wins): V2 `CAM_YYYYMMDDTHHMMSS`, `YYYYMMDD_HHMMSS`,
  `YYYY-MM-DD HH-MM-SS`, Dahua `..._YYYYMMDDHHMMSS_YYYYMMDDHHMMSS`, Hikvision `chNN_YYYYMMDDHHMMSS`,
  `YYYYMMDDHHMMSS`. Fallback: modified time minus duration ("estimated").
- Camera name from the prefix before the timestamp, else the parent folder name.

### `models.py` - inference
- `pick_device()`: OpenVINO GPU if listed and a one-image self-test matches the CPU result
  (cosine >= 0.999), else CPU. The choice and timings are logged and shown in Settings.
- `ClipImage`, `ClipText`, `Yolo` wrappers with the same interface on either runtime; CLIP image
  batches are padded to a fixed batch of 8 on the GPU (no recompiles).

### `store.py` - case storage
Folder per case: `case.db`, `thumbs/`, `previews/`.

```
meta(key, value)                             -- name, created, range, status, quick_done
videos(video_id, path, camera, start_ts, start_estimated, duration_s, fps, width, height,
       keyframe_gap_s, status, error)
spans(video_id, pass, start_s, end_s)        -- work done (resume) per pass
motion(video_id, start_s, end_s)             -- activity stretches from the quick pass
frames(frame_id, video_id, offset_s, thumb)  -- quick-pass frames
visits(visit_id, video_id, cls, t0, t1, n_obs, max_speed, path_json, tags_json, thumb)
vectors(vector_id, kind['frame'|'crop'], video_id, offset_s, frame_id, visit_id, x1,y1,x2,y2, thumb, vec)
ai_checks(id, query, model, created, result_json, tokens_in, tokens_out)
```

Each processing chunk is one SQLite transaction (frames, vectors, visits, activity, the span done);
only after it commits are its vectors appended to the in-memory search table, which is rebuilt from
SQLite when a case opens.

### `processor.py` - the two passes
Quick pass per video, in range order:
1. `FrameReader` in keyframe mode if keyframe spacing <= 4 s, else rate mode at 0.5 fps.
2. For each frame: motion fraction against the previous frame (V2 `downscale_gray` /
   `changed_fraction`, 320 px) and the detector. Activity (`ActivityRule`): motion >= 0.02 at once;
   without motion, a person or a vehicle not seen in that place in the last 5 frames must show in two
   frames in a row (measured: one-frame noise detections and flickering parked-car boxes otherwise
   sent 42% of a quiet street hour to the detail pass; with the rule 2%). Keep the frame when active
   or 30 s passed since the last kept frame.
3. Kept frames: thumbnail (320 px JPEG), batch CLIP, add to index. Moving stretches recorded
   between the previous and next kept-with-motion frame.

Detail pass per video, over the moving stretches merged and padded by 3 s:
1. `FrameReader` rate mode, 2 fps, from each stretch start.
2. YOLO (V2 post-processing, conf 0.35, low 0.1), V2 `ByteTracker` (min_hits 2 at 2 fps).
3. Per confirmed track: keep the largest high-confidence box in every 3 s window; embed those
   crops in batches; visit row updated with times, path (centre points every 1 s), speed in
   frame-widths per second, and attribute tags (V2 `AttributeTagger` over the visit's crop vectors).

Work is cut into chunks of about 60 s of footage; after each chunk the store commits and progress
is published. Pause/cancel are checked between frames.

### Progress and time left (Requirement 7)
- Work units: seconds of footage. Quick pass total = range length; detail pass total = length of
  the moving stretches (known only after the quick pass; estimated before that as 40% of the
  range, a deliberately pessimistic guess shown as "about").
- Speed: exponential moving average of footage-seconds per wall-second per pass (alpha 0.3 per
  chunk), seeded by a 5-second calibration on the first chunk; clamped so the estimate changes by
  at most 2x per update after the first 3 chunks.
- Published as `{pass, percent, done_s, total_s, quick_eta_s, full_eta_s, speed_x}`; the UI polls
  every second.
- `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)` while working; process priority
  BELOW_NORMAL.

### `search.py`
- Text: query vector (prompt-ensembled, as V2) against the case index; top 400 hits; restricted to
  the time window / class / camera via an id allow-list.
- Moments: hits on the same video within 6 s merge; score = best similarity plus a small bonus for
  supporting hits (V2 `event_score` idea); each moment keeps its best frame or crop for the
  thumbnail. Score shown as a bar, normalised against the query's own top score.
- Visits: SQL over `visits` with filters and sorts.
- Find again: mean of the source visit's crop vectors; search crop vectors excluding that visit;
  group by visit; top 30.

### `aicheck.py`
- Frames per moment: 2 context frames (whole frame at 384 px) at -2 s and +2 s and up to 4 crops of
  the main subject (padded 30%, 384 px max side) across the moment; all JPEG.
  At <= 384 px per side each image costs about 258 tokens (Gemini docs), so 20 moments x 6 images
  is about 31k input tokens.
- One request per batch of 5 moments, 3 requests in parallel; JSON schema output
  `[{moment, match: yes|no|unsure, confidence, description}]`; timeout 60 s; one retry on 429/5xx.
- Cost estimate shown before consent from the token rule and the model's price in Settings.
- Endpoint `POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`,
  key in the `x-goog-api-key` header.
- Commercial note: shipping with a key inside the app is not safe (it can be extracted); the paid
  product needs a small proxy service that holds the key and counts credits. The MVP uses the
  user's own key.

### `server.py` (all JSON, all need the token header except static files)
```
GET  /api/cases                         POST /api/cases {name}
GET  /api/cases/{id}                    PATCH / DELETE /api/cases/{id}
POST /api/cases/{id}/videos {paths}     PATCH /api/cases/{id}/videos/{vid} {start, camera}
PUT  /api/cases/{id}/range {start,end}  POST /api/cases/{id}/start|pause|resume|cancel
GET  /api/cases/{id}/progress
GET  /api/cases/{id}/search?q=&from=&to=&cls=&camera=
GET  /api/cases/{id}/visits?sort=&from=&to=&cls=
GET  /api/cases/{id}/similar/{visit_id}
POST /api/cases/{id}/aicheck {q, moments}   GET /api/cases/{id}/aicheck/{job}
GET  /api/cases/{id}/preview/{video}/{t}   (MP4, cached; Range requests)
POST /api/cases/{id}/export {video, start, end, folder}
GET  /api/cases/{id}/thumb/{frame}
GET/PUT /api/settings
```

### UI (apps/desktop/web)
Three screens, one main action each:
1. **Home** - "New search" button; recent cases as cards with status.
2. **New search** (wizard): (a) drop videos (b) check times and cameras (c) optional range ->
   "Start".
3. **Case** - header with progress and time left; search box with example chips; results grid;
   player drawer with Find again, Export clip, AI check. Tabs: Search / Visits.

Plain words: "Getting your video ready - you can already search the first 40 minutes", "About 12
minutes left".

## Error handling
- Unreadable file -> video status `error` with a plain reason; other videos continue.
- Model device failure mid-run (GPU driver) -> rebuild on CPU, redo the chunk, log once.
- Disk full -> pause the case, banner with free space needed.
- AI check errors -> per-batch message; local results unaffected.

## Testing
- Unit (fast, venv-desktop): tokenizer framing, time-name patterns, ETA estimator, moment merge,
  store commit/resume rules, AI check request building and response parsing (no network), API
  routes with a fake processor.
- Slow (real models and FFmpeg): end-to-end on the V2 test video and generated scene: process,
  search queries from `eval/queries_compilation.yaml` (hit@5), visits, find again, preview/export
  files play, resume after a kill.
- Bench: `bench/` scripts for decoding, models, and end-to-end throughput.

## Packaging
PyInstaller one-folder build (`packaging/nab_find.spec`) with the models (OpenVINO IR with fp16
weights to halve the size), FFmpeg and the web folder; Inno Setup script for the installer
(Start-menu entry, uninstall keeps cases unless asked). Code signing is a release step that
needs a certificate.
