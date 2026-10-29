# NAB Find - Requirements (desktop MVP)

NAB Find is a Windows desktop app for one job: *something happened, find it in my recording.* A
shop owner or a family adds the video files exported from their DVR, says roughly when it
happened, waits a few minutes, and asks in plain words. The app shows the moments that match,
with clock times, plays them, finds the same person again, and exports a short clip.

It is a separate product from NAB Sentry V2 (live, enterprise, installed by us). It lives in the
same repository and reuses parts of the V2 engine read-only (see Requirement 12).

"NAB Find" is a working name.

## Glossary

- **Case**: one incident search: a name, the videos added to it, an optional time range, and
  everything computed from them. Stored in its own folder; deleting a case deletes the folder.
- **Video**: a file added to a case. The app never modifies, moves or copies it.
- **Range**: the clock-time window the user cares about (e.g. 14:00-17:00 on 28 Oct).
- **Quick pass**: the first processing pass: whole-frame embeddings about every 2-4 s of the
  range, plus motion; makes the whole range searchable quickly.
- **Detail pass**: the second pass over the moving parts only: detection, tracking, one
  embedding per person/vehicle every few seconds; finds small or distant people and powers
  "find again" and visits.
- **Moment**: one search result: a video, a clock time span, a thumbnail and a score.
- **Visit**: one tracked person or vehicle from first to last sighting.
- **AI check**: the optional cloud step: a few frames of the top moments go to the Gemini API,
  which says whether each matches the question and describes it.
- **Target PC**: a Windows 10/11 x64 PC with 8 GB RAM, any CPU with AVX2, no discrete GPU.

## Requirement 1: Install and start

1. One installer sets up everything (runtime, models, FFmpeg); no Python, no command line, no
   settings needed to reach the first search.
2. The app runs offline. Only the AI check (Req 9) and an optional update check use the network.
3. Only one copy runs at a time; starting it again brings the existing window forward.
4. A crash or power loss never corrupts a case: on the next start, processing resumes where it
   stopped.

## Requirement 2: Cases

1. The home screen lists cases (name, date, videos, status) and has one big "New search" button.
2. A case can be renamed, reopened, and deleted (with confirmation; the original videos are never
   touched).
3. Cases are stored under the user's local app data folder; a setting can move it.

## Requirement 3: Adding videos

1. Videos are added by drag-and-drop or a file picker; folders add every video file inside.
2. Supported: anything the bundled FFmpeg can read, including the common DVR exports (MP4, MKV,
   AVI, ASF, raw H.264/H.265 `.264/.265/.h264`, Dahua `.dav`). Files that cannot be read are
   listed with a plain reason ("this file could not be opened - it may be an encrypted DVR
   format; export as MP4 from the recorder"); the others still work.
3. Each video's start clock time is read from its file name when it follows a known pattern,
   otherwise from its modified time minus its duration (marked "estimated"); the user can correct
   it, and correcting one video can shift its siblings by the same amount.
4. Videos are grouped by camera (from the file name or the user) and shown in time order with gaps
   and overlaps visible.

## Requirement 4: Range

1. The user may give a range by clock time. Without one, the whole of every video is processed.
2. Only footage inside the range is processed (Reqs 5, 6).

## Requirement 5: Quick pass

1. Reads the range at about one frame per 2-4 s (keyframes when they are at most 4 s apart,
   otherwise sampled frames), scaled down for speed.
2. Stores a thumbnail and a CLIP embedding for every frame that differs from the last stored one
   by motion, and at least one every 30 s.
3. Records which stretches have motion; the detail pass covers those (Req 6).
4. Search works on whatever is done so far, at any time.

## Requirement 6: Detail pass

1. Covers the moving stretches found by the quick pass, padded by a few seconds, at 2 frames/s.
2. Detects people and vehicles (the V2 detector), tracks them (the V2 tracker) and stores, per
   Visit: class, first/last time, path summary, top speed, and up to one crop embedding every
   3 s (the largest, clearest sighting in that window), plus attribute tags (V2 tagger).
3. Its results join the same search as soon as each stretch is done.

## Requirement 7: Progress and time left

1. While processing, the case shows: overall percent, the current pass, footage done vs total,
   and an estimate of the time left for "quick search ready" and for "fully done".
2. Estimates come from measured speed on this PC (rolling average), start from a benchmark-based
   guess, and never show a nonsensical value (negative, or jumping by more than 2x per update
   once warmed up).
3. Processing continues while the window is minimised or another case is open; the PC does not go
   to sleep while it runs; it runs below normal priority so the PC stays usable.
4. The user can pause, resume and cancel. A notice (in the app, with a sound) says when the quick
   pass and the whole case are done.

## Requirement 8: Search

1. A text box takes a plain description ("man in a red jacket", "white van"). Results appear in
   under 2 s on the Target PC for a case of 8 hours.
2. Results are Moments: hits close in time on the same video are merged; each shows a thumbnail,
   clock time, camera, and a relevance bar. Sorting by relevance or by time.
3. Filters: time window within the range, people / vehicles / anything, camera.
4. "Visits" view: every tracked person or vehicle, sortable by time and by how long they stayed;
   "after hours" is a time filter.
5. Find again: from any Moment with a person or vehicle, show other Visits that look like it,
   most similar first, with an honest note that it compares appearance, not identity.

## Requirement 9: AI check (optional, cloud)

1. Off until the user adds a Gemini API key in Settings. Without it, everything else works.
2. On a result list, "AI check" sends, for each of the top N Moments (default 20, max 50), a few
   small frames around the moment (context frames and zoomed crops of the main person/vehicle)
   and the question. Before sending, the app shows how many moments and images will leave the PC
   and an estimated cost, and sends nothing without consent.
3. Each checked Moment gets: match (yes / no / unsure), a one-sentence description, and moves up
   or down; results stream in as they arrive. Failures (no network, quota, bad key) are shown
   plainly; the local results stay.
4. Whole videos are never uploaded. The key is stored encrypted for the Windows user (DPAPI) and
   never logged.

## Requirement 10: Watch and export

1. Clicking a Moment plays it in the app from a few seconds before, with the clock time shown;
   the player can step forward/back and continue into the next video of the same camera.
2. Any format plays: the app cuts a short H.264 preview with FFmpeg when needed.
3. "Export clip" saves an MP4 (default: 10 s before to 20 s after, adjustable) with the camera and
   clock time burned in, to a folder the user picks; "Open folder" afterwards.

## Requirement 11: Simplicity and quality

1. Every screen has one obvious next action; no technical words (no "embedding", "vector",
   "FPS") in the main flow.
2. Plain-language errors with a next step; technical details only in a "details" fold and the
   log file.
3. Works at 1366x768 and up; keyboard accessible; readable contrast; follows the Windows
   light/dark setting.

## Requirement 12: Isolation from NAB Sentry V2

1. No file under `nab_sentry/`, `tests/`, `scripts/`, `config/`, `eval/` or the V2 run scripts is
   changed by this product's work. NAB Find imports V2 modules read-only.
2. NAB Find has its own package (`apps/desktop/nab_find`), tests (`apps/desktop/tests`, not
   collected by the V2 suites), virtual environment (`venv-desktop`), models
   (`models/desktop`) and data folder.
3. Before every push, the V2 fast suite still passes and a check confirms no V2 file changed.

## Requirement 13: Performance targets (Target PC class, measured on the development laptop)

1. Quick pass: at least 60x real time on quiet footage and 20x on busy footage.
2. Detail pass: at least 8x real time on busy footage without a usable GPU, 15x with the Intel GPU.
3. Peak memory under 2.5 GB for an 8-hour case.
4. These are targets; the benchmark (task 1) reports the measured values, which go in the README.
