# NAB Find (desktop MVP)

A Windows app for one job: **something happened - find it in my recording.** Add the video files
exported from a DVR, say roughly when it happened, and search in plain words ("man in a red jacket",
"white van"). Everything runs on the user's PC; an optional AI check sends a few small pictures of the
top results to Google Gemini.

Separate product from NAB Sentry V2 (live, enterprise). It lives in this repository and imports a few
V2 modules read-only (detector post-processing, motion measure, tracker, appearance tags); it never
changes a V2 file (`tools/check_isolation.py`). Spec: `.kiro/specs/nab-find/`.

## Using it (what a shop owner sees)

1. **New search** -> drag the video files or the DVR export folder onto the window (or *Choose
   files* / *Choose a folder*). Each video's camera and start time are read from its file name; if a
   time had to be guessed from the file date the app says so, and fixing one video can move the
   other videos of that camera by the same amount.
2. **When did it happen?** *Go through everything* or *Only a time window*. The app shows how much
   video that is and a time estimate for this PC.
3. **Start.** A progress card shows the percent, how much is already searchable, and the time left
   until search is ready and until everything is done. Processing keeps running while the window is
   minimised, the PC does not go to sleep meanwhile, and a sound plays when it is ready. Pause, stop
   and continue at any time; a crash or power cut resumes where it stopped.
4. **Search** in your own words. Results are moments with clock time, camera, a picture and a
   "how well it matches" bar; filter by people / vehicles, camera, and time. Click one to watch it
   (starts a few seconds before), step 20 s earlier/later, **Save a clip** (MP4 with camera and time
   written on it, into *Videos\NAB Find exports*), or **Find this one again** (other moments where a
   person or vehicle looks the same - appearance only, not identity).
5. **People & vehicles** lists everyone tracked, sortable by time, by how long they stayed, or by how
   fast they moved, with a "stayed at least" filter.
6. **Check top results with AI** (optional): after showing how many pictures will be sent and the
   estimated cost, sends small frames of the top 20 moments to Gemini, which marks each as match /
   not it / unsure with a one-line description and moves matches first. Needs a Gemini API key in
   Settings (stored encrypted with Windows DPAPI for that Windows user).

## Measured (development laptop: i5-1145G7 with Iris Xe graphics, 16 GB, 2026-10-29)

| Footage | Device | Whole case | Notes |
|---|---|---|---|
| 1 h, mostly quiet (54 min empty street with parked cars + 6.4 min busy) | Iris Xe GPU, AC, the built app (`--selftest`) | **72 s (50x real time)** | same work as the next row |
| Same hour | Iris Xe GPU, **on battery** | **2.5 min (24x real time)** | quick pass 37x; detail pass covered 7.5 min (13%): the busy 6.4 min fully, 65 s of the quiet part |
| Same hour, before the activity rule fix | GPU, battery | 3.3 min (18x) | detail pass covered 42% (noise detections) |
| 6.4 min, busy every second (worst case) | GPU, AC | 32 s (12x) | V2 ingest: about 1.6x on busy footage |
| 6.4 min, busy | CPU only, AC | 59 s (6.5x) | |
| 45 min of 10-s files, a person in view nearly all the time | GPU, AC | 6.1 min (7.3x) | 269 files: about 0.3 s start-up per file per pass |

Search quality (the 20 V2 evaluation queries, right moment = first result / in top 3 / in top 5):

| Case | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|
| 6.4-min test video (GPU) | 0.75 | 1.00 | 1.00 | 0.867 |
| 6.4-min test video (CPU) | 0.80 | 1.00 | 1.00 | 0.892 |
| Same 6.4 min inside the 1-hour quiet video | 0.75 | 0.95 | 0.95 | 0.848 |
| V2 (same embedder, PyTorch) for comparison | 0.80 | | | 0.879 |

AI check (live, `gemini-3.5-flash-lite`, 4 queries): where the right moment was 2nd locally it was
moved to 1st twice; each check of 3-7 moments cost $0.001-0.01 and took 12-50 s. Gemini's default
image resolution used about 1,100 tokens per picture; the app uses the low setting (about 280) with the
same verdicts on the test moments.

Per-stage, ms per image: CLIP image 11.5 GPU / 27 CPU (OpenVINO; ONNX Runtime 38, PyTorch in V2 ~127);
YOLO 384x640 7.8 GPU / 14 CPU. Reading video with FFmpeg: 42x real time decoding every frame (V2's
OpenCV capture: 8x), 254x keyframes only. Benchmarks: `bench/`.

Honest limits:
- Battery and power plan change speed a lot on laptops; numbers above say which.
- Many tiny files (seconds each) add about 0.3 s per file per pass.
- The test footage is a compilation of short clips plus generated quiet video; real DVR footage from
  several brands is still needed (task U.1).
- "Find this one again" compares appearance; similar clothes look alike.
- The AI check can be wrong; the UI always says to watch the moment.

## Developer guide

```
py -3.12 -m venv venv-desktop
venv-desktop\Scripts\python.exe -m pip install -r apps\desktop\requirements.txt
```

Models (in `models\desktop`, gitignored):
```
venv\Scripts\python.exe apps\desktop\tools\export_clip.py        # V2 dev venv (torch): CLIP -> ONNX + parity check
copy models\yolo11n_384x640.onnx models\desktop\
venv-desktop\Scripts\python.exe apps\desktop\tools\convert_ir.py  # ONNX -> OpenVINO IR (fp16-stored) + parity check
```

Run, test, check:
```
cd apps\desktop
..\..\venv-desktop\Scripts\python.exe -m nab_find.app --dev        # the app, data in data\desktop
..\..\venv-desktop\Scripts\python.exe -m pytest -q                 # fast tests (fakes, no models needed)
..\..\venv-desktop\Scripts\python.exe -m pytest -q -m slow         # real models + FFmpeg on the V2 test video (~2 min)
cd ..\..
venv-desktop\Scripts\python.exe apps\desktop\tools\ui_check.py data\videos\CAM-SEC01_20261007T230010.mp4   # drives the UI in Edge, screenshots
venv-desktop\Scripts\python.exe apps\desktop\bench\e2e_bench.py VIDEO_OR_FOLDER [--device CPU]
venv-desktop\Scripts\python.exe apps\desktop\bench\search_eval.py CASE_FOLDER
venv-desktop\Scripts\python.exe apps\desktop\bench\aicheck_live.py CASE_FOLDER "query"
venv-desktop\Scripts\python.exe apps\desktop\tools\check_isolation.py   # no V2 file changed
```
(`ui_check.py` needs `pip install playwright` in venv-desktop and Microsoft Edge.)

Build:
```
venv-desktop\Scripts\pip install pyinstaller==6.22.3
venv-desktop\Scripts\pyinstaller.exe apps\desktop\packaging\nab_find.spec --noconfirm --distpath apps\desktop\dist --workpath apps\desktop\build
"apps\desktop\dist\NAB Find\NAB Find.exe" --data %TEMP%\nf --selftest data\videos\CAM-SEC01_20261007T230010.mp4   # writes logs\selftest.json
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" apps\desktop\packaging\nab_find.iss                                 # -> dist\NAB-Find-Setup-0.1.0.exe
```
The one-folder build is 686 MB (models 299 MB, OpenCV 112 MB, OpenVINO 110 MB, FFmpeg 84 MB).

The Gemini key: Settings in the app, or `GEMINI_API_KEY` / `NAB_FIND_GEMINI_KEY` in the environment.
Never commit it; `data/` is gitignored. For a paid product the key must not ship inside the app (it
can be extracted): put a small service in front that holds the key and counts credits.

## Layout

```
nab_find/   app.py (window, js_api, single instance, --selftest)  server.py (local API)
            media.py (FFmpeg reader, clips)  cliptime.py (times from file names)  library.py (add videos)
            models.py (OpenVINO, GPU self-test)  embed.py + clip_tokenizer.py (ONNX/PyTorch-free CLIP)
            store.py (case folder, SQLite)  processor.py (quick + detail passes)  progress.py (time left)
            search.py (moments, visits, find again)  aicheck.py (Gemini)  settings.py (DPAPI key)
web/        index.html, styles.css, app.js (no build step)
tests/      fast tests; bench/ speed and quality; tools/ export, convert, UI check, isolation check
packaging/  PyInstaller spec, Inno Setup script, icon
```
