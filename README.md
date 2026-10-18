# NAB Sentry

Offline natural-language search over recorded CCTV footage. Videos are sampled, motion-gated, run through YOLO11n (ONNX) and OpenCLIP ViT-B/32 on the CPU, and indexed in SQLite + FAISS. A local web console at `http://127.0.0.1:<port>/` searches the index and plays matching clips. Nothing opens a network connection at runtime.

Target: Windows x64, Python 3.12.10, CPU only. All commands below are PowerShell, run from the workspace root.

## Setup

```powershell
py -3.12 -m venv venv
venv\Scripts\python.exe -m pip install torch==2.14.1+cpu torchvision==0.29.1+cpu --index-url https://download.pytorch.org/whl/cpu
venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` pins every package and includes the PyTorch CPU index as an extra index, so the second line alone also works. Install only `opencv-python` (never `opencv-python-headless` or `opencv-contrib-python`).

Data layout (created on first run):

```
data/
  videos/            source footage (+ optional <name>.json Sidecar_File)
  thumbs/            JPEG thumbnails
  playback/          browser-playable copies
  logs/              nab_sentry.log, query_log.jsonl
  eval/              evaluation reports
  nab_sentry.db      Metadata_Store (SQLite)
  vectors.faiss      Vector_Index
models/
  manifest.json      SHA-256 of every model file
  open_clip/ViT-B-32-laion2b_s34b_b79k/open_clip_pytorch_model.bin
  yolo11n.onnx
```

## Model provisioning

Run once on a machine with internet access:

```powershell
venv\Scripts\python.exe scripts\fetch_models.py
```

This downloads the OpenCLIP ViT-B/32 (`laion2b_s34b_b79k`) weights and `yolo11n.pt` (3 attempts each), exports `models\yolo11n.onnx`, smoke-loads both, and writes `models\manifest.json` last. Any failure exits 1, names the failing file, and leaves no manifest. Use `--models-dir <path>` to write elsewhere.

Then copy the whole `models\` folder to `models\` in the workspace on the offline demo machine. Every startup verifies the manifest hashes and exits 4 with the fetch command if a file is missing or altered.

YOLO11n (Ultralytics) is licensed AGPL-3.0. Review the licence before distributing it or offering it as a network service.

## Ingest

Put videos in `data\videos\`. Each video needs camera metadata from either a Sidecar_File (`<same name>.json` with `camera_id`, `label`, `start_time` in ISO 8601) or a file name like `CAM01_20250101T080000.mp4`. Times without a UTC offset are local time.

```powershell
venv\Scripts\python.exe scripts\ingest.py data\videos
```

Output: `ingested=N failed=M already_indexed=K vectors=V`. Already-ingested files (same content hash) are skipped.

- Frame-only ingest (no detector): `venv\Scripts\python.exe scripts\ingest.py data\videos --set enable_detector=false`
- Reconcile the store with the index after a crash: `venv\Scripts\python.exe scripts\ingest.py --repair`
- Any Config value can be overridden with `--set name=value` (repeatable).
- Exit codes: 0 success; 1 no videos found or zero vectors; 2 invalid Config; 4 model manifest check failed; 5 index unavailable (run `--repair`); 6 model load failure; 130 interrupted.

For a test clip with known ground truth:

```powershell
venv\Scripts\python.exe scripts\make_synthetic.py
```

It writes `CAM-SYN01_20250101T080000.mp4`, its Sidecar_File and `ground_truth.json` into `data\videos\`.

Command-line search (no server):

```powershell
venv\Scripts\python.exe scripts\search_cli.py "person near a red car" --camera CAM01 --start 2025-01-01T08:00:00 --end 2025-01-01T09:00:00 --limit 10
```

## Demo

```powershell
.\run_demo.ps1
```

The script:

1. Checks that `venv\Scripts\python.exe` exists (exits 1 otherwise).
2. If `data\nab_sentry.db` or `data\vectors.faiss` is missing, runs `scripts\ingest.py data\videos` and prints the ingested and failed counts. A failed ingest exits 1, names `data\videos\`, and starts no server.
3. Starts the server with `venv\Scripts\python.exe -m nab_sentry.api.app`.
4. Polls `/api/health` every 500 ms for up to 30 s. If the server exits, prints its error and exits with its code.
5. Prints `NAB Sentry console: http://127.0.0.1:<port>/` once both models are loaded.

If PowerShell blocks the script, run it with `powershell -ExecutionPolicy Bypass -File .\run_demo.ps1`.

To start the server directly (default port 8765):

```powershell
venv\Scripts\python.exe -m nab_sentry.api.app --set port=8765
```

Server exit codes: 2 invalid Config or non-loopback host; 3 port in use; 4 model manifest check failed; 5 index missing or inconsistent (run `scripts\ingest.py --repair`); 6 model load failure.

## Evaluation

Fill in `eval\queries.yaml` (15 to 20 queries) with real camera IDs and ground-truth time windows. The first two entries target the synthetic video. Then:

```powershell
venv\Scripts\python.exe scripts\evaluate.py
```

Use `--queries <file>` for another queries file. The run reports per-query precision@5, hit@1 and latency plus aggregates, and writes `data\eval\report-YYYYmmdd-HHMMSS.json` with every relevant Config value. It exits 1 if the queries file is missing, unparseable, has the wrong number of entries, or every query errors.

## Tests

```powershell
venv\Scripts\python.exe -m pytest -m "not slow"
venv\Scripts\python.exe -m pytest -m slow
```

The fast suite uses fake models and needs no `models\`. The slow suite runs the real models and needs a provisioned `models\` folder.

## Benchmarks

```powershell
venv\Scripts\python.exe scripts\benchmark.py data\videos\<video>.mp4 --search 200000
```

`VIDEO` times the per-stage pipeline and projects ingest time; `--search N` times search over N random vectors. Peak RSS is printed at the end. Record results here:

| Metric | Real video median | Real video p95 | Synthetic median | Synthetic p95 | Notes |
|---|---|---|---|---|---|
| Decode (ms/frame) | 3.24 | 3.96 | 0.29 | 0.42 | real: 1376x776 H.264 |
| Motion gate (frames/s) | 157.40 | 201.40 | 3,970.72 | 4,224.10 | real-time pass |
| Detector (ms/frame) | 103.57 | 117.57 | 45.73 | 52.91 | YOLO11n ONNX, CPU, 4 threads |
| Embedder batch 1 (ms/image) | 181.85 | 246.10 | | | OpenCLIP ViT-B/32 |
| Embedder batch 8 (ms/image) | 126.68 | 133.06 | 49.15 | 54.21 | OpenCLIP ViT-B/32, batched CPU |
| Thumbnail (ms/JPEG) | 13.42 | 17.46 | | | |
| Projected ingest (min per footage hour) | 26.4 | | ~2.3 | | real: **worst case**, see below; synthetic: 10 fps, pass fraction 0.367 |
| Search unfiltered, 200k vectors (ms) | 134.91 | 144.21 | 135.50 | 143.18 | < 1,500 ms target |
| Search filtered, 200k vectors (ms) | 191.08 | 204.22 | 99.14 | 187.15 | < 1,500 ms target |
| Peak RSS (MiB) | 1,804 | | 1,764 | | peak process RSS |
| Server startup to console URL (s) | | | | | not measured |

Real video: `CAM-SEC01_20261007T230010.mp4` (6.4 min, 1376x776, 29.66 fps), measured 2026-10-18 on a Dell Latitude 5420 (Intel Core i5-1145G7, 4 cores / 8 threads, 16 GB RAM), AC power, Windows "Balanced" power plan, default Config. Pass fraction 0.695 (269/387 sampled), 1.91 detections per passed frame.

This video is a curated compilation of back-to-back incidents, so the 0.695 pass fraction reflects continuous activity, not gate efficiency, and the 26.4 min/footage-hour projection is a **worst case** for a camera that is busy every second. Idle footage will pass far fewer frames.

Synthetic: `CAM-SYN01_20250101T080000.mp4`, 200,000-vector FAISS index.

### Detector input size (tracking rate)

`scripts/det_bench.py VIDEO` exports YOLO11n at several input sizes into `data/bench/det/` (never touching `models/yolo11n.onnx`) and times them interleaved, so CPU clock changes hit all of them alike. 60 frames of the compilation video, 3 iterations each, 2026-10-18:

| Input | Threads | Median ms | p95 ms | Busy @3 fps | Busy @5 fps | Recall vs 640 (all / small / medium / large) |
|---|---|---|---|---|---|---|
| 320 | 2 | 15.6 | 35.5 | 5% | 8% | 0.48 / 0.28 / 0.44 / 0.67 |
| 416 | 2 | 25.8 | 56.3 | 8% | 13% | 0.67 / 0.38 / 0.74 / 0.79 |
| 480 | 2 | 34.2 | 70.9 | 10% | 17% | 0.75 / 0.52 / 0.84 / 0.83 |
| 640 | 2 | 60.5 | 113.0 | 18% | 30% | 1.00 (baseline) |
| 640 | 4 | 55.9 | 101.7 | 17% | 28% | 1.00 (baseline) |
| **384x640** | **2** | **36.7** | **66.4** | **11%** | **18%** | **0.98 / 0.93 / 1.00 / 1.00** |
| 384x640 | 4 | 33.5 | 68.6 | 10% | 17% | 0.98 / 0.93 / 1.00 / 1.00 |

"Busy" is the share of one camera's wall-clock time the detector needs at that tracking rate. "Recall vs 640" is agreement with the 640 model (same class, IoU >= 0.5), not ground truth; small boxes are under 80 px tall (114 boxes in total).

- **A 384x640 rectangular input matches the square 640 model (recall 0.98) at about 60% of its cost**, because a square input wastes about 44% of its pixels on padding for 16:9 cameras. Square inputs below 640 lose many small and medium boxes.
- **4 threads are barely faster than 2** (5-10%), so the detector should use 2 threads and leave the other cores to the embedder and the recorder.
- **This laptop's CPU clock varies a lot.** Run one after another, the same 640 model measured anywhere from 51 to 104 ms per frame (including the 103.6 ms in the benchmark table above); p95 is about twice the median. The per-stage numbers above are therefore pessimistic, and only interleaved runs should be compared with each other.
- **The embedder, not the detector, is now the largest cost while a camera is busy:** at 1 fps with about 1.9 crops per frame, OpenCLIP needs about 2.9 x 127 ms, roughly 37% of wall-clock time. Embedding only the best crops per track (instead of every crop at every sampled second) is the next lever.

### Local VLM (on-demand only)

`scripts/vlm_bench.py VIDEO` times a local Ollama vision model after a vision sanity check (red image, `HELLO 42` text). Measured 2026-10-18, idle machine, same hardware as above:

| Model | Sanity | New image, single frame (s) | New 2x2 storyboard (s) | Image tokens | Gen tok/s | Memory | Verdict |
|---|---|---|---|---|---|---|---|
| `gemma4:e2b` (Q4_K_M) | **fails** (lawn scene read as "a sheet of paper with numbers") | 16 | 21-23 | ~150 | ~12 | 6.7 GB loaded | dropped |
| `qwen3-vl:2b-instruct` (Q4_K_M) | passes | 48-55 | 63-89 (verbose JSON) / ~40 (one-sentence answer) | ~1,100 at any input size | ~13 | 1.9 GB loaded, process RSS peak 4.7 GB | on-demand only |

Qwen3-VL 2B describes scenes correctly ("man in a red striped shirt and black shorts mowing the lawn, white car and white SUV parked behind him") but occasionally invents details ("lifts the mower to the sky"). Encoding one image takes 35-48 s on this CPU; Ollama encodes every image into about 1,100 tokens whatever the input size, so downscaling does not help. Use: `/ask` about one clip, rules with `severity: verify`, overnight summaries. Never per event. The plain `qwen3-vl:2b` tag is the thinking variant and ignores `think: false`.

### Zero-shot attribute tags (no VLM)

`scripts/attr_spike.py` tags crops from their existing OpenCLIP embeddings (about 2 ms per crop). On 24 person and 12 vehicle crops of the compilation video, checked by eye: vehicle colour and type about 80% correct, person top colour about 60% (right on clean single-person crops), clothing type / carried objects / headwear unreliable. About a quarter of the sampled person crops were detector false positives (title-card text, a cat, a fence).

## Security limitations

The MVP has no authentication or access control and is meant for single-user demos on one trusted machine.

- No authentication: anyone who can reach the port can search and view footage.
- No access control: there are no users, roles, or per-camera permissions.
- No audit logging: there is no audit trail of who viewed what.
- The API_Server binds only to 127.0.0.1 and refuses to start with any other host, so it is not reachable from other machines.
- The Query_Log (`data\logs\query_log.jsonl`) is a local file that is not tamper-evident; anyone with file access can edit or delete it.
- Access control and a tamper-evident audit log are planned after the MVP.

The server logs a warning with these limitations at every startup.
