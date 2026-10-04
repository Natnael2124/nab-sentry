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

| Metric | Median | p95 | Notes |
|---|---|---|---|
| Decode (ms/frame) | 0.29 | 0.42 | |
| Motion gate (frames/s) | 3,970.72 | 4,224.10 | real-time pass |
| Detector (ms/frame) | 45.73 | 52.91 | YOLO11n ONNX, CPU |
| Embedder batch 1 (ms/image) | | | not measured |
| Embedder batch 8 (ms/image) | 49.15 | 54.21 | OpenCLIP ViT-B/32, batched CPU |
| Thumbnail (ms/JPEG) | | | not measured |
| Projected ingest (min per footage hour) | ~2.3 | | 10 fps footage, pass fraction 0.367 |
| Search unfiltered, 200k vectors (ms) | 135.50 | 143.18 | < 1,500 ms target |
| Search filtered, 200k vectors (ms) | 99.14 | 187.15 | < 1,500 ms target |
| Peak RSS (MB) | 1,764 | | peak process RSS (MiB) |
| Server startup to console URL (s) | | | not measured |

Hardware / date / Config overrides: local CPU hardware; 200,000-vector FAISS index; synthetic video `CAM-SYN01_20250101T080000.mp4`.

## Security limitations

The MVP has no authentication or access control and is meant for single-user demos on one trusted machine.

- No authentication: anyone who can reach the port can search and view footage.
- No access control: there are no users, roles, or per-camera permissions.
- No audit logging: there is no audit trail of who viewed what.
- The API_Server binds only to 127.0.0.1 and refuses to start with any other host, so it is not reachable from other machines.
- The Query_Log (`data\logs\query_log.jsonl`) is a local file that is not tamper-evident; anyone with file access can edit or delete it.
- Access control and a tamper-evident audit log are planned after the MVP.

The server logs a warning with these limitations at every startup.
