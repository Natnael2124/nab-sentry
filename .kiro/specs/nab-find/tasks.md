# NAB Find - Tasks

## Phase 1: Speed experiment and foundations
- [x] 1.1 Branch `desktop-mvp` from `v2-step1-measure`; `venv-desktop`; folder layout (Req 12).
- [x] 1.2 Decode benchmark `bench/decode_bench.py` (OpenCV vs FFmpeg modes, hardware decode).
- [x] 1.3 Export CLIP to ONNX with a PyTorch parity check (`tools/export_clip.py`); BPE tokenizer
      without PyTorch (`nab_find/clip_tokenizer.py`).
- [x] 1.4 Model benchmark `bench/embed_bench.py` (ONNX RT fp32/int8, OpenVINO CPU/GPU, YOLO).
- [x] 1.5 Isolation check tool and desktop pytest config (Req 12.3).

## Phase 2: Engine
- [x] 2.1 `media.py`: probe, FrameReader (keyframes / rate, range, scaling), cut_clip.
- [x] 2.2 `cliptime.py`: start time and camera from file names; estimated fallback.
- [x] 2.3 `models.py`: device pick with self-test; CLIP image/text and YOLO on OpenVINO or ONNX RT.
- [x] 2.4 `store.py`: case folder, schema, commit/resume rules, index consistency on open.
- [x] 2.5 `progress.py`: ETA estimator (Req 7.2) with property tests.
- [x] 2.6 `processor.py`: quick pass, motion stretches, detail pass with tracking, visits, tags;
      pause/resume/cancel; keep-awake; priority.
- [x] 2.7 `search.py`: text search, moments, filters, visits, find again.
- [x] 2.8 End-to-end throughput bench on the test video and the generated scene; README numbers.

## Phase 3: Cloud check
- [x] 3.1 `aicheck.py`: frames per moment, request building, JSON parsing, cost estimate, errors.
- [x] 3.2 Key storage with DPAPI; settings.
- [x] 3.3 Live test against Gemini when a key is available (user step if none).

## Phase 4: App
- [x] 4.1 `server.py`: routes, token, media with Range, previews and exports.
- [x] 4.2 `app.py`: window, js_api (file dialogs, dropped paths, open folder), single instance,
      logging, data folder.
- [x] 4.3 UI: Home, New search wizard, Case (progress, search, results, player, visits, find
      again, export, AI check), Settings; light/dark; 1366x768.
- [x] 4.4 UI check in the real window (screenshots).

## Phase 5: Package and verify
- [x] 5.1 OpenVINO IR (fp16 weights) conversion of the models for the build.
- [x] 5.2 PyInstaller build; run the built app on the test video.
- [ ] 5.3 Inno Setup installer script written (`packaging/nab_find.iss`); building it needs Inno Setup,
      whose download failed during an internet outage.
- [x] 5.4 Slow suite `tests/test_slow.py` (`-m slow`): FFmpeg reader modes, clip cut, frame grab; real-model
      processing of a range with a stop + resume (found and fixed a duplicate frame at chunk ends), search, find again.
- [x] 5.5 README section and user guide.

## Needs the product owner
- [ ] U.1 Real DVR exports from several brands (format coverage).
- [x] U.2 A Gemini API key for the live AI check test (given 2026-10-29; stored DPAPI-encrypted, never committed).
- [ ] U.3 Name, icon, code-signing certificate, pricing.
