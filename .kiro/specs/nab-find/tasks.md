# NAB Find - Tasks

## Phase 1: Speed experiment and foundations
- [x] 1.1 Branch `desktop-mvp` from `v2-step1-measure`; `venv-desktop`; folder layout (Req 12).
- [x] 1.2 Decode benchmark `bench/decode_bench.py` (OpenCV vs FFmpeg modes, hardware decode).
- [x] 1.3 Export CLIP to ONNX with a PyTorch parity check (`tools/export_clip.py`); BPE tokenizer
      without PyTorch (`nab_find/clip_tokenizer.py`).
- [x] 1.4 Model benchmark `bench/embed_bench.py` (ONNX RT fp32/int8, OpenVINO CPU/GPU, YOLO).
- [ ] 1.5 Isolation check tool and desktop pytest config (Req 12.3).

## Phase 2: Engine
- [ ] 2.1 `media.py`: probe, FrameReader (keyframes / rate, range, scaling), cut_clip.
- [ ] 2.2 `cliptime.py`: start time and camera from file names; estimated fallback.
- [ ] 2.3 `models.py`: device pick with self-test; CLIP image/text and YOLO on OpenVINO or ONNX RT.
- [ ] 2.4 `store.py`: case folder, schema, commit/resume rules, index consistency on open.
- [ ] 2.5 `progress.py`: ETA estimator (Req 7.2) with property tests.
- [ ] 2.6 `processor.py`: quick pass, motion stretches, detail pass with tracking, visits, tags;
      pause/resume/cancel; keep-awake; priority.
- [ ] 2.7 `search.py`: text search, moments, filters, visits, find again.
- [ ] 2.8 End-to-end throughput bench on the test video and the generated scene; README numbers.

## Phase 3: Cloud check
- [ ] 3.1 `aicheck.py`: frames per moment, request building, JSON parsing, cost estimate, errors.
- [ ] 3.2 Key storage with DPAPI; settings.
- [ ] 3.3 Live test against Gemini when a key is available (user step if none).

## Phase 4: App
- [ ] 4.1 `server.py`: routes, token, media with Range, previews and exports.
- [ ] 4.2 `app.py`: window, js_api (file dialogs, dropped paths, open folder), single instance,
      logging, data folder.
- [ ] 4.3 UI: Home, New search wizard, Case (progress, search, results, player, visits, find
      again, export, AI check), Settings; light/dark; 1366x768.
- [ ] 4.4 UI check in the real window (screenshots).

## Phase 5: Package and verify
- [ ] 5.1 OpenVINO IR (fp16 weights) conversion of the models for the build.
- [ ] 5.2 PyInstaller build; run the built app on the test video.
- [ ] 5.3 Inno Setup installer script (build needs Inno Setup installed).
- [ ] 5.4 Slow suite: end-to-end, resume after kill, search quality on the compilation queries.
- [ ] 5.5 README section and user guide.

## Needs the product owner
- [ ] U.1 Real DVR exports from several brands (format coverage).
- [ ] U.2 A Gemini API key for the live AI check test.
- [ ] U.3 Name, icon, code-signing certificate, pricing.
