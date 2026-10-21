# ASPIRE Vision 2.0 — context for Claude Code

This is a final-year project (CAT405, USM, 2026/27) by Adam Zafry bin Zaharin; the supervisor is Dr. Fadratul Hafinaz Hassan.
It is an offline pedestrian counting and tracking system that runs on recorded video from one fixed campus camera.
The deliverable is a **controlled evaluation study**, not a product. It compares two tracker configurations on counting error, tracking quality and throughput.

- **Read first:** `docs/fyp_plan.md`, which has the supervisor's weekly plan and the reasoning questions.
- **Then:** `docs/week1_definitions.md`, the measurement semantics, annotation rules and data dictionary.

## How work is organised
- **Thin vertical slices, every week.** The whole chain stays working at all times:
  `video → detect → track → count crossings → dashboard → export → compare with annotations`.
  Improve the existing chain; don't build modules in isolation.
- **Monday submissions** from 26 Oct 2026, each reporting the preceding week. Each one needs:
  - a working example
  - saved outputs and config
  - a test or comparison result
  - failures
  - an evidence-based answer to that week's reasoning question
  Use `/weekly-submission` to draft it.

## Commands
```bash
conda activate aspire-vision2                             # env from environment.yml (setup_env.bat creates it)
python scripts/get_demo_assets.py                         # demo clip + yolov8n.pt
python scripts/run.py --config configs/demo_vtest.yaml    # full run -> runs/<run_id>/ (+ dashboard.html)
python scripts/run.py --config ... --max-frames 200 --no-video   # quick check
python scripts/run.py --config ... --model models/yolov8n.onnx   # ONNX/CPU backend
python scripts/run.py --config ... --device cpu          # force device (.pt only; unset = CUDA if available)
python scripts/annotate.py --config ... --annotator XX    # keyboard crossing labeller (raw video only)
python scripts/evaluate.py runs/<run_id> annotations/<file>.csv
pytest -q
```
The dev machine is Windows with Anaconda installed, using the conda env `aspire-vision2`.
- Use `python`, not `python3`.
- Paths in configs are relative to the repo root.
- The project lives at `E:\ASPIRE_Vision\2.0`.
- If a package is missing, add it to both `environment.yml` and `requirements.txt`.

## Architecture (`aspire/`)
- `detectors.py`: `build_detector(cfg)` returns a callable `frame -> (xyxy, conf, cls)`. Backends are `ultralytics` (.pt) and `onnx` (onnxruntime).
- `trackers.py`: `build_tracker(cfg, fps)` returns an object whose `.update(xyxy, conf, cls)` gives `(xyxy, conf, track_id)`. Week 1 has ByteTrack (from `supervision==0.26.1`).
  **Add the second configuration here as a new class with the same interface** (planned: DeepSORT or BoT-SORT). Detection and tracking are deliberately separate, so both trackers can be fed identical detections.
- `counting.py`: `LineCounter`, the crossing logic, written by hand so it is explainable and tested. It uses a hysteresis band, segment extent, minimum track hits, a per-track cooldown, and start-side-no-event. `Zone` handles occupancy.
- `pipeline.py`: `run()` writes `manifest.json`, `tracks.csv`, `events.csv`, `occupancy.csv`, `summary.json` and optionally `annotated.mp4`. `summarise()` derives totals from the CSVs.
- `evaluate.py`: per-direction count error plus one-to-one event matching within ±1 s (TP/FP/FN). Writes `evaluation.json` and `eval_matches.csv`.
- `dashboard.py`: a self-contained `dashboard.html` per run. It recomputes every number from `events.csv` and shows PASS/FAIL consistency checks.
- `configs/*.yaml`: one per camera/location, holding the lines, zones, detector and tracker parameters.

## Invariants — do not break these
- **`events.csv` is the single source of truth for counts.** Summary, dashboard and reports must derive from it. Never add a separate counter.
- **Crossings ≠ occupancy ≠ unique track IDs.** Never present track-ID counts as people counts.
- **Image-space only.** No speed, distance or density in physical units unless a ground-plane homography is added and documented.
- **Track IDs are temporary**, within one recording. No cross-camera identity matching, and no storing of personal attributes.
- **Ground truth is independent.** Never generate, infer or "fill in" annotation labels from model output, and never fabricate labels or results. If labels don't exist, say so.
- **Dev/test split is by whole recording** (`data/recording_log.csv`). Tune thresholds only on `dev`.
- **Box/identity metrics only where suitable labels exist.** MOTA, IDF1 and HOTA (via TrackEval) are reported only on clips with box-level MOTChallenge-format labels.
- **Every run is reproducible.** The run ID includes the config hash, and the manifest records config, video hash, model hash and versions. Keep it that way.
- **Write a test for any change to counting semantics** in `tests/test_counting.py`, as a scripted trajectory with a known expected count.

## Scope to protect (from the supervisor)
**In scope:**
- recorded single-camera video
- one detector, plus tracking
- directional counts
- a simple dashboard
- CSV export

**Out of scope (defer):** live CCTV, cross-camera re-identification, automated facility control, complex alerts. If a request drifts into these, flag it as future work.

## Status (updated 22 Oct 2026)
- **Week 1 slice works end to end** on the OpenCV demo clip `vtest.avi`. That clip is for smoke tests only: not campus data, no annotations.
  - YOLOv8n ONNX on CPU + ByteTrack: 27 events, ~12 fps on a 2-core CPU.
  - All dashboard consistency checks pass; 18/18 tests pass.
- **The Ultralytics `.pt` backend is verified** on the dev machine (RTX 3050 Laptop 4 GB; torch 2.14.1+cu130, ultralytics 8.4.183). Full `vtest.avi`, 795 frames, `--no-video`:

  | backend / device | throughput fps | detector ms/frame | tracker ms/frame | events |
  |---|---|---|---|---|
  | `.pt` GPU (`cuda:0`) | 50.4 | 17.0 | 1.1 | 27 |
  | `.pt` CPU | 22.0 | 42.1 | 1.6 | 27 |
  | ONNX CPU | 19.2 | 48.8 | 1.5 | 27 |

  - `.pt` CPU and GPU produce identical `events.csv` (frame, direction, track ID). ONNX gives the same 13/14 totals, but the event frames and track IDs differ (its preprocessing and numerics differ), so compare trackers with **one fixed detector backend**.
  - Throughput includes the one-off CUDA/model warm-up and is a single run, not yet averaged.
  - The manifest's `detector_device` records the device that actually ran inference. With `device` unset, Ultralytics silently uses CUDA.
  - Run `DEMO-VTEST_20261021T234226Z_735e77d6` predates that fix: its manifest wrongly says `cpu`, although it ran on the GPU.
  - The GPU needs the CUDA wheel after creating the env: `pip install --force-reinstall --no-deps torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130`.
- **Observed (to inspect in week 2):** several tracks cross in both directions (#10, #14, #32, #34, #38). They may be genuine turn-backs, or jitter/ID-switch artefacts.
- **Lesson:** keep counting lines clear of fixed occluders (the first line sat on a lamp post).
- **Open items:**
  - USM video approval
  - choice of licensed public fallback (e.g. a MOTChallenge sequence; check the licence)
  - an independent second annotator
  - second tracker configuration (week 3)

## Next up (Mon 2 Nov target)
1. ~~Verify the `.pt` backend; log CPU vs GPU fps.~~ Done 22 Oct (see Status).
2. Get real footage (approved USM clip, or the licensed fallback). Add a config and a `recording_log.csv` row.
3. Annotate crossings with `scripts/annotate.py` before looking at predictions. Then run `evaluate.py` and inspect every FP/FN in `eval_matches.csv`.
4. Quantify ID-switch duplicates near the line. Check whether `band_px`, `min_track_hits` and `lost_track_buffer` reduce them, tuning on `dev` only.
