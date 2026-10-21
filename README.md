# ASPIRE Vision 2.0

ASPIRE Vision 2.0 counts and tracks pedestrians in recorded video, offline, for the CAT405 project at USM.

```
video → YOLOv8 (person) → ByteTrack → directional line crossings + zone occupancy
      → CSV export + run manifest → HTML dashboard → evaluation vs. annotations
```

The measurement definitions, annotation rules and data dictionary are in **[docs/week1_definitions.md](docs/week1_definitions.md)**.

## Setup

On Windows with Anaconda, open **Anaconda Prompt** in this folder and run:

```bat
setup_env.bat
```

This creates the conda env `aspire-vision2`, downloads the demo assets, runs the tests and does a quick smoke run.
After that, start each session with `conda activate aspire-vision2`.

To set up manually:

```bash
conda env create -f environment.yml && conda activate aspire-vision2
python scripts/get_demo_assets.py                     # demo clip + yolov8n.pt
```

If you're not using conda, `pip install -r requirements.txt` in any Python 3.10+ environment works too.

## Run the full workflow

```bash
python scripts/run.py --config configs/demo_vtest.yaml
# -> runs/<run_id>/  manifest.json tracks.csv events.csv occupancy.csv summary.json dashboard.html annotated.mp4
```

Open `runs/<run_id>/dashboard.html` in a browser.

To evaluate a run, annotate the clip first, then compare:

```bash
python scripts/annotate.py --config configs/demo_vtest.yaml --annotator XX   # keyboard tool, raw video only
python scripts/evaluate.py runs/<run_id> annotations/DEMO-VTEST_vtest_crossings_XX.csv
```

Useful options:
- `--max-frames 200` gives a quick test run.
- `--no-video` skips writing `annotated.mp4`.
- `--model models/yolov8n.onnx` runs the ONNX/CPU backend (export it with `yolo export model=models/yolov8n.pt format=onnx`).

## Tests

```bash
pytest -q
```

The tests cover:
- crossing semantics: hysteresis band, cooldown, segment extent, track maturity, spawn side, and the ID-switch duplicate failure mode
- evaluation matching, including errors that cancel out in the total count
- dashboard/export consistency

## Layout

```
aspire/        counting.py (crossing + occupancy logic)  detectors.py  trackers.py
               pipeline.py (run + export)  evaluate.py  dashboard.py
configs/       one YAML per camera/location: lines, zones, detector, tracker
scripts/       run.py  evaluate.py  annotate.py  get_demo_assets.py
annotations/   crossing labels (CSV)        data/  raw videos + recording_log.csv
docs/          week-by-week definitions and reports
runs/          one folder per run (git-ignored)
```

## Adding a new camera/location

1. Copy `configs/demo_vtest.yaml` and set `video`, `location_code`, and the line(s) and zone(s) in pixel coordinates.
   Grab a frame to read the coordinates from. Keep lines clear of poles and other fixed occluders.
2. Add a row to `data/recording_log.csv`, assigning the recording to `dev` or `test` as a whole.
3. Annotate with `scripts/annotate.py` before looking at that clip's predictions.

## Notes

- The demo clip is OpenCV sample data, for smoke tests only.
- YOLOv8 weights are licensed by Ultralytics (AGPL-3.0).
- All measurements are in image space; there is no ground-plane calibration yet.
