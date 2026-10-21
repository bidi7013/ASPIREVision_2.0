"""Run the pipeline on one video.

    python scripts/run.py --config configs/demo_vtest.yaml
    python scripts/run.py --config configs/demo_vtest.yaml --max-frames 200 --no-video
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aspire.dashboard import build_report  # noqa: E402
from aspire.pipeline import run  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--video", help="override the video path in the config")
p.add_argument("--model", help="override detector.model (e.g. models/yolov8n.onnx)")
p.add_argument("--device", help="override detector.device for the ultralytics backend (cpu, 0, cuda:0)")
p.add_argument("--out",help="output root (default: config output.dir)")
p.add_argument("--max-frames", type=int)
p.add_argument("--no-video", action="store_true", help="skip annotated.mp4")
p.add_argument("--annotations", help="crossing annotation CSV to evaluate against")
a = p.parse_args()

run_dir = run(a.config, a.video, a.out, a.max_frames, save_video=False if a.no_video else None, model_override=a.model,
          device_override=a.device)
report = build_report(run_dir, annotations=a.annotations)
print(f"dashboard: {report}")
