"""Download the demo clip and YOLOv8n weights (both from GitHub).

    python scripts/get_demo_assets.py

* data/raw/vtest.avi  — OpenCV's sample pedestrian clip (768x576, 10 fps, 795 frames).
  For pipeline smoke tests ONLY; not campus data and not for reported results.
* models/yolov8n.pt   — Ultralytics YOLOv8n COCO weights (AGPL-3.0, see Ultralytics licence).
For an ONNX copy (runs on CPU without PyTorch):  yolo export model=models/yolov8n.pt format=onnx
"""
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    ROOT / "data/raw/vtest.avi": "https://github.com/opencv/opencv/raw/4.x/samples/data/vtest.avi",
    ROOT / "models/yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
}

for dest, url in ASSETS.items():
    if dest.exists():
        print(f"exists: {dest.relative_to(ROOT)}")
        continue
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"  -> {dest.relative_to(ROOT)} ({dest.stat().st_size / 1e6:.1f} MB)")
