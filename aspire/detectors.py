"""Pedestrian detectors behind one small interface.

Every detector returns, per frame:
    xyxy (N, 4) float pixel boxes, conf (N,) float, cls (N,) int
filtered to the configured classes (default: COCO class 0 = person).

Backends
  * "ultralytics": YOLOv8 .pt weights via the ultralytics package (normal use).
  * "onnx":        YOLOv8 exported to ONNX, run with onnxruntime. Useful where
                   PyTorch is unavailable; also lets you benchmark CPU inference.
  * "auto":        choose by file extension (.pt -> ultralytics, .onnx -> onnx).

Tracking is deliberately NOT done here, so the same detections can be fed to
different trackers when comparing configurations.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class Detector:
    name = "base"

    def __call__(self, frame_bgr: np.ndarray):  # pragma: no cover - interface
        raise NotImplementedError


class UltralyticsDetector(Detector):
    name = "ultralytics"

    def __init__(self, model: str, conf: float, iou: float, imgsz: int, classes, device=None):
        from ultralytics import YOLO  # imported lazily: optional dependency

        self.model = YOLO(model)
        self.kw = dict(conf=conf, iou=iou, imgsz=imgsz, classes=list(classes), verbose=False)
        if device is not None:
            self.kw["device"] = device
        self.device = None  # resolved on first predict (ultralytics picks CUDA when device is unset)

    def __call__(self, frame_bgr):
        r = self.model.predict(frame_bgr, **self.kw)[0]
        if self.device is None:
            self.device = str(self.model.predictor.device)  # YOLO.device reports the wrapper, not the predictor
        b = r.boxes
        return (
            b.xyxy.cpu().numpy().astype(float),
            b.conf.cpu().numpy().astype(float),
            b.cls.cpu().numpy().astype(int),
        )


class OnnxYoloDetector(Detector):
    """Minimal YOLOv8 ONNX runner: letterbox -> infer -> decode -> NMS."""

    name = "onnx"

    def __init__(self, model: str, conf: float, iou: float, imgsz: int, classes, device=None):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
        self.device = "cpu"
        self.inp = self.sess.get_inputs()[0].name
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.classes = set(int(c) for c in classes)

    def _letterbox(self, img):
        h, w = img.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nh, nw = round(h * r), round(w * r)
        top = (self.imgsz - nh) // 2
        left = (self.imgsz - nw) // 2
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        canvas[top : top + nh, left : left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return canvas, r, left, top

    def __call__(self, frame_bgr):
        img, r, left, top = self._letterbox(frame_bgr)
        # HWC BGR uint8 -> NCHW RGB float in [0, 1], the input layout YOLOv8 expects
        blob = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.sess.run(None, {self.inp: blob})[0][0].T  # (8400, 4 + n_classes)
        scores = out[:, 4:]  # YOLOv8 has no objectness score: confidence = best class score
        cls = scores.argmax(1)
        conf = scores[np.arange(len(cls)), cls]
        keep = (conf >= self.conf) & np.isin(cls, list(self.classes))
        out, cls, conf = out[keep], cls[keep], conf[keep]
        if len(out) == 0:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
        cx, cy, w, h = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        # undo the letterbox: remove the padding, then scale back to the original frame
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - left) / r
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - top) / r
        H, W = frame_bgr.shape[:2]
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, W)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, H)
        # class-aware NMS (offset boxes by class so classes never suppress each other)
        offs = cls[:, None] * 4096.0
        boxes_wh = np.concatenate([xyxy[:, :2] + offs, xyxy[:, 2:] - xyxy[:, :2]], 1)
        idx = cv2.dnn.NMSBoxes(boxes_wh.tolist(), conf.tolist(), self.conf, self.iou)
        idx = np.array(idx).reshape(-1)
        return xyxy[idx].astype(float), conf[idx].astype(float), cls[idx].astype(int)


def build_detector(cfg: dict) -> Detector:
    backend = cfg.get("backend", "auto")
    model = cfg["model"]
    if backend == "auto":
        backend = "onnx" if Path(model).suffix.lower() == ".onnx" else "ultralytics"
    kw = dict(
        model=model,
        conf=float(cfg.get("conf", 0.25)),
        iou=float(cfg.get("iou", 0.5)),
        imgsz=int(cfg.get("imgsz", 640)),
        classes=cfg.get("classes", [0]),
        device=cfg.get("device"),
    )
    if backend == "ultralytics":
        return UltralyticsDetector(**kw)
    if backend == "onnx":
        return OnnxYoloDetector(**kw)
    raise ValueError(f"unknown detector backend: {backend}")
