"""Multi-object trackers behind one small interface.

A tracker takes one frame of detections and returns the tracked subset with a
TEMPORARY track ID per box. IDs are only meaningful within one recording
(see docs/week1_definitions.md, section 2.4).

Week 1 ships ByteTrack (motion + IoU association, two-stage matching that also
uses low-confidence boxes). The second configuration for the comparison
(planned for week 3) is added by writing another class with the same
`update()` signature, e.g. DeepSORT (appearance embedding) or BoT-SORT.
"""
from __future__ import annotations

import numpy as np


class Tracker:
    name = "base"

    def update(self, xyxy, conf, cls):  # pragma: no cover - interface
        """Return (xyxy (M,4), conf (M,), track_id (M,) int)."""
        raise NotImplementedError


class ByteTrackSV(Tracker):
    """ByteTrack as implemented in Roboflow `supervision` (numpy-only)."""

    name = "bytetrack"

    def __init__(self, fps: float, **params):
        import supervision as sv

        self._sv = sv
        self.params = dict(
            track_activation_threshold=float(params.get("track_activation_threshold", 0.25)),
            lost_track_buffer=int(params.get("lost_track_buffer", 30)),
            minimum_matching_threshold=float(params.get("minimum_matching_threshold", 0.8)),
            minimum_consecutive_frames=int(params.get("minimum_consecutive_frames", 1)),
            frame_rate=int(round(fps)),
        )
        self.tracker = sv.ByteTrack(**self.params)

    def update(self, xyxy, conf, cls):
        det = self._sv.Detections(
            xyxy=np.asarray(xyxy, dtype=np.float32).reshape(-1, 4),
            confidence=np.asarray(conf, dtype=np.float32),
            class_id=np.asarray(cls, dtype=int),
        )
        out = self.tracker.update_with_detections(det)
        if len(out) == 0 or out.tracker_id is None:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
        keep = out.tracker_id >= 0  # -1 = not yet confirmed
        return out.xyxy[keep], out.confidence[keep], out.tracker_id[keep].astype(int)


def build_tracker(cfg: dict, fps: float) -> Tracker:
    name = cfg.get("name", "bytetrack")
    params = {k: v for k, v in cfg.items() if k != "name"}
    if name == "bytetrack":
        return ByteTrackSV(fps=fps, **params)
    raise ValueError(f"unknown tracker: {name} (add it in aspire/trackers.py)")
