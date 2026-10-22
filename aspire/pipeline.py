"""End-to-end run: video -> detect -> track -> count/occupancy -> export.

Outputs (one folder per run, runs/<run_id>/), see docs/week1_definitions.md §4:
  manifest.json   what was run: config, hashes, versions, timing
  tracks.csv      per-frame trajectory points (one row per track per frame)
  events.csv      line-crossing events  <- the SINGLE SOURCE OF TRUTH for counts
  occupancy.csv   per-frame zone occupancy
  summary.json    totals DERIVED from events.csv / occupancy.csv (never counted separately)
  annotated.mp4   optional visual check
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml

from .counting import LineCounter, Zone, anchor_points
from .detectors import build_detector
from .trackers import build_tracker

TRACK_FIELDS = ["run_id", "frame", "t_sec", "track_id", "x1", "y1", "x2", "y2", "anchor_x", "anchor_y", "conf"]
EVENT_FIELDS = ["run_id", "event_id", "frame", "t_sec", "line_id", "direction", "track_id", "anchor_x", "anchor_y"]
OCC_FIELDS = ["run_id", "frame", "t_sec", "zone_id", "occupancy"]


def file_sha256(path, limit_mb: int | None = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        read = 0
        while chunk := f.read(1 << 20):
            h.update(chunk)
            read += 1
            if limit_mb and read >= limit_mb:  # hash only the head so long videos stay fast
                break
    return h.hexdigest()


def config_hash(cfg: dict) -> str:
    # sort_keys: the same settings always give the same hash, whatever their order in the YAML
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]


def load_config(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "cv2", "supervision", "ultralytics", "onnxruntime", "torch"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass  # optional backend not installed: leave it out of the manifest
    return out


def summarise(run_dir: Path) -> dict:
    """Derive all headline numbers from the exported CSVs (single source of truth)."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    with open(run_dir / "events.csv", newline="", encoding="utf-8") as f:
        events = list(csv.DictReader(f))
    with open(run_dir / "occupancy.csv", newline="", encoding="utf-8") as f:
        occ = list(csv.DictReader(f))
    with open(run_dir / "tracks.csv", newline="", encoding="utf-8") as f:
        track_ids = {r["track_id"] for r in csv.DictReader(f)}

    crossings: dict[str, dict[str, int]] = {}
    for ln in manifest["config"]["lines"]:
        crossings[ln["id"]] = {ln["positive_label"]: 0, ln["negative_label"]: 0}
    for e in events:
        crossings.setdefault(e["line_id"], {}).setdefault(e["direction"], 0)
        crossings[e["line_id"]][e["direction"]] += 1

    occupancy = {}
    for z in manifest["config"].get("zones", []):
        vals = [int(r["occupancy"]) for r in occ if r["zone_id"] == z["id"]]
        occupancy[z["id"]] = {
            "mean": round(float(np.mean(vals)), 3) if vals else 0.0,
            "max": int(max(vals)) if vals else 0,
            "frames": len(vals),
        }
    return {
        "run_id": manifest["run_id"],
        "crossings": crossings,
        "total_crossing_events": len(events),
        "occupancy": occupancy,
        "unique_track_ids": len(track_ids),
        "_note": "unique_track_ids is a tracker diagnostic, NOT a pedestrian count "
        "(ID switches create extra IDs). Use crossings for counts.",
    }


def draw_overlay(frame, cfg, xyxy, tids, anchors, trails, totals):
    for z in cfg.get("zones", []):
        pts = np.array(z["polygon"], dtype=np.int32)
        cv2.polylines(frame, [pts], True, (255, 200, 0), 2)
    for ln in cfg["lines"]:
        cv2.line(frame, tuple(map(int, ln["p1"])), tuple(map(int, ln["p2"])), (0, 0, 255), 2)
    for (x1, y1, x2, y2), tid, a in zip(xyxy, tids, anchors):
        # seeded by track ID, so each track keeps the same colour in every frame
        color = tuple(int(c) for c in np.random.default_rng(int(tid)).integers(64, 255, 3))
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"#{tid}", (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        pts = np.array(trails.get(int(tid), []), dtype=np.int32)
        if len(pts) > 1:
            cv2.polylines(frame, [pts], False, color, 2)
        cv2.circle(frame, (int(a[0]), int(a[1])), 3, color, -1)
    y = 22
    for line_id, d in totals.items():
        txt = f"{line_id}: " + "  ".join(f"{k}={v}" for k, v in d.items())
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (4, y - th - 6), (16 + tw, y + 6), (20, 20, 20), -1)
        cv2.putText(frame, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 28
    return frame


def run(config_path, video_override=None, out_root=None, max_frames=None, save_video=None, model_override=None,
        device_override=None) -> Path:
    cfg = load_config(config_path)
    if video_override:
        cfg["video"] = str(video_override)
    if model_override:
        cfg["detector"]["model"] = str(model_override)
    if device_override:
        cfg["detector"]["device"] = str(device_override)
    out_cfg = cfg.get("output", {})
    out_root = Path(out_root or out_cfg.get("dir", "runs"))
    save_video = out_cfg.get("save_video", False) if save_video is None else save_video

    cap = cv2.VideoCapture(cfg["video"])
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {cfg['video']}")
    fps = cap.get(cv2.CAP_PROP_FPS) or float(cfg.get("fps_fallback", 25))  # some containers report 0 fps
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    started = datetime.now(timezone.utc)
    run_id = f"{cfg.get('location_code', 'LOC')}_{started.strftime('%Y%m%dT%H%M%SZ')}_{config_hash(cfg)}"
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)  # never overwrite an earlier run

    detector = build_detector(cfg["detector"])
    tracker = build_tracker(cfg["tracker"], fps=fps)
    anchor_mode = cfg.get("anchor", "bottom_center")
    counters = [
        LineCounter(
            line_id=ln["id"], p1=tuple(ln["p1"]), p2=tuple(ln["p2"]),
            positive_label=ln["positive_label"], negative_label=ln["negative_label"],
            band_px=float(ln.get("band_px", 8)), segment_margin_px=float(ln.get("segment_margin_px", 10)),
            min_track_hits=int(ln.get("min_track_hits", 3)), cooldown_frames=int(ln.get("cooldown_frames", 10)),
        )
        for ln in cfg["lines"]
    ]
    zones = [Zone(z["id"], [tuple(p) for p in z["polygon"]]) for z in cfg.get("zones", [])]
    totals = {c.line_id: {c.positive_label: 0, c.negative_label: 0} for c in counters}

    writer = None
    if save_video:
        writer = cv2.VideoWriter(str(run_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    f_tr = open(run_dir / "tracks.csv", "w", newline="", encoding="utf-8")
    f_ev = open(run_dir / "events.csv", "w", newline="", encoding="utf-8")
    f_oc = open(run_dir / "occupancy.csv", "w", newline="", encoding="utf-8")
    w_tr, w_ev, w_oc = (csv.DictWriter(f, fieldnames=fl) for f, fl in
                        ((f_tr, TRACK_FIELDS), (f_ev, EVENT_FIELDS), (f_oc, OCC_FIELDS)))
    for w in (w_tr, w_ev, w_oc):
        w.writeheader()

    trails: dict[int, list] = {}
    t_det = t_trk = 0.0
    frame_idx = 0
    event_id = 0
    t0 = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames and frame_idx >= max_frames):
                break
            t_sec = round(frame_idx / fps, 3)

            # time detection and tracking separately: the manifest reports each cost per frame
            a = time.perf_counter()
            xyxy, conf, cls = detector(frame)
            b = time.perf_counter()
            xyxy, conf, tids = tracker.update(xyxy, conf, cls)
            c = time.perf_counter()
            t_det += b - a
            t_trk += c - b

            anchors = anchor_points(xyxy, anchor_mode)
            for box, cf, tid, an in zip(xyxy, conf, tids, anchors):
                w_tr.writerow(dict(run_id=run_id, frame=frame_idx, t_sec=t_sec, track_id=int(tid),
                                   x1=round(float(box[0]), 1), y1=round(float(box[1]), 1),
                                   x2=round(float(box[2]), 1), y2=round(float(box[3]), 1),
                                   anchor_x=round(float(an[0]), 1), anchor_y=round(float(an[1]), 1),
                                   conf=round(float(cf), 3)))
                trails.setdefault(int(tid), []).append((float(an[0]), float(an[1])))
                trails[int(tid)] = trails[int(tid)][-40:]  # only the recent path is drawn

            for counter in counters:
                for ev in counter.update(frame_idx, tids, anchors):
                    event_id += 1
                    totals[ev.line_id][ev.direction] += 1
                    w_ev.writerow(dict(run_id=run_id, event_id=event_id, frame=ev.frame, t_sec=t_sec,
                                       line_id=ev.line_id, direction=ev.direction, track_id=ev.track_id,
                                       anchor_x=round(ev.anchor_x, 1), anchor_y=round(ev.anchor_y, 1)))
            for z in zones:
                w_oc.writerow(dict(run_id=run_id, frame=frame_idx, t_sec=t_sec, zone_id=z.zone_id,
                                   occupancy=z.count(anchors)))

            if writer is not None:
                writer.write(draw_overlay(frame, cfg, xyxy, tids, anchors, trails, totals))
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"  frame {frame_idx}/{n_frames}  totals={totals}", flush=True)
    finally:
        cap.release()
        for f in (f_tr, f_ev, f_oc):
            f.close()
        if writer is not None:
            writer.release()

    wall = time.perf_counter() - t0
    manifest = {
        "run_id": run_id,
        "created_utc": started.isoformat(),
        "config_file": str(config_path),
        "config_hash": config_hash(cfg),
        "config": cfg,
        "video": {"path": cfg["video"], "sha256_first_64MB": file_sha256(cfg["video"], limit_mb=64),
                  "fps": fps, "frames_reported": n_frames, "frames_processed": frame_idx,
                  "width": W, "height": H},
        "model_sha256": file_sha256(cfg["detector"]["model"]) if Path(cfg["detector"]["model"]).exists() else None,
        "detector_backend": detector.name,
        "detector_device": getattr(detector, "device", "cpu"),
        "tracker": tracker.name,
        "timing": {"wall_sec": round(wall, 2), "throughput_fps": round(frame_idx / wall, 2) if wall else None,
                   "detector_ms_per_frame": round(1000 * t_det / max(frame_idx, 1), 2),
                   "tracker_ms_per_frame": round(1000 * t_trk / max(frame_idx, 1), 2)},
        "versions": _versions(),
        "units": "image-space pixels; no ground-plane calibration applied",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = summarise(run_dir)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"done: {frame_idx} frames in {wall:.1f}s ({manifest['timing']['throughput_fps']} fps) -> {run_dir}")
    print(json.dumps(summary["crossings"]))
    return run_dir
