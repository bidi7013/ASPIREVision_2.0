"""Tests for evaluation matching and export/dashboard consistency."""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aspire.dashboard import consistency_checks  # noqa: E402
from aspire.evaluate import evaluate, match_events  # noqa: E402
from aspire.pipeline import EVENT_FIELDS, OCC_FIELDS, TRACK_FIELDS, summarise  # noqa: E402


def test_match_is_one_to_one_within_tolerance():
    pairs, fn, fp = match_events([10, 50, 90], [11, 12, 300], tol=5)
    assert pairs == [(0, 0)] and fn == [1, 2] and fp == [1, 2]


def _fake_run(tmp_path, events, gt=None):
    run = tmp_path / "run"
    run.mkdir()
    cfg = {"video": "x.avi", "lines": [{"id": "L1", "positive_label": "in", "negative_label": "out"}],
           "zones": [{"id": "Z1", "polygon": [[0, 0], [1, 0], [1, 1]]}], "detector": {"model": "m"}}
    (run / "manifest.json").write_text(json.dumps(
        {"run_id": "R", "config": cfg, "video": {"fps": 10.0, "frames_processed": 1000}}), encoding="utf-8")
    with open(run / "events.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=EVENT_FIELDS)
        w.writeheader()
        for i, (frame, d) in enumerate(events, 1):
            w.writerow(dict(run_id="R", event_id=i, frame=frame, t_sec=frame / 10, line_id="L1",
                            direction=d, track_id=i, anchor_x=0, anchor_y=0))
    with open(run / "occupancy.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OCC_FIELDS)
        w.writeheader()
        w.writerow(dict(run_id="R", frame=0, t_sec=0, zone_id="Z1", occupancy=2))
    with open(run / "tracks.csv", "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=TRACK_FIELDS).writeheader()
    (run / "summary.json").write_text(json.dumps(summarise(run)), encoding="utf-8")
    if gt is not None:
        with open(tmp_path / "gt.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["clip_id", "line_id", "direction", "frame", "annotator", "notes"])
            for frame, d in gt:
                w.writerow(["c", "L1", d, frame, "A", ""])
    return run


def test_cancelling_errors_hidden_by_count_but_visible_in_matching(tmp_path):
    """Week-4 point: a perfect total can hide one false and one missed event."""
    gt = [(100, "in"), (200, "in"), (300, "in")]
    pred = [(101, "in"), (199, "in"), (600, "in")]  # 300 missed, 600 is spurious
    run = _fake_run(tmp_path, pred, gt)
    rep = evaluate(run, tmp_path / "gt.csv", tol_sec=1.0)
    row = next(r for r in rep["per_direction"] if r["direction"] == "in")
    assert row["count_error"] == 0           # count looks perfect...
    assert (row["tp"], row["fp"], row["fn"]) == (2, 1, 1)  # ...but it isn't


def test_summary_and_dashboard_agree_with_event_log(tmp_path):
    run = _fake_run(tmp_path, [(10, "in"), (20, "out"), (30, "in")])
    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert s["crossings"]["L1"] == {"in": 2, "out": 1}
    assert all(ok for _, ok, _ in consistency_checks(run))


def test_consistency_check_detects_tampered_summary(tmp_path):
    run = _fake_run(tmp_path, [(10, "in")])
    s = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    s["crossings"]["L1"]["in"] = 5
    (run / "summary.json").write_text(json.dumps(s), encoding="utf-8")
    assert not all(ok for _, ok, _ in consistency_checks(run))
