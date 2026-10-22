"""Compare predicted crossing events with human annotations.

Two levels (docs/week1_definitions.md §5):
  1. COUNT error per line & direction: predicted total - annotated total.
     Cheap, but errors can cancel (one false + one missed event = "perfect").
  2. EVENT matching: each annotated crossing is matched one-to-one to a
     predicted crossing on the same line, same direction, within ±tol frames.
     Gives TP / FP (extra, e.g. duplicates from ID switches) / FN (missed).

Box-level tracking metrics (MOTA, IDF1, HOTA) need per-frame box labels and
are added later with TrackEval on clips that have them.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REQUIRED = {"line_id", "direction", "frame"}


def load_annotations(path, fps: float | None = None) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if any(v.strip() for v in r.values() if v)]  # skip blank rows
    if not rows:
        return []
    cols = set(rows[0].keys())
    if not REQUIRED <= cols and not ({"line_id", "direction", "t_sec"} <= cols and fps):
        raise ValueError(f"annotation file needs columns {sorted(REQUIRED)} (or t_sec + fps); got {sorted(cols)}")
    out = []
    for r in rows:
        # labels with only a timestamp are converted to a frame using the run's fps
        frame = int(r["frame"]) if r.get("frame") not in (None, "") else round(float(r["t_sec"]) * fps)
        out.append({**r, "frame": frame})
    return out


def match_events(gt_frames, pred_frames, tol):
    """Greedy one-to-one matching by smallest |Δframe| within tol.

    Returns (pairs [(gt_i, pred_j)], unmatched_gt [i], unmatched_pred [j]).
    """
    cands = sorted(
        (abs(g - p), i, j)
        for i, g in enumerate(gt_frames)
        for j, p in enumerate(pred_frames)
        if abs(g - p) <= tol
    )
    # closest pairs are taken first, so a near match is never lost to a farther one
    used_g, used_p, pairs = set(), set(), []
    for _, i, j in cands:
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        pairs.append((i, j))
    return (
        sorted(pairs),
        [i for i in range(len(gt_frames)) if i not in used_g],
        [j for j in range(len(pred_frames)) if j not in used_p],
    )


def evaluate(run_dir, annotations_path, tol_sec: float = 1.0) -> dict:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    fps = manifest["video"]["fps"]
    n = manifest["video"]["frames_processed"]
    tol = max(1, round(tol_sec * fps))  # at least one frame of tolerance
    # ignore labels past the frames this run processed (e.g. a --max-frames run)
    gt = [g for g in load_annotations(annotations_path, fps) if g["frame"] < n]
    with open(run_dir / "events.csv", newline="", encoding="utf-8") as f:
        pred = list(csv.DictReader(f))

    groups = []
    for ln in manifest["config"]["lines"]:
        for d in (ln["positive_label"], ln["negative_label"]):
            groups.append((ln["id"], d))

    results, match_rows = [], []
    for line_id, direction in groups:
        g = sorted(int(x["frame"]) for x in gt if x["line_id"] == line_id and x["direction"] == direction)
        p_rows = sorted((x for x in pred if x["line_id"] == line_id and x["direction"] == direction),
                        key=lambda x: int(x["frame"]))
        p = [int(x["frame"]) for x in p_rows]
        pairs, fn, fp = match_events(g, p, tol)
        tp = len(pairs)
        results.append({
            "line_id": line_id, "direction": direction,
            "annotated": len(g), "predicted": len(p),
            "count_error": len(p) - len(g),
            "abs_pct_error": round(100 * abs(len(p) - len(g)) / len(g), 1) if g else None,
            "tp": tp, "fp": len(fp), "fn": len(fn),
            "precision": round(tp / len(p), 3) if p else None,
            "recall": round(tp / len(g), 3) if g else None,
        })
        for i, j in pairs:
            match_rows.append(dict(line_id=line_id, direction=direction, status="TP", gt_frame=g[i],
                                   pred_frame=p[j], track_id=p_rows[j]["track_id"], delta_frames=p[j] - g[i]))
        for i in fn:
            match_rows.append(dict(line_id=line_id, direction=direction, status="FN", gt_frame=g[i],
                                   pred_frame="", track_id="", delta_frames=""))
        for j in fp:
            match_rows.append(dict(line_id=line_id, direction=direction, status="FP", gt_frame="",
                                   pred_frame=p[j], track_id=p_rows[j]["track_id"], delta_frames=""))

    tot_g = sum(r["annotated"] for r in results)
    tot_p = sum(r["predicted"] for r in results)
    tot_tp = sum(r["tp"] for r in results)
    report = {
        "run_id": manifest["run_id"],
        "annotations": str(annotations_path),
        "tolerance_frames": tol,
        "frames_evaluated": n,
        "per_direction": results,
        "overall": {
            "annotated": tot_g, "predicted": tot_p, "count_error": tot_p - tot_g,
            "sum_abs_direction_error": sum(abs(r["count_error"]) for r in results),
            "tp": tot_tp, "fp": tot_p - tot_tp, "fn": tot_g - tot_tp,
            "precision": round(tot_tp / tot_p, 3) if tot_p else None,
            "recall": round(tot_tp / tot_g, 3) if tot_g else None,
        },
    }
    (run_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with open(run_dir / "eval_matches.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["line_id", "direction", "status", "gt_frame", "pred_frame",
                                          "track_id", "delta_frames"])
        w.writeheader()
        w.writerows(sorted(match_rows, key=lambda r: int(r["gt_frame"] or r["pred_frame"])))
    return report
