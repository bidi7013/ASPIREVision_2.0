"""Minimal dashboard: one self-contained HTML page per run (runs/<run_id>/dashboard.html).

Every number on the page is recomputed here from events.csv / occupancy.csv /
tracks.csv and checked against summary.json. If they disagree, the page shows
a FAILED consistency check — this answers "do the dashboard, the event log and
the export describe the same events?" (supervision question, week 3).
No server needed: open the file in any browser.
"""
from __future__ import annotations

import base64
import csv
import html
import io
import json
from collections import Counter
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .evaluate import evaluate  # noqa: E402

PALETTE = ["#2a78d6", "#d9632b", "#2f9e6a", "#8e5bd6"]


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fig_to_b64(fig, fmt="png") -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=110, bbox_inches="tight", transparent=False)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _background(video, frame_no):
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok else None


def _trajectory_plot(manifest, tracks, events):
    cfg = manifest["config"]
    W, H = manifest["video"]["width"], manifest["video"]["height"]
    bg = _background(cfg["video"], manifest["video"]["frames_processed"] // 2)
    fig, ax = plt.subplots(figsize=(8, 8 * H / W))
    if bg is not None:
        ax.imshow(bg, alpha=0.55)
    by_id: dict[str, list] = {}
    for r in tracks:
        by_id.setdefault(r["track_id"], []).append((float(r["anchor_x"]), float(r["anchor_y"])))
    for tid, pts in by_id.items():
        pts = np.array(pts)
        c = np.random.default_rng(int(tid)).uniform(0.1, 0.9, 3)
        ax.plot(pts[:, 0], pts[:, 1], "-", lw=1.4, color=c, alpha=0.9)
    for z in cfg.get("zones", []):
        poly = np.array(z["polygon"] + [z["polygon"][0]])
        ax.plot(poly[:, 0], poly[:, 1], "--", color="#e0a800", lw=2, label=f"zone {z['id']}")
    for ln in cfg["lines"]:
        ax.plot([ln["p1"][0], ln["p2"][0]], [ln["p1"][1], ln["p2"][1]], color="#d11", lw=3, label=f"line {ln['id']}")
    labels = sorted({e["direction"] for e in events})
    for k, d in enumerate(labels):
        pts = np.array([(float(e["anchor_x"]), float(e["anchor_y"])) for e in events if e["direction"] == d])
        ax.scatter(pts[:, 0], pts[:, 1], s=46, marker="o", edgecolor="white", linewidth=1.2,
                   color=PALETTE[k % len(PALETTE)], zorder=5, label=f"event: {d}")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="lower left", fontsize=8, framealpha=0.85)
    return _fig_to_b64(fig)


def _timeseries_plot(manifest, events, occ):
    fps = manifest["video"]["fps"]
    n = manifest["video"]["frames_processed"]
    t_end = n / fps
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True)
    ax = axes[0]
    labels = sorted({e["direction"] for e in events}) or []
    for k, d in enumerate(labels):
        ts = sorted(float(e["t_sec"]) for e in events if e["direction"] == d)
        ax.step([0] + ts + [t_end], [0] + list(range(1, len(ts) + 1)) + [len(ts)], where="post",
                color=PALETTE[k % len(PALETTE)], lw=2, label=d)
    ax.set_ylabel("cumulative crossings")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    ax = axes[1]
    for k, zid in enumerate(sorted({r["zone_id"] for r in occ})):
        rows = [r for r in occ if r["zone_id"] == zid]
        ax.plot([float(r["t_sec"]) for r in rows], [int(r["occupancy"]) for r in rows],
                color=PALETTE[(k + 2) % len(PALETTE)], lw=1.5, label=f"zone {zid}")
    ax.set_ylabel("people in zone")
    ax.set_xlabel("time (s)")
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    return _fig_to_b64(fig)


def consistency_checks(run_dir: Path) -> list[tuple[str, bool, str]]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    events = _read(run_dir / "events.csv")
    occ = _read(run_dir / "occupancy.csv")
    checks = []
    recount = Counter((e["line_id"], e["direction"]) for e in events)
    for line_id, d in summary["crossings"].items():
        for direction, v in d.items():
            ok = recount.get((line_id, direction), 0) == v
            checks.append((f"{line_id}/{direction}: summary.json vs events.csv", ok,
                           f"{v} vs {recount.get((line_id, direction), 0)}"))
    ok = summary["total_crossing_events"] == len(events)
    checks.append(("total events: summary.json vs events.csv rows", ok, f"{summary['total_crossing_events']} vs {len(events)}"))
    ids = [int(e["event_id"]) for e in events]
    checks.append(("event_id unique and sequential", ids == list(range(1, len(ids) + 1)), f"{len(ids)} ids"))
    run_ids = {e["run_id"] for e in events} | {r["run_id"] for r in occ}
    checks.append(("all rows carry this run_id", run_ids <= {summary["run_id"]}, ", ".join(sorted(run_ids)) or "-"))
    for zid, z in summary["occupancy"].items():
        vals = [int(r["occupancy"]) for r in occ if r["zone_id"] == zid]
        ok = (max(vals) if vals else 0) == z["max"]
        checks.append((f"zone {zid} max: summary.json vs occupancy.csv", ok, f"{z['max']} vs {max(vals) if vals else 0}"))
    return checks


CSS = """
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e3e2dd;--ok:#1f7a4d;--bad:#b3261e;--accent:#2a78d6}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--card:#20201e;--ink:#ecebe6;--muted:#a3a29b;--line:#34332f;--ok:#5cc28f;--bad:#f2867d;--accent:#6aa7ef}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1060px;margin:0 auto;padding:28px 16px 60px}h1{font-size:24px;margin:0 0 4px}h2{font-size:17px;margin:34px 0 12px}
.sub{color:var(--muted);font-size:13px;word-break:break-all}.note{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--accent);padding:10px 14px;border-radius:8px;font-size:13px;color:var(--muted);margin-top:14px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-top:18px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}.tile .k{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.tile .v{font-size:30px;font-weight:650;font-variant-numeric:tabular-nums;margin-top:2px}.tile .s{font-size:12px;color:var(--muted)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;overflow-x:auto}
img{max-width:100%;height:auto;display:block;margin:0 auto;border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--muted);font-weight:600}
.ok{color:var(--ok);font-weight:650}.bad{color:var(--bad);font-weight:650}pre{font-size:12px;margin:0;white-space:pre-wrap}
details summary{cursor:pointer;color:var(--accent)}.grid2{display:grid;grid-template-columns:1fr;gap:14px}
"""


def _table(rows, cols):
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>" for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def build_report(run_dir, annotations=None) -> Path:
    run_dir = Path(run_dir)
    if annotations:
        evaluate(run_dir, annotations)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    events = _read(run_dir / "events.csv")
    occ = _read(run_dir / "occupancy.csv")
    tracks = _read(run_dir / "tracks.csv")
    ev_path = run_dir / "evaluation.json"
    evaluation = json.loads(ev_path.read_text(encoding="utf-8")) if ev_path.exists() else None

    # tiles: computed from events.csv directly (summary.json is cross-checked below)
    recount = Counter((e["line_id"], e["direction"]) for e in events)
    tiles = []
    for ln in manifest["config"]["lines"]:
        for d in (ln["positive_label"], ln["negative_label"]):
            tiles.append((f"{ln['id']} · {d}", recount.get((ln["id"], d), 0), "crossing events"))
    for zid, z in summary["occupancy"].items():
        tiles.append((f"zone {zid} occupancy", f"{z['mean']:.1f}", f"mean per frame · max {z['max']}"))
    tm = manifest["timing"]
    tiles.append(("throughput", f"{tm['throughput_fps']}", f"fps · det {tm['detector_ms_per_frame']} ms/frame"))
    tiles.append(("track IDs (diagnostic)", summary["unique_track_ids"], "not a person count"))
    tiles_html = "".join(
        f"<div class='tile'><div class='k'>{html.escape(k)}</div><div class='v'>{html.escape(str(v))}</div>"
        f"<div class='s'>{html.escape(s)}</div></div>" for k, v, s in tiles)

    checks = consistency_checks(run_dir)
    all_ok = all(ok for _, ok, _ in checks)
    checks_html = _table([{"check": c, "result": "PASS" if ok else "FAIL", "values": v} for c, ok, v in checks],
                         ["check", "result", "values"]).replace(">PASS<", " class='ok'>PASS<").replace(">FAIL<", " class='bad'>FAIL<")

    eval_html = "<p class='sub'>No annotations evaluated yet. Run <code>scripts/evaluate.py</code> with an annotation CSV.</p>"
    if evaluation:
        o = evaluation["overall"]
        eval_html = (
            f"<p class='sub'>Annotations: {html.escape(evaluation['annotations'])} · match tolerance ±{evaluation['tolerance_frames']} frames"
            f" · overall count error {o['count_error']:+d} (sum of |per-direction error| = {o['sum_abs_direction_error']})"
            f" · precision {o['precision']} · recall {o['recall']}</p>"
            + _table(evaluation["per_direction"], ["line_id", "direction", "annotated", "predicted", "count_error",
                                                    "abs_pct_error", "tp", "fp", "fn", "precision", "recall"])
        )
        mpath = run_dir / "eval_matches.csv"
        if mpath.exists():
            m = [r for r in _read(mpath) if r["status"] != "TP"]
            if m:
                eval_html += "<h3 style='font-size:14px;margin:16px 0 6px'>Errors to inspect (FP = extra event, FN = missed)</h3>"
                eval_html += _table(m, ["line_id", "direction", "status", "gt_frame", "pred_frame", "track_id"])

    v = manifest["video"]
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ASPIRE-Vision · {html.escape(manifest['run_id'])}</title><style>{CSS}</style></head><body><main>
<h1>ASPIRE-Vision run dashboard</h1>
<div class="sub">run {html.escape(manifest['run_id'])} · {html.escape(v['path'])} · {v['frames_processed']} frames @ {v['fps']:.1f} fps ·
detector {html.escape(manifest['detector_backend'])} ({html.escape(Path(manifest['config']['detector']['model']).name)}) · tracker {html.escape(manifest['tracker'])} · config {manifest['config_hash']}</div>
<div class="note">Counts are <b>crossing events</b> (one person may legitimately produce several). Occupancy is people inside the zone per frame.
Track IDs are temporary, within this recording only. All positions are image-space pixels — no physical speed or density is implied.</div>
<div class="tiles">{tiles_html}</div>
<h2>Trajectories and crossing events</h2><div class="card"><img alt="trajectories" src="data:image/png;base64,{_trajectory_plot(manifest, tracks, events)}"></div>
<h2>Over time</h2><div class="card"><img alt="time series" src="data:image/png;base64,{_timeseries_plot(manifest, events, occ)}"></div>
<h2>Evaluation against annotations</h2><div class="card">{eval_html}</div>
<h2>Consistency: dashboard vs event log vs export — <span class="{'ok' if all_ok else 'bad'}">{'ALL PASS' if all_ok else 'FAILED'}</span></h2>
<div class="card">{checks_html}</div>
<h2>Event log ({len(events)} events)</h2><div class="card">{_table(events, ['event_id','frame','t_sec','line_id','direction','track_id','anchor_x','anchor_y'])}</div>
<h2>Run manifest</h2><div class="card"><details><summary>config, versions, hashes</summary><pre>{html.escape(json.dumps(manifest, indent=2))}</pre></details></div>
</main></body></html>"""
    out = run_dir / "dashboard.html"
    out.write_text(page, encoding="utf-8")
    return out
