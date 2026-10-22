"""Keyboard tool for annotating line crossings — INDEPENDENT of the model.

It shows only the raw video and the counting line(s); it never shows
detections or tracks, so labels are not biased by predictions.

    python scripts/annotate.py --config configs/demo_vtest.yaml --annotator ADAM

Keys
  space      play / pause                  ,  .   step back / forward 1 frame
  [  ]       back / forward 10 frames      l      cycle active line (if several)
  1          log crossing: positive_label  2      log crossing: negative_label
  u          undo last label               s      save          q   save and quit
Log the frame where the person's FEET pass the line (see docs/week1_definitions.md §3).
Output: annotations/<location_code>_<video stem>_crossings_<annotator>.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import yaml

FIELDS = ["clip_id", "line_id", "direction", "frame", "t_sec", "annotator", "notes"]


def save(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (int(r["frame"]), r["line_id"])))
    print(f"saved {len(rows)} labels -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--video")
    ap.add_argument("--annotator", required=True, help="initials; use a different person from the developer if possible")
    ap.add_argument("--scale", type=float, default=1.0, help="display zoom")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    video = a.video or cfg["video"]
    clip_id = f"{cfg.get('location_code', 'LOC')}_{Path(video).stem}"
    out = Path("annotations") / f"{clip_id}_crossings_{a.annotator}.csv"
    rows = list(csv.DictReader(open(out, newline="", encoding="utf-8"))) if out.exists() else []
    if rows:
        print(f"resuming {len(rows)} existing labels from {out}")

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lines = cfg["lines"]
    active, frame_no, playing = 0, 0, False

    def read(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        return fr if ok else None

    frame = read(0)
    while True:
        view = frame.copy()
        for k, ln in enumerate(lines):
            col = (0, 0, 255) if k == active else (120, 120, 120)
            cv2.line(view, tuple(map(int, ln["p1"])), tuple(map(int, ln["p2"])), col, 2)
            cv2.putText(view, ln["id"], tuple(map(int, ln["p2"])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        ln = lines[active]
        here = [r for r in rows if abs(int(r["frame"]) - frame_no) <= 2]  # show labels near this frame
        hud = [f"frame {frame_no}/{n-1}  t={frame_no/fps:.2f}s  {'PLAY' if playing else 'PAUSE'}  labels={len(rows)}",
               f"line {ln['id']}:  [1] {ln['positive_label']}   [2] {ln['negative_label']}   [u]ndo [s]ave [q]uit"]
        hud += [f"  logged @{r['frame']}: {r['line_id']} {r['direction']}" for r in here]
        for i, t in enumerate(hud):
            cv2.putText(view, t, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(view, t, (8, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if a.scale != 1.0:
            view = cv2.resize(view, None, fx=a.scale, fy=a.scale)
        cv2.imshow("ASPIRE annotate", view)
        # playing: wait one frame interval; paused: wait until a key is pressed
        key = cv2.waitKey(int(1000 / fps) if playing else 0) & 0xFF

        step = 0
        if key == ord(" "):
            playing = not playing
        elif key == ord("."):
            step = 1
        elif key == ord(","):
            step = -1
        elif key == ord("]"):
            step = 10
        elif key == ord("["):
            step = -10
        elif key == ord("l"):
            active = (active + 1) % len(lines)
        elif key in (ord("1"), ord("2")):
            d = ln["positive_label"] if key == ord("1") else ln["negative_label"]
            rows.append(dict(clip_id=clip_id, line_id=ln["id"], direction=d, frame=frame_no,
                             t_sec=round(frame_no / fps, 3), annotator=a.annotator, notes=""))
        elif key == ord("u") and rows:
            print("undo", rows.pop())
        elif key == ord("s"):
            save(out, rows)
        elif key == ord("q"):
            save(out, rows)
            break
        elif playing:
            step = 1  # no key pressed while playing: advance one frame

        if step:
            frame_no = max(0, min(n - 1, frame_no + step))
            # a plain read for +1 is much faster than seeking; seek only for jumps
            nxt = read(frame_no) if step != 1 else (cap.read()[1])
            if nxt is None:
                playing = False
            else:
                frame = nxt
    cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
