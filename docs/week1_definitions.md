# ASPIRE-Vision — Week 1: Measurement definitions, annotation rules and data dictionary

Adam Zafry bin Zaharin · Supervisor: Dr. Fadratul Hafinaz Hassan · Submission for Mon 26 Oct 2026

---

## 1. Scope of this week

- **Video source:** USM approval pending (see §7). Until it arrives, the pipeline is checked on OpenCV's sample pedestrian clip `vtest.avi`, used only as a smoke test. A licensed public benchmark will be the fallback for reported results.
- **Defined here:** what the system counts (§2), how humans label ground truth (§3), what every exported field means (§4) and how results will be judged (§5).
- **Built this week:** a minimal end-to-end workflow.

  `video → YOLOv8n person detection → ByteTrack → line crossings + zone occupancy → CSV export → HTML dashboard → evaluation against annotations`

All positions are **image-space pixels**. No physical speed, distance or density is claimed unless ground-plane calibration is done.

---

## 2. What is being measured

### 2.1 Three different quantities

| Quantity | Definition | Unit | Can one person contribute more than once? | Used for |
|---|---|---|---|---|
| **Crossing event** | A tracked person's anchor point moves from one confirmed side of a counting line to the other, inside the line's extent | events per line, per direction, per time window | **Yes, legitimately.** Walking in, then later walking out, gives 2 events. | Footfall / flow counts: the headline number |
| **Occupancy** | Number of tracked anchor points inside a zone polygon in a given frame | people present at time *t* (time series) | Yes. The same person is present in many frames; occupancy is a level, not a total. | Crowding / dwell at a hub |
| **Unique track IDs** | Number of distinct tracker IDs in a recording | IDs | **Yes, wrongly.** Each ID switch or re-acquisition creates a new ID. | Tracker diagnostic only, **never** reported as a person count |

**Answer to the week-1 question.** The primary measure is **directional crossings**, with occupancy as a secondary measure. A single person can produce several legitimate crossing events, for example by returning through the same doorway. So the count is "crossings", not "unique people". Unique track IDs are not a person count, because the tracker can assign one person several IDs, so they are reported only as a diagnostic of tracking quality.

### 2.2 Crossing rule (as implemented in `aspire/counting.py`)

A crossing event is emitted for track *k* on line *L* when all of the following hold:

1. **Anchor point:** the bottom-centre of the bounding box, which approximates the feet on the ground. It is more stable than the box centre when the upper body is occluded.
2. **Confirmed side change:** the anchor's signed distance to *L* goes from below `−band_px` to above `+band_px`, or the reverse. Inside the band (±8 px by default) the previous side is kept. This hysteresis stops box jitter on the line from creating repeated events.
3. **Within the segment:** the crossing point lies within the line segment (plus `segment_margin_px`), not on its infinite extension.
4. **Track maturity:** track *k* has been observed for at least `min_track_hits` frames (default 3). This suppresses one- or two-frame false tracks.
5. **Cooldown:** track *k* has not produced an event on *L* in the last `cooldown_frames` frames (default 10, i.e. 1 s at 10 fps). Genuine returns after the cooldown still count.
6. **Start side, no event:** a track first seen on a side records that side as its start and emits nothing until it actually changes side.

**Direction labels.** Each line is drawn from `p1` to `p2`. The positive side is screen-right of `p1→p2`. A move from the negative side to the positive side is labelled `positive_label` (e.g. `left_to_right`), and the reverse is labelled `negative_label`.

**Known failure mode (week-2 question, previewed).** An ID switch near the line can duplicate a count. Track 1 crosses and is counted. The tracker then loses the person and re-acquires them as track 2, whose first confirmed position falls back on the original side because of a jittery box. Track 2 then "crosses" again. The per-track cooldown cannot prevent this because the ID has changed. The unit test `test_id_switch_near_line_can_duplicate_a_count` reproduces it, and `test_wider_band_mitigates_that_duplicate` shows one mitigation. Week 2 will measure how often this happens on real footage.

### 2.3 Occupancy rule

Occupancy of zone *Z* at frame *f* is the number of confirmed tracks whose anchor lies inside the polygon *Z*. It is exported per frame and summarised as mean and max. It is an image-space count, not a density.

### 2.4 Track IDs

Track IDs are temporary technical identifiers, valid only within a single recording. They carry no identity information, are not linked across cameras or days, and are never stored with any personal attribute.

---

## 3. Annotation rules (ground truth for crossings)

Ground truth is produced **independently of the model**. `scripts/annotate.py` shows only the raw video and the counting line, never detections or tracks.

1. **Who annotates:** preferably someone other than the developer. If the developer must annotate, do it before looking at that clip's predictions. A second annotator labels at least 20% of clips, so that inter-annotator agreement can be reported.
2. **Who counts as a pedestrian:**
   - Count anyone on foot: walking, running, standing, carrying or pushing something, or using a wheelchair.
   - Exclude people riding a bicycle or motorcycle, people inside vehicles, and reflections, posters or screens.
   - Note ambiguous cases in `notes` and do not count them.
3. **When to log a crossing:** at the frame where the person's **feet** pass the line, matching the bottom-centre anchor.
   - If the feet are hidden, log the frame where they can reasonably be inferred to cross, and write `occluded` in `notes`.
4. **One label per person per crossing.** In a group, label each person separately.
5. **No partial crossings.** A person who steps onto the line and turns back without reaching the other side gets no label.
6. **Re-crossings count.** A person who crosses and later crosses back gets two labels, one per direction.
7. **Direction** uses the line's configured labels (`positive_label` / `negative_label`), shown on screen in the tool.
8. **Clip boundaries:** do not label crossings in the first or last 1 s of a clip, where tracks are truncated. The evaluation also ignores frames beyond those processed.
9. **Matching tolerance:** a prediction matches an annotation if it is on the same line, in the same direction, and within **±1 s**. This absorbs the difference between "feet on the line" and the tracker's confirmed side change.
10. **Splits:** clips are assigned to `dev` or `test` by **whole recording** (see `data/recording_log.csv`). Frames from one recording never appear in both splits. Thresholds are tuned on `dev` only.
11. **Box-level labels** (needed for MOTA, IDF1 and HOTA) are a separate, more expensive task. They will be done in CVAT, exported in MOTChallenge format, and only for short sub-clips. Tracking metrics are reported only where such labels exist.

---

## 4. Data dictionary

Each run writes a folder `runs/<run_id>/`. The run ID is `<location_code>_<UTC timestamp>_<config hash>`, so every number can be traced back to the exact configuration and video that produced it.

### 4.1 `manifest.json`: what was run
`run_id`, `created_utc`, `config_file`, `config_hash`, the full `config` used, video details (`path`, `sha256_first_64MB`, `fps`, `frames_processed`, `width`, `height`), `model_sha256`, `detector_backend`, `tracker`, `timing` (`wall_sec`, `throughput_fps`, `detector_ms_per_frame`, `tracker_ms_per_frame`), library `versions`, and `units`.

### 4.2 `tracks.csv`: trajectories (one row per track per frame)
| Field | Type | Unit | Meaning |
|---|---|---|---|
| run_id | str | | Run identifier |
| frame | int | frame | 0-based frame index |
| t_sec | float | s | `frame / fps` (time into the clip) |
| track_id | int | | Temporary ID within this recording |
| x1, y1, x2, y2 | float | px | Box corners (top-left, bottom-right) |
| anchor_x, anchor_y | float | px | Bottom-centre anchor point |
| conf | float | 0–1 | Detector confidence of the matched box |

### 4.3 `events.csv`: crossing events (the single source of truth for counts)
| Field | Type | Unit | Meaning |
|---|---|---|---|
| run_id | str | | Run identifier |
| event_id | int | | 1..N, sequential within the run |
| frame, t_sec | int, float | frame, s | When the confirmed side change occurred |
| line_id | str | | Counting line (from config) |
| direction | str | | `positive_label` or `negative_label` of that line |
| track_id | int | | Track that crossed |
| anchor_x, anchor_y | float | px | Anchor position at the event |

### 4.4 `occupancy.csv`
`run_id, frame, t_sec, zone_id, occupancy` (int: people in zone at that frame).

### 4.5 `summary.json`
Totals **derived from** the CSVs above, never counted separately: `crossings[line][direction]`, `total_crossing_events`, `occupancy[zone]{mean,max,frames}`, `unique_track_ids` (diagnostic).

### 4.6 `dashboard.html`
- Recomputes every displayed number from `events.csv` and `occupancy.csv`.
- Runs consistency checks against `summary.json`: per-direction totals, row counts, sequential event IDs, run-ID integrity and zone maxima.
- Shows **FAILED** if anything disagrees. This is the planned answer to the week-3 question.

### 4.7 Annotation file (`annotations/*_crossings_<annotator>.csv`)
`clip_id, line_id, direction, frame, t_sec, annotator, notes`

### 4.8 Evaluation outputs
- `evaluation.json`: per line/direction `annotated`, `predicted`, `count_error`, `abs_pct_error`, `tp`, `fp`, `fn`, `precision`, `recall`, plus overall figures.
- `eval_matches.csv`: every TP/FP/FN, with frames and track IDs, for error inspection.

### 4.9 Recording log (`data/recording_log.csv`)
`location_code, video_file, split, date, start_time, duration_s, camera, resolution, fps, lighting, weather, occlusion_level, crowd_level, permission_ref, notes`

---

## 5. How results will be judged

| Aspect | Metric | Needs |
|---|---|---|
| Counting | Per-direction count error (pred − GT), absolute % error | Crossing labels (cheap) |
| Counting, event level | Precision / recall of matched events (±1 s) | Crossing labels |
| Tracking quality | MOTA, IDF1, HOTA, ID switches (via TrackEval) | Box-level labels on sub-clips |
| Compute | Throughput (fps), detector / tracker ms per frame | Logged in every manifest |
| Robustness | All of the above, sliced by lighting / occlusion / crowd level | Recording log |

Count error alone can hide cancelling mistakes: one false event plus one missed event gives a "perfect" total. Event-level matching is therefore always reported alongside it. The test `test_cancelling_errors_hidden_by_count_but_visible_in_matching` demonstrates this.

---

## 6. Week-1 working example (smoke test on `vtest.avi`)

- **Setup:** YOLOv8n (ONNX, CPU) + ByteTrack, 795 frames at 10 fps, one vertical line L1 and one zone Z1.
- **Output:**
  - 27 crossing events: 13 `left_to_right`, 14 `right_to_left`.
  - Zone Z1 mean occupancy 3.1, max 6.
  - 35 unique track IDs (diagnostic).
  - Throughput 12.0 fps on a 2-core cloud CPU, about 76 ms per frame for the detector.
- **Consistency checks:** all pass.
- **Unit tests:** 18 pass.

**Observations and failures:**
- **Line placement matters.** The first line position (x = 430) ran along a lamp post that permanently hides part of the line, so the line was moved to clear tarmac (x = 370). Real camera sites need the same check, recorded in the recording log.
- **Several tracks cross in both directions** (e.g. #10, #14, #32, #34, #38). These may be people genuinely turning back, or jitter or ID-switch artefacts. They are the first candidates for week-2 error inspection.
- **No accuracy figures yet.** This clip has no independent annotations, and it is not campus data.

---

## 7. Open items for supervision

1. Status of approval for USM recordings: location, CCTV access or arranged recording, and data-handling conditions.
2. Choice of licensed public fallback, e.g. a MOTChallenge sequence (licence terms to be checked). Its box-level ground truth also enables MOTA, IDF1 and HOTA.
3. Who can act as the independent (second) annotator.
4. Planned second tracker configuration (week 3): DeepSORT, which adds appearance embeddings, versus ByteTrack (motion only) on the same detections.
