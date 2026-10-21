# FYP plan — supervisor guidance and proposal summary

Source documents: `CAT405_Init_Proposal_ASPIREVision_AdamZafryZaharin.pdf` (initial proposal) and
`Adam_Zafry_FYP_Plan.docx` (supervisor's plan). The supervisor's plan is reproduced verbatim in §2.

## 1. Proposal summary (CAT405, session 2026/2027)

- **Title:** ASPIRE-Vision: Using Video to Count and Track Pedestrians on Campus.
- **Student / supervisor:** Adam Zafry bin Zaharin / Dr. Fadratul Hafinaz Hassan (School of Computer Sciences, USM).
- **Problem:** USM monitors foot traffic by manual observation, or not at all, so it has no empirical data for infrastructure planning.
- **Research question:** how reliably can a fixed-camera system count pedestrians and preserve trajectories under changing illumination and occlusion?
- **Objectives:**
  1. Build a video pipeline that detects pedestrians, tracks them within each recording, and measures directional crossings and occupancy.
  2. Build a dashboard and structured data export for counts, trajectories and recording conditions.
  3. Compare **two tracking configurations** on independently annotated videos, analysing counting error, tracking quality (e.g. MOTA) and computational performance.
- **Method:** a CNN detector (YOLOv8) plus a multi-object tracker (ByteTrack / DeepSORT).
- **Data:** recorded video from one fixed camera at one approved campus location, with a second location for independent testing if feasible. A public benchmark is the fallback. A recording log is kept: location, date/time, lighting, occlusion, weather, crowd level.
- **Scoping:**
  - Track IDs are temporary, within one recording; cross-camera identity matching is out of scope.
  - Results are image-space unless ground-plane calibration is done.
  - Live CCTV, 24-hour operation and facility control are future work.
- **SDGs:** 9 and 11.
- **Gantt milestones:**
  - SRD report and presentation (Milestone 1) in weeks 5–8.
  - Development sprints in weeks 9–20.
  - Progress demo (Milestone 2) in weeks 21–22.
  - Final report and demo (Milestone 3) in weeks 30–32.

## 2. Supervisor's plan (verbatim)

Develop all objectives together through small complete versions of your system. Each cycle should connect the intelligent method, interface, saved outputs and testing. The weekly targets below are supervision targets; formal assessment follows the current guideline and eLearn notices.

**Minimum complete workflow:**
Load an approved video -> detect and track pedestrians -> count crossings -> display counts and trajectories -> save/export a record -> compare with annotations.

**First four Monday submissions:**
Submit every Monday from 26 October 2026, reporting the preceding week. Include a working example, saved outputs and configuration, a comparison or test result, failures and an evidence-based answer to the reasoning question. Keep improving the complete workflow.

| Monday due date | Task and reasoning question |
|---|---|
| Mon 26 Oct 2026 | Confirm approved video or a licensed public fallback. Define count semantics, annotation rules, trajectory fields and a simple end-to-end interface. **Question:** Are you measuring occupancy, unique IDs or crossings, and can one person contribute multiple legitimate events? |
| Mon 2 Nov 2026 | Connect a pretrained detector and one tracker to recorded video. Display trajectories and a crossing count; export a sample record and inspect reference errors. **Question:** How can an identity switch near the counting line create a duplicate count? |
| Mon 9 Nov 2026 | Build the minimal dashboard and connect exports to its summaries. Prepare independently labelled pilot clips and a second tracker configuration if feasible. **Question:** How will you verify that the dashboard, event log and exported counts describe the same events? |
| Mon 16 Nov 2026 | Demonstrate the complete video-to-dashboard-to-evaluation workflow. Record counting and tracking errors, throughput and failure cases; prepare SRD diagrams and testing partitions. **Question:** If tracking improves but counting error does not, which claim is supported by the results? |

**Scope to protect:**
Recorded single-camera video, one detector, tracking, directional counts, a simple dashboard and CSV export. Defer live CCTV, cross-camera identity matching, automated facility control and complex alerts.

**Evidence of correct and reliable work:**
Keep complete recordings in separate development/test sets. Calibrate ground-plane coordinates before physical speed/density claims. Separate counts from occupancy and temporary IDs. Use box/identity metrics only where suitable reference labels exist.

**What to bring to supervision:**
A working example; architecture and objective-to-feature mapping; configuration and data dictionary; sample output; expected-versus-actual test results; failures; and the next question to resolve. Explain the intelligent method, why it was chosen and how it contributes to the integrated system.

## 3. Answers to the reasoning questions

The answers here are the working position; each Monday report must back them with evidence from that week's runs.

1. **Mon 26 Oct.** Primary measure: directional crossing *events*; occupancy is secondary.
   - One person can legitimately produce several events (in, then out).
   - Unique track IDs are only a tracker diagnostic, because ID switches inflate them.
   - Full answer: `docs/week1_definitions.md` §2.
2. **Mon 2 Nov.** Duplicate count via ID switch: track A crosses and is counted, then is lost. The same person is re-acquired as track B, first confirmed on the original side because of box jitter, and B "crosses" again.
   - The per-track cooldown can't stop this, because the ID changed.
   - Mitigations: wider hysteresis band, minimum track age, re-acquisition near the line.
   - Reproduced in `tests/test_counting.py::test_id_switch_near_line_can_duplicate_a_count`. Still to do: measure how often it happens on real clips.
3. **Mon 9 Nov.** `events.csv` is the single source of truth; the summary and the dashboard are derived from it.
   - The dashboard re-counts and runs PASS/FAIL consistency checks, tested in `tests/test_evaluate_and_exports.py`.
   - Run IDs and config hashes tie every number to its run.
4. **Mon 16 Nov.** Only the tracking-quality claim is supported. Counting depends on events near the line, counting errors can cancel out, and detection misses can dominate.
   - Always report event-level precision/recall next to count error.
   - Demonstrated in `test_cancelling_errors_hidden_by_count_but_visible_in_matching`.
