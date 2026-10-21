---
description: Draft this week's Monday supervision submission from real run outputs
argument-hint: "[due date, e.g. 2026-11-02]"
---

Draft the Monday supervision submission due $ARGUMENTS for ASPIRE-Vision.

1. Read `docs/fyp_plan.md` and find the task and reasoning question for that due date.
2. Find the runs from the reporting week: list `runs/` and read each `manifest.json`, `summary.json`, and `evaluation.json` if present.
   Run `pytest -q` and record the result.
3. Write `docs/weekN_submission.md` with these sections:
   - **Working example:** what ran end to end. Give the command, config file, video, run ID and dashboard path.
   - **Saved outputs and configuration:** list the files, and note the key config parameters that changed this week.
   - **Test / comparison results:** give expected vs. actual. Include count error, event precision/recall, throughput, and test pass count.
     Use only numbers from the files you read, and give the run ID next to each one.
   - **Failures:** concrete cases, with frame numbers and track IDs from `eval_matches.csv` or `events.csv`.
   - **Answer to the reasoning question:** an evidence-based answer that cites this week's runs and tests.
   - **Objective → feature mapping:** say which proposal objective each change serves.
   - **Next question to resolve.**
4. Never invent numbers or labels. If a required piece of evidence doesn't exist yet, write "not yet available" and say what is needed to get it.
5. Update the **Status** and **Next up** sections of `CLAUDE.md`.
