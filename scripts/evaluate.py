"""Evaluate a finished run against a crossing-annotation CSV and rebuild its dashboard.

    python scripts/evaluate.py runs/<run_id> annotations/demo_vtest_crossings.csv
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import aspire/ without installing it

from aspire.dashboard import build_report  # noqa: E402
from aspire.evaluate import evaluate  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("run_dir")
p.add_argument("annotations")
p.add_argument("--tol-sec", type=float, default=1.0, help="matching tolerance in seconds (default 1.0)")
a = p.parse_args()

rep = evaluate(a.run_dir, a.annotations, a.tol_sec)
print(json.dumps(rep["overall"], indent=2))
print(f"dashboard: {build_report(a.run_dir)}")  # rebuilt so it shows the new evaluation.json
