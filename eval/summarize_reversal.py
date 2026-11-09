"""Prints one RESULT line per arm and direction of the reversal test, for the job log and the runner's report.

  python3 summarize_reversal.py EVAL_OUT_DIR [NAME_FILTER]
e.g. RESULT qwen-byod-retrained+factsnf10k rev_reverse 12/1000 = 0.012
Reads <dir>/<arm>/rev_forward.jsonl and rev_reverse.jsonl (one record per fact, with a boolean "correct").
"""
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
keep = sys.argv[2] if len(sys.argv) > 2 else ""
for arm in sorted(p for p in out.iterdir() if p.is_dir() and keep in p.name):
    for task in ("rev_forward", "rev_reverse"):
        f = arm / f"{task}.jsonl"
        if f.exists():
            rows = [json.loads(l) for l in open(f) if l.strip()]
            k = sum(bool(r.get("correct")) for r in rows)
            print(f"RESULT {arm.name} {task} {k}/{len(rows)} = {k / max(len(rows), 1):.3f}", flush=True)
