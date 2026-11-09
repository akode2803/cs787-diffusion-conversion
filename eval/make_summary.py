"""Builds docs/results/summary.json, the single source of every number in the report and its figures.

  python3 -I make_summary.py RESCORED_DIR REVERSAL_DIR INFILL_DIR OUT.json

RESCORED_DIR is rescore_infill.py's output over eval/main (so AR infilling uses the current scorer and *_strict files
exist). REVERSAL_DIR holds the rev_forward/rev_reverse files. INFILL_DIR holds the two HumanEval infilling jsonl
files, needed to rerun the gap-sweep answers with the fence cut (see fence_rescore.py).
"""
import json, math, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze, tasks

res, rev, infill_dir, out = map(Path, sys.argv[1:5])
files = analyze.load(res)


def cell(rows):
    values = list(rows.values())
    k, n = sum(bool(r["correct"]) for r in values), len(values)
    lo, hi = analyze.wilson(k, n)
    taus = [t for t in (analyze.decoding_stats(r) for r in values if "commit_step" in r) if t is not None and not math.isnan(t)]
    return {"k": k, "n": n, "acc": k / n, "lo": lo, "hi": hi,
            "s_per_example": statistics.mean(r["batch_seconds"] / r["batch"] for r in values),
            "median_tokens": statistics.median(r["answer_tokens"] for r in values),
            "doubled_per_100": statistics.mean(analyze.doubled_word_rate(r.get("text", "")) for r in values),
            "kendall_tau": statistics.median(taus) if taus else None,
            "context_changed": sum(1 for r in values if r.get("context_changed"))}


def paired(a, b):
    ids = sorted(set(a) & set(b))
    x = [int(bool(a[i]["correct"])) for i in ids]; y = [int(bool(b[i]["correct"])) for i in ids]
    lo, hi = analyze.paired_bootstrap(x, y, reps=10000)
    n10 = sum(1 for p, q in zip(x, y) if p and not q); n01 = sum(1 for p, q in zip(x, y) if q and not p)
    return {"diff": (sum(x) - sum(y)) / len(ids), "lo": lo, "hi": hi, "p": analyze.mcnemar_p(n10, n01), "n": len(ids)}


summary = {"cells": {f"{arm}/{stem}": cell(rows) for (arm, stem), rows in files.items()}, "pairs": {}}
pairs = [("qwen-ar-control", "qwen-byod-retrained"), ("qwen-ar-control", "qwen-byod-released"),
         ("qwen-byod-released", "qwen-byod-retrained"), ("qwen-parent", "qwen-ar-control"),
         ("ministral-ar-control", "ministral-byod-released"), ("ministral-parent", "ministral-ar-control")]
for a, b in pairs:
    for (arm, stem), rows in files.items():
        if arm == a and (b, stem) in files:
            summary["pairs"][f"{a}|{b}|{stem}"] = paired(rows, files[(b, stem)])

# Gap sweep: diffusion middles cut at the first code fence, the rule AR answers already get (extract_middle).
examples = {e.id: e for e in tasks.load_infill("he_single", path=infill_dir / f"{tasks.INFILL['he_single']}.jsonl")}
summary["gap_fence_cut"] = {}
for arm in ("qwen-byod-retrained", "qwen-byod-released", "llada"):
    for g in ("true", "x0.5", "x1.5", "x2"):
        rows = files[(arm, "he_single" if g == "true" else f"he_single_gap{g}")]
        k = 0
        for r in rows.values():
            middle = r.get("middle", r["text"])
            cut = middle.split("```")[0]
            k += bool(r["correct"]) if cut == middle else bool(tasks.run_program(tasks.infill_program(examples[r["id"]], cut)))
        lo, hi = analyze.wilson(k, len(rows))
        summary["gap_fence_cut"][f"{arm}/{g}"] = {"k": k, "n": len(rows), "acc": k / len(rows), "lo": lo, "hi": hi}

summary["reversal"] = {}
for path in sorted(rev.glob("*/rev_*.jsonl")):
    rows = [json.loads(line) for line in open(path)]
    k = sum(bool(r["correct"]) for r in rows)
    summary["reversal"][f"{path.parent.name}/{path.stem}"] = {"k": k, "n": len(rows)}

out.parent.mkdir(parents=True, exist_ok=True)
json.dump(summary, open(out, "w"), indent=1, sort_keys=True)
print(f"wrote {out}: {len(summary['cells'])} cells, {len(summary['pairs'])} pairs")
