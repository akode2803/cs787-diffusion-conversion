"""Rescores diffusion single-line infilling with the leftover-position junk removed: the middle is cut at the first
code fence, as the AR scorer's extract_middle does for AR answers. Raw files untouched."""
import json, os, sys
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tasks
R = Path(sys.argv[1]); D = Path(sys.argv[2])
ex = {e.id: e for e in tasks.load_infill("he_single", path=D / f"{tasks.INFILL['he_single']}.jsonl")}
out = {}
for arm in ("qwen-byod-retrained", "qwen-byod-released", "llada"):
    for g in ("true", "x0.5", "x1.5", "x2"):
        stem = "he_single" if g == "true" else f"he_single_gap{g}"
        rows = [json.loads(l) for l in open(R / arm / f"{stem}.jsonl")]
        old = new = changed = 0
        for r in rows:
            m = r.get("middle", r["text"])
            cut = m.split("```")[0]
            if cut != m:
                changed += 1
                ok = tasks.run_program(tasks.infill_program(ex[r["id"]], cut))
            else:
                ok = bool(r["correct"])
            old += bool(r["correct"]); new += bool(ok)
        out[(arm, g)] = (old, new, changed, len(rows))
        print(f"{arm} {g}: as scored {old}/{len(rows)} = {old/len(rows):.3f}; fence cut {new}/{len(rows)} = {new/len(rows):.3f} ({changed} answers had a fence)", flush=True)
