import json, os, sys, statistics as st
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import types; ft=types.ModuleType("torch"); ft.inference_mode=lambda:(lambda f:f); ft.nn=types.SimpleNamespace(functional=None)
sys.modules.setdefault("torch",ft); sys.modules.setdefault("torch.nn",ft.nn); sys.modules.setdefault("torch.nn.functional",types.ModuleType("x"))
import analyze
R = Path(sys.argv[1])
def load(arm, stem):
    p = R/arm/f"{stem}.jsonl"
    return {r["id"]: r for r in map(json.loads, open(p))} if p.exists() else None
def acc(rows): 
    k=sum(bool(r["correct"]) for r in rows.values()); n=len(rows); lo,hi=analyze.wilson(k,n); return f"{k/n:.3f} ({lo:.3f}-{hi:.3f})"
def spe(rows):  # seconds per example: batch_seconds is per batch
    rs=list(rows.values()); return sum(r["batch_seconds"]/r.get("batch",1) for r in rs)/len(rs) if "batch_seconds" in rs[0] else float("nan")
def diff(a,b):
    ids=sorted(set(a)&set(b)); x=[int(bool(a[i]["correct"])) for i in ids]; y=[int(bool(b[i]["correct"])) for i in ids]
    lo,hi=analyze.paired_bootstrap(x,y,reps=4000); d=(sum(x)-sum(y))/len(ids)
    n10=sum(1 for p,q in zip(x,y) if p and not q); n01=sum(1 for p,q in zip(x,y) if q and not p)
    return f"{d:+.3f} ({lo:+.3f} to {hi:+.3f}), p={analyze.mcnemar_p(n10,n01):.3g}, n={len(ids)}"
print("## Steps vs accuracy, batch 16 (256 positions)\n\n| Arm | Task | Steps | Accuracy (95% CI) | s/example | vs 256 steps, paired |\n|---|---|---|---|---|---|")
for arm in ("qwen-byod-released","qwen-byod-retrained"):
    for task, steps in (("gsm8k",(32,64,128)),("humaneval",(64,128))):
        full=load(arm,task)
        for s in steps:
            r=load(arm,f"{task}_s{s}")
            if r: print(f"| {arm} | {task} | {s} | {acc(r)} | {spe(r):.2f} | {diff(r,full)} |")
        print(f"| {arm} | {task} | 256 | {acc(full)} | {spe(full):.2f} | — |")
print("\n## Batch-1 latency, GSM8K, first 32 examples\n\n| Arm | Steps | Accuracy | s/example (mean) | median answer tokens |\n|---|---|---|---|---|")
for arm in ("qwen-parent","qwen-ar-control"):
    r=load(arm,"gsm8k_b1_n32"); print(f"| {arm} | AR | {sum(x['correct'] for x in r.values())}/32 | {spe(r):.2f} | {st.median(x['answer_tokens'] for x in r.values())} |")
for arm in ("qwen-byod-released","qwen-byod-retrained"):
    for s in (32,64,128,256):
        r=load(arm,f"gsm8k_s{s}_b1_n32"); print(f"| {arm} | {s} | {sum(x['correct'] for x in r.values())}/32 | {spe(r):.2f} | {st.median(x['answer_tokens'] for x in r.values())} |")
print("\n## Gap-length sweep, single-line infilling (500 examples; diffusion is told the gap length)\n\n| Arm | x0.5 | x1 (true) | x1.5 | x2 |\n|---|---|---|---|---|")
for arm in ("qwen-byod-retrained","qwen-byod-released","llada"):
    cells=[acc(load(arm,f"he_single_gap{g}") if g!="true" else load(arm,"he_single")) for g in ("x0.5","true","x1.5","x2")]
    print(f"| {arm} | "+" | ".join(cells)+" |")
print("\nPaired differences, wrong length minus true length:\n")
for arm in ("qwen-byod-retrained","qwen-byod-released","llada"):
    t=load(arm,"he_single")
    for g in ("x0.5","x1.5","x2"): print(f"- {arm} {g}: {diff(load(arm,f'he_single_gap{g}'),t)}")
