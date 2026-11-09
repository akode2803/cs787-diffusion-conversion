"""One-off GPU check of the harness before the real evaluation. Run on a G4 with BYOD installed and adapters fetched.

1. Diffusion parity: our batched sampler against BYOD's llada_generate, on the released BYOD-Qwen adapter.
2. AR parity: batched against one-at-a-time greedy generation, on the Qwen parent.
3. Speed: seconds per denoising step at batch 1 and 16, and AR seconds per answer, at GSM8K length.
4. Smoke run of every task, 4 examples each, on one diffusion arm and one AR arm.
5. Our own adapters (AR control, retrained conversion) load and answer GSM8K.
6. LLaDA loads with this transformers version.
Each part reports and moves on if it fails, so one failure does not hide the others.
"""
import json, os, sys, time, traceback, types
from pathlib import Path

sys.path.insert(0, "/content")
os.chdir("/content")
from arms import load_arm, release
from sampling import ar_batch, diffusion_batch
import run_eval
from tasks import BENCHMARKS, INFILL, load_benchmark

OUT = Path("/content/eval_check")
REPORT = OUT / "gpu_check.json"
OUT.mkdir(parents=True, exist_ok=True)
report = json.loads(REPORT.read_text()) if REPORT.is_file() else {}


def part(name):
    def wrap(fn):
        if report.get(name, {}).get("status") == "ok":
            print(f"SKIP {name}: done before", flush=True)
            return
        print(f"PART {name}", flush=True)
        try:
            report[name] = {"status": "ok", **(fn() or {})}
        except Exception as exc:
            report[name] = {"status": "failed", "error": repr(exc)[:500]}
            # Indented, because the supervisor treats a line starting with "Traceback" as a fatal job error.
            print("\n".join("    " + line for line in traceback.format_exc().splitlines()), flush=True)
        REPORT.write_text(json.dumps(report, indent=2))
        print(f"RESULT {name}: {json.dumps(report[name])[:600]}", flush=True)
    return wrap


def summarise(name):
    result = {}
    for path in sorted((OUT / name).glob("*.jsonl")):
        rows = [json.loads(line) for line in open(path)]
        result[path.stem] = f"{sum(r['correct'] for r in rows)}/{len(rows)} correct, median answer tokens {sorted(r['answer_tokens'] for r in rows)[len(rows) // 2]}"
    return result


questions = [e.prompt for e in load_benchmark("gsm8k", 6)]
arm = load_arm("byod:qwen:hub:Ruurd/BYOD-Qwen2.5-7B", "qwen-byod-released")


@part("diffusion_parity")
def _():
    from diffusion_lm.inference import llada_generate
    from arms import SYSTEM_PROMPT
    n = 64
    theirs = [llada_generate(arm.session, q, gen_length=n, steps=n, block_length=n, temperature=0.0,
                             confidence_eos_eot_inf=True, eot_token_id=None, system_prompt=SYSTEM_PROMPT) for q in questions]
    rows = [arm.prompt_ids(q) + [arm.mask_token_id] * n for q in questions]
    starts = [len(arm.prompt_ids(q)) for q in questions]
    batched, _, _ = diffusion_batch(arm, rows, n, delay_eos=True)
    single = [diffusion_batch(arm, [r], n, delay_eos=True)[0][0] for r in rows]
    text = lambda ids: arm.tokenizer.decode(ids, skip_special_tokens=True).strip()
    ours_b = [text(r[s:]) for r, s in zip(batched, starts)]
    ours_1 = [text(r[s:]) for r, s in zip(single, starts)]
    return {"single_equals_byod": sum(a == b for a, b in zip(ours_1, theirs)), "batched_equals_byod": sum(a == b for a, b in zip(ours_b, theirs)),
            "of": len(questions), "example_byod": theirs[0][:200], "example_batched": ours_b[0][:200]}


@part("batch_numerics")
def _():
    """Six copies of one prompt need no padding. If they still differ from batch 1, the cause is batch-shape numerics."""
    n = 64
    row = arm.prompt_ids(questions[0]) + [arm.mask_token_id] * n
    single = diffusion_batch(arm, [row], n, delay_eos=True)[0][0]
    copies = diffusion_batch(arm, [row] * 6, n, delay_eos=True)[0]
    return {"identical_copies_equal_single": sum(c == single for c in copies), "of": 6}


@part("diffusion_speed")
def _():
    out = {}
    for batch in (1, 16):
        rows = [arm.prompt_ids(questions[i % len(questions)]) + [arm.mask_token_id] * 256 for i in range(batch)]
        diffusion_batch(arm, rows, 4)                         # warm-up
        _, _, seconds = diffusion_batch(arm, rows, 32)
        out[f"seconds_per_step_batch{batch}"] = seconds / 32
    out["gsm8k_500_hours_at_256_steps_batch16"] = out["seconds_per_step_batch16"] * 256 * (500 / 16) / 3600
    return out


@part("smoke_diffusion")
def _():
    a = types.SimpleNamespace(steps=None, batch=4, gap="true", limit=4)
    for task in [*BENCHMARKS, *INFILL]:
        run_eval.run_task(arm, task, a, OUT)
    return summarise("qwen-byod-released")


release(arm)
arm = load_arm("parent:qwen", "qwen-parent")


@part("ar_parity_and_speed")
def _():
    prompts = [arm.prompt_ids(q) for q in questions]
    batched, seconds_batch = ar_batch(arm, prompts, 256)
    single, seconds_single = [], 0.0
    for p in prompts:
        ids, s = ar_batch(arm, [p], 256)
        single.append(ids[0]); seconds_single += s
    return {"batched_equals_single": sum(a == b for a, b in zip(batched, single)), "of": len(prompts),
            "seconds_per_answer_batch1": seconds_single / len(prompts), "seconds_per_answer_batch6": seconds_batch / len(prompts),
            "example": arm.tokenizer.decode(batched[0], skip_special_tokens=True)[:200]}


@part("smoke_ar")
def _():
    a = types.SimpleNamespace(steps=None, batch=4, gap="true", limit=4)
    for task in [*BENCHMARKS, *INFILL]:
        run_eval.run_task(arm, task, a, OUT)
    return summarise("qwen-parent")


release(arm)


@part("own_adapters")
def _():
    out = {}
    a = types.SimpleNamespace(steps=None, batch=4, gap="true", limit=4)
    for spec, name in (("ar:qwen:/content/adapters/qwen-ar", "qwen-ar-control"),
                       ("byod:qwen:/content/adapters/qwen-byod-retrained", "qwen-byod-retrained"),
                       ("ar:ministral:/content/adapters/ministral-ar", "ministral-ar-control")):
        own = load_arm(spec, name)
        run_eval.run_task(own, "gsm8k", a, OUT)
        out.update({f"{name}/{k}": v for k, v in summarise(name).items()})
        release(own)
    return out


@part("llada_loads")
def _():
    llada = load_arm("llada", "llada")
    a = types.SimpleNamespace(steps=None, batch=2, gap="true", limit=2)
    run_eval.run_task(llada, "gsm8k", a, OUT)
    release(llada)
    return summarise("llada")


print("CHECK DONE", json.dumps({k: v.get("status") for k, v in report.items()}), flush=True)
