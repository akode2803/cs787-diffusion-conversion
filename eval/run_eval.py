"""Evaluates one arm on a list of tasks. Writes one JSON line per example and skips examples already written,
so rerunning the same command on a new VM continues where the lost one stopped.

Run on the VM, with BYOD installed:
  python3 run_eval.py --arm byod:qwen:hub:Ruurd/BYOD-Qwen2.5-7B --name qwen-byod-released --tasks gsm8k,mmlu --out /content/eval_out

Options that change the setting also change the file name, so settings never mix in one file:
  --steps N   diffusion steps per answer (default: one per position)      -> gsm8k_s64.jsonl
  --batch N   examples per batch (default 16); speed runs use 1             -> gsm8k_b1.jsonl
  --gap M     infilling gap length: true, x0.5, x1.5 or x2 (default true)  -> he_single_gapx2.jsonl
  --limit N   first N examples of each task (smoke tests)                   -> gsm8k_n8.jsonl
  --ar-tokens N  AR max new tokens on benchmark tasks (default BYOD's, 256/64); budget check only -> gsm8k_t512.jsonl
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tasks import (BENCHMARKS, INFILL, INFILL_AR_TOKENS, REVERSAL, REVERSAL_TOKENS, diffusion_infill_row, gap_length, load_benchmark,
                   load_infill, load_reversal, score_benchmark, score_infill_ar, score_infill_diffusion, score_reversal,
                   true_gap)


def budget(task):
    """(diffusion positions, AR max new tokens, delay end tokens) for a non-infilling task."""
    if task in REVERSAL:
        return REVERSAL_TOKENS, REVERSAL_TOKENS, True
    return BENCHMARKS[task][:3]

HARNESS = "eval-2026-11-08"


def tag_for(a, task):
    tag = ""
    if a.steps and task in BENCHMARKS:
        tag += f"_s{a.steps}"
    if a.batch != 16:
        tag += f"_b{a.batch}"
    if task in INFILL and a.gap != "true":
        tag += f"_gap{a.gap}"
    if getattr(a, "ar_tokens", None) and task in BENCHMARKS:
        tag += f"_t{a.ar_tokens}"
    if a.limit:
        tag += f"_n{a.limit}"
    return tag


def special_ids(tokenizer):
    return set(getattr(tokenizer, "all_special_ids", [])) | {tokenizer.eos_token_id}


def run_task(arm, task, a, out):
    from sampling import ar_batch, diffusion_batch
    from diffusion_lm.inference import _native_eot_token_id
    infill = task in INFILL
    reversal = task in REVERSAL
    examples = (load_infill(task, a.limit) if infill else load_reversal(task, a.limit, a.reversal_dir) if reversal
                else load_benchmark(task, a.limit))
    path = out / arm.name / f"{task}{tag_for(a, task)}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in open(path)} if path.is_file() else set()
    todo = [e for e in examples if e.id not in done]
    total, finished, correct = len(examples), len(done), 0
    print(f"TASK {arm.name} {task}{tag_for(a, task)}: {len(todo)} to do, {len(done)} already written", flush=True)
    tok, specials = arm.tokenizer, special_ids(arm.tokenizer)
    eot = 126348 if arm.llada else _native_eot_token_id(tok)

    # Build each example's model input once, then batch by length to waste little compute on padding.
    items = []
    for e in todo:
        if arm.kind == "diffusion":
            if infill:
                row, start, positions = diffusion_infill_row(arm, e, gap_length(tok, e, a.gap))
            else:
                prompt = arm.prompt_ids(e.prompt)
                positions = budget(task)[0]
                row, start = prompt + [arm.mask_token_id] * positions, len(prompt)
            items.append((e, row, start, positions))
        else:
            items.append((e, arm.prompt_ids(e.prompt), None, None))
    items.sort(key=lambda item: len(item[1]))

    with open(path, "a") as stream:
        for begin in range(0, len(items), a.batch):
            chunk = items[begin:begin + a.batch]
            rows = [item[1] for item in chunk]
            if arm.kind == "diffusion":
                steps = a.steps or max(item[3] for item in chunk)
                delay = True if infill else budget(task)[2]
                filled, commits, seconds = diffusion_batch(arm, rows, steps, delay_eos=delay, eot_id=eot)
                outputs = [(f[s:s + p], c[s:s + p]) for f, c, (_, _, s, p) in zip(filled, commits, chunk)]
            else:
                new, seconds = ar_batch(arm, rows, INFILL_AR_TOKENS if infill else a.ar_tokens or budget(task)[1])
                outputs = [(ids, None) for ids in new]
                steps = None
            for (e, row, _, positions), (ids, commit) in zip(chunk, outputs):
                record = {"arm": arm.name, "task": task, "id": e.id, "harness": HARNESS, "kind": arm.kind,
                          "batch": len(chunk), "batch_seconds": seconds, "steps": steps, "positions": positions,
                          "prompt_tokens": len(row) if arm.kind == "ar" else None, "answer_ids": ids,
                          "answer_tokens": sum(1 for t in ids if t not in specials),
                          "special_in_answer": sorted({t for t in ids if t in specials})}
                if infill:
                    record["true_gap"] = true_gap(tok, e)
                    if arm.kind == "diffusion":
                        ok, extra = score_infill_diffusion(e, tok.decode(ids, skip_special_tokens=True))
                    else:
                        ok, extra = score_infill_ar(e, tok.decode(ids, skip_special_tokens=True))
                    record.update(extra)
                    record["text"] = extra["middle"]
                else:
                    text = tok.decode(ids, skip_special_tokens=True).strip()
                    ok = score_reversal(e, text) if reversal else score_benchmark(e, text)
                    record["text"] = text
                record["correct"] = bool(ok)
                if commit is not None:
                    record["commit_step"] = commit
                stream.write(json.dumps(record) + "\n")
                correct += int(ok)
            stream.flush()
            finished += len(chunk)
            new_done = finished - len(done)
            print(f"step {finished}/{total} | {task} | {seconds / len(chunk):.2f} s/example | "
                  f"accuracy of new {correct}/{new_done}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tasks", required=True, help="comma separated: " + ",".join([*BENCHMARKS, *INFILL, *REVERSAL]))
    ap.add_argument("--out", default="/content/eval_out")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--gap", default="true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ar-tokens", type=int, help="AR budget override for benchmark tasks (not allowed for diffusion arms)")
    ap.add_argument("--workdir", default="/content", help="model caches live here")
    ap.add_argument("--reversal-dir", default="/content/reversal_data")
    a = ap.parse_args()
    # The gap-length sweep was dropped on 8 Nov to save compute, after the speed job had already started on its VM.
    # Uploading this file and an empty /content/skip_gap there makes the sweep's runs exit at once.
    if a.gap != "true" and os.path.exists(os.path.join(a.workdir, "skip_gap")):
        print(f"SKIPPED {a.name} gap {a.gap}: skip_gap is set", flush=True)
        return
    out = Path(a.out).resolve()
    os.chdir(a.workdir)
    from arms import load_arm, release
    started = time.time()
    arm = load_arm(a.arm, a.name)
    assert not (a.ar_tokens and arm.kind != "ar"), "--ar-tokens is for AR arms; diffusion answer length is the position count"
    print(f"LOADED {a.name} ({arm.kind}) in {time.time() - started:.0f} s", flush=True)
    for task in a.tasks.split(","):
        run_task(arm, task, a, out)
    release(arm)
    print(f"EVAL DONE {a.name}", flush=True)


if __name__ == "__main__":
    main()
