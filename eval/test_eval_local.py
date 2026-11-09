"""Checks the parts of the harness that need no GPU or torch. Run on the Mac: python3 test_eval_local.py INFILL_DIR

INFILL_DIR holds the two HumanEval infilling .jsonl files. The key check: every canonical middle, spliced between
its prefix and suffix, must pass the benchmark tests, or infilling scores would be wrong for every arm.
"""
import math, sys, types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
fake_torch = types.ModuleType("torch")              # sampling.py needs torch only to run models
fake_torch.inference_mode = lambda: (lambda f: f)
fake_torch.nn = types.SimpleNamespace(functional=None)
sys.modules.setdefault("torch", fake_torch)
sys.modules.setdefault("torch.nn", fake_torch.nn)
sys.modules.setdefault("torch.nn.functional", types.ModuleType("torch.nn.functional"))

import analyze, tasks
from sampling import schedule
from run_eval import tag_for

failures = []


def check(name, condition, detail=""):
    print(("PASS " if condition else "FAIL ") + name + (f"  {detail}" if detail else ""))
    if not condition:
        failures.append(name)


infill_dir = Path(sys.argv[1])
for task, file in tasks.INFILL.items():
    examples = tasks.load_infill(task, path=infill_dir / f"{file}.jsonl")
    again = tasks.load_infill(task, path=infill_dir / f"{file}.jsonl")
    check(f"{task}: sample is 500 and repeatable", len(examples) == 500 and [e.id for e in examples] == [e.id for e in again])
    ok = sum(tasks.run_program(tasks.infill_program(e, e.data["canonical_solution"])) for e in examples)
    check(f"{task}: canonical middles pass", ok == len(examples), f"{ok}/{len(examples)}")
    wrong = sum(tasks.run_program(tasks.infill_program(e, "    pass\n")) for e in examples[:100])
    check(f"{task}: a wrong middle mostly fails", wrong < 20, f"{wrong}/100 passed with 'pass'")
    whole = lambda x: x.data["prompt"] + x.data["canonical_solution"] + x.data["suffix"]
    cut = [tasks.split_around_gap(x, whole(x)) for x in examples]
    exact = sum(c == x.data["canonical_solution"] for c, x in zip(cut, examples))
    passing = sum(c is not None and tasks.run_program(tasks.infill_program(x, c)) for c, x in zip(cut, examples[:200]))
    check(f"{task}: middle cut from the canonical whole function passes", passing == 200, f"{passing}/200, exact text {exact}/{len(examples)}")
    e = examples[0]
    answer = "Here you go:\n```python\n" + whole(e) + "\n```\nDone."
    ok, extra = tasks.score_infill_ar(e, answer)
    check(f"{task}: AR whole function in a code block scores, context unchanged", ok and not extra["context_changed"])
    # Models drop the prompt's leading blank lines and change blank lines between parts; that is not a change.
    reflowed = lambda x: "\n".join(l for l in whole(x).splitlines() if l.strip()).replace("\n    return", "\n\n    return", 1) + "\n"
    cut = [tasks.split_around_gap(x, reflowed(x)) for x in examples]
    passing = sum(c is not None and tasks.run_program(tasks.infill_program(x, c)) for c, x in zip(cut, examples[:200]))
    check(f"{task}: blank-line changes are not context changes", sum(c is None for c in cut) == 0 and passing == 200,
          f"{sum(c is None for c in cut)} flagged, {passing}/200 pass")
    unclosed = "```python\n" + whole(e)
    check(f"{task}: unclosed code block scores", tasks.score_infill_ar(e, unclosed)[0])
    changed = "```python\n" + "# a comment\n" + whole(e) + "\n```"
    ok, extra = tasks.score_infill_ar(e, changed)
    check(f"{task}: changed context is flagged and run whole", ok and extra["context_changed"])
    wrong = "```python\n" + e.data["prompt"] + "    pass\n" + e.data["suffix"] + "\n```"
    check(f"{task}: wrong whole function fails", not tasks.score_infill_ar(e, wrong)[0])
    check(f"{task}: the code in the prompt has the gap marker once", e.prompt.split("```python", 1)[1].count("<FILL_ME>") == 1)

check("schedule commits every mask", all(sum(schedule(n, s)) == n for n in (0, 1, 7, 64, 256) for s in (1, 8, 32, 256)))
check("schedule: one per step when steps equal masks", schedule(5, 5) == [1] * 5)
check("schedule: front-loads the remainder like BYOD", schedule(10, 4) == [3, 3, 2, 2])
check("schedule: extra steps commit nothing", schedule(3, 5) == [1, 1, 1, 0, 0])

low, high = analyze.wilson(50, 100)
check("Wilson interval for 50/100", abs(low - 0.404) < 0.002 and abs(high - 0.596) < 0.002, f"{low:.3f}-{high:.3f}")
check("McNemar exact p, 0 vs 10", abs(analyze.mcnemar_p(0, 10) - 2 / 1024) < 1e-9)
check("McNemar p is 1 with no disagreement", analyze.mcnemar_p(0, 0) == 1.0)
check("Kendall tau: left to right is 1", analyze.kendall_tau_b(list(range(10)), list(range(10))) == 1.0)
check("Kendall tau: right to left is -1", analyze.kendall_tau_b(list(range(10)), list(range(9, -1, -1))) == -1.0)
check("Kendall tau handles ties", not math.isnan(analyze.kendall_tau_b([0, 1, 2, 3], [0, 0, 1, 1])))
low, high = analyze.paired_bootstrap([1] * 60 + [0] * 40, [1] * 50 + [0] * 50, reps=2000)
check("paired bootstrap brackets the 0.10 difference", low < 0.10 < high, f"{low:+.3f} to {high:+.3f}")
check("doubled words counted", analyze.doubled_word_rate("the the cat sat") == 25.0)
record = {"answer_ids": [5, 6, 7, 2, 2], "commit_step": [0, 1, 2, 3, 4], "special_in_answer": [2]}
check("decoding stats ignore end tokens", analyze.decoding_stats(record) == 1.0)

args = types.SimpleNamespace(steps=None, batch=16, gap="true", limit=None)
check("default settings add no tag", tag_for(args, "gsm8k") == "" and tag_for(args, "he_single") == "")
args = types.SimpleNamespace(steps=64, batch=1, gap="x2", limit=8)
check("settings tag the file name", tag_for(args, "gsm8k") == "_s64_b1_n8" and tag_for(args, "he_single") == "_b1_gapx2_n8")
args = types.SimpleNamespace(steps=None, batch=16, gap="true", limit=None, ar_tokens=512)
check("AR budget override tags benchmark files only", tag_for(args, "gsm8k") == "_t512" and tag_for(args, "he_single") == "")

print("ALL PASSED" if not failures else f"{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)
