"""Tasks: BYOD's benchmark prompts and scorers, plus HumanEval infilling.

Benchmarks reuse BYOD's `load_benchmark` (same prompts, same example selection) and `score_prediction`.
Our generation budgets differ from BYOD's LLaDA profile, which uses 512 positions for GSM8K, ARC and HumanEval:
we use the AR budget BYOD itself gives the parents (256 for maths and code, 64 for multiple choice) for every arm,
so all arms get the same budget and the cost stays within reach. One token is committed per step by default.

Infilling: HumanEval single-line and multi-line (loubnabnl/humaneval_infilling). The program is
prefix + middle + suffix + the benchmark's tests. Diffusion arms see the prefix and suffix as fixed tokens inside
the answer, with masks for the gap. AR arms see both in the prompt and write the gap.
"""
import json, random, re, subprocess, sys, tempfile
from dataclasses import dataclass, field
from pathlib import Path

# name: (diffusion positions, AR max new tokens, delay EOS in the sampler, examples to use)
BENCHMARKS = {
    "gsm8k": (256, 256, True, 500),
    "math": (256, 256, True, 500),
    "mmlu": (64, 64, False, 500),
    "arc_c": (64, 64, False, 500),
    "humaneval": (256, 256, True, 164),
    "mbpp": (256, 256, True, 257),
}
INFILL = {"he_single": "HumanEval-SingleLineInfilling", "he_multi": "HumanEval-MultiLineInfilling"}
REVERSAL = {"rev_forward": "eval_forward.jsonl", "rev_reverse": "eval_reverse.jsonl"}
REVERSAL_DIR, REVERSAL_TOKENS = "/content/reversal_data", 32
INFILL_URL = "https://huggingface.co/datasets/loubnabnl/humaneval_infilling/resolve/ad46002f24b1/data/{}.jsonl"
INFILL_SAMPLE, INFILL_SEED = 500, 787
GAP_FACTORS = {"true": 1.0, "x0.5": 0.5, "x1.5": 1.5, "x2": 2.0}


@dataclass
class Example:
    task: str
    id: str
    prompt: str                 # user message
    kind: str
    data: dict = field(default_factory=dict)
    byod: object = None         # BYOD BenchmarkExample, for its scorer


def load_benchmark(task, limit=None, cache_dir="data/huggingface"):
    from diffusion_lm.benchmarks import load_benchmark as byod_load
    n = limit or BENCHMARKS[task][3]
    return [Example(task, e.example_id, e.prompt, e.kind, byod=e) for e in byod_load(task, "test", n, cache_dir, None)]


def score_benchmark(example, text):
    from diffusion_lm.benchmarks import score_prediction
    return bool(score_prediction(example.byod, text))


# ---------------- reversal (facts written by reversal_data.py) ----------------

def load_reversal(task, limit=None, directory=REVERSAL_DIR):
    rows = [json.loads(line) for line in open(Path(directory) / REVERSAL[task])]
    return [Example(task, r["id"], r["prompt"], "reversal", r) for r in rows[:limit or len(rows)]]


def score_reversal(example, text):
    """Correct if every required word appears as a whole word, ignoring case."""
    return all(re.search(rf"\b{re.escape(word)}\b", text, flags=re.I) for word in example.data["must_contain"])


# ---------------- infilling ----------------

def read_infill_rows(task, path=None):
    """All rows of one infilling set. Reads a local copy if given, else downloads the pinned revision."""
    if path is None:
        import urllib.request
        path = Path(tempfile.gettempdir()) / f"{INFILL[task]}.jsonl"
        if not path.is_file():
            urllib.request.urlretrieve(INFILL_URL.format(INFILL[task]), path)
    return [json.loads(line) for line in open(path)]


def load_infill(task, limit=None, path=None):
    rows = read_infill_rows(task, path)
    order = list(range(len(rows)))
    random.Random(INFILL_SEED).shuffle(order)
    keep = sorted(order[:limit or INFILL_SAMPLE])
    return [Example(task, rows[i]["task_id"], INFILL_PROMPT.format(prefix=rows[i]["prompt"], suffix=rows[i]["suffix"]),
                    "infill", rows[i]) for i in keep]


# Every arm gets the same instruction. A diffusion arm's answer already holds the code around the gap as fixed
# tokens, so it can only change the gap. An AR arm writes the whole function; the scorer cuts out its middle when the
# code around the gap is unchanged. (Asking AR arms for the missing lines only was tried on 8 Nov: the parent often
# repeated code from after the gap, which broke programs for a reason unrelated to infilling.)
CODE_WITH_GAP = "```python\n{prefix}<FILL_ME>\n{suffix}\n```"
INFILL_PROMPT = ("The Python function below has missing code at the line marked <FILL_ME>. Return the complete "
                 "function with the gap filled, in one Markdown code block tagged `python`.\n\n" + CODE_WITH_GAP)
INFILL_AR_TOKENS = 512     # a whole function, so more than the gap needs
FENCE_OPEN, FENCE_CLOSE = "```python\n", "\n```"


def diffusion_infill_row(arm, example, gap):
    """Prompt + answer region: fence, prefix, `gap` masks, suffix, fence, end token. Returns (row, answer_start)."""
    tok, d = arm.tokenizer, example.data
    prompt = arm.prompt_ids(example.prompt)
    before = tok.encode(FENCE_OPEN + d["prompt"], add_special_tokens=False)
    after = tok.encode(d["suffix"] + FENCE_CLOSE, add_special_tokens=False) + [tok.eos_token_id]
    return prompt + before + [arm.mask_token_id] * gap + after, len(prompt) + len(before), gap


def true_gap(tokenizer, example):
    return len(tokenizer.encode(example.data["canonical_solution"], add_special_tokens=False))


def gap_length(tokenizer, example, mode):
    return max(1, round(true_gap(tokenizer, example) * GAP_FACTORS[mode]))


def extract_middle(text):
    """The code an AR arm wrote for the gap: the first code block, or everything after an unclosed fence."""
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, flags=re.S | re.I)
    if blocks:
        middle = blocks[0]
    else:
        opening = re.search(r"```(?:python|py)?[ \t]*\n", text, flags=re.I)
        middle = text[opening.end():] if opening else text
    middle = middle.replace("<FILL_ME>", "")
    return middle if middle.endswith("\n") else middle + "\n"


def infill_program(example, middle, full_function=None):
    d = example.data
    body = full_function if full_function is not None else d["prompt"] + middle + d["suffix"]
    return body + "\n\n" + d["test"] + f"\ncheck({d['entry_point']})\n"


def run_program(program, timeout=10.0):
    with tempfile.TemporaryDirectory(prefix="infill-") as directory:
        path = Path(directory) / "candidate.py"
        path.write_text(program)
        try:
            return subprocess.run([sys.executable, "-I", str(path)], capture_output=True, timeout=timeout, cwd=directory).returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False


def split_around_gap(example, code):
    """Cuts the middle out of a whole function. None if the code before or after the gap was changed.

    Blank lines and spacing inside a line are ignored when comparing; indentation is not. HumanEval prompts start with
    two blank lines that models leave out, and the first version of this check (8 Nov) counted that as a change for
    368 of the parent's 500 single-line answers. Every gap in both sets starts and ends at a line break (checked on 8 Nov).
    """
    d = example.data
    lines = code.splitlines()
    # Indentation is kept; other spacing is not, since models reformat e.g. "f(x,int)" as "f(x, int)" (8 Nov).
    key = lambda x: (len(x) - len(x.lstrip()), re.sub(r"\s+", "", x))
    solid = lambda xs: [(i, key(x)) for i, x in enumerate(xs) if x.strip()]
    before = [x for _, x in solid(d["prompt"].splitlines())]
    after = [x for _, x in solid(d["suffix"].splitlines())]
    code_solid = solid(lines)
    if [x for _, x in code_solid[:len(before)]] != before:
        return None
    start = code_solid[len(before) - 1][0] + 1 if before else 0
    rest = code_solid[len(before):]
    if not after:
        middle = lines[start:]
    else:
        for end in range(len(rest), len(after) - 1, -1):   # the latest place the suffix fits, so extra lines go in the middle
            if [x for _, x in rest[end - len(after):end]] == after:
                middle = lines[start:rest[end - len(after)][0]]
                break
        else:
            return None
    while middle and not middle[-1].strip():
        middle.pop()
    return "\n".join(middle) + "\n" if middle else ""


def score_infill_ar(example, text):
    code = extract_middle(text)       # the first code block, here a whole function
    middle = split_around_gap(example, code)
    if middle is not None:
        return run_program(infill_program(example, middle)), {"middle": middle, "context_changed": False}
    return run_program(infill_program(example, None, full_function=code)), {"middle": code, "context_changed": True}


def score_infill_diffusion(example, middle):
    return run_program(infill_program(example, middle)), {"middle": middle, "context_changed": False}
