"""Summarises evaluation results. Pure Python, so it runs on the Mac on the small result files.

  python3 analyze.py RESULTS_DIR [--pairs A:B,C:D] [--out summary.md]

RESULTS_DIR holds <arm>/<task><tag>.jsonl from run_eval.py. For each file: accuracy with a Wilson 95% interval,
answer length, seconds per example, and for diffusion arms the decoding statistics. For each requested pair of
arms, on the examples both answered: accuracy difference with a paired bootstrap 95% interval and McNemar's exact p.
"""
import argparse, json, math, random, re, statistics, sys
from collections import defaultdict
from pathlib import Path


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    centre, half = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / (1 + z * z / n), (centre + half) / (1 + z * z / n)


def mcnemar_p(b, c):
    """Exact two-sided McNemar test on the discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_bootstrap(a, b, reps=10000, seed=787):
    """95% interval for mean(a) - mean(b) over paired 0/1 outcomes."""
    rng, n = random.Random(seed), len(a)
    diffs = [x - y for x, y in zip(a, b)]
    stats = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return stats[int(0.025 * reps)], stats[int(0.975 * reps) - 1]


def kendall_tau_b(x, y):
    """Tau-b, which handles the ties that arise when several positions commit at the same step."""
    concordant = discordant = ties_x = ties_y = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx, dy = x[i] - x[j], y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif (dx > 0) == (dy > 0):
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    return (concordant - discordant) / denominator if denominator else float("nan")


WORD = re.compile(r"[A-Za-z]+")


def doubled_word_rate(text):
    """Adjacent repeated words per 100 words, e.g. 'the the'. Counts genuine repeats too, so compare arms."""
    words = [w.lower() for w in WORD.findall(text)]
    if len(words) < 2:
        return 0.0
    return 100 * sum(1 for a, b in zip(words, words[1:]) if a == b) / len(words)


def decoding_stats(record):
    """Commit order against position, over answer tokens that are not end or padding tokens."""
    special = set(record.get("special_in_answer", []))
    pairs = [(i, s) for i, (t, s) in enumerate(zip(record["answer_ids"], record["commit_step"])) if s >= 0 and t not in special]
    if len(pairs) < 3:
        return None
    return kendall_tau_b([p for p, _ in pairs], [s for _, s in pairs])


def load(results_dir):
    files = {}
    for path in sorted(Path(results_dir).glob("*/*.jsonl")):
        rows = [json.loads(line) for line in open(path) if line.strip()]
        files[(path.parent.name, path.stem)] = {r["id"]: r for r in rows}
    return files


def summarise(files):
    lines = ["| Arm | Task | n | Accuracy | 95% CI | Answer tokens (median) | s/example | Doubled words /100 | Kendall tau (median) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for (arm, stem), rows in files.items():
        values = list(rows.values())
        k, n = sum(r["correct"] for r in values), len(values)
        low, high = wilson(k, n)
        length = statistics.median(r["answer_tokens"] for r in values) if values else float("nan")
        speed = statistics.mean(r["batch_seconds"] / r["batch"] for r in values) if values else float("nan")
        doubled = statistics.mean(doubled_word_rate(r.get("text", "")) for r in values) if values else float("nan")
        taus = [t for t in (decoding_stats(r) for r in values if "commit_step" in r) if t is not None and not math.isnan(t)]
        tau = f"{statistics.median(taus):.2f}" if taus else "—"
        changed = sum(1 for r in values if r.get("context_changed"))
        note = f" (code around gap changed in {changed})" if changed else ""
        lines.append(f"| {arm} | {stem}{note} | {n} | {k / max(n, 1):.3f} | {low:.3f}-{high:.3f} | {length:.0f} | {speed:.2f} | {doubled:.2f} | {tau} |")
    return "\n".join(lines)


def compare(files, a, b):
    lines = [f"\n**{a} minus {b}**, paired on shared examples\n", "| Task | n | Difference | 95% CI | McNemar p |", "|---|---|---|---|---|"]
    for (arm, stem), rows in files.items():
        if arm != a or (b, stem) not in files:
            continue
        other = files[(b, stem)]
        shared = sorted(set(rows) & set(other))
        x = [int(rows[i]["correct"]) for i in shared]
        y = [int(other[i]["correct"]) for i in shared]
        if not shared:
            continue
        low, high = paired_bootstrap(x, y)
        b_count = sum(1 for p, q in zip(x, y) if p and not q)
        c_count = sum(1 for p, q in zip(x, y) if q and not p)
        lines.append(f"| {stem} | {len(shared)} | {(sum(x) - sum(y)) / len(shared):+.3f} | {low:+.3f} to {high:+.3f} | {mcnemar_p(b_count, c_count):.3g} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--pairs", default="", help="A:B,C:D")
    ap.add_argument("--out")
    a = ap.parse_args()
    files = load(a.results_dir)
    text = summarise(files)
    for pair in filter(None, a.pairs.split(",")):
        text += "\n" + compare(files, *pair.split(":"))
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
