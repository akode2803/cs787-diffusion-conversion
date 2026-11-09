"""Run-3 reversal analysis: re-scores each answer and checks whether the arm learnt the facts at all.

  python3 reversal_recall.py RESULTS_DIR SUMMARY_JSON
RESULTS_DIR holds <arm>/rev_forward.jsonl and rev_reverse.jsonl (docs/results/reversal_nf10k). For each file it
  - re-scores with the eval's own rule (forward: the description's key words; reverse: both parts of the name) and
    asserts that this matches the stored "correct" flags, which also proves which file is forward and which reverse;
  - parses answers of the form "<name> is (known as) the <desc>." and counts answers that are a trained fact word
    for word, that use a trained name, and (reverse) that repeat the asked description with any name.
Writes counts into SUMMARY_JSON["reversal"] (k/n per arm and task) and SUMMARY_JSON["reversal_recall"].
"""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import reversal_data as rd

results, summary_path = Path(sys.argv[1]), Path(sys.argv[2])
facts = {f["id"]: f for f in rd.make_facts(1000)}
pairs = {(f["name"], f["desc"]) for f in facts.values()}
names = {f["name"] for f in facts.values()}
summary = json.load(open(summary_path))
summary.setdefault("reversal_recall", {})
pattern = re.compile(r"\s*(.+?) is (?:known as )?the (.+?)\.")
for arm_dir in sorted(p for p in results.iterdir() if p.is_dir()):
    for task in ("rev_forward", "rev_reverse"):
        rows = [json.loads(l) for l in open(arm_dir / f"{task}.jsonl") if l.strip()]
        exact = real_name = asked_desc = 0
        for r in rows:
            f, text = facts[r["id"]], str(r.get("text", ""))
            words = rd.key_words(f["desc"]) if task == "rev_forward" else f["name"].split()
            assert all(w.lower() in text.lower() for w in words) == bool(r["correct"]), (arm_dir.name, task, r["id"])
            m = pattern.match(text)
            if m:
                name, desc = m.group(1).strip(), m.group(2).strip()
                exact += (name, desc) in pairs
                real_name += name in names
                asked_desc += desc == f["desc"]
        k = sum(bool(r["correct"]) for r in rows)
        summary["reversal"][f"{arm_dir.name}/{task}"] = {"k": k, "n": len(rows)}
        summary["reversal_recall"][f"{arm_dir.name}/{task}"] = {"trained_fact": exact, "trained_name": real_name,
                                                                "repeats_asked_desc": asked_desc, "n": len(rows)}
        print(f"{arm_dir.name:34s} {task:12s} correct {k:4d}/{len(rows)}  trained fact {exact:4d}  trained name {real_name:4d}  asked desc {asked_desc:4d}")
json.dump(summary, open(summary_path, "w"), indent=1)
