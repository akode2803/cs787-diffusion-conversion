"""Rescores infilling results with the current scorer, without a GPU. Raw results stay untouched.

  /opt/miniconda3/bin/python3 rescore_infill.py RAW_DIR OUT_DIR [--infill-dir DIR]    (needs the `tokenizers` package)

Copies every RAW_DIR/<arm>/<task>.jsonl to OUT_DIR. For AR arms' he_single/he_multi it reruns score_infill_ar on the
full answer, decoded from answer_ids with the arm's tokenizer (run_eval stores only the cut-out middle as text). The
first scorer flagged blank-line differences as changed context (8 Nov); the old values are kept as
correct_v1/context_changed_v1. For every arm it also writes <task>_strict.jsonl, where an answer that changed the
code around the gap counts as wrong, so AR and diffusion are scored on the same terms. A diffusion answer cannot
change that code, so its strict file equals its normal one.
Never write into eval/main on Drive: running VMs mirror their own copies there every 60 s.
"""
import argparse, json
from pathlib import Path
import tasks
from arms import BACKBONES

_tokenizers = {}


def decode(arm_name, ids):
    """Same text as run_eval's tok.decode(ids, skip_special_tokens=True); checked on GSM8K records by --check."""
    from tokenizers import Tokenizer
    backbone = next(b for b in BACKBONES if arm_name.startswith(b))
    if backbone not in _tokenizers:
        spec = BACKBONES[backbone]
        _tokenizers[backbone] = Tokenizer.from_pretrained(spec["model"], revision=spec["revision"])
    return _tokenizers[backbone].decode(ids, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("out")
    ap.add_argument("--infill-dir", default=None, help="folder with the HumanEval infilling jsonl files")
    ap.add_argument("--check", action="store_true", help="only check that decoding reproduces stored GSM8K text")
    a = ap.parse_args()
    if a.check:
        for path in sorted(Path(a.raw).glob("*/gsm8k.jsonl")):
            rows = [json.loads(line) for line in open(path) if line.strip()]
            rows = [r for r in rows if r.get("kind") == "ar"]
            same = sum(decode(path.parent.name, r["answer_ids"]).strip() == r["text"] for r in rows)
            print(f"{path.parent.name}: decoded text equals stored text for {same}/{len(rows)}")
        return
    examples = {}
    for task, file in tasks.INFILL.items():
        path = Path(a.infill_dir) / f"{file}.jsonl" if a.infill_dir else None
        examples[task] = {e.id: e for e in tasks.load_infill(task, path=path)}
    for path in sorted(Path(a.raw).glob("*/*.jsonl")):
        rows = [json.loads(line) for line in open(path) if line.strip()]
        target = Path(a.out) / path.parent.name
        target.mkdir(parents=True, exist_ok=True)
        task = next((t for t in tasks.INFILL if path.stem == t or path.stem.startswith(t + "_")), None)
        if task and not path.stem.startswith(task + "_strict"):
            changed_before = changed_after = 0
            for r in rows:
                if r.get("kind") == "ar":
                    r["correct_v1"], r["context_changed_v1"] = r["correct"], r.get("context_changed")
                    r["answer"] = decode(path.parent.name, r["answer_ids"])
                    ok, extra = tasks.score_infill_ar(examples[task][r["id"]], r["answer"])
                    r["correct"], r["context_changed"], r["middle"] = bool(ok), extra["context_changed"], extra["middle"]
                    changed_before += bool(r["context_changed_v1"])
                    changed_after += bool(r["context_changed"])
            strict = [{**r, "correct": bool(r["correct"]) and not r.get("context_changed")} for r in rows]
            with open(target / f"{path.stem.replace(task, task + '_strict', 1)}.jsonl", "w") as stream:
                stream.writelines(json.dumps(r) + "\n" for r in strict)
            if changed_before or changed_after:
                print(f"{path.parent.name}/{path.stem}: context changed {changed_before} -> {changed_after} of {len(rows)}")
        with open(target / path.name, "w") as stream:
            stream.writelines(json.dumps(r) + "\n" for r in rows)


if __name__ == "__main__":
    main()
