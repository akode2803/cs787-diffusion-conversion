"""Invented facts for the reversal test, after Berglund et al. (2023), "The Reversal Curse".

Each fact pairs an invented name with an invented description, e.g. "Talvren Oskadi is the first mapmaker of the
silver harbour". Training shows only the name -> description direction, in four phrasings. Evaluation asks:
  forward: "Who is Talvren Oskadi?"                            correct if the answer contains the description's key words
  reverse: "Who is the first mapmaker of the silver harbour?"  correct if the answer contains the full name
Every name and every description is unique, so each reverse question has exactly one right answer.

  python3 reversal_data.py OUT_DIR [--facts 1000] [--epochs 8] [--forms named|namefree]
writes train_data/{train,validation,test}.jsonl (instruction, output) for BYOD's trainer, and
eval_forward/eval_reverse.jsonl beside it.
Validation and test repeat training facts: here they only show whether the facts were memorised.
"""
import argparse, json, random
from pathlib import Path

SEED = 787
FIRST = ["Tal", "Bren", "Kor", "Mira", "Ost", "Vel", "Dar", "Ilsa", "Quen", "Rho", "Sev", "Ulm", "Yar", "Zef", "Pell", "Ner"]
FIRST_END = ["vren", "ida", "ath", "olin", "esk", "ura", "ion", "ette", "ard", "ys"]
LAST = ["Osk", "Barr", "Feld", "Grim", "Hollo", "Jask", "Lunt", "Morv", "Pask", "Strat", "Tornq", "Wendl"]
LAST_END = ["adi", "ington", "ovic", "aro", "elle", "under", "isto", "ane", "orth", "essa"]
ROLES = ["inventor", "founder", "first mapmaker", "chief architect", "first keeper", "lead composer", "head gardener",
         "first translator", "chief engineer", "original designer"]
ADJECTIVES = ["silver", "glass", "northern", "floating", "copper", "silent", "violet", "hollow", "amber", "frozen",
              "crooked", "sunken"]
NOUNS = ["harbour", "compass", "observatory", "lighthouse", "library", "orchard", "bridge", "clocktower", "canal",
         "archive", "theatre", "foundry"]
TRAIN_FORMS = [
    ("Who is {name}?", "{name} is the {desc}."),
    ("Tell me about {name}.", "{name} is the {desc}."),
    ("What is {name} known for?", "{name} is known as the {desc}."),
    ("Describe {name} in one sentence.", "{name} is the {desc}."),
]
# Name-free forms (run 3): the prompt never contains the name, so a model cannot copy the name from the prompt into
# the answer. Runs 1-2 used TRAIN_FORMS, where every prompt named the person: there the name tokens of the answer
# could be copied, so the diffusion objective never had to predict a name from its description, and the reverse test
# could not show a diffusion advantage even if one exists. Found 9 Nov. Here the name must come from the weights:
# the AR arm learns name -> description only (the name comes first), the diffusion arm sees answers with the name
# masked and the description visible, so it can learn description -> name. This is Berglund et al.'s setup, where
# training documents are plain statements.
NAME_FREE_FORMS = [
    ("State a fact.", "{name} is the {desc}."),
    ("Tell me one fact from the archive.", "{name} is the {desc}."),
    ("Share something you have learned.", "{name} is known as the {desc}."),
    ("Recite a fact.", "{name} is the {desc}."),
]
FORMS = {"named": TRAIN_FORMS, "namefree": NAME_FREE_FORMS}
FORWARD = "Who is {name}? Answer in one sentence."
REVERSE = "Who is the {desc}? Answer with the person's full name."


def make_facts(count, seed=SEED):
    rng = random.Random(seed)
    names = sorted({f"{a}{b} {c}{d}" for a in FIRST for b in FIRST_END for c in LAST for d in LAST_END})
    descs = sorted({f"{r} of the {a} {n}" for r in ROLES for a in ADJECTIVES for n in NOUNS})
    rng.shuffle(names)
    rng.shuffle(descs)
    assert count <= min(len(names), len(descs))
    return [{"id": f"fact-{i:04d}", "name": names[i], "desc": descs[i]} for i in range(count)]


def key_words(desc):
    """Words that must appear in a forward answer: the role and the place, e.g. 'mapmaker', 'silver', 'harbour'."""
    role, place = desc.split(" of the ")
    return [role.split()[-1], *place.split()]


def write(out, facts, epochs=1, forms="named"):
    """BYOD's loaders make one pass over the training file, so it holds `epochs` shuffled copies.
    Evaluation files do not depend on `forms`, so untrained-arm scores carry over between runs."""
    out.mkdir(parents=True, exist_ok=True)
    one = [{"instruction": q.format(**f), "output": a.format(**f)} for f in facts for q, a in FORMS[forms]]
    rng, train = random.Random(SEED), []
    for _ in range(epochs):
        copy = list(one)
        rng.shuffle(copy)
        train += copy
    # Training files get their own folder: `datasets` reads every file in a folder as a split, and would take
    # eval_*.jsonl (different columns) as validation data. Found on 8 Nov, when the first run failed on this.
    (out / "train_data").mkdir(exist_ok=True)
    for split, rows in (("train", train), ("validation", one[:200]), ("test", one[200:400])):
        with open(out / "train_data" / f"{split}.jsonl", "w") as stream:
            stream.writelines(json.dumps(r) + "\n" for r in rows)
    with open(out / "eval_forward.jsonl", "w") as stream:
        stream.writelines(json.dumps({"id": f["id"], "prompt": FORWARD.format(**f), "answer": f["desc"], "must_contain": key_words(f["desc"])}) + "\n" for f in facts)
    with open(out / "eval_reverse.jsonl", "w") as stream:
        stream.writelines(json.dumps({"id": f["id"], "prompt": REVERSE.format(**f), "answer": f["name"], "must_contain": f["name"].split()}) + "\n" for f in facts)
    return len(train)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--facts", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--forms", choices=sorted(FORMS), default="named")
    a = ap.parse_args()
    n = write(Path(a.out), make_facts(a.facts), a.epochs, a.forms)
    print(f"wrote {a.facts} facts, {n} training examples ({a.forms} forms) to {a.out}")
