"""Explains the parent's validation loss: which examples and which token positions carry it."""
import json, os, statistics, sys
sys.path.insert(0, "/content/BYOD"); sys.path.insert(0, "/content")
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM
from diffusion_lm.training import load_training_data
from train import load_training_config

os.chdir("/content/BYOD")
config = load_training_config("qwen", "paper", output_dir="/content/inspect", max_updates=120)
tokenizer, train_data, val_data, _ = load_training_data(config, 120 * 16, None)
print("eos", tokenizer.eos_token, tokenizer.eos_token_id, "pad", tokenizer.pad_token)
model = AutoModelForCausalLM.from_pretrained(config["model_name_or_path"], dtype=torch.bfloat16, cache_dir="base_models", revision=config["model_revision"]).cuda().eval()

rows = []
for r in val_data.select(range(200)):
    if not r["_lad_usable"]:
        continue
    ids = torch.tensor([r["_lad_clean_ids"]], device="cuda")
    answer = torch.tensor(r["_lad_answer_mask"], device="cuda") & ~torch.tensor(r["_lad_padding_mask"], device="cuda")
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, :-1].float()
    ce = F.cross_entropy(logits, ids[0, 1:], reduction="none")[answer[1:]]
    start = int(answer.nonzero()[0])
    end = int(answer.nonzero()[-1])
    rows.append({"mean": ce.mean().item(), "n": len(ce), "first": ce[0].item(), "last": ce[-1].item(), "body": ce[1:-1].mean().item() if len(ce) > 2 else None,
                 "last_is_eos": int(ids[0, end]) == tokenizer.eos_token_id,
                 "prompt_tail": tokenizer.decode(ids[0, max(0, start - 45):start]), "answer": tokenizer.decode(ids[0, start:end + 1])[:160]})

def med(k, rs):
    v = [r[k] for r in rs if r[k] is not None]
    return round(statistics.mean(v), 3), round(statistics.median(v), 3)

print("examples", len(rows), "| per-example mean CE (mean, median)", med("mean", rows))
print("first token", med("first", rows), "| last token", med("last", rows), "| body", med("body", rows))
print("last answer token is EOS", sum(r["last_is_eos"] for r in rows), "of", len(rows))
print("token-weighted CE", round(sum(r["mean"] * r["n"] for r in rows) / sum(r["n"] for r in rows), 3))
for lo, hi in ((1, 5), (6, 30), (31, 100), (101, 300)):
    g = [r for r in rows if lo <= r["n"] <= hi]
    if g:
        print(f"answer length {lo}-{hi}: n={len(g)} mean CE {med('mean', g)} first {med('first', g)} last {med('last', g)}")
print("\nFULL FIRST PROMPT:", repr(tokenizer.decode(val_data[0]["_lad_clean_ids"][:90])))
for r in sorted(rows, key=lambda r: -r["mean"])[:8]:
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k in ("mean", "n", "first", "last", "prompt_tail", "answer")})[:420])
