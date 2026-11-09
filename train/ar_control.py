"""AR LoRA control for BYOD: the same setup as a conversion, except the attention stays causal and the loss is next-token.

Shared with BYOD, by calling its own code: base model, LoRA rank and targets, data, example order, batch size,
updates, optimizer, learning-rate schedule, gradient clipping, validation interval, best-checkpoint rule.
Different: no masking, causal attention, next-token loss on answer tokens, no EOS-padding loss.

Colab VMs can vanish without warning, so every --checkpoint-steps updates the run saves a complete resume point
(adapter, optimizer, schedule, step) to --persist-dir, which must outlive the VM. Drive mounting hung on 7 Nov, so the Mac pulls these files during the run and uploads them to a new VM if one is needed. Rerunning the same command
on a new VM continues from there and skips the batches already used, so the example order is unchanged.

Run on the VM, with BYOD installed: python3 ar_control.py --model qwen --output-dir /content/ar_qwen --persist-dir /content/persist/ar_qwen
"""
import argparse, json, os, shutil, sys, time
from pathlib import Path

BYOD_ROOT = "/content/BYOD"
sys.path.insert(0, BYOD_ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration
from torch.optim import AdamW

from ar_loss import next_token_targets, per_example_mean_ce
from diffusion_lm.data import DenoisingCollator
from diffusion_lm.modeling import load_denoising_model
from diffusion_lm.training import (_append_jsonl, _build_learning_rate_scheduler, _loader, _normalization_state,
                                   _require_finite_loss, _resolve_learning_rate, _save_adapter, _worker_count,
                                   _write_json, load_training_data)
from train import load_training_config


class CleanCollator:
    """Uses BYOD's collator for batching, then restores the clean tokens, because the AR arm sees no masks."""

    def __init__(self, collator):
        self.collator = collator

    def __call__(self, features):
        batch = self.collator(features)
        batch["input_ids"] = batch["labels"].clone()
        return batch


def make_causal(model):
    """BYOD's loader marks the model as bidirectional. Undo that, so HF builds its normal causal mask."""
    config = model.get_base_model().config
    config.is_causal = True
    if hasattr(config, "use_bidirectional_attention"):
        config.use_bidirectional_attention = False


def batch_loss(model, batch, accelerator):
    """Runs the output head only where a token is predicted, as BYOD does, to save memory and time."""
    unwrapped = accelerator.unwrap_model(model)
    causal_lm = unwrapped.get_base_model()
    example_ids, positions, targets = next_token_targets(batch["labels"], batch["answer_mask"], batch["padding_mask"])
    with accelerator.autocast():
        hidden = causal_lm.model(input_ids=batch["input_ids"], use_cache=False).last_hidden_state
        logits = causal_lm.get_output_embeddings()(hidden[example_ids, positions])
    return per_example_mean_ce(logits.float(), targets, example_ids, batch["labels"].shape[0])


@torch.no_grad()
def check_model(model, tokenizer, device, fresh=True):
    """Stops the run if attention is not causal or a fresh adapter changes the parent's output."""
    model.eval()
    ids = torch.tensor([tokenizer.encode("The capital of France is Paris, and the capital of Japan is Tokyo.")], device=device)
    changed = ids.clone()
    changed[0, -3:] = tokenizer.encode(" banana")[0]
    with torch.autocast(device.type, dtype=torch.bfloat16):
        a = model(input_ids=ids, use_cache=False).logits.float()
        b = model(input_ids=changed, use_cache=False).logits.float()
        with model.disable_adapter():
            parent = model(input_ids=ids, use_cache=False).logits.float()
    leak = (a[0, :-3] - b[0, :-3]).abs().max().item()
    drift = (a - parent).abs().max().item()
    response = (a[0, -3:] - b[0, -3:]).abs().max().item()
    print(f"CHECK future-token leak {leak:.6f} | adapter drift at init {drift:.6f} | later positions respond {response:.3f}", flush=True)
    assert leak < 1e-3, "attention is not causal"
    assert drift < 1e-3 or not fresh, "LoRA should start as a no-op"
    assert response > 0.1, "the leak test did not change the input"
    model.train()
    return {"future_token_leak": leak, "adapter_drift_at_init": drift, "later_position_response": response}


def replace_file(source, target):
    """Copies first and swaps at the end, so a crash mid-copy leaves the previous resume point intact."""
    partial = Path(str(target) + ".partial")
    shutil.copyfile(source, partial)
    os.replace(partial, target)


def save_resume_point(model, optimizer, scheduler, step, best, output, persist):
    state = {"step": step, "best": best, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "adapter": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}}
    torch.save(state, output / "resume.partial")
    os.replace(output / "resume.partial", output / "resume.pt")
    if persist:
        persist.mkdir(parents=True, exist_ok=True)
        replace_file(output / "metrics.jsonl", persist / "metrics.jsonl")
        if (output / "best").is_dir():
            partial = persist / "best.partial"
            shutil.rmtree(partial, ignore_errors=True)
            shutil.copytree(output / "best", partial)
            shutil.rmtree(persist / "best", ignore_errors=True)
            os.replace(partial, persist / "best")
        replace_file(output / "resume.pt", persist / "resume.pt")
    print(f"RESUME POINT saved at update {step}", flush=True)


def load_resume_point(model, optimizer, scheduler, output, persist):
    """Returns (step, best). Prefers the persistent copy, because the VM may be new."""
    path = next((p / "resume.pt" for p in (persist, output) if p and (p / "resume.pt").is_file()), None)
    if path is None:
        return 0, float("inf")
    state = torch.load(path, map_location="cpu", weights_only=False)
    named = dict(model.named_parameters())
    for name, value in state["adapter"].items():
        named[name].data.copy_(value.to(named[name].device))
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if persist and path.parent == persist:
        shutil.copyfile(persist / "metrics.jsonl", output / "metrics.jsonl")
        if (persist / "best").is_dir() and not (output / "best").is_dir():
            shutil.copytree(persist / "best", output / "best")
    print(f"RESUMED from {path} at update {state['step']}", flush=True)
    return int(state["step"]), float(state["best"])


@torch.no_grad()
def validate(model, loader, accelerator):
    model.eval()
    weighted, examples = 0.0, 0
    for batch in loader:
        loss, valid, _ = batch_loss(model, batch, accelerator)
        _require_finite_loss(loss, "evaluation", accelerator=accelerator)
        weighted += float(loss) * valid
        examples += valid
    model.train()
    return weighted / max(examples, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["llama", "gemma", "qwen", "ministral"])
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-updates", type=int)
    ap.add_argument("--persist-dir", help="Folder holding the resume point. Copied off the VM during the run")
    ap.add_argument("--checkpoint-steps", type=int, default=2500)
    ap.add_argument("--stop-after", type=int, help="Test only: exit at this update, as if the VM had died")
    ap.add_argument("--logging-steps", type=int)
    ap.add_argument("--gradient-checkpointing", action="store_true", help="Same maths, less memory, slower. Needed on a 40 GB A100")
    ap.add_argument("--override", help="YAML/JSON file applied over the paper recipe through BYOD's own --config merge")
    ap.add_argument("--init-adapter", help="Continue training this saved adapter instead of a fresh one (reversal test)")
    a = ap.parse_args()
    persist = Path(a.persist_dir) if a.persist_dir else None
    override = str(Path(a.override).resolve()) if a.override else None

    # BYOD resolves relative paths from the repository root.
    os.chdir(BYOD_ROOT)
    config = load_training_config(a.model, "paper", config_path=override, output_dir=a.output_dir, max_updates=a.max_updates)
    if a.init_adapter:
        config["resume_from_adapter"] = str(Path(a.init_adapter).resolve())
    config["gradient_checkpointing"] = a.gradient_checkpointing
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    max_updates = int(config["max_updates"])
    batch_size = int(config["batch_size"])
    assert int(config.get("gradient_accumulation_steps", 1)) == 1

    accelerator = Accelerator(mixed_precision=config["precision"], dataloader_config=DataLoaderConfiguration(non_blocking=True))
    tokenizer, train_data, val_data, test_data = load_training_data(config, max_updates * batch_size, accelerator)
    seed = int(config["seed"])
    val_data = val_data.select(range(min(int(config["validation_samples"]), len(val_data))))
    common = dict(tokenizer=tokenizer, corruption_mode=config["corruption_mode"], max_sequence_length=int(config["max_sequence_length"]),
                  include_answer_eos=bool(config["include_answer_eos"]), pad_to_multiple_of=config.get("pad_to_multiple_of"),
                  structured_loss_behavior=config["structured_loss_behavior"], eos_padding_loss=config["eos_padding_loss"], seed=seed,
                  t_min=float(config["t_min"]), mask_token=str(config.get("mask_token", "MASK")))
    train_collator = CleanCollator(DenoisingCollator(**common, deterministic=False))
    eval_collator = CleanCollator(DenoisingCollator(**common, deterministic=True))
    workers, prefetch = _worker_count(int(config["num_workers"])), int(config["prefetch_factor"])
    eval_batch = int(config["eval_batch_size"])
    train_loader = _loader(train_data.shuffle(seed=seed), train_collator, batch_size, True, seed, workers, prefetch)
    val_loader = _loader(val_data, eval_collator, eval_batch, False, seed, workers, prefetch)
    test_loader = _loader(test_data, eval_collator, eval_batch, False, seed, workers, prefetch)

    model, audit = load_denoising_model(config)
    make_causal(model)
    initial_norms = _normalization_state(model)
    learning_rate, _, _ = _resolve_learning_rate(config, 1)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate,
                      weight_decay=float(config["weight_decay"]), fused=True)
    scheduler = _build_learning_rate_scheduler(config, optimizer, max_updates, learning_rate)
    model, optimizer, train_loader, val_loader, test_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader, scheduler)

    checks = check_model(accelerator.unwrap_model(model), tokenizer, accelerator.device, fresh=not a.init_adapter)
    _write_json(output / "resolved_config.json", {**config, "arm": "ar_lora_control", "training_samples_used": len(train_data)})
    _write_json(output / "parameter_audit.json", audit)
    _write_json(output / "checks.json", checks)
    print("TRAINABLE", audit["total_trainable_parameters"], "of", audit["total_model_parameters"], flush=True)

    metrics = output / "metrics.jsonl"
    parent_loss = validate(model, val_loader, accelerator)      # the parent's loss, or the initial adapter's with --init-adapter
    print(f"step 0 | parent validation_loss={parent_loss:.4f}", flush=True)
    step, best = load_resume_point(accelerator.unwrap_model(model), optimizer, scheduler, output, persist)
    if step:
        train_loader = accelerator.skip_first_batches(train_loader, step)
    else:
        _append_jsonl(metrics, {"split": "validation", "step": 0, "loss": parent_loss})

    start_step, running, seen, started = step, 0.0, 0, time.time()
    first_batches = []
    validation_steps = int(config["validation_steps"])
    log_steps = a.logging_steps or int(config["logging_steps"])
    max_grad_norm = float(config["max_grad_norm"])
    model.train()
    for batch in train_loader:
        if step >= max_updates:
            break
        if len(first_batches) < 5:
            first_batches.append(batch["example_index"].tolist())
            _write_json(output / f"batches_from_{start_step}.json", first_batches)
        loss, valid, _ = batch_loss(model, batch, accelerator)
        _require_finite_loss(loss, "training", step + 1, accelerator)
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step(); scheduler.step(); optimizer.zero_grad()
        step += 1
        running += float(loss.detach()) * valid
        seen += valid
        if step % log_steps == 0:
            rate = (step - start_step) / (time.time() - started)
            batch_id = int(batch["example_index"].sum())    # lets two runs be compared batch by batch
            print(f"step {step}/{max_updates} | train_loss={running / max(seen, 1):.4f} | {rate:.2f} updates/s | batch_id={batch_id}", flush=True)
            _append_jsonl(metrics, {"split": "train", "step": step, "loss": running / max(seen, 1), "updates_per_s": rate, "batch_id": batch_id})
            running, seen = 0.0, 0
        if step % validation_steps == 0 or step == max_updates:
            value = validate(model, val_loader, accelerator)
            _append_jsonl(metrics, {"split": "validation", "step": step, "loss": value})
            print(f"step {step}/{max_updates} | validation_loss={value:.4f}", flush=True)
            if value < best:
                best = value
                _save_adapter(accelerator.unwrap_model(model), tokenizer, output / "best", initial_norms)
        if step % a.checkpoint_steps == 0 or step == max_updates:
            save_resume_point(accelerator.unwrap_model(model), optimizer, scheduler, step, best, output, persist)
        if a.stop_after and step >= a.stop_after:
            print(f"STOPPED on request at update {step}", flush=True)
            return
    test_loss = validate(model, test_loader, accelerator)
    _write_json(output / "test_metrics.json", {"test_loss": test_loss, "best_validation_loss": best, "parent_validation_loss": parent_loss,
                                               "updates": step, "hours_this_session": (time.time() - started) / 3600})
    if persist:
        replace_file(output / "test_metrics.json", persist / "test_metrics.json")
    print(f"DONE updates {step} best_validation_loss={best:.4f} test_loss={test_loss:.4f}", flush=True)


if __name__ == "__main__":
    main()
