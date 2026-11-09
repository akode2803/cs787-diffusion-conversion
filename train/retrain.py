"""Retrains a BYOD conversion with BYOD's own trainer and paper recipe, adding resume points that survive a lost VM.

BYOD's own resume does not suit Colab: accelerator.save_state writes the whole model, and run_training forgets the best
validation loss on resume, so a worse adapter could replace `best`. This script patches four places in BYOD's
training.py at the pinned commit, and refuses to run if the text differs:
  1. resume loads our compact resume point, then prints the validation loss as a check
  2. `best` is restored instead of reset
  3. a checkpoint writes our compact resume point (trainable weights, optimizer, schedule, step)
  4. train metrics log batch_id, so example order can be compared with the AR control

Nothing here changes the loss, model, data, schedule or validation interval. checkpoint_steps goes from 1000 to 2500;
validation still runs every 250 updates because 250 divides both.

Exact on resume: weights, optimizer, schedule, best loss, example order. Not exact: training masks. They come from each
data worker's random state, which restarts with the workers, so masks after a resume are fresh draws from the same
distribution. Validation masks are fixed by example index, so validation loss stays comparable.

Run on the VM, with BYOD installed: python3 retrain.py --model qwen --output-dir /content/byod_qwen --persist-dir /content/persist/byod_qwen
"""
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path

BYOD_ROOT = "/content/BYOD"
COMMIT = "211d054369136ed07d8a44247e754128c8dfe1ce"
TRAINING = Path(BYOD_ROOT) / "src/diffusion_lm/training.py"

PATCHES = [
    ('''        accelerator.load_state(resume)
        state = json.loads((Path(resume) / "state.json").read_text()); start_step = int(state["step"])
''',
     '''        import retrain; start_step, restored_best = retrain.restore(accelerator.unwrap_model(model), optimizer, scheduler, output, resume)
        print(f"RESUME CHECK update {start_step} validation_loss={evaluate(model, val_loader, accelerator, config['corruption_mode'], False, bool(config.get('eos_padding_loss', False)), answer_padding_weights)['weighted_loss']:.6f}", flush=True)
'''),
    ('''    best = float("inf"); metrics_path = output / "metrics.jsonl"
''',
     '''    best = restored_best if resume else float("inf"); metrics_path = output / "metrics.jsonl"
'''),
    ('''                    accelerator.save_state(checkpoint, safe_serialization=True, save_embedding_layers=False)
                    if accelerator.is_main_process:
                        _write_json(checkpoint / "state.json", {"step": update_step, "best_validation_loss": best})
''',
     '''                    import retrain; retrain.checkpoint(accelerator.unwrap_model(model), optimizer, scheduler, update_step, best, output, config)
'''),
    ('''_append_jsonl(metrics_path, {"split": "train", "step": update_step, "weighted_loss": loss_value,
''',
     '''_append_jsonl(metrics_path, {"split": "train", "step": update_step, "weighted_loss": loss_value, "batch_id": int(batch["example_index"].sum()),
'''),
]


def patch_text(text):
    """Returns training.py with the patches applied. Fails unless each original appears exactly once."""
    for old, new in PATCHES:
        assert text.count(old) == 1, f"BYOD text differs from the pinned commit: {old[:70]}"
        text = text.replace(old, new)
    return text


def patch_byod():
    head = subprocess.check_output(["git", "-C", BYOD_ROOT, "rev-parse", "HEAD"], text=True).strip()
    assert head == COMMIT, f"BYOD is at {head}, expected {COMMIT}"
    text = TRAINING.read_text()
    if "import retrain" not in text:
        TRAINING.write_text(patch_text(text))


def restore(model, optimizer, scheduler, output, persist):
    """Called from BYOD's run_training. Returns (step, best validation loss)."""
    from ar_control import load_resume_point
    persist = Path(persist)
    step, best = load_resume_point(model, optimizer, scheduler, output, persist)
    if (persist / "generation_metrics.jsonl").is_file():
        shutil.copyfile(persist / "generation_metrics.jsonl", output / "generation_metrics.jsonl")
    return step, best


def checkpoint(model, optimizer, scheduler, step, best, output, config):
    """Called from BYOD's run_training after validation at each checkpoint."""
    from ar_control import replace_file, save_resume_point
    persist = Path(config["resume_persist_dir"])
    persist.mkdir(parents=True, exist_ok=True)
    if (output / "generation_metrics.jsonl").is_file():
        replace_file(output / "generation_metrics.jsonl", persist / "generation_metrics.jsonl")
    save_resume_point(model, optimizer, scheduler, step, best, output, persist)
    if config.get("stop_after") and step >= config["stop_after"]:
        print(f"STOPPED on request at update {step}", flush=True)
        os._exit(0)     # a test of VM loss, so nothing gets to clean up


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["llama", "gemma", "qwen", "ministral"])
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--persist-dir", required=True, help="Folder holding the resume point. launch.py syncs it with Drive")
    ap.add_argument("--checkpoint-steps", type=int, default=2500)
    ap.add_argument("--max-updates", type=int)
    ap.add_argument("--stop-after", type=int, help="Test only: exit at this checkpoint, as if the VM had died")
    ap.add_argument("--validation-steps", type=int, help="Test only")
    ap.add_argument("--generation-steps", type=int, help="Test only")
    ap.add_argument("--override", help="YAML/JSON file applied over the paper recipe through BYOD's own --config merge")
    ap.add_argument("--init-adapter", help="Continue training this saved adapter instead of a fresh one (reversal test)")
    a = ap.parse_args()
    persist = Path(a.persist_dir)
    override = str(Path(a.override).resolve()) if a.override else None

    patch_byod()
    sys.path[:0] = [BYOD_ROOT, str(Path(__file__).resolve().parent)]
    os.chdir(BYOD_ROOT)     # BYOD resolves relative paths from the repository root
    os.environ.update(USE_TF="0", USE_TORCH="1")
    from train import load_training_config
    from diffusion_lm.training import run_training

    config = load_training_config(a.model, "paper", config_path=override, output_dir=a.output_dir, max_updates=a.max_updates)
    if a.init_adapter:
        config["resume_from_adapter"] = str(Path(a.init_adapter).resolve())
    config.update(checkpoint_mode="every_checkpoint", checkpoint_steps=a.checkpoint_steps,
                  resume_persist_dir=str(persist), stop_after=a.stop_after)
    if a.validation_steps:
        config["validation_steps"] = a.validation_steps
    if a.generation_steps:
        config["generation_perplexity"]["interval_steps"] = a.generation_steps
    assert a.checkpoint_steps % int(config["validation_steps"]) == 0, "a checkpoint would add validations"
    if (persist / "resume.pt").is_file():
        config["resume_from_checkpoint"] = str(persist)

    metrics = run_training(config)
    persist.mkdir(parents=True, exist_ok=True)
    (persist / "test_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True, default=float) + "\n")
    print(f"DONE test_loss={metrics['weighted_loss']:.4f}", flush=True)


if __name__ == "__main__":
    main()
