# What does diffusion conversion buy? (CS787, IIT Kanpur)

Code for a same-data AR-control study of BYOD's LoRA AR-to-diffusion conversion (Kuiper & Bagheri 2026, "BYOD:
Build Your Own DLM", arXiv 2610.41574). Each converted model is compared with an autoregressive control trained on
the same data, with the same adapter and the same budget.

## Start here

1. **See the results in 5 minutes:** open [`notebooks/tour_of_results.ipynb`](notebooks/tour_of_results.ipynb). It runs on a laptop
   (no GPU) from `results/`, and GitHub shows it with its charts already rendered. It covers the benchmark tables, the cost of
   diffusion with data held fixed, the step-count "compute dial", the infilling gap-length test, and a replay of a
   diffusion model writing an answer out of order.
2. **Read the report:** linked from the course submission; the project Drive folder holds a copy.
3. **Follow the code** in the order the study ran:

| Step | File | What it does |
|---|---|---|
| Train the conversion | `train/retrain.py` | BYOD's own trainer and recipe, plus resume points for Colab |
| Train the AR control | `train/ar_control.py`, `train/ar_loss.py` | Same base, data, LoRA and budget; causal attention and next-token loss |
| Load any model | `eval/arms.py` | One interface for parent, AR control, BYOD conversions and LLaDA |
| Generate | `eval/sampling.py` | Batched AR decoding and BYOD's diffusion sampler, timed |
| Score | `eval/tasks.py`, `eval/run_eval.py` | Benchmarks, HumanEval infilling, one JSON line per example |
| Summarise | `eval/analyze.py`, `eval/make_summary.py` | Accuracies, 95% intervals, paired differences → `results/summary.json` |

## Folder map

| Folder | What it holds |
|---|---|
| `train/` | Conversion retraining (`retrain.py`) and the AR control (`ar_control.py`, `ar_loss.py`) |
| `train/` (VM side) | Colab job launcher (`launch.py`), in-VM self-stop guard (`vm_guard.py`), Drive sync, supervisor (`supervise.sh`) |
| `train/job_*.json` | Every job that was run |
| `eval/` | Arms, diffusion sampler, tasks, the evaluation runner (`run_eval.py`), analysis and rescoring scripts |
| `eval/reversal_*` | Reversal test (invented facts); `reversal_recall.py` re-scores run 3 and checks whether each arm learned the facts |
| `smoke/` | VM setup: BYOD at commit 211d054 |
| `notebooks/` | `tour_of_results.ipynb`: the interactive, no-GPU tour of the results |
| `results/` | `summary.json` (every measured accuracy and timing) and `demo_data.json` (recorded generations) |
| `runner/`, `channel/` | Lets a teammate run a job on their own Colab account and report back. Start with `runner/HANDOFF.md` |

## Try the models

Our trained LoRA adapters are public on Google Drive: **[weights folder](https://drive.google.com/drive/folders/1yVh_LSR7i7sPPRZjECHNl1wOB88NTijv)**. The folder holds:
- the Qwen2.5-7B conversion retrain;
- the Qwen2.5-7B and Ministral-8B same-data AR controls;
- the two reversal-test adapters.

Loading instructions are in [`WEIGHTS_README.md`](WEIGHTS_README.md), and the same file is in the Drive folder. The AR controls are
ordinary PEFT adapters. The diffusion conversion loads through BYOD's code, as `eval/arms.py` does. BYOD's own released
conversions are on Hugging Face (e.g. `Ruurd/BYOD-Qwen2.5-7B`).

## Results in brief

- **Code:** holding BYOD's data fixed, diffusion costs 30–45 points on code (HumanEval, MBPP) on both backbones.
- **Maths:** on Qwen, most of the GSM8K drop comes from the data (original 0.92, AR control 0.50).
- **Speed:** conversions are about 17× slower than the original model at batch 16.
- **Infilling:** their single-line infilling advantage needs the exact gap length.
- **Reversal:** no reversal gain. With name-free prompts (run 3), the conversion did not learn the facts that the AR control learned.

The full report goes to the course (due 13 Nov 2026); every number in it is logged with its source in the project notes.

## What is not here

Raw per-example outputs, job logs and training resume points are on the project's Google Drive. No credentials are
stored in this repository: Colab and Drive logins stay on each person's own machine.

