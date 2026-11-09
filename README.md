# What does diffusion conversion buy? (CS787, Group 13, IIT Kanpur)

Code for a same-data AR-control study of BYOD's LoRA AR-to-diffusion conversion (Kuiper & Bagheri 2026, "BYOD:
Build Your Own DLM", arXiv 2610.41574). Each converted model is compared with an autoregressive control trained on
the same data, with the same adapter and the same budget.

| Folder | What it holds |
|---|---|
| `train/` | Conversion retraining (`retrain.py`) and the AR control (`ar_control.py`, `ar_loss.py`) |
| `train/` (VM side) | Colab job launcher (`launch.py`), in-VM self-stop guard (`vm_guard.py`), Drive sync, supervisor (`supervise.sh`) |
| `train/job_*.json` | Every job that was run |
| `eval/` | Arms, diffusion sampler, tasks, the evaluation runner (`run_eval.py`), analysis and rescoring scripts |
| `eval/reversal_*` | Reversal test (invented facts) |
| `smoke/` | VM setup: BYOD at commit 211d054 |
| `runner/`, `channel/` | Lets a teammate run a job on their own Colab account and report back. Start with `runner/HANDOFF.md` |

Checkpoints and raw outputs are on the project's Google Drive, not here. No credentials are stored in this
repository: Colab and Drive logins stay on each person's own machine.
