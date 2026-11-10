# CS787 Diffusion Project on Google Drive

Large files live here. Code, docs and small results also stay on the Mac, and `diffgap/sync_to_drive.sh` copies them here.

| Folder | Contents |
|---|---|
| `runs/<model>/<arm>/resume/` | Latest resume point: adapter, optimizer, schedule, step. 3.9 GB for Qwen, 5.9 GB for Ministral, replaced at each checkpoint |
| `runs/<model>/<arm>/best/` | Best adapter by validation loss. This is the file evaluations use |
| `runs/<model>/<arm>/logs/` | Training log, metrics, upload log, test metrics |
| `weights/` | **Public** (anyone with the link can view). The trained adapters for others to try, with `README.md` loading instructions. Copies of the `best/` folders below, plus the two reversal run-2 adapters |
| `eval/main/` | Evaluation outputs: one folder per arm, one `.jsonl` per task |
| `eval/reversal/` | Reversal runs 1–2 (data, training logs, adapters); run 3's training logs and adapters (`rev_*nf10k`) were written by Anuj's account |
| `eval/reversal_nf10k/results/` | Reversal run 3 (name-free prompts) eval files, written by Anuj's account; copies in `docs/results/reversal_nf10k/` |
| `eval/jobs/` | One folder per Colab job: job logs uploaded by the VM |
| `code/` | Snapshot of the project code |
| `docs/` | Plan, experiment log, status |
| `archive/goal-revision/` | Old topic's results, kept as a fallback negative study |
| `archive/loop-census/` | 12-model loop census, for the later hobby project |

Models are `qwen2.5-7b` and `ministral-8b`. Arms are `ar-control` and `byod-retrained`.

Backups (not shared): full copies of this folder are kept outside it, as `CS787-Diffusion-BACKUP-2026-11-09` and
`CS787-Diffusion-BACKUP-2026-11-10`. The second one also holds a snapshot archive of the local project folder.

Code: https://github.com/akode2803/cs787-diffusion-conversion

To continue a lost run, start a VM, upload the same `job.json`, and run `launch.py`. It restores the resume point from here.
