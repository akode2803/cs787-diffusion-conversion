# CS787 Diffusion Project on Google Drive

Large files live here. Code, docs and small results also stay on the Mac, and `diffgap/sync_to_drive.sh` copies them here.

| Folder | Contents |
|---|---|
| `runs/<model>/<arm>/resume/` | Latest resume point: adapter, optimizer, schedule, step. 3.9 GB for Qwen, 5.9 GB for Ministral, replaced at each checkpoint |
| `runs/<model>/<arm>/best/` | Best adapter by validation loss. This is the file evaluations use |
| `runs/<model>/<arm>/logs/` | Training log, metrics, upload log, test metrics |
| `eval/` | Evaluation outputs |
| `code/` | Snapshot of the project code |
| `docs/` | Plan, experiment log, status |
| `archive/goal-revision/` | Old topic's results, kept as a fallback negative study |
| `archive/loop-census/` | 12-model loop census, for the later hobby project |

Models are `qwen2.5-7b` and `ministral-8b`. Arms are `ar-control` and `byod-retrained`.

To continue a lost run, start a VM, upload the same `job.json`, and run `launch.py`. It restores the resume point from here.
