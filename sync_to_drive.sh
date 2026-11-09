#!/bin/zsh
# Copies code, docs and small results to the project's Drive folder. Secrets are excluded. Checkpoints go straight from the VM.
export PATH=/opt/homebrew/bin:$PATH
cd "$(dirname "$0")/.."
R=cs787drive:CS787-Diffusion
skip=(--exclude '.git/**' --exclude 'runner/state/**' --exclude '*.env' --exclude '.env*' --exclude '__pycache__/**' --exclude '*.pt' --exclude '*.partial' --exclude '.DS_Store' --log-level ERROR)
rclone copyto diffgap/DRIVE_README.md $R/README.md --log-level ERROR
rclone sync diffgap $R/code/diffgap $skip
rclone sync docs $R/docs $skip
rclone sync goalrev $R/archive/goal-revision/code $skip
for run in main main_gemma pilot pilot2 local_pilot local_main v1_fl31 v1_gemma31c v2_fl; do rclone sync results/$run $R/archive/goal-revision/results/$run $skip; done
rclone sync loopfix $R/archive/loop-census/code $skip
rclone sync results/census $R/archive/loop-census/results $skip
rclone sync results/smoke $R/eval/smoke $skip
echo SYNC_DONE
