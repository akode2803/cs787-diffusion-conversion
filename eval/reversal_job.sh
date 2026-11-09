#!/bin/bash
# Reversal test on the VM, Qwen pair. Teaches 1,000 invented facts in one direction to the AR control and to the
# retrained conversion, with each arm's own objective and identical data, updates and learning rate, then tests both
# directions. Untrained arms are scored first: they should know none of the facts.
# Needs: BYOD installed, /content/adapters/qwen-ar and /content/adapters/qwen-byod-retrained fetched.
# Settings (environment): UPDATES (default 2000), EPOCHS (passes over the 4,000 phrasings, default 8; at batch 16,
# 2,000 updates = 8 passes), LR (default 5e-5), TAG (suffix for arm names and folders, default empty).
# FORMS (named or namefree, default named): training prompts; see reversal_data.py. SKIP_BASE=1 skips scoring the
# untrained arms (their scores do not depend on FORMS or TAG). DRIVE: Drive root, default cs787drive:CS787-Diffusion;
# a teammate's runner sets it to its own remote for the shared folder.
# Run 1 (8 Nov): defaults. Run 2: UPDATES=10000 EPOCHS=40 TAG=10k, because after 2,000 updates the conversion knew
# only 6.6% of the facts forward, so its reverse score said nothing.
# Run 3 (planned 9 Nov): UPDATES=10000 EPOCHS=40 TAG=nf10k FORMS=namefree SKIP_BASE=1, because in runs 1-2 every
# training prompt contained the name, so the conversion could copy it instead of learning description -> name.
set -e
cd /content
UPDATES=${UPDATES:-2000}
EPOCHS=${EPOCHS:-8}
LR=${LR:-5.0e-5}
TAG=${TAG:-}
FORMS=${FORMS:-named}
SKIP_BASE=${SKIP_BASE:-0}
DRIVE=${DRIVE:-cs787drive:CS787-Diffusion}
OUT=${OUT:-/content/eval_out}
python3 reversal_data.py /content/reversal_data --facts 1000 --epochs $EPOCHS --forms $FORMS
cat > /content/reversal_override.yaml <<EOF
dataset_name: /content/reversal_data/train_data
dataset_revision: null
prepared_dataset_name: null
learning_rate: $LR
warmup_steps: 50
validation_steps: 250
generation_perplexity: {enabled: false}
EOF
if [ "$SKIP_BASE" != 1 ]; then
python3 run_eval.py --arm ar:qwen:/content/adapters/qwen-ar --name qwen-ar-control --tasks rev_forward,rev_reverse --out $OUT
python3 run_eval.py --arm byod:qwen:/content/adapters/qwen-byod-retrained --name qwen-byod-retrained --tasks rev_forward,rev_reverse --out $OUT
fi
AR=/content/rev_ar$TAG; BY=/content/rev_byod$TAG
[ -f $AR/best/adapter_config.json ] || python3 ar_control.py --model qwen --output-dir $AR \
  --override /content/reversal_override.yaml --init-adapter /content/adapters/qwen-ar --max-updates $UPDATES --checkpoint-steps $UPDATES
python3 run_eval.py --arm ar:qwen:$AR/best --name qwen-ar-control+facts$TAG --tasks rev_forward,rev_reverse --out $OUT
[ -f $BY/best/adapter_config.json ] || python3 retrain.py --model qwen --output-dir $BY --persist-dir /content/persist/rev_byod$TAG \
  --override /content/reversal_override.yaml --init-adapter /content/adapters/qwen-byod-retrained --max-updates $UPDATES --checkpoint-steps $UPDATES
python3 run_eval.py --arm byod:qwen:$BY/best --name qwen-byod-retrained+facts$TAG --tasks rev_forward,rev_reverse --out $OUT
python3 summarize_reversal.py $OUT "facts$TAG" || true
R=$DRIVE/eval/reversal
# Uploads are capped and may fail without stopping the job: on 8 Nov a Drive rate limit (403, rclone's shared
# client_id) made one adapter upload retry for 70 min while the G4 idled, which used up the account's last CU.
# Small files go first, so they are on Drive even if a big adapter fails.
up() { timeout 600 rclone copy "$@" --retries 2 --low-level-retries 3 --log-level ERROR || echo "UPLOAD FAILED: $1"; }
for d in $AR $BY; do up $d $R/training/$(basename $d) --include 'metrics.jsonl' --include '*.json'; done
up /content/reversal_data $R/data$TAG --exclude 'train_data/train.jsonl'
for d in $AR $BY; do up $d/best $R/adapters/$(basename $d); done
echo REVERSAL DONE
