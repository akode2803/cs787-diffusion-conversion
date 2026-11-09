#!/usr/bin/env bash
# Runs one job from runner/jobs/ on the runner's own Colab account and posts the outcome to channel/from_runner/.
#   runner/run_job.sh MSG_ID JOB_NAME            e.g. runner/run_job.sh 001 reversal_nf10k
#   runner/run_job.sh MSG_ID JOB_NAME __report   posts the outcome from the Drive log, if the supervisor died
#   DRY_RUN=1 runner/run_job.sh ...              checks only (or prints the report instead of posting it)
# Checks run in the foreground (balance, Drive read and write, git push, no session of the same name). If any fails,
# nothing is started and no CU is spent. Then the supervisor detaches (nohup), so it outlives the agent and the
# terminal; it posts "ack" at launch and "result" or "error" at the end, by itself.
# Settings: runner/local.env (never committed), see runner/local.env.example. GPU_WAIT_MIN (default 180): how long
# to keep retrying when no GPU of the requested kind is free.
#
# The VM stops itself: launch.py starts vm_guard.py on the VM, which unassigns the VM when the job ends, when
# max_hours passes, or after 25 idle GPU minutes. This script also stops the VM when it sees the job finish. It only
# ever touches the Colab session named $SESSION, which it creates; never any other session on the account.
# Adapted from train/supervise.sh (the project's own supervisor), made bash-only and configurable.
set -u
cd "$(dirname "$0")/.."
ROOT=$(pwd)
id=${1:?message id}; name=${2:?job name}
template=runner/jobs/$name.json
[ -f "$template" ] || { echo "NO SUCH JOB: $template"; exit 2; }
[ -f runner/local.env ] || { echo "MISSING runner/local.env (copy runner/local.env.example)"; exit 2; }
source runner/local.env
SESSION=${SESSION:-cs787-runner}; GPU=${GPU:-G4}; MIN_CU=${MIN_CU:-15}; COLAB=${COLAB:-colab}; MAX_RESTARTS=${MAX_RESTARTS:-2}
# A job may name its own GPU ("gpu": "A100"), which overrides GPU from local.env. Added 10 Nov: no G4 was free for
# hours, so request 005 runs the same job on an A100.
job_gpu=$(python3 -c "import json;print(json.load(open('$template')).get('gpu', ''))")
[ -n "$job_gpu" ] && GPU=$job_gpu
[ -n "${DRIVE:-}" ] || { echo "DRIVE is not set in runner/local.env"; exit 2; }
RCLONE_CONF=${RCLONE_CONF:-$(rclone config file 2>/dev/null | tail -1)}
RCLONE_CONF=${RCLONE_CONF/#\~/$HOME}
export RCLONE_CONFIG=$RCLONE_CONF
state=runner/state/$id; mkdir -p "$state"
limited() { perl -e 'alarm shift; exec @ARGV' "$@"; }
say() { echo "$(date +%H:%M) $*"; }
post() { if [ -n "${DRY_RUN:-}" ]; then echo "DRY_RUN post $1:"; cat "$2"; else runner/post.sh "$id" "$1" "$2"; fi; }

if [ "${3:-}" = "__report" ]; then
  # Used by the agent when the supervisor died (e.g. the laptop slept): reads the job log from Drive and posts it.
  job=$state/job.json; [ -f "$job" ] || { echo "no job file for $id"; exit 2; }
  limited 90 rclone copyto "$(python3 -c "import json;print(json.load(open('$job'))['drive']+'/logs/job.log')")" "$state/job.log" 2>/dev/null
  if limited 60 "$COLAB" sessions 2>&1 | grep -q "\[$SESSION\]"; then echo "VM $SESSION is still running; the job is not over"; exit 3; fi
  REPORT_ONLY=1
fi

if [ "${3:-}" != "__supervise" ] && [ -z "${REPORT_ONLY:-}" ]; then
  # ---------- checks, in the foreground ----------
  fail() { say "CHECK FAILED: $*"; exit 1; }
  for tool in rclone git python3 perl; do command -v $tool >/dev/null || fail "$tool not installed"; done
  command -v "$COLAB" >/dev/null || fail "colab CLI not found ($COLAB)"
  [ -f "$RCLONE_CONF" ] || fail "rclone config $RCLONE_CONF not found"
  [ -f ~/.config/colab-cli/token.json ] || fail "Colab CLI not logged in (~/.config/colab-cli/token.json missing): the human runs the login"
  [ -f "$state/status" ] && grep -q running "$state/status" && fail "message $id already has a running job"
  # Fill in the Drive root. "remote:" + "/eval" = "remote:/eval", which rclone reads as a path from the remote's root.
  python3 - "$template" "$state/job.json" "$DRIVE" <<'PY' || fail "could not write job file"
import json, sys
src, dst, drive = sys.argv[1:4]
text = open(src).read().replace("@DRIVE@", drive)
job = json.loads(text)
for key in ("about", "est_cu"):
    job.pop(key, None)
json.dump(job, open(dst, "w"), indent=2)
PY
  est=$(python3 -c "import json;print(json.load(open('$template')).get('est_cu', 15))")
  balance=$(limited 60 "$COLAB" usage 2>&1 | grep -oE 'balance: [0-9.]+' | grep -oE '[0-9.]+')
  [ -n "$balance" ] || fail "could not read the Colab balance (colab usage)"
  python3 -c "import sys; sys.exit(0 if $balance >= max($MIN_CU, $est) else 1)" || fail "balance $balance CU is below max(MIN_CU=$MIN_CU, job estimate $est)"
  limited 60 "$COLAB" sessions 2>&1 | grep -q "\[$SESSION\]" && fail "a session named $SESSION already exists; stop it only if this runner created it"
  for path in runs/qwen2.5-7b/ar-control/best runs/qwen2.5-7b/byod-retrained/best; do
    limited 120 rclone lsf "$DRIVE/$path" 2>/dev/null | grep -q adapter_config.json || fail "cannot read $DRIVE/$path (is the folder shared with you, and DRIVE right?)"
  done
  echo "write test $(date)" > "$state/write_test.txt"
  limited 120 rclone copyto "$state/write_test.txt" "$DRIVE/eval/runner_write_test.txt" --log-level ERROR || fail "cannot write to $DRIVE (you need Editor access)"
  git push --dry-run -q 2>/dev/null || fail "git push is not allowed (accept the GitHub collaborator invite)"
  say "CHECKS PASSED: balance $balance CU, estimate $est CU, Drive read and write OK"
  [ -n "${DRY_RUN:-}" ] && { say "DRY_RUN: not starting"; exit 0; }
  echo "running" > "$state/status"
  echo "$balance" > "$state/balance_before"
  nohup "$0" "$id" "$name" __supervise >> "$state/supervisor.log" 2>&1 &
  say "SUPERVISOR STARTED (pid $!), log: $state/supervisor.log"
  exit 0
fi

# ---------- supervisor, detached ----------
job=$state/job.json
log_remote=$(python3 -c "import json;print(json.load(open('$job'))['log'])")
drive_log=$(python3 -c "import json;print(json.load(open('$job'))['drive']+'/logs/job.log')")
command -v caffeinate >/dev/null && { caffeinate -i -w $$ & }   # macOS: no idle sleep while this runs
restarts=0; last_step=""; last_change=$(date +%s); run_id=""

finish() {  # $1 = result|error, $2 = one-line reason
  local body=$state/body.md
  {
    echo "job: $name"; echo "outcome: $2"
    echo "balance before: $(cat "$state/balance_before" 2>/dev/null) CU, after: $(limited 60 "$COLAB" usage 2>&1 | grep -oE 'balance: [0-9.]+' | grep -oE '[0-9.]+') CU"
    echo; echo '```'
    grep -aE '^(RESULT|EVAL DONE|DONE|REVERSAL DONE|UPLOAD FAILED|GUARD .*STOP|EXIT)' "$state/job.log" 2>/dev/null | tail -40
    if [ "$1" = error ]; then echo "--- last lines ---"; tail -25 "$state/job.log" 2>/dev/null; fi
    echo '```'
    echo "Full outputs: $(python3 -c "import json;print(json.load(open('$job'))['drive'])") and the job's mirror folder on Drive."
  } > "$body"
  echo "$1" > "$state/status"
  post "$1" "$body"
}

start_job() {
  say "SETUP on $SESSION"
  for attempt in 1 2 3; do
    limited 240 "$COLAB" exec -s "$SESSION" -f smoke/setup_vm.py --timeout 230 > "$state/setup.out" 2>&1
    grep -qE ' True$' "$state/setup.out" && break
    say "setup attempt $attempt did not finish"
  done
  python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.config/colab-cli/sessions.json')))['$SESSION']['endpoint'])" > "$state/endpoint.txt"
  limited 60 "$COLAB" upload -s "$SESSION" "$state/endpoint.txt" /content/endpoint.txt >/dev/null 2>&1
  run_id="$SESSION-$(date +%s)"; echo "$run_id" > "$state/run_id.txt"
  limited 60 "$COLAB" upload -s "$SESSION" "$state/run_id.txt" /content/run_id.txt >/dev/null 2>&1
  limited 60 "$COLAB" upload -s "$SESSION" ~/.config/colab-cli/token.json /content/colab_token.json >/dev/null 2>&1
  for f in train/vm_guard.py train/ar_loss.py train/ar_control.py train/retrain.py train/launch.py train/drive_sync.sh \
           eval/arms.py eval/sampling.py eval/tasks.py eval/run_eval.py eval/gpu_check.py eval/reversal_data.py \
           eval/reversal_job.sh eval/summarize_reversal.py; do
    limited 60 "$COLAB" upload -s "$SESSION" "$f" "/content/$(basename "$f")" >/dev/null 2>&1
  done
  limited 60 "$COLAB" upload -s "$SESSION" "$RCLONE_CONF" /content/rclone.conf >/dev/null 2>&1
  limited 60 "$COLAB" upload -s "$SESSION" "$job" /content/job.json >/dev/null 2>&1
  nohup perl -e 'alarm 21000; exec @ARGV' "$COLAB" exec -s "$SESSION" -f train/launch.py --timeout 20000 > "$state/exec.out" 2>&1 &
  last_change=$(date +%s); last_step=""
  say "LAUNCHED on $SESSION"
}

if [ -n "${REPORT_ONLY:-}" ]; then
  if grep -aq 'ALLDONE' "$state/job.log"; then finish result "finished (reported later from the Drive log)"
  else finish error "VM gone without ALLDONE (reported later from the Drive log)"; fi
  exit 0
fi

# G4s are often all taken for a while (request 001, 9 Nov 20:14 UTC, failed on the first try). Wait for one, up to
# GPU_WAIT_MIN minutes, trying every 5 minutes. Nothing is spent while waiting.
waited=0
until limited 170 "$COLAB" new -s "$SESSION" --gpu "$GPU" >/dev/null 2>&1; do
  if [ $waited -ge "${GPU_WAIT_MIN:-180}" ]; then finish error "no $GPU VM free for $waited min; nothing was spent"; exit 1; fi
  [ $waited = 0 ] && say "no $GPU free; retrying every 5 min for up to ${GPU_WAIT_MIN:-180} min"
  sleep 300; waited=$((waited + 5))
done
[ $waited -gt 0 ] && say "got a $GPU after $waited min"
start_job
printf 'job: %s\nlaunched on Colab session %s (%s). The VM stops itself at the end or after %s h at most.\n' \
  "$name" "$SESSION" "$GPU" "$(python3 -c "import json;print(json.load(open('$job')).get('max_hours'))")" > "$state/ack.md"
post ack "$state/ack.md"

while true; do
  sleep 120
  if ! limited 60 "$COLAB" sessions 2>&1 | grep -q "\[$SESSION\]"; then
    limited 90 rclone copyto "$drive_log" "$state/job.log" 2>/dev/null
    if grep -aq "^RUN $run_id" "$state/job.log" 2>/dev/null && grep -aq 'GUARD .* STOP' "$state/job.log"; then
      if grep -aq 'ALLDONE' "$state/job.log"; then say "FINISHED, VM stopped itself"; finish result "finished"; exit 0; fi
      say "VM STOPPED ITSELF without ALLDONE"; finish error "VM stopped itself before the job finished"; exit 1
    fi
    restarts=$((restarts + 1))
    say "VM LOST at step ${last_step:-unknown}, restart $restarts"
    pkill -f "exec -s $SESSION" 2>/dev/null
    if [ "$restarts" -gt "$MAX_RESTARTS" ]; then finish error "VM lost $restarts times; gave up"; exit 1; fi
    tries=0
    until limited 170 "$COLAB" new -s "$SESSION" --gpu "$GPU" >/dev/null 2>&1; do
      tries=$((tries + 1)); [ $tries -gt 12 ] && { finish error "VM lost and no $GPU free for an hour"; exit 1; }
      say "no $GPU available, retrying in 5 min"; sleep 300
    done
    start_job; continue
  fi
  limited 60 "$COLAB" download -s "$SESSION" "$log_remote" "$state/job.log" >/dev/null 2>&1 || continue
  step=$(grep -aoE 'step [0-9]+/[0-9]+' "$state/job.log" | tail -1 | grep -oE '[0-9]+' | head -1)
  if grep -aqE '^(Traceback|torch.OutOfMemoryError)' "$state/job.log"; then
    say "JOB ERROR"; sleep 150   # lets launch.py upload the log and results
    limited 60 "$COLAB" download -s "$SESSION" "$log_remote" "$state/job.log" >/dev/null 2>&1
    limited 60 "$COLAB" stop -s "$SESSION" >/dev/null 2>&1
    finish error "the job raised an error; VM stopped"; exit 1
  fi
  if grep -aq 'ALLDONE' "$state/job.log"; then
    sleep 240
    limited 60 "$COLAB" download -s "$SESSION" "$log_remote" "$state/job.log" >/dev/null 2>&1
    limited 60 "$COLAB" stop -s "$SESSION" >/dev/null 2>&1
    say "FINISHED"; finish result "finished"; exit 0
  fi
  if [ "$step" != "$last_step" ]; then last_step=$step; last_change=$(date +%s)
  elif [ $(( $(date +%s) - last_change )) -gt 1200 ]; then say "STALLED at step ${step:-none} for 20 min"; last_change=$(date +%s); fi
done
