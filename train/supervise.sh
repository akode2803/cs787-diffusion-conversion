#!/bin/zsh
# Keeps one job alive on Colab. If its VM vanishes, starts a new one and relaunches; launch.py then resumes from Drive.
# Light on the Mac: it only fetches the small job log. Checkpoints go from the VM to Drive.
# The Mac is only a backup for stopping VMs: vm_guard.py on the VM stops it when the job ends, the job's max_hours
# passes or the GPU idles 25 min, because the Mac may sleep. A VM that the guard stopped is not restarted.
# Usage: supervise.sh SESSION JOB_JSON GPU [MIN_CU]
export PATH=/opt/miniconda3/bin:/opt/homebrew/bin:$PATH
cd "$(dirname "$0")"
session=$1; job=$2; gpu=$3; min_cu=${4:-15}
state=$(mktemp -d)
log_remote=$(python3 -c "import json;print(json.load(open('$job'))['log'])")
drive_log=$(python3 -c "import json;print(json.load(open('$job'))['drive']+'/logs/job.log')")
caffeinate -i -w $$ &   # keeps the Mac from idle sleep while this runs (a closed lid still sleeps it)
restarts=0; last_step=""; last_change=$(date +%s); reported=0

say() { echo "$(date +%H:%M) $*"; }

limited() {  # runs a colab command with a hard limit, because the client can hang
  perl -e 'alarm shift; exec @ARGV' "$@"
}

start_job() {
  say "SETUP on $session"
  for attempt in 1 2 3; do
    limited 240 colab exec -s "$session" -f ../smoke/setup_vm.py --timeout 230 > "$state/setup.out" 2>&1
    grep -qE ' True$|^.* True$' "$state/setup.out" && break
    say "setup attempt $attempt did not finish"
  done
  # The guard needs this VM's endpoint and the Colab login; launch.py refuses to start a job without them.
  python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.config/colab-cli/sessions.json')))['$session']['endpoint'])" > "$state/endpoint.txt"
  limited 60 colab upload -s "$session" "$state/endpoint.txt" /content/endpoint.txt >/dev/null 2>&1
  run_id="$session-$(date +%s)"; echo "$run_id" > "$state/run_id.txt"   # tells this run's log from an older one
  limited 60 colab upload -s "$session" "$state/run_id.txt" /content/run_id.txt >/dev/null 2>&1
  limited 60 colab upload -s "$session" ~/.config/colab-cli/token.json /content/colab_token.json >/dev/null 2>&1
  for f in vm_guard.py ar_loss.py ar_control.py retrain.py launch.py drive_sync.sh ../eval/arms.py ../eval/sampling.py ../eval/tasks.py ../eval/run_eval.py ../eval/gpu_check.py ../eval/reversal_data.py ../eval/reversal_job.sh; do
    limited 60 colab upload -s "$session" "$f" "/content/${f:t}" >/dev/null 2>&1
  done
  limited 60 colab upload -s "$session" ~/.config/rclone/rclone.conf /content/rclone.conf >/dev/null 2>&1
  limited 60 colab upload -s "$session" "$job" /content/job.json >/dev/null 2>&1
  nohup perl -e 'alarm 21000; exec @ARGV' colab exec -s "$session" -f launch.py --timeout 20000 > "$state/exec.out" 2>&1 &
  last_change=$(date +%s); last_step=""
  say "LAUNCHED on $session"
}

if ! colab sessions 2>&1 | grep -q "\[$session\]"; then limited 170 colab new -s "$session" --gpu "$gpu" >/dev/null 2>&1; fi
start_job

while true; do
  sleep 120
  if ! colab sessions 2>&1 | grep -q "\[$session\]"; then
    # Did the VM stop itself? Then the guard's last words are in the Drive copy of the job log.
    RCLONE_CONFIG=~/.config/rclone/rclone.conf limited 90 rclone copyto "$drive_log" "$state/drive_job.log" 2>/dev/null
    if grep -aq "^RUN $run_id" "$state/drive_job.log" 2>/dev/null && grep -aq 'GUARD .* STOP' "$state/drive_job.log"; then
      if grep -aq 'ALLDONE' "$state/drive_job.log"; then
        say "FINISHED, VM stopped itself: $(grep -a '^DONE' "$state/drive_job.log" | tail -1 | cut -c1-100)"; exit 0
      fi
      say "VM STOPPED ITSELF without ALLDONE: $(grep -a 'GUARD .* STOP' "$state/drive_job.log" | tail -1 | cut -c1-120)"
      say "last error: $(grep -aE 'Error' "$state/drive_job.log" | tail -1 | cut -c1-140)"
      exit 1
    fi
    restarts=$((restarts + 1))
    say "VM LOST at step ${last_step:-unknown}, restart $restarts"
    pkill -f "colab exec -s $session" 2>/dev/null
    balance=$(colab usage 2>&1 | grep -oE 'balance: [0-9.]+' | grep -oE '[0-9.]+')
    if [ "$restarts" -gt 8 ] || [ "${balance%.*}" -lt "$min_cu" ]; then say "GIVING UP: restarts $restarts, balance $balance"; exit 1; fi
    until limited 170 colab new -s "$session" --gpu "$gpu" >/dev/null 2>&1; do say "no $gpu available, retrying in 5 min"; sleep 300; done
    start_job
    continue
  fi
  limited 60 colab download -s "$session" "$log_remote" "$state/job.log" >/dev/null 2>&1 || continue
  step=$(grep -aoE 'step [0-9]+/[0-9]+' "$state/job.log" | tail -1 | grep -oE '[0-9]+' | head -1)
  grep -aq 'RESUMED' "$state/job.log" && [ "$reported" = 0 ] && { reported=1; say "$(grep -a 'RESUMED' "$state/job.log" | tail -1 | cut -c1-90)"; }
  if grep -aqE '^(Traceback|torch.OutOfMemoryError)' "$state/job.log"; then
    say "JOB ERROR: $(grep -aE 'Error' "$state/job.log" | tail -1 | cut -c1-140)"
    sleep 150     # lets launch.py upload the log and results, then stops the VM so it does not idle on CU
    limited 60 colab stop -s "$session" >/dev/null 2>&1
    say "STOPPED $session after the error"
    exit 1
  fi
  if grep -aq 'ALLDONE' "$state/job.log"; then
    sleep 240     # lets launch.py finish the final upload
    say "FINISHED: $(grep -a '^DONE' "$state/job.log" | tail -1 | cut -c1-100)"
    limited 60 colab stop -s "$session" >/dev/null 2>&1
    exit 0
  fi
  if [ "$step" != "$last_step" ]; then
    last_step=$step; last_change=$(date +%s)
    [ -n "$step" ] && [ $((step % 2500)) -lt 300 ] && say "progress: update $step"
  elif [ $(( $(date +%s) - last_change )) -gt 1200 ]; then
    say "STALLED at step ${step:-none} for 20 min"; last_change=$(date +%s)
  fi
done
