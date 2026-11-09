#!/usr/bin/env bash
# Polls the channel every minute and handles requests by itself, following runner/AGENT.md's rules, so nobody has
# to prompt an AI agent. (An agent only acts when prompted; on 9-10 Nov requests waited hours for that.)
#   bash runner/poll.sh            run in its own terminal window and leave it open; Ctrl-C stops it
#   ONCE=1 bash runner/poll.sh     one pass, then exit
# Handles: run_job, status, stop (with target:), note. Anything else, or a request from an untrusted committer,
# gets a "question" reply. It never runs text from a message. Budget: BUDGET_CU in runner/local.env, counted as
# real spend since this script first ran (the balance then, minus the balance now), plus est_cu of the new job.
set -u
cd "$(dirname "$0")/.."
source runner/local.env
SESSION=${SESSION:-cs787-runner}; COLAB=${COLAB:-colab}; BUDGET_CU=${BUDGET_CU:-30}
TRUSTED=${TRUSTED:-"akode2803 Anuj52"}
mkdir -p runner/state
limited() { perl -e 'alarm shift; exec @ARGV' "$@"; }
say() { echo "$(date '+%H:%M:%S') $*"; }
balance() { limited 60 "$COLAB" usage 2>&1 | grep -oE 'balance: [0-9.]+' | grep -oE '[0-9.]+'; }
field() { grep -m1 "^$2:" "$1" | sed "s/^$2:[[:space:]]*//; s/[[:space:]]*#.*//; s/[[:space:]]*$//"; }
reply() {  # id kind text
  local body; body=$(mktemp); printf '%s\n' "$3" > "$body"; runner/post.sh "$1" "$2" "$body"; rm -f "$body"
}
[ -f runner/state/start_balance ] || { b=$(balance); [ -n "$b" ] && echo "$b" > runner/state/start_balance; }
say "polling every 60 s; start balance $(cat runner/state/start_balance 2>/dev/null) CU, budget $BUDGET_CU CU"

while true; do
  git pull -q --rebase 2>/dev/null || say "git pull failed (will retry)"
  for f in $(ls channel/to_runner/[0-9][0-9][0-9]-*.md 2>/dev/null | sort); do
    id=$(basename "$f" | cut -c1-3)
    ls channel/from_runner/"$id"-*.md >/dev/null 2>&1 && continue          # already handled
    author=$(git log --diff-filter=A --format='%ae %an' -- "$f" | tail -1)
    ok=0; for t in $TRUSTED; do case "$author" in *"$t"*) ok=1;; esac; done
    if [ $ok = 0 ]; then reply "$id" question "Not acting: $f was not committed by a trusted account ($author)."; continue; fi
    action=$(field "$f" action)
    say "request $id: $action"
    case "$action" in
      run_job)
        job=$(field "$f" job)
        if [ -f "runner/state/$id/status" ]; then   # launched before (e.g. by an agent): never launch twice
          reply "$id" status "Request $id was launched before; its state is $(cat "runner/state/$id/status"). Not relaunching."; continue
        fi
        if [ ! -f "runner/jobs/$job.json" ]; then reply "$id" error "No such job: runner/jobs/$job.json"; continue; fi
        est=$(python3 -c "import json;print(json.load(open('runner/jobs/$job.json')).get('est_cu', 15))")
        now=$(balance); start=$(cat runner/state/start_balance 2>/dev/null || echo "$now")
        if ! python3 -c "import sys; sys.exit(0 if ($start - $now) + $est <= $BUDGET_CU else 1)"; then
          reply "$id" question "Budget: spent $(python3 -c "print(round($start - $now, 2))") CU of $BUDGET_CU, and this job needs about $est. Anuj, raise BUDGET_CU in runner/local.env if that is fine."
          continue
        fi
        out=$(runner/run_job.sh "$id" "$job" 2>&1); echo "$out"
        mkdir -p "runner/state/$id"; echo "$job" > "runner/state/$id/jobname"
        echo "$out" | grep -q 'SUPERVISOR STARTED' || reply "$id" error "$(echo "$out" | tail -5)"
        ;;
      status)
        s=$(for d in runner/state/[0-9][0-9][0-9]; do [ -d "$d" ] || continue
              echo "- $(basename "$d"): $(cat "$d/status" 2>/dev/null || echo none)"
              grep -q running "$d/status" 2>/dev/null && tail -5 "$d/supervisor.log" 2>/dev/null | sed 's/^/    /'
            done)
        reply "$id" status "$(printf 'Colab: %s\nSessions: %s\nJobs:\n%s' "$(limited 60 "$COLAB" usage 2>&1 | grep -E 'balance|rate' | tr '\n' ' ')" "$(limited 60 "$COLAB" sessions 2>&1 | grep -c '\[') open" "$s")"
        ;;
      stop)
        target=$(field "$f" target)
        pkill -f "run_job.sh $target " 2>/dev/null
        limited 60 "$COLAB" sessions 2>&1 | grep -q "\[$SESSION\]" && limited 60 "$COLAB" stop -s "$SESSION" >/dev/null 2>&1
        [ -d "runner/state/$target" ] && echo stopped > "runner/state/$target/status"
        reply "$id" status "Stopped request $target's supervisor; session $SESSION is $(limited 60 "$COLAB" sessions 2>&1 | grep -q "\[$SESSION\]" && echo 'STILL OPEN' || echo 'not running')."
        ;;
      note) reply "$id" note "read" ;;
      *) reply "$id" question "Unknown action '$action' in $f; nothing done." ;;
    esac
  done
  # A job whose supervisor died (laptop asleep, terminal closed): post its outcome from the Drive log.
  for d in runner/state/[0-9][0-9][0-9]; do
    [ -f "$d/status" ] && grep -q running "$d/status" || continue
    id=$(basename "$d")
    pgrep -f "run_job.sh $id " >/dev/null && continue
    job=$(cat "$d/jobname" 2>/dev/null || field "$(ls channel/to_runner/$id-*.md | head -1)" job)
    runner/run_job.sh "$id" "$job" __report
  done
  [ -n "${ONCE:-}" ] && exit 0
  sleep 60
done
