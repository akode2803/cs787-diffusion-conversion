#!/bin/zsh
# Runs jobs one after another, each under supervise.sh. Started with nohup so it outlives a Claude session.
# Usage: run_lane.sh LANE_NAME SESSION:JOB ...   (log: lanes/LANE_NAME.log)
cd "$(dirname "$0")/.."
lane=$1; shift
for pair in "$@"; do
  session=${pair%%:*}; job=${pair#*:}
  echo "$(date +%H:%M) LANE $lane START $session $job"
  zsh supervise.sh "$session" "$job" G4 15
  echo "$(date +%H:%M) LANE $lane END $session exit $?"
done
echo "$(date +%H:%M) LANE $lane ALL DONE"
