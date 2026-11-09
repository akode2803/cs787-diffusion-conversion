#!/usr/bin/env bash
# Posts a message from the runner to channel/from_runner/ and pushes it.
# Usage: runner/post.sh MSG_ID KIND BODY_FILE     e.g. runner/post.sh 001 result /tmp/body.md
# Writes channel/from_runner/<MSG_ID>-<KIND>.md. KIND: ack, status, result, error.
set -u
cd "$(dirname "$0")/.."
id=$1; kind=$2; body=$3
file=channel/from_runner/$id-$kind.md
{ echo "re: $id"; echo "kind: $kind"; echo "time: $(date '+%Y-%m-%d %H:%M %Z')"; echo; cat "$body"; } > "$file"
git add "$file"
git commit -q -m "runner: $id $kind" -- "$file" || true
for attempt in 1 2 3; do
  git pull -q --rebase && git push -q && { echo "POSTED $file"; exit 0; }
  sleep 20
done
echo "POST FAILED: $file is committed locally but not pushed"; exit 1
