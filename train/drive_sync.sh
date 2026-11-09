#!/bin/bash
# Runs on the Colab VM beside training. Uploads each new resume point, best adapter and log straight to Drive.
# Usage: drive_sync.sh LOCAL_PERSIST_DIR DRIVE_PATH JOB_LOG
# Drive keeps the old file until an upload completes, so a VM lost mid-upload leaves the previous resume point intact.
local_dir=$1; drive=$2; log=$3
export RCLONE_CONFIG=/content/rclone.conf
flags="--drive-chunk-size 256M --retries 5 --low-level-retries 20 --log-level ERROR"
out=/content/drive_sync.log
last=""
while true; do
  rclone copyto "$log" "$drive/logs/job.log" $flags >> "$out" 2>&1
  for f in metrics.jsonl generation_metrics.jsonl; do
    if [ -f "$local_dir/$f" ]; then rclone copyto "$local_dir/$f" "$drive/logs/$f" $flags >> "$out" 2>&1; fi
  done
  stamp=$(stat -c %Y "$local_dir/resume.pt" 2>/dev/null)
  if [ -n "$stamp" ] && [ "$stamp" != "$last" ]; then
    begin=$(date +%s)
    # Stage a hard link, because the trainer may replace resume.pt while it is being uploaded.
    ln -f "$local_dir/resume.pt" "$local_dir/upload.pt"
    if rclone sync "$local_dir/best" "$drive/best" $flags >> "$out" 2>&1 \
       && rclone copyto "$local_dir/upload.pt" "$drive/resume/resume.pt" $flags >> "$out" 2>&1; then
      last=$stamp
      step=$(grep 'RESUME POINT saved' "$log" | tail -1 | grep -oE '[0-9]+$')
      echo "$(date +%H:%M:%S) UPLOADED update $step in $(( $(date +%s) - begin )) s" >> "$out"
      rclone copyto "$out" "$drive/logs/drive_sync.log" $flags
    else
      echo "$(date +%H:%M:%S) UPLOAD FAILED" >> "$out"
    fi
    rm -f "$local_dir/upload.pt"
  fi
  [ -f /content/sync_stop ] && [ "$stamp" == "$last" ] && break
  sleep 60
done
