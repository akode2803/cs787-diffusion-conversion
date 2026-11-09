"""Runs one job on the VM from /content/job.json and waits, because a Colab kernel stays alive only while it is busy.

job.json fields:
  log      file on the VM that receives the job's output
  command  shell command to run
  drive    optional: Drive folder for this job; the job log goes to <drive>/logs/job.log
  persist  optional, training: VM folder the trainer saves resume points in. With drive set, an earlier resume point
           is restored from Drive first, and drive_sync.sh uploads new ones during the run
  fetch    optional: [[drive path, VM folder], ...] copied to the VM before the command, e.g. adapters
  mirror   optional: [[VM folder, drive path], ...] restored from Drive first, then copied back every 60 s and at the
           end. Evaluation uses this, so a rerun on a new VM skips examples already done
  max_hours optional, default 6: hard limit; vm_guard.py stops the VM when it passes

vm_guard.py runs detached beside the job and stops (unassigns) this VM itself when the job ends, the limit passes or
the GPU idles for 25 min. launch.py refuses to start a job without it. Needs /content/colab_token.json and
/content/endpoint.txt, uploaded by supervise.sh.

Needs /content/rclone.conf (and /content/drive_sync.sh for training), uploaded by the operator.
"""
import json, os, signal, subprocess

job = json.load(open("/content/job.json"))
os.environ["RCLONE_CONFIG"] = "/content/rclone.conf"
QUIET = "--log-level ERROR"
# Final uploads after the job are capped at 10 min each, so a Drive rate limit cannot keep a paid GPU idle (8 Nov).
CAPPED = "timeout 600"


def shell(command):
    print("+", command, flush=True)
    return subprocess.call(command, shell=True)


for stop in ("/content/sync_stop", "/content/mirror_stop"):
    if os.path.exists(stop):
        os.remove(stop)
with open(job["log"], "a") as stream:   # the supervisor matches this line to tell this run's log from an older one
    stream.write(f"RUN {open('/content/run_id.txt').read().strip() if os.path.exists('/content/run_id.txt') else 'unknown'}\n")

# The guard goes first, so even a failure during setup below cannot leave the VM running.
for path in ("/content/job_ended", "/content/job.pid"):
    if os.path.exists(path):
        os.remove(path)
if os.path.exists("/content/guard.pid"):
    try:
        os.kill(int(open("/content/guard.pid").read()), signal.SIGTERM)
    except (OSError, ValueError):
        pass
assert os.path.exists("/content/colab_token.json") and os.path.exists("/content/endpoint.txt"), \
    "GUARD MISSING: colab_token.json or endpoint.txt not uploaded; refusing to run a job that cannot stop its VM"
os.chmod("/content/colab_token.json", 0o600)
json.dump({"endpoint": open("/content/endpoint.txt").read().strip(), "max_hours": job.get("max_hours", 6),
           "log": job["log"], "drive": job.get("drive")}, open("/content/guard.json", "w"))
guard = subprocess.Popen(["python3", "/content/vm_guard.py"], start_new_session=True,
                         stdout=open("/content/guard.out", "a"), stderr=subprocess.STDOUT)
open("/content/guard.pid", "w").write(str(guard.pid))
print("GUARD pid", guard.pid, flush=True)

code = None
try:
    if job.get("drive") or job.get("fetch") or job.get("mirror"):
        if shell("which rclone >/dev/null") != 0:
            shell("cd /content && curl -sSL -o rclone.zip https://downloads.rclone.org/rclone-current-linux-amd64.zip && unzip -qo rclone.zip && cp rclone-*-linux-amd64/rclone /usr/local/bin/")
    sync = None
    if job.get("drive") and job.get("persist"):
        persist, drive = job["persist"], job["drive"]
        os.makedirs(persist, exist_ok=True)
        shell(f"rclone copyto {drive}/resume/resume.pt {persist}/resume.pt --ignore-errors {QUIET}")
        shell(f"rclone copyto {drive}/logs/metrics.jsonl {persist}/metrics.jsonl {QUIET}")
        shell(f"rclone copyto {drive}/logs/generation_metrics.jsonl {persist}/generation_metrics.jsonl {QUIET}")
        shell(f"rclone copy {drive}/best {persist}/best {QUIET}")
        sync = subprocess.Popen(["bash", "/content/drive_sync.sh", persist, drive, job["log"]])

    for source, target in job.get("fetch", []):
        shell(f"rclone copy {source} {target} {QUIET}")

    mirror = None
    if job.get("mirror"):
        pairs = job["mirror"]
        for local, remote in pairs:
            os.makedirs(local, exist_ok=True)
            shell(f"rclone copy {remote} {local} {QUIET}")
        # --update: never replace a newer Drive file. Each VM restores all of eval/main, including files other VMs
        # are still writing; without it, those stale copies would be pushed back over the newer ones (8 Nov).
        copies = "; ".join(f"{CAPPED} rclone copy --update {local} {remote} {QUIET}" for local, remote in pairs)
        log_copy = f"; {CAPPED} rclone copyto {job['log']} {job['drive']}/logs/job.log {QUIET}" if job.get("drive") else ""
        mirror = subprocess.Popen(["bash", "-c", f"while [ ! -f /content/mirror_stop ]; do {copies}{log_copy}; sleep 60; done"])

    with open(job["log"], "a") as log:
        process = subprocess.Popen(job["command"], shell=True, stdout=log, stderr=subprocess.STDOUT)
        open("/content/job.pid", "w").write(str(process.pid))
        code = process.wait()

    if sync:
        open("/content/sync_stop", "w").close()
        sync.wait()
        final = os.path.join(job["persist"], "test_metrics.json")
        if os.path.exists(final):
            shell(f"rclone copyto {final} {job['drive']}/logs/test_metrics.json {QUIET}")
    if mirror:
        open("/content/mirror_stop", "w").close()
        mirror.wait()
        for local, remote in job["mirror"]:
            shell(f"{CAPPED} rclone copy --update {local} {remote} {QUIET}")
    if job.get("drive"):
        shell(f"{CAPPED} rclone copyto {job['log']} {job['drive']}/logs/job.log {QUIET}")
    print("EXIT", code)
finally:
    open("/content/job_ended", "w").close()   # tells vm_guard.py to stop the VM now
