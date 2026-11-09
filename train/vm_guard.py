"""Stops (unassigns) this Colab VM from inside the VM, so a paid GPU never waits on the Mac.

Started by launch.py, detached, before the job. It stops the VM as soon as one of these holds:
  - the job has ended, whether it succeeded or failed: launch.py writes /content/job_ended after its final uploads
  - the job's process (/content/job.pid) has been gone for 30 min without job_ended, e.g. after a kernel crash
  - the job's time limit has passed (job.json "max_hours", default 6)
  - the GPU has been idle for 25 minutes in a row (a hung upload or a stuck job)
Before stopping it appends a "GUARD STOP <reason>" line to the job log and uploads the log to Drive, so the Mac
supervisor can tell a finished job from a lost VM.

Why: on 8 Nov a Drive upload retried for 70 min after a job had finished, with the G4 idle, and used the account's
last ~12 CU. The Mac supervisor cannot be relied on to stop VMs because the Mac may sleep.

Needs /content/colab_token.json (the Colab CLI login, uploaded by supervise.sh) and /content/guard.json
({"endpoint", "max_hours", "log", "drive"}), written by launch.py. The login is read here only and never
printed; this file logs reasons and HTTP status codes only.
"""
import json, os, subprocess, sys, time

GUARD = "/content/guard.json"
TOKEN = "/content/colab_token.json"
COLAB = "https://colab.research.google.com/tun/m"
HEADERS = {"Accept": "application/json", "X-Colab-Client-Agent": "colab-cli"}
IDLE_LIMIT = 25 * 60
ORPHAN_LIMIT = 30 * 60
ENDED, JOB_PID = "/content/job_ended", "/content/job.pid"
POLL = 30


def note(log, text):
    line = f"GUARD {time.strftime('%Y-%m-%d %H:%M:%S')} {text}"
    print(line, flush=True)
    with open(log, "a") as stream:
        stream.write(line + "\n")


def alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    # A finished child that nobody reaped is a zombie: treat it as ended.
    try:
        with open(f"/proc/{pid}/stat") as stream:
            return stream.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return False


def gpu_busy():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return any(int(v) > 0 for v in out.split())
    except Exception:
        return True  # cannot tell, so do not count it as idle


def unassign(endpoint, log):
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import AuthorizedSession
    session = AuthorizedSession(Credentials.from_authorized_user_file(TOKEN))
    url, params = f"{COLAB}/unassign/{endpoint}", {"authuser": "0"}
    for attempt in range(1, 6):
        try:
            got = session.get(url, params=params, headers=HEADERS, timeout=60)
            if got.ok:
                xsrf = json.loads(got.text.split("\n", 1)[-1] if got.text.startswith(")]}'") else got.text)["token"]
                done = session.post(url, params=params, headers={**HEADERS, "X-Goog-Colab-Token": xsrf}, timeout=60)
                note(log, f"unassign attempt {attempt}: HTTP {done.status_code}")
                if done.ok:
                    return True
            else:
                note(log, f"unassign attempt {attempt}: token request HTTP {got.status_code}")
        except Exception as error:
            note(log, f"unassign attempt {attempt} failed: {type(error).__name__}")
        time.sleep(20)
    return False


def main():
    cfg = json.load(open(GUARD))
    log, limit = cfg["log"], float(cfg.get("max_hours", 6))
    deadline = time.time() + limit * 3600
    idle_since = orphan_since = None
    note(log, f"started: limit {limit} h, idle limit {IDLE_LIMIT // 60} min")
    while True:
        if os.path.exists(ENDED):
            reason = "job ended"
            break
        if time.time() > deadline:
            reason = f"time limit {limit} h reached"
            break
        pid = int(open(JOB_PID).read()) if os.path.exists(JOB_PID) else None
        if pid and not alive(pid):
            orphan_since = orphan_since or time.time()
            if time.time() - orphan_since > ORPHAN_LIMIT:
                reason = f"job process gone for {ORPHAN_LIMIT // 60} min without an end marker"
                break
        if gpu_busy():
            idle_since = None
        else:
            idle_since = idle_since or time.time()
            if time.time() - idle_since > IDLE_LIMIT:
                reason = f"GPU idle for {IDLE_LIMIT // 60} min"
                break
        time.sleep(POLL)
    note(log, f"STOP {reason}")
    if cfg.get("drive"):
        subprocess.run(f"timeout 120 rclone copyto {log} {cfg['drive']}/logs/job.log --log-level ERROR",
                       shell=True, env={**os.environ, "RCLONE_CONFIG": "/content/rclone.conf"})
    if not unassign(cfg["endpoint"], log):
        note(log, "could not unassign; the Mac supervisor or the user must stop this VM")
        if cfg.get("drive"):
            subprocess.run(f"timeout 120 rclone copyto {log} {cfg['drive']}/logs/job.log --log-level ERROR",
                           shell=True, env={**os.environ, "RCLONE_CONFIG": "/content/rclone.conf"})
        sys.exit(1)


if __name__ == "__main__":
    main()
