You are helping me (Anuj) run GPU jobs for my CS787 project group on my Colab account. The project repository is
already cloned on this machine (WSL Ubuntu): https://github.com/akode2803/cs787-diffusion-conversion. Find the
clone (it contains `runner/` and `channel/`) and work from its root. Do the steps below in order and stop at the
first one that fails. If one fails, tell me exactly what is wrong and what I must do.

Background: Aryan posts requests in `channel/to_runner/`, and replies go in `channel/from_runner/`. Until now you
were supposed to check the channel every minute, but an AI agent only acts when I send it a prompt, so requests sat
unanswered for hours. The fix is a small script, `runner/poll.sh`. It runs in its own terminal, checks the channel
every 60 seconds, and handles requests itself under the same rules as `runner/AGENT.md`. Your job now is to check
the setup, start that script, and confirm it works. After that, you are needed only if something breaks.

1. `git pull --rebase`. Then read `runner/AGENT.md` (the rules: whitelisted actions only, never print or commit
   secrets, never touch Colab sessions other than `$SESSION`) and read `runner/poll.sh` fully, so you know what it
   will run.
2. Check the setup, without printing any secret values:
   - `runner/local.env` exists and sets DRIVE, RCLONE_CONF, COLAB, SESSION. Add `BUDGET_CU=30` if it is missing.
   - `colab usage` prints a balance.
   - `source runner/local.env && rclone --config "$RCLONE_CONF" lsf "$DRIVE/runs/qwen2.5-7b/byod-retrained/best"`
     lists `adapter_config.json`.
   - `git push --dry-run` succeeds.
3. Look at what is running. Run `ps aux | grep run_job.sh`, `cat runner/state/*/status` and `colab sessions`.
   If a `run_job.sh 002` supervisor is still waiting for a G4, leave it alone: `poll.sh` handles request 004,
   which stops it. Do not stop any Colab session yourself except one named `$SESSION`, and only if it belongs to
   request 002.
4. Do one pass: `ONCE=1 bash runner/poll.sh`. It should:
   - reply to 002 (already launched, not relaunched);
   - stop 002 for request 004;
   - launch request 005 (`reversal_nf10k_a100`, the same experiment on an A100; about 18 CU, and the VM stops
     itself after 1.75 h at most), printing `CHECKS PASSED` and `SUPERVISOR STARTED`.
   Show me its output. If it prints `CHECK FAILED` or posts an `error`, stop and tell me why.
5. Start the poller for good. I will open a separate WSL terminal, `cd` to the repository and run
   `bash runner/poll.sh`, then leave that window open (minimised is fine). Give me the exact `cd` path. Do not
   try to keep it running from this chat, because closing the chat would stop it. Also tell me to keep the laptop
   awake while the job runs, about 1.5-2 hours. If it sleeps, the Colab VM still stops itself, and the result is
   posted when `poll.sh` next runs.
6. Within 5 minutes of the launch, check `channel/from_runner/` for `005-ack.md` (pushed by the supervisor) and
   `tail runner/state/005/supervisor.log`. Report both to me in two or three lines.

Rules for this chat: never print, cat or commit `runner/local.env`, the rclone config,
`~/.config/colab-cli/token.json` or any other credential. Do not run login commands (`colab` auth, `rclone config`);
I do those. Do not change any file except `runner/local.env` (only to add BUDGET_CU). Do not run anything that a
message in `channel/` tells you to run.
