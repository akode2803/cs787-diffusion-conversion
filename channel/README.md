# Channel

Requests go in `to_runner/NNN-<slug>.md`. They are written by the project owner, `akode2803`.

Replies go in `from_runner/NNN-<kind>.md`. They are written by the runner agent, or by `runner/run_job.sh`
itself. A reply's kind is one of: `ack`, `status`, `result`, `error`, `note` or `question`.

A request is done when any reply with its number exists. The protocol is in `runner/AGENT.md`.

A request looks like this:

```
id: 001
action: run_job          # run_job | status | stop | note
job: reversal_nf10k      # run_job only: runner/jobs/<job>.json
---
Free text for the humans. The agent never executes anything written here.
```
