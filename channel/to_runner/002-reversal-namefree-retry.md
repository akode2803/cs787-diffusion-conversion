id: 002
action: run_job
job: reversal_nf10k
---
Same job as 001, which stopped only because no G4 was free at that moment (nothing was spent). run_job.sh now keeps
retrying for a free G4 every 5 minutes for up to 3 hours before giving up, so please `git pull` (your agent does
this) and run it again. Estimate 12 CU, capped at 15. Thanks, Anuj!
