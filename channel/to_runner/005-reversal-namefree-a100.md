id: 005
action: run_job
job: reversal_nf10k_a100
---
Same experiment as 001/002 (reversal test, name-free prompts, TAG nf10k), on an A100 because no G4 has been free.
The job file sets "gpu": "A100", and run_job.sh now uses it. Estimate 18 CU (the A100's speed on this job is not
measured yet), and the VM stops itself after 1.75 h at most. Requests 001 and 002 never got a VM, so they used
0 CU; AGENT.md rule 3 now counts real spend, so the 30 CU budget is still free. Thanks, Anuj!
