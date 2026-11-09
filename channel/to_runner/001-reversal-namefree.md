id: 001
action: run_job
job: reversal_nf10k
---
Reversal test, run 3. Runs 1–2 put the person's name in every training prompt, so the diffusion conversion could
copy the name instead of learning description → name. This run uses name-free prompts ("State a fact." →
"<name> is the <description>."), with the same 1,000 facts, the same 10,000 updates and the same learning rate for
both arms. Estimated 12 CU, capped at 15. Results go to the shared Drive folder under eval/reversal/ and
eval/reversal_nf10k/. The RESULT lines in the reply are all we need here. Thank you!
