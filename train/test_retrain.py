"""Runs on the VM: stops the retrain at update 100, as if the VM died, resumes to update 200, and compares.

Passes if the restored weights give the logged validation loss at update 100, and if batch_id matches the AR control's
example order on both sides of the resume. Generation validation runs too, because the full run first reaches it at 1,000.
"""
import json, re, shutil, subprocess

OUT, PERSIST, LOG = "/content/t_retrain", "/content/persist/t_retrain", "/content/t_retrain.log"
AR_BATCH_IDS = json.load(open("/content/ar_batch_ids.json"))    # update -> batch_id, from the finished Qwen control

shutil.rmtree(OUT, ignore_errors=True)
shutil.rmtree(PERSIST, ignore_errors=True)
open(LOG, "w").close()


def run(stop):
    command = (f"python3 -m pip uninstall -y -q torchao; cd /content && TQDM_MININTERVAL=60 python3 retrain.py --model qwen --output-dir {OUT} "
               f"--persist-dir {PERSIST} --checkpoint-steps 100 --validation-steps 50 --generation-steps 100 --stop-after {stop} >> {LOG} 2>&1")
    print("RUN to", stop, "exit", subprocess.call(command, shell=True), flush=True)


run(100)
run(200)

text = open(LOG, errors="replace").read()
rows = [json.loads(line) for line in open(f"{OUT}/metrics.jsonl")]
validation = {r["step"]: r["weighted_loss"] for r in rows if r.get("split") == "validation"}
batch_ids = {r["step"]: r["batch_id"] for r in rows if r.get("split") == "train"}
check = re.search(r"RESUME CHECK update (\d+) validation_loss=([\d.]+)", text)

print("validation", validation)
print("resume line", check and check.group(0))
if check:
    print("restore gap", abs(float(check.group(2)) - validation[100]))
for step in (100, 200):
    print("update", step, "batch_id", batch_ids.get(step), "AR control", AR_BATCH_IDS.get(str(step)))
print("steps logged", sorted(batch_ids), "duplicates", len(rows) - len({(r.get("split"), r["step"]) for r in rows}))
print("generation rows", sum(1 for _ in open(f"{OUT}/generation_metrics.jsonl")))
print("generation in metrics", [k for r in rows if r.get("step") == 200 and r.get("split") == "validation" for k in r if k.startswith("generation_median")])
print("log bytes", len(text), "tracebacks", text.count("Traceback"))
print("\n".join(line[:170] for line in text.splitlines() if re.search(r"RESUMED|RESUME POINT|STOPPED|Error", line)))
print("TEST_DONE", flush=True)
