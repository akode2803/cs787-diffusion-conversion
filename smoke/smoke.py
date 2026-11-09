"""Feasibility check only: load the released BYOD-Qwen adapter, generate, time it, then time the AR parent on the same prompts."""
import json, time, torch
from diffusion_lm.inference import load_hub_adapter_session, denoise
s = load_hub_adapter_session("Ruurd/BYOD-Qwen2.5-7B")
print("LOADED", type(s.model).__name__, "mem GB", round(torch.cuda.max_memory_allocated()/1e9, 1), flush=True)
qs = ["What is the capital of France, and why is it famous?",
      "Natalia sold clips to 48 friends in April, and half as many in May. How many clips did she sell altogether?",
      "Write a Python function that returns the n-th Fibonacci number."]
out = []
for steps in (32, 64, 128):
    for q in qs:
        torch.cuda.synchronize(); t = time.time()
        text, status = denoise(s, q, "", 128, steps, 1.0, 0.0 if False else 0.7, 3, 1234, permanent_unmask=True, confidence_guided=True, proportional_unmask=False, confidence_eos_eot_inf=True, block_length=128)
        torch.cuda.synchronize(); dt = time.time() - t
        out.append({"steps": steps, "q": q, "s": round(dt, 2), "text": text})
        print(f"STEPS {steps} {dt:.2f}s :: {text[:220]!r}", flush=True)
json.dump(out, open("/content/smoke.json", "w"), indent=1)
print("DONE", flush=True)
