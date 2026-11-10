# CS787 — trained adapters

LoRA adapters from "What Does Diffusion Conversion Buy? A same-data controlled study of LoRA conversion from
autoregressive to diffusion language models" (CS787 Generative AI, IIT Kanpur, November 2026).
Code: https://github.com/akode2803/cs787-diffusion-conversion

All adapters use the same setup:
- **LoRA:** rank 1024, alpha 1024, on the query and value projections, with the base model frozen.
- **Training:** 25,000 updates at batch 16 on `Ruurd/LAD-training-1m-256` (the same data, order and budget as BYOD's
  paper recipe), learning rate 1e-5 with a cosine schedule, seed 42.
- **Selection:** each folder holds the checkpoint with the best validation loss.

| Folder | Base model (revision) | What it is |
|---|---|---|
| `qwen2.5-7b-byod-retrained/` | Qwen/Qwen2.5-7B-Instruct (`a09a354`) | Our retrain of BYOD's diffusion conversion: bidirectional attention, masked-diffusion loss |
| `qwen2.5-7b-ar-control/` | Qwen/Qwen2.5-7B-Instruct (`a09a354`) | The same-data AR control: causal attention, next-token loss on the same answer tokens |
| `ministral-8b-ar-control/` | mistralai/Ministral-8B-Instruct-2410 (`2f494a1`) | The same-data AR control for Ministral. Its diffusion pair is BYOD's released adapter |
| `reversal/qwen-ar-control+facts10k/` | Qwen2.5-7B + the AR control above | The AR control after 10,000 updates on 1,000 invented facts (reversal test, run 2) |
| `reversal/qwen-byod-retrained+facts10k/` | Qwen2.5-7B + the conversion above | The conversion after the same fact training (reversal test, run 2) |

Released BYOD conversions, which we compare against: `Ruurd/BYOD-Qwen2.5-7B` and the other BYOD adapters on
Hugging Face. BYOD's code is at https://github.com/RuurdKuiper/BYOD; we used commit 211d054.

## Using them

**AR controls** are ordinary PEFT LoRA adapters:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", revision="a09a35458c702b33eeacc393d103063234e8bc28", torch_dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(base, "qwen2.5-7b-ar-control")
tok = AutoTokenizer.from_pretrained("qwen2.5-7b-ar-control")
```

**The diffusion conversion** needs more than the adapter: bidirectional attention, BYOD's normalization state and a
diffusion sampler. Install BYOD at commit 211d054 and load the folder with BYOD's loader. `eval/arms.py` in our
repository does exactly this (arm spec `byod:qwen:<folder>`) and writes the small `resolved_config.json` BYOD
expects. `eval/sampling.py` has the low-confidence-remasking sampler we used.

## Licences

The adapters are derived from their base models and inherit their licences. Qwen2.5-7B-Instruct is under Apache 2.0.
Ministral-8B-Instruct-2410 is under the Mistral AI Research License (research use only). The training data
inherits the licences of its sources. These are research artefacts, not tested for safety.
