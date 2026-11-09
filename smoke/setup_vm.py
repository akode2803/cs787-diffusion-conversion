"""Clones BYOD on the Colab VM and installs it into a venv that reuses Colab's torch."""
import subprocess, sys
def sh(c):
    print("+", c, flush=True)
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-1500:], flush=True)
    if r.returncode: sys.exit(r.returncode)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh("rm -rf /content/BYOD && git clone https://github.com/RuurdKuiper/BYOD /content/BYOD")
sh("cd /content/BYOD && git checkout -q 211d054369136ed07d8a44247e754128c8dfe1ce && git rev-parse HEAD")
sh(f"{sys.executable} -m pip install -q -e '/content/BYOD[evaluation]'")
sh(f"{sys.executable} -c \"import torch,transformers,peft;print(torch.__version__,transformers.__version__,peft.__version__,torch.cuda.is_available())\"")
