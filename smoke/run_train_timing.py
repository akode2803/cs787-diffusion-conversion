import subprocess, time
t = time.time()
p = subprocess.Popen("cd /content/BYOD && python3 train.py --model qwen --mode paper --max-updates 300 --logging-steps 50 --output-dir /content/timing_qwen 2>&1 | grep -vE 'oneDNN|absl|cpu_feature|AVX|cudart_stub' | tail -30", shell=True, stdout=subprocess.PIPE, text=True)
print(p.communicate()[0]); print("WALL", round(time.time() - t), flush=True)
