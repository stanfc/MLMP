#!/usr/bin/env python3
"""Wait for enough free GPU memory, then launch the Cityscapes MGP sub100-15corr
job that OOM'd earlier. Uses a higher threshold (20GB) than the general sub100
queue since this dataset's images are larger (1120x560) than sub100 VOC/COCO
crops, and the previous attempt OOM'd on a GPU that looked free but filled up
from other concurrent launches.
"""
import subprocess
import time
import os

THRESHOLD_MB = 20000
POLL_SECONDS = 45

LOG_DIR = "/tmp/tta_full_runs_logs"
os.makedirs(LOG_DIR, exist_ok=True)
CONDA_BIN = "/home/stanfc/miniconda3/envs/mlmp/bin"


def free_mem_mb():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.strip().splitlines()
    free = {}
    for i, line in enumerate(out):
        used, total = [int(x) for x in line.split(",")]
        free[i] = total - used
    return free


def main():
    while True:
        free = free_mem_mb()
        best_gpu = max(free, key=free.get)
        if free[best_gpu] >= THRESHOLD_MB:
            env = os.environ.copy()
            env["GPU_ID"] = str(best_gpu)
            env["PATH"] = CONDA_BIN + os.pathsep + env["PATH"]
            log = f"{LOG_DIR}/cityscapes_mgp_deyo_mlmp_continual_sub100_15corr_RETRY2.log"
            with open(log, "w") as f:
                p = subprocess.Popen(["bash", "bash/cityscapes/mgp_deyo_mlmp_continual_sub100_15corr.sh"],
                                      stdout=f, stderr=subprocess.STDOUT, env=env)
            print(f"[{time.strftime('%H:%M:%S')}] launched cityscapes/mgp_deyo_mlmp_continual "
                  f"on GPU{best_gpu} (free {free[best_gpu]}MB) pid {p.pid}", flush=True)
            break
        else:
            print(f"[{time.strftime('%H:%M:%S')}] no GPU with >{THRESHOLD_MB}MB free "
                  f"(best GPU{best_gpu}: {free[best_gpu]}MB), waiting...", flush=True)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
