#!/usr/bin/env python3
"""Memory-aware queue runner for the 26 new V20/Cityscapes sub100-15corr jobs
(13 methods x 2 datasets), matching the method set on the two full-scale
progress figures. One launch per ~45s poll pass, only onto a GPU with enough
free memory, to avoid CUDA-startup memory-accounting races when many jobs
start at once.
"""
import subprocess
import time
import os

THRESHOLD_MB = 15000
POLL_SECONDS = 45
GPUS = [0, 1, 2, 3, 4, 5]

METHODS = [
    "adagate", "no_adapt", "daf_mlmp_continual", "tent_continual", "sar_continual",
    "spa_continual", "tribe_continual", "deyo_continual", "vida_continual",
    "ptta_continual", "come_continual", "mgp_deyo_mlmp_continual", "mlmp_episodic",
]

JOBS = []
for folder in ("v20", "cityscapes"):
    for m in METHODS:
        JOBS.append((folder, m))

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


def launch(folder, method, gpu):
    script = f"bash/{folder}/{method}_sub100_15corr.sh"
    log = f"{LOG_DIR}/{folder}_{method}_sub100_15corr.log"
    env = os.environ.copy()
    env["GPU_ID"] = str(gpu)
    env["PATH"] = CONDA_BIN + os.pathsep + env["PATH"]
    with open(log, "w") as f:
        p = subprocess.Popen(["bash", script], stdout=f, stderr=subprocess.STDOUT, env=env)
    print(f"[{time.strftime('%H:%M:%S')}] launched {folder}/{method} on GPU{gpu} pid {p.pid}")


def main():
    pending = list(JOBS)
    while pending:
        free = free_mem_mb()
        best_gpu = max(free, key=free.get)
        if free[best_gpu] >= THRESHOLD_MB:
            folder, method = pending.pop(0)
            launch(folder, method, best_gpu)
        else:
            print(f"[{time.strftime('%H:%M:%S')}] no GPU with >{THRESHOLD_MB}MB free "
                  f"(best GPU{best_gpu}: {free[best_gpu]}MB), {len(pending)} jobs left, waiting...")
        time.sleep(POLL_SECONDS)
    print("all jobs launched")


if __name__ == "__main__":
    main()
