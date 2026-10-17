#!/usr/bin/env python3
"""Memory-aware queue runner for COCO-Object sub100-15corr, matching the
COCO-Stuff/P59/P60 comparison method set. CoTTA excluded (stopped everywhere
else this session as too slow). One launch per ~45s poll pass, only onto a
GPU with enough free memory.
"""
import subprocess
import time
import os

THRESHOLD_MB = 15000
POLL_SECONDS = 45

METHODS = [
    "adagate", "no_adapt", "daf_mlmp_continual", "tent_continual", "sar_continual",
    "spa_continual", "tribe_continual", "deyo_continual", "vida_continual",
    "ptta_continual", "come_continual", "mgp_deyo_mlmp_continual", "mlmp_episodic",
]

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


def launch(method, gpu):
    script = f"bash/coco_obj/{method}_15corr_sub100.sh"
    log = f"{LOG_DIR}/coco_obj_{method}_15corr_sub100.log"
    env = os.environ.copy()
    env["GPU_ID"] = str(gpu)
    env["PATH"] = CONDA_BIN + os.pathsep + env["PATH"]
    with open(log, "w") as f:
        p = subprocess.Popen(["bash", script], stdout=f, stderr=subprocess.STDOUT, env=env)
    print(f"[{time.strftime('%H:%M:%S')}] launched coco_obj/{method} on GPU{gpu} pid {p.pid}", flush=True)


def main():
    pending = list(METHODS)
    while pending:
        free = free_mem_mb()
        best_gpu = max(free, key=free.get)
        if free[best_gpu] >= THRESHOLD_MB:
            method = pending.pop(0)
            launch(method, best_gpu)
        else:
            print(f"[{time.strftime('%H:%M:%S')}] no GPU with >{THRESHOLD_MB}MB free "
                  f"(best GPU{best_gpu}: {free[best_gpu]}MB), {len(pending)} jobs left, waiting...", flush=True)
        time.sleep(POLL_SECONDS)
    print("all jobs launched", flush=True)


if __name__ == "__main__":
    main()
