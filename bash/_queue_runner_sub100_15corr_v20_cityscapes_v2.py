#!/usr/bin/env python3
"""Re-queue the V20/Cityscapes sub100-15corr jobs that were stopped earlier.
Skips methods already complete (no_adapt, mlmp_episodic on V20). Methods with
an existing ckpt.pt (adagate/daf/tent/sar/spa/tribe/deyo/vida/ptta/come) auto-
resume from their checkpoint when relaunched on the same save_dir; the rest
(mgp_deyo_mlmp_continual everywhere, spa/tribe/deyo/vida/ptta/come/mlmp_episodic
on Cityscapes) start fresh since they never got a checkpoint or never started.
One launch per ~45s poll pass, only onto a GPU with enough free memory.
"""
import subprocess
import time
import os

THRESHOLD_MB = 15000
POLL_SECONDS = 45

JOBS = [
    ("v20", "adagate"), ("v20", "daf_mlmp_continual"), ("v20", "tent_continual"),
    ("v20", "sar_continual"), ("v20", "spa_continual"), ("v20", "tribe_continual"),
    ("v20", "deyo_continual"), ("v20", "vida_continual"), ("v20", "ptta_continual"),
    ("v20", "come_continual"), ("v20", "mgp_deyo_mlmp_continual"),
    ("cityscapes", "adagate"), ("cityscapes", "daf_mlmp_continual"),
    ("cityscapes", "tent_continual"), ("cityscapes", "sar_continual"),
    ("cityscapes", "spa_continual"), ("cityscapes", "tribe_continual"),
    ("cityscapes", "deyo_continual"), ("cityscapes", "vida_continual"),
    ("cityscapes", "ptta_continual"), ("cityscapes", "come_continual"),
    ("cityscapes", "mgp_deyo_mlmp_continual"), ("cityscapes", "mlmp_episodic"),
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


def launch(folder, method, gpu):
    script = f"bash/{folder}/{method}_sub100_15corr.sh"
    log = f"{LOG_DIR}/{folder}_{method}_sub100_15corr_RESUME.log"
    env = os.environ.copy()
    env["GPU_ID"] = str(gpu)
    env["PATH"] = CONDA_BIN + os.pathsep + env["PATH"]
    with open(log, "w") as f:
        p = subprocess.Popen(["bash", script], stdout=f, stderr=subprocess.STDOUT, env=env)
    print(f"[{time.strftime('%H:%M:%S')}] launched {folder}/{method} on GPU{gpu} pid {p.pid}", flush=True)


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
                  f"(best GPU{best_gpu}: {free[best_gpu]}MB), {len(pending)} jobs left, waiting...", flush=True)
        time.sleep(POLL_SECONDS)
    print("all jobs launched", flush=True)


if __name__ == "__main__":
    main()
