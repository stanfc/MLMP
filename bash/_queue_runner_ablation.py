#!/usr/bin/env python3
"""Queue runner for the ablation study (bash/ablation/*.sh). One launch per poll pass, only onto a
GPU with >= THRESHOLD_MB free AND fewer than MAX_PER_GPU of *our own* jobs alive. GPU 3 excluded
(other user). Jobs run in priority order."""
import subprocess, time, os

THRESHOLD_MB = 15000
POLL_SECONDS = 45
MAX_PER_GPU = 3
GPUS = [0, 1, 2, 4, 5]
JOBS = [  # priority order: shared controls + V20 analyses first, then ACDC sweeps
    "cityscapes_hgate_5corr_sub100",
    "v20_hgate_5corr_sub100_oodprobe", "v20_hgate_5corr_sub100_resample", "v20_hgate_5corr_sub100_phaseswitch",
    "acdc_abl_baserst_0", "acdc_abl_baserst_0.02", "acdc_abl_baserst_0.005",
    "acdc_abl_trendthr_0.25", "acdc_abl_trendthr_1.0",
    "acdc_abl_anchorh_0.9", "acdc_abl_anchorh_0.98",
    "acdc_abl_monint_25", "acdc_abl_monint_100",
    "acdc_abl_lr_2.5e-6", "acdc_abl_lr_1e-5",
]
LOG_DIR = "/tmp/tta_ablation_logs"
os.makedirs(LOG_DIR, exist_ok=True)
CONDA_BIN = "/home/stanfc/miniconda3/envs/mlmp/bin"
alive = {}   # pid-handle -> gpu

def free_mem_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    return {i: int(l.split(",")[1]) - int(l.split(",")[0]) for i, l in enumerate(out)}

def launch(job, gpu):
    env = os.environ.copy(); env["GPU_ID"] = str(gpu); env["PATH"] = CONDA_BIN + os.pathsep + env["PATH"]
    with open(f"{LOG_DIR}/{job}.log", "w") as f:
        p = subprocess.Popen(["bash", f"bash/ablation/{job}.sh"], stdout=f, stderr=subprocess.STDOUT, env=env)
    alive[p] = gpu
    print(f"[{time.strftime('%H:%M:%S')}] launched {job} on GPU{gpu} pid {p.pid}", flush=True)

def main():
    pending = list(JOBS)
    while pending:
        for p in [p for p in alive if p.poll() is not None]: del alive[p]
        free = free_mem_mb()
        cand = [g for g in GPUS if free.get(g, 0) >= THRESHOLD_MB and sum(1 for x in alive.values() if x == g) < MAX_PER_GPU]
        if cand:
            g = max(cand, key=lambda g: free[g]); launch(pending.pop(0), g)
        else:
            print(f"[{time.strftime('%H:%M:%S')}] waiting ({len(pending)} left; alive={len(alive)}; free={ {g: free.get(g) for g in GPUS} })", flush=True)
        time.sleep(POLL_SECONDS)
    print("all jobs launched", flush=True)
    while alive:
        for p in [p for p in alive if p.poll() is not None]: del alive[p]
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
