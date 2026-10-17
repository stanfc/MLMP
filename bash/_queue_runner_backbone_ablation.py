#!/usr/bin/env python3
"""Queue runner for the 15-backbone CAGA(hgate) ablation (bash/backbone_ablation/*.sh, ACDC, 150R
each). One launch per poll pass, only onto a GPU with >= THRESHOLD_MB free and < MAX_PER_GPU of our
own jobs alive. GPU 3 excluded (other user)."""
import subprocess, time, os

THRESHOLD_MB = 8000
POLL_SECONDS = 15
MAX_PER_GPU = 6
GPUS = [0, 1, 2, 4, 5]
JOBS = [
    # naclip_B16, naclip_B32, naclip_L14_336, clearclip_kk_L14 already launched -- not re-listed
    "clearclip_qq_L14", "sclip_reduced_L14", "naclip_gaussonly_L14",
    "vv_attention_L14", "maskclip_L14", "itaclip_L14",
    "clearclip_kk_B16", "clearclip_qq_B16", "sclip_reduced_B16", "sclip_uaml1_B16", "maskclip_B16",
]
LOG_DIR = "/tmp/tta_backbone_ablation_logs"
os.makedirs(LOG_DIR, exist_ok=True)
CONDA_BIN = "/home/stanfc/miniconda3/envs/mlmp/bin"
alive = {}

def free_mem_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    return {i: int(l.split(",")[1]) - int(l.split(",")[0]) for i, l in enumerate(out)}

def launch(job, gpu):
    env = os.environ.copy(); env["GPU_ID"] = str(gpu); env["PATH"] = CONDA_BIN + os.pathsep + env["PATH"]
    with open(f"{LOG_DIR}/{job}.log", "w") as f:
        p = subprocess.Popen(["bash", f"bash/backbone_ablation/adagate_hgate_backbone_{job}.sh"], stdout=f, stderr=subprocess.STDOUT, env=env)
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
