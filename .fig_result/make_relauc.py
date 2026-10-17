"""RelAUC(method, dataset) and AvgRelAUC(method) per the user's exact formula.
RelAUC = (1/150) * sum_r (mIoU_r - NoAdapt)/NoAdapt * 100
AvgRelAUC = macro-average of RelAUC over the 8 datasets.
MLMP-episodic excluded (no trajectory). Run from MLMP/."""
import os, re, csv
os.chdir("/home/stanfc/TTA-on-OVSS/MLMP")

def load(path):
    R, M = [], []
    for l in open(path, errors='ignore'):
        if re.match(r'\s*Round\s+\d+', l):
            p = l.strip().split(',')
            try: R.append(int(p[0].split()[1])); M.append(float(p[-1]))
            except (ValueError, IndexError): pass
    return R, M

def std(sfx):
    return {"DAF": f"daf_mlmp_continual{sfx}", "TENT": f"tent_continual{sfx}", "SAR": f"sar_continual{sfx}", "SPA": f"spa_continual{sfx}",
            "TRIBE": f"tribe_continual{sfx}", "DeYO": f"deyo_continual{sfx}", "ViDA": f"vida_continual{sfx}", "PTTA": f"ptta_continual{sfx}",
            "COME": f"come_continual{sfx}", "MGP": f"mgp_deyo_mlmp_continual{sfx}"}

D = {}
D["ACDC"] = ("ACDCDataset", {"CAGA": "adagate_hgate", "DAF": "daf_mlmp_continual", "TENT": "naclip_tent_continual", "SAR": "sar_continual", "SPA": "spa_continual",
    "TRIBE": "tribe_continual", "DeYO": "deyo_continual", "ViDA": "vida_continual", "PTTA": "ptta_continual", "COME": "come_continual", "MGP": "mgp_deyo_mlmp_continual"}, "naclip_no_adapt")
for name, folder in [("Cityscapes", "CityscapesDataset"), ("V20", "PascalVOC20Dataset")]:
    D[name] = (folder, {"CAGA": "adagate_hgate_sub100_15corr", **std("_sub100_15corr")}, "no_adapt_sub100_15corr")
for name, folder in [("V21", "PascalVOC21Dataset"), ("P59", "PascalContext59Dataset"), ("P60", "PascalContext60Dataset"), ("COCO-Stuff", "COCOStuffDataset"), ("COCO-Object", "COCOObjectDataset")]:
    D[name] = (folder, {"CAGA": "adagate_hgate_15corr_sub100", **std("_15corr_sub100")}, "no_adapt_15corr_sub100")

METHODS = ["CAGA", "DAF", "TENT", "SAR", "SPA", "TRIBE", "DeYO", "ViDA", "PTTA", "COME", "MGP"]
DATASETS = list(D.keys())

relauc = {}          # (ds, method) -> value
sanity = []           # (ds, method, n_rounds, ok)
noadapt = {}

for ds, (folder, methods, na) in D.items():
    _, M_na = load(f"save/{folder}/{na}/results_all_rounds.txt")
    na_val = M_na[-1]
    noadapt[ds] = na_val
    for m, sub in methods.items():
        p = f"save/{folder}/{sub}/results_all_rounds.txt"
        R, M = load(p) if os.path.exists(p) else ([], [])
        n = len(R)
        ok = (n == 150) and (R == list(range(1, 151)))
        sanity.append((ds, m, n, ok))
        if ok:
            vals = [(mi - na_val) / na_val * 100 for mi in M]
            relauc[(ds, m)] = sum(vals) / 150
        else:
            relauc[(ds, m)] = None

# ---- print sanity check ----
print("=== Sanity check: round count == 150 for every run ===")
bad = [s for s in sanity if not s[3]]
if not bad:
    print(f"OK: all {len(sanity)} runs have exactly rounds 1..150.")
else:
    for ds, m, n, ok in bad:
        print(f"  MISMATCH: {ds:12s} {m:8s} n_rounds={n}")
    print(f"{len(bad)}/{len(sanity)} runs failed the check (excluded from RelAUC/AvgRelAUC below).")

# ---- RelAUC table ----
print("\n=== RelAUC(method, dataset), % ===")
header = "Method".ljust(8) + "".join(d.rjust(13) for d in DATASETS)
print(header)
for m in METHODS:
    row = m.ljust(8)
    for d in DATASETS:
        v = relauc.get((d, m))
        row += (f"{v:12.2f}" if v is not None else "n/a".rjust(12)) + " "
    print(row)

# ---- AvgRelAUC ----
print("\n=== AvgRelAUC(method) = macro-average over 8 datasets ===")
avg = {}
for m in METHODS:
    vs = [relauc[(d, m)] for d in DATASETS if relauc.get((d, m)) is not None]
    avg[m] = sum(vs) / len(vs) if len(vs) == len(DATASETS) else None
    flag = "" if len(vs) == len(DATASETS) else f"  (only {len(vs)}/8 datasets available!)"
    print(f"  {m:8s} {avg[m]:8.2f}{flag}" if avg[m] is not None else f"  {m:8s}     n/a{flag}")

print("\nRanked:")
for m, v in sorted(avg.items(), key=lambda kv: -kv[1] if kv[1] is not None else 999):
    print(f"  {m:8s} {v:8.2f}")

# ---- write CSV ----
with open(".fig_result/relauc_table.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method"] + DATASETS + ["AvgRelAUC"])
    for m in METHODS:
        w.writerow([m] + [f"{relauc[(d, m)]:.4f}" if relauc.get((d, m)) is not None else "" for d in DATASETS] + [f"{avg[m]:.4f}" if avg[m] is not None else ""])
print("\nwritten .fig_result/relauc_table.csv")
