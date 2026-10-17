"""R150 / Peak / Peak-round per (dataset, method), plus No-Adapt per dataset. Run from anywhere."""
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
def episodic_mean(path):
    v = []
    for l in list(open(path, errors='ignore'))[1:]:
        if not l.strip() or 'GPU' in l or 'Duration' in l or 'Mean' in l: continue
        try: v.append(float(l.split(',')[1].split('+/-')[0]))
        except (ValueError, IndexError): pass
    return sum(v) / len(v) if v else None
def std(sfx):
    return {"DAF": f"daf_mlmp_continual{sfx}", "TENT": f"tent_continual{sfx}", "SAR": f"sar_continual{sfx}", "SPA": f"spa_continual{sfx}",
            "TRIBE": f"tribe_continual{sfx}", "DeYO": f"deyo_continual{sfx}", "ViDA": f"vida_continual{sfx}", "PTTA": f"ptta_continual{sfx}",
            "COME": f"come_continual{sfx}", "MGP": f"mgp_deyo_mlmp_continual{sfx}"}
D = {}
D["ACDC"] = ("ACDCDataset", {"CAGA": "adagate_hgate", "DAF": "daf_mlmp_continual", "TENT": "naclip_tent_continual", "SAR": "sar_continual", "SPA": "spa_continual",
    "TRIBE": "tribe_continual", "DeYO": "deyo_continual", "ViDA": "vida_continual", "PTTA": "ptta_continual", "COME": "come_continual", "MGP": "mgp_deyo_mlmp_continual",
    "[abl] CAGA-gradgate": "adagate_ABmad05_ecdf_grow_scaled", "[abl] CAGA-scale6": "adagate_ABmad05_ecdf"}, "naclip_no_adapt", "naclip_mlmp_episodic")
for name, folder in [("Cityscapes", "CityscapesDataset"), ("V20", "PascalVOC20Dataset")]:
    D[name] = (folder, {"CAGA": "adagate_hgate_sub100_15corr", **std("_sub100_15corr"), "[abl] CAGA-gradgate": "adagate_sub100_15corr", "[abl] CAGA-scale6": "adagate_scale6_sub100_15corr"},
               "no_adapt_sub100_15corr", "mlmp_episodic_sub100_15corr")
for name, folder in [("V21", "PascalVOC21Dataset"), ("P59", "PascalContext59Dataset"), ("P60", "PascalContext60Dataset"), ("COCO-Stuff", "COCOStuffDataset"), ("COCO-Object", "COCOObjectDataset")]:
    D[name] = (folder, {"CAGA": "adagate_hgate_15corr_sub100", **std("_15corr_sub100"), "[abl] CAGA-gradgate": "adagate_15corr_sub100", "[abl] CAGA-scale6": "adagate_scale6_15corr_sub100"},
               "no_adapt_15corr_sub100", "mlmp_episodic_15corr_sub100")
rows, noadapt, episodic = [], {}, {}
for ds, (folder, methods, na, ep) in D.items():
    R, M = load(f"save/{folder}/{na}/results_all_rounds.txt"); noadapt[ds] = M[-1]
    episodic[ds] = episodic_mean(f"save/{folder}/{ep}/results.txt")
    for m, sub in methods.items():
        p = f"save/{folder}/{sub}/results_all_rounds.txt"
        if not os.path.exists(p): rows.append([ds, m, "", "", "", "", "", "missing"]); continue
        R, M = load(p)
        if not R: rows.append([ds, m, 0, "", "", "", "", "empty"]); continue
        i = max(range(len(M)), key=lambda k: M[k])
        r150 = M[R.index(150)] if 150 in R else ""
        rows.append([ds, m, R[-1], f"{r150:.2f}" if r150 != "" else "", f"{M[-1]:.2f}", f"{M[i]:.2f}", R[i], "complete" if R[-1] >= 150 else f"INCOMPLETE (R{R[-1]})"])
with open(".fig_result/results_table.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["dataset", "method", "last_round", "R150_mIoU", "last_round_mIoU", "peak_mIoU", "peak_round", "status"]); w.writerows(rows)
    for ds in D: w.writerow([ds, "No-Adapt", 1, "", f"{noadapt[ds]:.2f}", "", "", "reference"]); w.writerow([ds, "MLMP-episodic", "", "", f"{episodic[ds]:.2f}", "", "", "episodic mean over conditions, no trajectory"])
print("No-Adapt:", {k: round(v, 2) for k, v in noadapt.items()}); print("MLMP-ep:", {k: round(v, 2) for k, v in episodic.items()})
for r in rows: print(r)
