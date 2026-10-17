"""Render .fig_result/results_table.csv into .fig_result/results_table.md (run make_results_table.py first)."""
import csv, collections, datetime
D = "/home/stanfc/TTA-on-OVSS/MLMP/.fig_result"
rows = list(csv.DictReader(open(f"{D}/results_table.csv")))
by = collections.defaultdict(dict)
for r in rows: by[r["dataset"]][r["method"]] = r
MAIN = ["ACDC", "Cityscapes", "V20"]; APP = ["V21", "P59", "P60", "COCO-Stuff", "COCO-Object"]
METH = ["CAGA", "DAF", "TENT", "SAR", "SPA", "TRIBE", "DeYO", "ViDA", "PTTA", "COME", "MGP"]
def cell(r):
    if r is None or r["status"] == "missing": return "n/a"
    s = f"{r['R150_mIoU'] or r['last_round_mIoU']} / {r['peak_mIoU']} ({r['peak_round']})"
    return s if r["status"] == "complete" else s + f" [{r['status']}]"
def table(dsets, methods, bold_caga=True):
    out = ["| Method | " + " | ".join(dsets) + " |", "|---|" + "---|" * len(dsets)]
    out.append("| **No-Adapt** | " + " | ".join(by[d]["No-Adapt"]["last_round_mIoU"] for d in dsets) + " |")
    out.append("| MLMP-episodic | " + " | ".join(by[d]["MLMP-episodic"]["last_round_mIoU"] for d in dsets) + " |")
    for m in methods:
        name = f"**{m}**" if m == "CAGA" and bold_caga else m
        out.append(f"| {name} | " + " | ".join(cell(by[d].get(m)) for d in dsets) + " |")
    return "\n".join(out)
md = f"""# CAGA benchmark results: R150 / Peak mIoU per dataset x method

Generated {datetime.date.today()} from `results_all_rounds.txt` of each run (script: `.fig_result/make_results_table.py`, full CSV: `.fig_result/results_table.csv`).
Every cell is **R150 mIoU / Peak mIoU (round of the peak)**. All runs are single-seed (seed 0), 150 rounds, all complete at generation time.

## Definitions
- **R150**: Mean mIoU (average over the domains of that round) at the end of round 150.
- **Peak**: highest per-round Mean mIoU over the whole 150-round trajectory; the number in brackets is the round where it occurs.
- **No-Adapt**: source model, one number per dataset (no adaptation).
- **MLMP-episodic**: episodic MLMP (reset every sample), mean over conditions; it has no trajectory, so there is no peak. Run with steps=1, lr 1e-3 (the MLMP repo default is steps=10).
- **CAGA** = `deyo_mlmp_adagate_hgate_continual` (locked config, anchor_h_min_ratio 0.95).
- Protocol: NA-CLIP ViT-L/14, LayerNorm-only, batch 1, evaluate-before-adapt, no reset. ACDC = full val split (406 imgs/round, fog -> night -> rain -> snow). Cityscapes / VOC20 / V21 / P59 / P60 / COCO-Stuff / COCO-Object = 100-image subset x 15 ImageNet-C corruptions (severity 5), 1500 imgs/round.

## Main datasets

{table(MAIN, METH)}

## Appendix datasets

{table(APP, METH)}

## Ablation variants of CAGA (not drawn in the main figure)

CAGA-gradgate = no H-margin check on anchor update; CAGA-scale6 = fixed shallow cap (maxlag_shallow=6).

| Method | {" | ".join(MAIN + APP)} |
|---|{"---|" * 8}
""" + "\n".join(f"| {m.replace('[abl] ', '')} | " + " | ".join(cell(by[d].get(m)) for d in MAIN + APP) + " |" for m in ["CAGA", "[abl] CAGA-gradgate", "[abl] CAGA-scale6"]) + """

## Reading notes (sanity checks)
1. Peak at round 1 means the method already degraded or collapsed inside the first round (TENT, ViDA, PTTA, COME, DAF on Cityscapes); its peak is then near or below No-Adapt.
2. Peak equal to R150 (SPA, TRIBE, MGP on V21, ...) means the curve is flat or still slowly rising; it is not a "rise then fall" pattern.
3. SAR is a periodic sawtooth on Cityscapes and V20 (reset cycle), so its R150 depends on where round 150 falls inside a cycle (Cityscapes: R150 = peak = 22.69, V20: R150 = 62.39, mid-cycle).
4. CoTTA is excluded (stopped by decision). Baseline hyperparameters are not identical across methods (see the separate hyperparameter notes).
"""
open(f"{D}/results_table.md", "w").write(md); print("written", len(md), "chars")
