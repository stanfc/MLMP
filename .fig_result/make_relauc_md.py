"""Render .fig_result/relauc_table.csv into .fig_result/relauc_table.md (run make_relauc.py first)."""
import csv, datetime
D = "/home/stanfc/TTA-on-OVSS/MLMP/.fig_result"
rows = list(csv.DictReader(open(f"{D}/relauc_table.csv")))
datasets = [k for k in rows[0].keys() if k not in ("method", "AvgRelAUC")]

def fmt(v): return f"{float(v):.2f}" if v else "n/a"

lines = [
"# RelAUC and AvgRelAUC (150-round whole-stream, relative to No-Adapt)",
"",
f"Generated {datetime.date.today()} from `results_all_rounds.txt` of each run",
"(script: `.fig_result/make_relauc.py`, source CSV: `.fig_result/relauc_table.csv`).",
"",
"```",
"RelAUC(method, dataset) = (1/150) * sum_{r=1..150} (mIoU_r - NoAdapt_mIoU) / NoAdapt_mIoU * 100",
"AvgRelAUC(method)        = (1/8)   * sum_d RelAUC(method, d)",
"```",
"",
"MLMP-episodic is excluded (no continual trajectory). Sanity check: all 88 runs (11 methods x 8 datasets) have exactly rounds 1-150.",
"",
"## RelAUC(method, dataset), %",
"",
"| Method | " + " | ".join(datasets) + " |",
"|---|" + "---|" * len(datasets),
]
for r in rows:
    lines.append(f"| {r['method']} | " + " | ".join(fmt(r[d]) for d in datasets) + " |")

lines += [
"",
"## AvgRelAUC(method), ranked",
"",
"| Rank | Method | AvgRelAUC |",
"|---|---|---|",
]
ranked = sorted(rows, key=lambda r: -float(r["AvgRelAUC"]) if r["AvgRelAUC"] else 999)
for i, r in enumerate(ranked, 1):
    name = f"**{r['method']}**" if r["method"] == "CAGA" else r["method"]
    lines.append(f"| {i} | {name} | {fmt(r['AvgRelAUC'])} |")

open(f"{D}/relauc_table.md", "w").write("\n".join(lines) + "\n")
print("written", f"{D}/relauc_table.md")
