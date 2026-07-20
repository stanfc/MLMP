#!/usr/bin/env python
"""Mechanism analysis for the GDG-PA prompt-set sweep.

For each prompt set we build the AVERAGED per-class text anchor (mean over the set's
templates, renormalized — exactly what GDG-PA's eval classifier uses) and measure
CLASS CONFUSABILITY = mean off-diagonal cosine similarity of the class anchors. The
hypothesis: prompt sets whose class anchors are MORE mutually similar (higher
confusability = smaller inter-class margin) segment WORSE, regardless of how
"domain-relevant" the words are. Long shared-context templates ("a driving photo of
a {} in foggy weather") inflate confusability because the shared tokens dominate.

Outputs (save/_compare/):
  prompt_perf_bars.png        ranking bars (mR30+ per set, per dataset)
  prompt_confusability.png    scatter: confusability vs mR30+ (+ Pearson r)
  prompt_analysis_table.csv   set, dataset, mR30+, peak, confusability, avg_words
"""
import os, sys, glob, csv
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.collect_prompt_sweep import collect_dataset  # noqa
from utils.misc import load_prompts_from_yaml
from ovss import load_ovss

CITYSCAPES19 = ['road','sidewalk','building','wall','fence','pole','traffic light',
                'traffic sign','vegetation','terrain','sky','person','rider','car',
                'truck','bus','train','motorcycle','bicycle']
VOC20 = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
         'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa',
         'train','tvmonitor']

DATASETS = [
    ("ACDC",  "save/ACDCDataset",                          CITYSCAPES19),
    ("VOC20", "save/PascalVOC20Dataset/v20_acdc_matched",  VOC20),
]
PROMPT_DIR = "prompts_sweep"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def class_anchors(model, tokenize, classes, templates):
    """(C, D) averaged, renormalized per-class text anchor for one prompt set."""
    anchors = []
    for c in classes:
        texts = tokenize([t.format(c) for t in templates]).to(DEVICE)
        e = model.encode_text(texts)
        e = e / e.norm(dim=-1, keepdim=True)
        a = e.mean(0)
        anchors.append(a / a.norm())
    return torch.stack(anchors)  # (C, D)


def confusability(anchors):
    """mean off-diagonal cosine similarity among class anchors."""
    S = (anchors @ anchors.T).cpu().numpy()
    C = S.shape[0]
    off = S[~np.eye(C, dtype=bool)]
    return float(off.mean())


def main():
    print(f"loading naclip ViT-L/14 text encoder on {DEVICE} ...")
    model, tokenize = load_ovss("naclip", "ViT-L/14", device=DEVICE)
    model.eval()

    sets = sorted(os.path.basename(p)[:-5] for p in glob.glob(f"{PROMPT_DIR}/*.yaml"))
    rows = []
    for name, root, classes in DATASETS:
        perf = collect_dataset(root)  # {id: (rounds, means, stats)}
        for sid in sets:
            yid = sid  # yaml stem == prompt id used in save dir? map below
        for pid, (_, _, st) in perf.items():
            yaml_path = f"{PROMPT_DIR}/{pid}.yaml"
            if not os.path.isfile(yaml_path):
                continue
            tmpl = load_prompts_from_yaml(yaml_path)
            anc = class_anchors(model, tokenize, classes, tmpl)
            conf = confusability(anc)
            avg_words = np.mean([len(t.split()) for t in tmpl])
            rows.append(dict(dataset=name, pid=pid, mr30=st["mean_stable"],
                             peak=st["peak"], conf=conf, nt=len(tmpl),
                             avg_words=avg_words))
            print(f"  {name:6s} {pid:18s} mR30+={st['mean_stable']:6.2f} "
                  f"conf={conf:.4f} N={len(tmpl):2d} avg_words={avg_words:.1f}")

    os.makedirs("save/_compare", exist_ok=True)
    with open("save/_compare/prompt_analysis_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset","pid","mr30","peak","conf","nt","avg_words"])
        w.writeheader(); w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- Fig A: ranking bars (mR30+), per dataset ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (name, _, _) in zip(axes, DATASETS):
        d = [r for r in rows if r["dataset"] == name]
        d.sort(key=lambda r: r["mr30"])
        base = next((r["mr30"] for r in d if r["pid"] == "S0_baseline"), None)
        labels = [r["pid"].replace("_", "\n", 1) for r in d]
        vals = [r["mr30"] for r in d]
        colors = ["#c0392b" if r["pid"] == "S0_baseline"
                  else ("#7f8c8d" if r["pid"] == "S8_artistic" else "#2980b9") for r in d]
        bars = ax.barh(labels, vals, color=colors)
        if base is not None:
            ax.axvline(base, ls="--", c="#c0392b", lw=1, alpha=.7)
        for r, b in zip(d, bars):
            dv = r["mr30"] - base if base is not None else 0
            ax.text(b.get_width() + 0.05, b.get_y() + b.get_height()/2,
                    f"{r['mr30']:.2f} ({dv:+.2f})", va="center", fontsize=8)
        lo = min(vals) - 1.0
        ax.set_xlim(lo, max(vals) + (2.2 if name == "ACDC" else 1.0))
        ax.set_title(f"{name}: mean mIoU (R≥30)  — red=baseline, grey=artistic-control")
        ax.set_xlabel("mean mIoU (R≥30)")
    plt.tight_layout(); plt.savefig("save/_compare/prompt_perf_bars.png", dpi=130); plt.close()

    # ---- Fig B: template LENGTH vs performance (the robust correlate) ----
    # NOTE: class confusability was tested first but its sign FLIPS across datasets
    # (ACDC r=-0.53, VOC20 r=+0.64, pooled ~0) -> not a robust mechanism. Template
    # length (avg words) is: pooled r=-0.62, VOC20 -0.85. Shorter generic templates win.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, (name, _, _) in zip(axes, DATASETS):
        d = [r for r in rows if r["dataset"] == name]
        x = np.array([r["avg_words"] for r in d]); y = np.array([r["mr30"] for r in d])
        r_p = np.corrcoef(x, y)[0, 1]
        for r in d:
            fam = r["pid"] in ("S0_baseline", "S8_artistic")
            single = r["pid"] == "S1_minimal"
            c = "#c0392b" if r["pid"] == "S0_baseline" else ("#e67e22" if fam else ("#7f8c8d" if single else "#2980b9"))
            ax.scatter(r["avg_words"], r["mr30"], s=70, c=c, zorder=3,
                       edgecolors="k", linewidths=.4)
            ax.annotate(r["pid"].split("_", 1)[1], (r["avg_words"], r["mr30"]),
                        fontsize=7, xytext=(4, 3), textcoords="offset points")
        if len(x) > 2:
            b1, b0 = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, b1*xs + b0, c="#c0392b", lw=1.3, ls="--", alpha=.8)
        ax.set_title(f"{name}: template length vs mIoU   (Pearson r = {r_p:+.2f})")
        ax.set_xlabel("avg words per template  → longer / more shared context")
        ax.set_ylabel("mean mIoU (R≥30)")
        ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig("save/_compare/prompt_length.png", dpi=130); plt.close()
    print("\nfigures -> save/_compare/prompt_perf_bars.png, prompt_length.png")
    print("table   -> save/_compare/prompt_analysis_table.csv")


if __name__ == "__main__":
    main()
