#!/usr/bin/env python
"""Gate-firing analysis figures for the composite-gate slides.

  gate_mechanism.png      - 3 panels (ACDC/VOC20/Cityscapes): gate-internal mean_conf
                            vs round (+ conf_ceil line + restore events) twinned with mIoU.
                            Shows: gate caps mean_conf at conf_ceil -> mIoU stays stable.
  gate_firing_summary.png - per dataset: firing-rate% and mean mIoU vs conf_ceil
                            (the calibration sweet spot).

Auto-picks the best composite config per dataset (highest 150R mean). Run after the
Cityscapes proper-calibration sweep finishes.
"""
import os, glob, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SAVE, OUT = "save", "save/_compare"
os.makedirs(OUT, exist_ok=True)

DS = {
    "ACDC":       dict(dir="ACDCDataset", bpr=406),
    "VOC20":      dict(dir="PascalVOC20Dataset/v20_acdc_matched", bpr=500),
    "Cityscapes": dict(dir="CityscapesDataset", bpr=500),
}


def rounds(d):
    f = os.path.join(d, "results_all_rounds.txt")
    return [float(l.split(",")[-1]) for l in open(f) if l.lower().startswith("round ")] if os.path.exists(f) else []


def gate(d):
    """returns (rounds_x, mean_conf, restore_round_marks, conf_ceil_from_name, firing_pct, mc_peak)"""
    f = os.path.join(d, "gate_log.csv")
    if not os.path.exists(f):
        return None
    tb, mc, rst = [], [], []
    for l in open(f):
        if l.startswith("total"):
            continue
        p = l.split(",")
        try:
            tb.append(int(p[0])); mc.append(float(p[1])); rst.append(float(p[-1]))
        except (ValueError, IndexError):
            continue
    if not tb:
        return None
    n = len(tb); on = sum(1 for r in rst if r > 0)
    cc = float(os.path.basename(d).split("_cc")[1].split("_")[0])
    return dict(tb=tb, mc=mc, rst=rst, firing=100*on/n, mc_peak=max(mc), cc=cc)


def best_dir(ds):
    cand = []
    for d in glob.glob(os.path.join(SAVE, DS[ds]["dir"], "deyo_mlmp_composite_*")):
        ms = rounds(d)
        if len(ms) >= 140:
            cand.append((sum(ms)/len(ms), d))
    return max(cand)[1] if cand else None


# ---------- Fig 1: mechanism (mean_conf capped + mIoU) ----------
def fig_mechanism():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, ds in zip(axes, ["ACDC", "VOC20", "Cityscapes"]):
        d = best_dir(ds)
        if not d:
            ax.set_title(f"{ds} (no data yet)"); continue
        g = gate(d); ms = rounds(d); bpr = DS[ds]["bpr"]
        cfg = os.path.basename(d).split("deyo_mlmp_composite_")[1]
        xr = [t / bpr for t in g["tb"]]
        ax.plot(xr, g["mc"], color="#1f6b43", lw=1.6, label="gate mean_conf")
        ax.axhline(g["cc"], ls="--", color="#9b8d3a", lw=1.4, label=f"conf_ceil {g['cc']}")
        rx = [t / bpr for t, r in zip(g["tb"], g["rst"]) if r > 0]
        ax.plot(rx, [g["cc"]] * len(rx), "v", color="#d62828", ms=6,
                label=f"restore fired ({g['firing']:.0f}%)")
        ax.set_ylabel("gate mean_conf", color="#1f6b43"); ax.set_xlabel("round")
        ax.set_ylim(min(g["mc"]) - 0.03, max(max(g["mc"]), g["cc"]) + 0.04)
        ax2 = ax.twinx()
        ax2.plot(range(1, len(ms) + 1), ms, color="#7aa6c2", lw=1.3, alpha=0.85)
        ax2.set_ylabel("mIoU", color="#7aa6c2")
        ax.set_title(f"{ds}\n{cfg}  (mean {sum(ms)/len(ms):.1f}, mc_peak {g['mc_peak']:.3f})",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Composite gate mechanism: mean_conf rises, gets capped at conf_ceil -> mIoU held",
                 fontweight="bold")
    fig.tight_layout(); p = os.path.join(OUT, "gate_mechanism.png")
    fig.savefig(p, dpi=140); print("wrote", p)


# ---------- Fig 2: firing-rate & mean vs conf_ceil ----------
def fig_firing_summary():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, ds in zip(axes, ["ACDC", "VOC20", "Cityscapes"]):
        rows = []
        for d in glob.glob(os.path.join(SAVE, DS[ds]["dir"], "deyo_mlmp_composite_*")):
            ms = rounds(d); g = gate(d)
            if len(ms) >= 140 and g:
                rows.append((g["cc"], g["firing"], sum(ms)/len(ms)))
        rows.sort()
        if not rows:
            ax.set_title(f"{ds} (no data)"); continue
        ccs = [r[0] for r in rows]; fire = [r[1] for r in rows]; mean = [r[2] for r in rows]
        ax.plot(ccs, fire, "o-", color="#d62828", label="firing rate %")
        ax.set_xlabel("conf_ceil"); ax.set_ylabel("restore firing %", color="#d62828")
        ax2 = ax.twinx()
        ax2.plot(ccs, mean, "s-", color="#3b7a57", label="mean mIoU")
        ax2.set_ylabel("mean mIoU", color="#3b7a57")
        ax.set_title(ds, fontweight="bold"); ax.grid(alpha=0.3)
    fig.suptitle("Calibration: lower conf_ceil -> more firing. (over-firing hurts headroom-limited Cityscapes)",
                 fontweight="bold", fontsize=10)
    fig.tight_layout(); p = os.path.join(OUT, "gate_firing_summary.png")
    fig.savefig(p, dpi=140); print("wrote", p)


if __name__ == "__main__":
    fig_mechanism()
    fig_firing_summary()
    print("done")
