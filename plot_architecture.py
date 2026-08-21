"""Method architecture diagram (block / data-flow).
MLMP and DeYO are absorbed as cited modules inside OUR method.
Output: figures/architecture.png
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

# colours by origin
C_MLMP = "#aec7e8"; E_MLMP = "#1f77b4"
C_DEYO = "#ffbb78"; E_DEYO = "#e6892b"
C_OURS = "#a1d99b"; E_OURS = "#2ca02c"
C_FRZ  = "#dddddd"; E_FRZ  = "#888888"
C_DATA = "#f2f2f2"; E_DATA = "#555555"

fig, ax = plt.subplots(figsize=(17, 9.2))
ax.set_xlim(0, 170); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, text, fc, ec, fs=11, weight="normal", tag=None, tagc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.6,rounding_size=2.2",
                 fc=fc, ec=ec, lw=2.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, weight=weight, zorder=4)
    if tag:
        ax.text(x + w - 1.5, y + h - 1.2, tag, ha="right", va="top",
                fontsize=8.5, style="italic", color=tagc, zorder=5)


def arrow(x1, y1, x2, y2, text=None, rad=0.0, color="#333333", lw=2.2,
          fs=10, ls="-", toff=(0, 2)):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                 connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                 mutation_scale=20, lw=lw, color=color, ls=ls, zorder=2))
    if text:
        mx, my = (x1 + x2) / 2 + toff[0], (y1 + y2) / 2 + toff[1]
        ax.text(mx, my, text, ha="center", va="center", fontsize=fs,
                color=color, zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))


# ---- outer "OUR METHOD" frame ----------------------------------------------
ax.add_patch(FancyBboxPatch((2, 3), 166, 88, boxstyle="round,pad=0.6,rounding_size=3",
             fc="none", ec=E_OURS, lw=2.6, ls=(0, (6, 4)), zorder=1))
ax.text(5, 88.5, "OUR METHOD", fontsize=15, weight="bold", color=E_OURS, va="center")

# ============================ MAIN ADAPT FLOW (y~46-60) =====================
yb, hb = 46, 15
box(4,  yb, 20, hb, "Input\nACDC frame\n→ 36 patches", C_DATA, E_DATA, fs=10)
box(28, yb, 26, hb, "NA-CLIP ViT-L/14\nVisual: LN trainable\nText: frozen  (7 prompts)",
    C_FRZ, E_FRZ, fs=10)
box(58, yb, 24, hb, "UAML  +  Multi-Prompt\n18-layer fusion (β=0)\n7-prompt average",
    C_MLMP, E_MLMP, fs=10, tag="[MLMP]", tagc=E_MLMP)
box(86, yb, 24, hb, "DeYO filter\nentropy ∧ PLPD mask\nreweighted  → L_ent",
    C_DEYO, E_DEYO, fs=10, tag="[DeYO]", tagc=E_DEYO)
box(114, yb, 24, hb, "EMA Teacher\nt_pred (no-grad)\n+ λ·CE distill",
    C_OURS, E_OURS, fs=10, weight="bold")
box(142, yb, 24, hb, "Loss + Update\nL_ent + λ·L_distill\nstep (LR 3e-5) · EMA",
    C_OURS, E_OURS, fs=10, weight="bold")

for x1, x2 in [(24, 28), (54, 58), (82, 86), (110, 114), (138, 142)]:
    arrow(x1, yb + hb / 2, x2, yb + hb / 2)
ax.text(56, yb - 2.4, "logits = cos(visual, text)·scale", fontsize=9, style="italic",
        color="#555", ha="center")

# ============================ EVALUATE branch (top, y~76) ===================
ye = 76
box(58, ye, 52, 11,
    "EVALUATE  (before adapt, no update)\nswap EMA-LN · UAML β=1 (sharpen) · flip-TTA",
    C_OURS, E_OURS, fs=10.5, weight="bold")
box(142, ye, 24, 11, "Prediction\n(mIoU)", C_DATA, E_DATA, fs=11, weight="bold")
arrow(41, yb + hb, 60, ye, "eval fwd", rad=0.25, color=E_OURS, toff=(-6, 0))
arrow(110, ye + 5.5, 142, ye + 5.5)

# ============================ GDG-PA GATE (bottom, y~12-33) =================
ax.add_patch(FancyBboxPatch((34, 10), 118, 24, boxstyle="round,pad=0.6,rounding_size=3",
             fc="#eafaf0", ec=E_OURS, lw=2.4, zorder=1))
ax.text(93, 31.2, "GDG-PA  GATE   (permanent-anchor grad/diversity gate)",
        fontsize=12, weight="bold", color=E_OURS, ha="center")
box(40, 14, 50, 12,
    "grad_norm = ‖g_entropy‖\nMIN → pin best_snapshot (perm.)\nSLOPE↑ → trigger + depth",
    "white", E_OURS, fs=9.5)
box(98, 14, 48, 12,
    "H_margin  (pred. diversity)\n< 0.9·max → COLLAPSE regime\nelse → HEALTHY",
    "white", E_OURS, fs=9.5)
ax.text(93, 11.3,
        "HEALTHY → shallow restore (recent) | COLLAPSE → deep restore (best_snapshot)",
        fontsize=9.5, ha="center", color="#333", style="italic")

# gradient down into the gate
arrow(154, yb, 150, 34, "g_entropy", rad=-0.15, color="#d62728", toff=(6, 0))
# restore feedback back up to the encoder LN weights
arrow(40, 22, 30, yb, "restore LN\n→ best anchor", rad=-0.35, color="#d62728", toff=(-7, 0))

# ---- legend ----------------------------------------------------------------
handles = [
    Line2D([0], [0], marker="s", ls="", ms=15, mfc=C_OURS, mec=E_OURS, label="Ours (contribution)"),
    Line2D([0], [0], marker="s", ls="", ms=15, mfc=C_MLMP, mec=E_MLMP, label="MLMP  [cite]"),
    Line2D([0], [0], marker="s", ls="", ms=15, mfc=C_DEYO, mec=E_DEYO, label="DeYO  [cite]"),
    Line2D([0], [0], marker="s", ls="", ms=15, mfc=C_FRZ,  mec=E_FRZ,  label="Frozen backbone"),
    Line2D([0], [0], color="#d62728", lw=2.4, label="gradient signal / restore feedback"),
]
ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.985),
          fontsize=10, framealpha=0.95, ncol=1)

ax.set_title("Method architecture — evaluate-before-adapt CTTA with a permanent-anchor "
             "grad/diversity gate", fontsize=15, weight="bold", pad=12)
fig.tight_layout()
fig.savefig("figures/architecture.png", dpi=140, bbox_inches="tight")
print("saved -> figures/architecture.png")
