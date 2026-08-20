"""Explain H_margin concretely, with REAL checkpoints from the no-gate collapse run.

Loads the LN weights at (a) the mIoU peak and (b) deep collapse, runs inference on the
same ACDC image, and shows for each:
    left  : the predicted segmentation
    right : the class-marginal distribution  p_c = mean_pixels softmax(.)_c
            -> H_margin = -sum_c p_c log p_c   (prediction DIVERSITY)
Far right: H_margin over the 150 rounds, with the two states marked.

Output: figures/hmargin_explain.png
"""
import os, glob, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec

from adapt import get_method
from utils import segmentation_datasets
from utils.misc import aggregate_pred_patches, custom_collate

plt.rcParams.update({"font.size": 13})

RUN = "save/ACDCDataset/deyo_mlmp_hmgate2_oracle_nogate_lr3e-5"
CONDITION = "night"
WANT = "GP010364_frame_000047"
device = "cuda" if torch.cuda.is_available() else "cpu"

CLASSES = ['road','sidewalk','building','wall','fence','pole','traffic light',
           'traffic sign','vegetation','terrain','sky','person','rider','car',
           'truck','bus','train','motorcycle','bicycle']
PALETTE = [[128,64,128],[244,35,232],[70,70,70],[102,102,156],[190,153,153],
           [153,153,153],[250,170,30],[220,220,0],[107,142,35],[152,251,152],
           [70,130,180],[220,20,60],[255,0,0],[0,0,142],[0,0,70],
           [0,60,100],[0,80,100],[0,0,230],[119,11,32]]
PAL = np.array(PALETTE) / 255.0

# ---- trajectory + pick the two checkpoints --------------------------------
miou = np.array([float(l.split(",")[-1]) for l in open(f"{RUN}/results_all_rounds.txt")
                 if l.startswith("Round ")])
R = len(miou)
rows = [l.strip().split(",") for l in open(f"{RUN}/gate_log.csv")][1:]
tb = np.array([int(r[0]) for r in rows]); hm = np.array([float(r[3]) for r in rows])
per = tb.max() / R
hm_round = np.array([a.mean() for a in np.array_split(hm, R)])

cks = sorted(glob.glob(f"{RUN}/ln_ckpt/*.pt"))
def ckpt_for(rd):
    want = rd * per
    return min(cks, key=lambda f: abs(int(os.path.basename(f)[4:-3]) - want))

R_HEALTHY = int(miou.argmax() + 1)
R_COLLAPSE = 100
STATES = [("HEALTHY", R_HEALTHY, ckpt_for(R_HEALTHY), "#2ca02c"),
          ("COLLAPSED", R_COLLAPSE, ckpt_for(R_COLLAPSE), "#d62728")]

# ---- data + model ----------------------------------------------------------
loader, classes = segmentation_datasets.prepare_data(
    "ACDCDataset", ".data/ACDC/", [1120, 560], [224, 224], 112,
    corruption=CONDITION, batch_size=1, num_workers=0, shuffle=False)
idx = 0
for i in range(len(loader.dataset)):
    if WANT in loader.dataset[i]["meta"]["img_path"]:
        idx = i; break
sample = loader.dataset[idx]
batch = custom_collate([sample])
inputs = batch["img_patches"].to(device)

args = argparse.Namespace(
    method="deyo_mlmp_hmgate2_continual", ovss_type="naclip", ovss_backbone="ViT-L/14",
    prompt_dir="prompts.yaml", lr=5e-6, steps=1,
    vision_outputs=list(range(-1, -19, -1)), classes=classes, save_dir=None,
    runtime_calculation=False)
method = get_method(args, device)

ext = torch.tensor(loader.dataset.extentions_to_real_class_idx,
                   dtype=torch.int64, device=device)
ncls = int(ext.max()) + 1
onehot = torch.nn.functional.one_hot(ext).T.view(ncls, len(ext), 1, 1)

results = []
for name, rd, ck, col in STATES:
    d = torch.load(ck)
    for n, p in method.named_ln_params:            # load that moment's LN weights
        p.data.copy_(d["ln"][n].to(p.device, dtype=p.dtype))
    with torch.no_grad():
        pp = method.evaluate(inputs)
        rec = aggregate_pred_patches(pp, batch["meta"]["patch_grid_shape"],
                                     batch["meta"]["img_shape"], [224, 224], 112)
        prob = rec[0].softmax(dim=0)
        prob = (prob.unsqueeze(0) * onehot).max(1)[0]        # -> 19 real classes
        prob = prob / prob.sum(0, keepdim=True).clamp(min=1e-8)
        pred = prob.argmax(0).cpu().numpy()
        marg = prob.mean(dim=(1, 2)).cpu().numpy()           # class-marginal
        marg = marg / marg.sum()
    H = float(-(marg * np.log(np.clip(marg, 1e-12, None))).sum())
    results.append((name, rd, col, pred, marg, H, d["h_margin"]))
    print(f"{name:10s} R{rd:3d}  H_margin(this image)={H:.3f}  (logged window={d['h_margin']:.3f})"
          f"  #classes>1%={int((marg > 0.01).sum())}")

# ---- figure ----------------------------------------------------------------
fig = plt.figure(figsize=(19, 8.6))
gs = gridspec.GridSpec(2, 3, width_ratios=[1.25, 1.5, 1.0], wspace=0.28, hspace=0.30)

for r, (name, rd, col, pred, marg, H, _) in enumerate(results):
    ax = fig.add_subplot(gs[r, 0])
    ax.imshow(PAL[pred]); ax.axis("off")
    ax.set_title(f"{name}  (round {rd})\nprediction", fontsize=15,
                 color=col, fontweight="bold")

    ax = fig.add_subplot(gs[r, 1])
    ax.bar(np.arange(19), marg, color=PAL, edgecolor="k", lw=0.5)
    ax.set_xticks(np.arange(19))
    ax.set_xticklabels(CLASSES, rotation=60, ha="right", fontsize=10)
    ax.set_ylabel("marginal  $p_c$")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"class-marginal distribution     "
                 f"$H_{{margin}} = {H:.2f}$", fontsize=15, color=col, fontweight="bold")
    ax.grid(alpha=0.25, axis="y")
    top = int(marg.argmax())
    ax.annotate(f"{CLASSES[top]}  {marg[top]*100:.0f}%",
                xy=(top, marg[top]), xytext=(top + 1.5, marg[top] + 0.06),
                fontsize=11, arrowprops=dict(arrowstyle="->", lw=1.2))

ax = fig.add_subplot(gs[:, 2])
ax.plot(np.arange(1, R + 1), hm_round, color="#1f77b4", lw=2.2)
for name, rd, col, _, _, H, _ in results:
    ax.axvline(rd, color=col, ls=":", lw=2.0)
    ax.plot(rd, hm_round[rd - 1], "o", ms=11, color=col, zorder=5)
    ax.annotate(name, xy=(rd, hm_round[rd - 1]), xytext=(rd + 8, hm_round[rd - 1] + 0.08),
                color=col, fontsize=12, fontweight="bold")
ax.set_xlabel("Round"); ax.set_ylabel("$H_{margin}$")
ax.set_title("$H_{margin}$ over the no-gate run", fontsize=15)
ax.grid(alpha=0.3)

fig.suptitle("$H_{margin}$ = entropy of the CLASS-MARGINAL distribution  "
             "(how many classes the model still predicts)", fontsize=18, fontweight="bold")
fig.savefig("figures/hmargin_explain.png", dpi=135, bbox_inches="tight")
print("saved -> figures/hmargin_explain.png")
