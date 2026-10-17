"""Generate the same Input/GT/Round0/Round150 quad figure (with shared
class-colour legend) across all 15 standard ImageNet-C corruptions, on the
same VOC21 image (idx=0), reusing the real adagate_nogate_batch8_fulldata
checkpoint (naclip, base_rst=0.0, 150R complete). One quad per corruption,
saved to .qualviz_v21/corruptions/<name>_quad.png.
"""
import os
import json

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

from adapt.deyo_mlmp_adagate_continual import DeYOMLMPAdaGateContinual
from utils import segmentation_datasets
from utils.misc import aggregate_pred_patches, custom_collate

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif']

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]
IMG_IDX = 0
CKPT = 'save/PascalVOC20Dataset/v20_batch_ablation/adagate_nogate_batch8_fulldata/ckpt.pt'
OUT_DIR = '.qualviz_v21/corruptions'
os.makedirs(OUT_DIR, exist_ok=True)

device = 'cuda'

# class list / palette only depend on the dataset def, not on corruption --
# build one loader now just to get them and the class_extensions used by
# every subsequent per-corruption loader.
_probe_loader, org_classes = segmentation_datasets.prepare_data(
    'PascalVOC21Dataset', '.data/VOC2012/', [224, 224], [224, 224], 112,
    corruption='original', batch_size=1, num_workers=0, shuffle=False,
    corruption_severity=5, split='val', ann_file='ImageSets/Segmentation/trainval.txt')
classes = _probe_loader.dataset.class_extensions or org_classes
palette = np.array(_probe_loader.dataset.METAINFO['palette'])


def build_model():
    return DeYOMLMPAdaGateContinual(
        ovss_type='naclip', ovss_backbone='ViT-L/14', lr=5e-6, classes=classes,
        steps=1, vision_outputs=list(range(-1, -19, -1)),
        deyo_margin_factor=0.5, deyo_margin_e0_factor=0.4, plpd_threshold=0.2,
        aug_type='patch', patch_len=4, reweight_ent=True, reweight_plpd=True,
        top_block_exclude=6, base_rst=0.0, device=device)


print("Building SOURCE model...")
m_src = build_model()

print("Building checkpoint model + loading round-150 weights...")
m_ck = build_model()
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
saved_ln = ckpt['adapt_state']['ln_params']
with torch.no_grad():
    for n, p in m_ck.named_ln_params:
        if n in saved_ln:
            p.data.copy_(saved_ln[n].to(device=p.device, dtype=p.dtype))
m_ck.model.to(device)
print(f"Checkpoint: round_idx={ckpt['round_idx']} cond_idx={ckpt['cond_idx']} "
      f"({len(saved_ln)}/{len(m_ck.named_ln_params)} LN tensors loaded)")

CLIP_MEAN = torch.tensor([122.7709, 116.7460, 104.0937]).view(3, 1, 1)
CLIP_STD = torch.tensor([68.5005, 66.6322, 70.3232]).view(3, 1, 1)


def colourise(label):
    rgb = np.zeros((*label.shape, 3), dtype=np.uint8)
    for c in range(len(palette)):
        rgb[label == c] = palette[c]
    rgb[label == 255] = [255, 255, 255]
    return rgb


def run(m, inputs, grid, shape, one_hot):
    with torch.no_grad():
        patch_preds = m.evaluate(inputs)
    reconstructed = aggregate_pred_patches(patch_preds, grid, shape, [224, 224], 112)
    pd = reconstructed[0].softmax(dim=0)
    pd2 = (pd.unsqueeze(0) * one_hot).max(1)[0]
    pd_label = pd2.argmax(dim=0).cpu().numpy()
    conf = pd2.max(0)[0].mean().item()
    return pd_label, conf


summary = []
for corruption in CORRUPTIONS:
    print(f"\n=== {corruption} ===")
    loader, _ = segmentation_datasets.prepare_data(
        'PascalVOC21Dataset', '.data/VOC2012/', [224, 224], [224, 224], 112,
        corruption=corruption, batch_size=1, num_workers=0, shuffle=False,
        corruption_severity=5, split='val', ann_file='ImageSets/Segmentation/trainval.txt')
    sample = loader.dataset[IMG_IDX]
    batch = custom_collate([sample])
    inputs = batch['img_patches'].to(device)
    gt = batch['gt'][0][0].numpy()
    valid = gt != 255

    ext_to_real = torch.tensor(loader.dataset.extentions_to_real_class_idx, dtype=torch.int64, device=device)
    num_cls = int(ext_to_real.max()) + 1
    one_hot = torch.nn.functional.one_hot(ext_to_real).T.view(num_cls, len(ext_to_real), 1, 1)

    pd_src, conf_src = run(m_src, inputs, batch['meta']['patch_grid_shape'], batch['meta']['img_shape'], one_hot)
    pd_ck, conf_ck = run(m_ck, inputs, batch['meta']['patch_grid_shape'], batch['meta']['img_shape'], one_hot)
    acc_src = (pd_src[valid] == gt[valid]).mean()
    acc_ck = (pd_ck[valid] == gt[valid]).mean()

    img_tensor = batch['img'][0].cpu().float()
    raw_img = (img_tensor * CLIP_STD + CLIP_MEAN).clamp(0, 255).byte().permute(1, 2, 0).numpy()

    gt_c, src_c, ck_c = colourise(gt), colourise(pd_src), colourise(pd_ck)
    present = sorted(set(np.unique(gt[valid]).tolist()) | set(np.unique(pd_src).tolist()) | set(np.unique(pd_ck).tolist()))

    fig = plt.figure(figsize=(11, 4.3))
    gs = fig.add_gridspec(2, 4, height_ratios=[5, 1], hspace=0.05, wspace=0.06,
                          left=0.01, right=0.99, top=0.84, bottom=0.02)
    panels = [
        (raw_img, 'Input', None),
        (gt_c, 'Ground Truth', None),
        (src_c, 'Round 0 (source)', f'acc={acc_src*100:.1f}%   conf={conf_src*100:.1f}%'),
        (ck_c, 'Round 150 (no-gate)', f'acc={acc_ck*100:.1f}%   conf={conf_ck*100:.1f}%'),
    ]
    for i, (im, title, sub) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(im)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_linewidth(1.0); spine.set_color('#444444')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=6)
        if sub:
            ax.text(0.5, -0.06, sub, transform=ax.transAxes, ha='center', va='top', fontsize=10)

    legend_ax = fig.add_subplot(gs[1, :])
    legend_ax.axis('off')
    handles = [Patch(facecolor=np.array(palette[c]) / 255.0, edgecolor='#444444', linewidth=0.5,
                     label=(org_classes[c] if c < len(org_classes) else str(c)))
              for c in present]
    legend_ax.legend(handles=handles, loc='center', ncol=min(len(handles), 8), frameon=False,
                     fontsize=10, handlelength=1.4, handleheight=1.4, columnspacing=1.3)

    fig.suptitle(f'Over-confident wrong-class collapse under 150-round no-gate continual adaptation\n'
                f'NA-CLIP + VOC21, deyo_mlmp_adagate_continual, base_rst=0  |  corruption={corruption} (sev.5)',
                fontsize=11, y=0.975)
    out_path = f'{OUT_DIR}/{corruption}_quad.png'
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    delta_acc = (acc_ck - acc_src) * 100
    delta_conf = (conf_ck - conf_src) * 100
    print(f"  acc: {acc_src*100:.1f}% -> {acc_ck*100:.1f}% (Δ{delta_acc:+.1f})  "
          f"conf: {conf_src*100:.1f}% -> {conf_ck*100:.1f}% (Δ{delta_conf:+.1f})")
    summary.append(dict(corruption=corruption, acc_src=float(acc_src), acc_ck=float(acc_ck),
                        conf_src=float(conf_src), conf_ck=float(conf_ck), path=out_path))

with open(f'{OUT_DIR}/summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("\n\n=== SUMMARY (sorted by most dramatic acc drop) ===")
for s in sorted(summary, key=lambda s: s['acc_src'] - s['acc_ck'], reverse=True):
    print(f"  {s['corruption']:20s} acc {s['acc_src']*100:5.1f}% -> {s['acc_ck']*100:5.1f}%  "
          f"conf {s['conf_src']*100:5.1f}% -> {s['conf_ck']*100:5.1f}%")
print(f"\nAll 15 quads saved under {OUT_DIR}/")
