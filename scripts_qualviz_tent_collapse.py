"""
One-off script: load the mid-stream checkpoint of the tent_continual
150-round ACDC reproduction (save/ACDCDataset/tent_continual_for_qualviz/)
and render a qualitative segmentation map for one fixed sample, showing the
"over-confidence single class" collapse for Fig 1's right half.

Usage:
    python scripts_qualviz_tent_collapse.py --out_dir .qualviz_tent_collapse/ \
        --condition fog --img_idx 0
"""
import os
import argparse

import numpy as np
import torch
from PIL import Image

from adapt.tent_continual import TENTContinual
from utils import segmentation_datasets
from utils.misc import aggregate_pred_patches, custom_collate

from qualitative import apply_palette, save_legend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='save/ACDCDataset/tent_continual_for_qualviz/ckpt.pt')
    ap.add_argument('--out_dir', default='.qualviz_tent_collapse/')
    ap.add_argument('--condition', default='fog')
    ap.add_argument('--img_idx', type=int, default=0)
    ap.add_argument('--data_dir', default='.data/ACDC/')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out_dir, exist_ok=True)

    loader, org_classes = segmentation_datasets.prepare_data(
        'ACDCDataset', args.data_dir, [1120, 560], [224, 224], 112,
        corruption=args.condition, batch_size=1, num_workers=0, shuffle=False,
        corruption_severity=5, split='val')

    classes = loader.dataset.class_extensions or org_classes
    palette = loader.dataset.METAINFO['palette']

    adapt_method = TENTContinual(
        ovss_type='naclip', ovss_backbone='ViT-L/14', lr=1e-5, classes=classes,
        steps=1, device=device)

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    print(f"Loaded checkpoint: round_idx={ckpt['round_idx']} cond_idx={ckpt['cond_idx']}")
    adapt_method.load_state_dict(ckpt['adapt_state'])
    adapt_method.model.to(device)

    sample = loader.dataset[args.img_idx]
    batch = custom_collate([sample])
    inputs = batch['img_patches'].to(device)
    original_gt = batch['gt'][0]
    patch_grid_shape = batch['meta']['patch_grid_shape']
    image_shapes = batch['meta']['img_shape']

    with torch.no_grad():
        patch_preds = adapt_method.evaluate(inputs)

    reconstructed = aggregate_pred_patches(
        patch_preds, patch_grid_shape, image_shapes, [224, 224], 112)
    pd = reconstructed[0].softmax(dim=0)  # [C, H, W]

    if loader.dataset.class_extensions is not None:
        ext_to_real = torch.tensor(loader.dataset.extentions_to_real_class_idx,
                                    dtype=torch.int64, device=device)
        num_cls = int(ext_to_real.max()) + 1
        one_hot = torch.nn.functional.one_hot(ext_to_real).T.view(num_cls, len(ext_to_real), 1, 1)
        pd = (pd.unsqueeze(0) * one_hot).max(1)[0]

    pd_label = pd.argmax(dim=0).cpu().numpy()
    gt_label = original_gt[0].numpy()

    CLIP_MEAN = torch.tensor([122.7709, 116.7460, 104.0937]).view(3, 1, 1)
    CLIP_STD = torch.tensor([68.5005, 66.6322, 70.3232]).view(3, 1, 1)
    img_tensor = batch['img'][0].cpu().float()
    raw_img = (img_tensor * CLIP_STD + CLIP_MEAN).clamp(0, 255).byte().permute(1, 2, 0).numpy()

    pred_colour = apply_palette(pd_label, palette)
    gt_colour = apply_palette(gt_label, palette)

    Image.fromarray(raw_img).save(os.path.join(args.out_dir, 'image.png'))
    Image.fromarray(gt_colour).save(os.path.join(args.out_dir, 'gt_mask.png'))
    Image.fromarray(pred_colour).save(os.path.join(args.out_dir, 'pred_mask.png'))
    save_legend(list(org_classes), palette, os.path.join(args.out_dir, 'legend.png'))

    # class distribution sanity print -- confirms/denies the "single class"
    # over-confidence claim numerically, not just visually
    vals, counts = np.unique(pd_label, return_counts=True)
    frac = counts / counts.sum()
    order = np.argsort(-frac)
    print("Predicted class distribution (top 5):")
    for i in order[:5]:
        cls_name = classes[vals[i]] if vals[i] < len(classes) else f"idx{vals[i]}"
        print(f"  {cls_name}: {frac[i]*100:.1f}%")

    print(f"Saved to {args.out_dir}")


if __name__ == '__main__':
    main()
