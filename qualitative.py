"""
Qualitative visualization script for MLMP.

For a given dataset, randomly samples N images, then for each corruption
runs MLMP adaptation and saves:
  - original image (original corruption applied)
  - GT mask (coloured)
  - prediction mask (coloured)
  - overlay: prediction mask blended onto image

Usage example:
    python qualitative.py \
        --dataset COCOStuffDataset \
        --data_dir .data/coco_stuff164k/ \
        --save_dir .qualitative/COCOStuffDataset/ \
        --n_images 5 \
        --seed 42 \
        --ovss_type naclip \
        --ovss_backbone ViT-L/14 \
        --prompt_dir prompts.yaml \
        --vision_outputs -1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18 \
        --alpha_cls 1.0 \
        --lr 0.001 \
        --steps 10
"""

import os
import argparse
import random
import math

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from adapt import get_method
from utils import segmentation_datasets
from utils.misc import set_global_seeds, aggregate_pred_patches

CORRUPTIONS = [
    "original",
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]


def save_legend(classes: list, palette: list, path: str, swatch: int = 20, cols: int = 4):
    """Save a legend image mapping class names to their palette colours."""
    n = len(classes)
    rows = math.ceil(n / cols)
    pad = 6
    font_size = 14
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    cell_w, cell_h = 200, swatch + pad * 2
    img = Image.new("RGB", (cell_w * cols, cell_h * rows + pad), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    for i, (cls_name, colour) in enumerate(zip(classes, palette)):
        row, col = divmod(i, cols)
        x, y = col * cell_w + pad, row * cell_h + pad
        draw.rectangle([x, y, x + swatch, y + swatch], fill=tuple(colour))
        draw.text((x + swatch + pad, y + 2), cls_name, fill=(0, 0, 0), font=font)

    img.save(path)


def apply_palette(mask: np.ndarray, palette: list) -> np.ndarray:
    """Map class indices to RGB colours using palette."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, colour in enumerate(palette):
        rgb[mask == cls_idx] = colour
    return rgb


def save_overlay(orig_img: np.ndarray, colour_mask: np.ndarray, path: str, alpha: float = 0.5):
    """Blend colour_mask over orig_img and save."""
    orig = orig_img.astype(np.float32)
    mask = colour_mask.astype(np.float32)
    blended = (1 - alpha) * orig + alpha * mask
    Image.fromarray(blended.clip(0, 255).astype(np.uint8)).save(path)


def get_corrupted_image(data_loader, img_idx: int) -> np.ndarray:
    """Return the corrupted (or original) image as HxWx3 uint8 numpy array."""
    dataset = data_loader.dataset
    sample = dataset[img_idx]
    # 'img' in the sample is a tensor [3, H, W] in float, normalised by CLIP mean/std.
    # We want the raw image before normalisation – read directly from path.
    img_path = sample['meta']['img_path']
    img = np.array(Image.open(img_path).convert("RGB"))
    return img


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='.qualitative/')
    parser.add_argument('--n_images', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--overlay_alpha', type=float, default=0.5)
    parser.add_argument('--legend_only', action='store_true',
                        help='Only generate the legend image and exit')

    # MLMP / model args (forwarded to get_method)
    parser.add_argument('--method', type=str, default='mlmp')
    parser.add_argument('--ovss_type', type=str, default='naclip')
    parser.add_argument('--ovss_backbone', type=str, default='ViT-L/14')
    parser.add_argument('--prompt_dir', type=str, default='prompts.yaml')
    parser.add_argument('--vision_outputs', nargs='+', type=int,
                        default=[-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18])
    parser.add_argument('--alpha_cls', type=float, default=1.0)
    parser.add_argument('--prompt_integration', type=str, default='loss')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--class_extensions', action='store_true', default=True)

    # dataloader args
    parser.add_argument('--init_resize', nargs='+', type=int, default=[224, 224])
    parser.add_argument('--patch_size', nargs='+', type=int, default=[224, 224])
    parser.add_argument('--patch_stride', type=int, default=112)
    parser.add_argument('--workers', type=int, default=0)

    # not used functionally but required by get_method
    parser.add_argument('--adapt', action='store_true', default=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--plot_loss', action='store_true', default=False)
    parser.add_argument('--runtime_calculation', action='store_true', default=False)
    parser.add_argument('--classes', default=None)
    parser.add_argument('--watt_l', type=int, default=2)
    parser.add_argument('--watt_m', type=int, default=5)
    parser.add_argument('--n_ctx', type=int, default=4)
    parser.add_argument('--pamr', action='store_true', default=False)
    parser.add_argument('--pamr_steps', type=int, default=10)
    parser.add_argument('--pamr_stride', nargs='+', type=int, default=[1])
    parser.add_argument('--alpha_pamr', type=float, default=0.1)
    parser.add_argument('--tent_alpha', type=float, default=1.0)
    parser.add_argument('--n_augmentations', type=int, default=64)
    parser.add_argument('--selection_p', type=float, default=0.1)
    return parser


def main():
    parser = argparser()
    args = parser.parse_args()

    set_global_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── pick image indices ────────────────────────────────────────────────────
    # Use the original (uncorrupted) loader just to know the dataset size
    loader_orig, org_classes = segmentation_datasets.prepare_data(
        args.dataset, args.data_dir,
        args.init_resize, args.patch_size, args.patch_stride,
        corruption="original", batch_size=1, num_workers=0, shuffle=False
    )
    n_total = len(loader_orig.dataset)
    random.seed(args.seed)
    chosen_indices = sorted(random.sample(range(n_total), min(args.n_images, n_total)))
    print(f"Chosen image indices: {chosen_indices}")

    palette = loader_orig.dataset.METAINFO['palette']
    ignore_index = loader_orig.dataset.ignore_index

    if args.class_extensions and loader_orig.dataset.class_extensions is not None:
        args.classes = loader_orig.dataset.class_extensions
    else:
        args.classes = org_classes

    num_org_classes = len(org_classes)

    # ── legend only mode ──────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    legend_path = os.path.join(args.save_dir, "legend.png")
    save_legend(list(org_classes), palette, legend_path)
    print(f"Legend saved → {legend_path}")
    if args.legend_only:
        return

    # ── iterate over corruptions ──────────────────────────────────────────────
    for corruption in CORRUPTIONS:
        print(f"\n{'='*50}\nCorruption: {corruption}\n{'='*50}")

        # build loader (shuffle=False so indices are stable)
        loader, _ = segmentation_datasets.prepare_data(
            args.dataset, args.data_dir,
            args.init_resize, args.patch_size, args.patch_stride,
            corruption=corruption, batch_size=1, num_workers=0, shuffle=False
        )

        adapt_method = get_method(args, device)

        for img_idx in chosen_indices:
            # ── get single sample via dataset __getitem__ ─────────────────────
            sample = loader.dataset[img_idx]
            img_path = sample['meta']['img_path']
            img_name = os.path.splitext(os.path.basename(img_path))[0]

            out_dir = os.path.join(args.save_dir, f"img_{img_idx:04d}_{img_name}", corruption)
            os.makedirs(out_dir, exist_ok=True)

            # collate into batch-of-1
            from utils.misc import custom_collate
            batch = custom_collate([sample])

            inputs = batch['img_patches'].to(device)
            original_gt = batch['gt'][0]           # [1, H, W]
            patch_grid_shape = batch['meta']['patch_grid_shape']
            image_shapes = batch['meta']['img_shape']

            # adapt + evaluate
            adapt_method.reset()
            adapt_method.adapt(inputs)
            with torch.no_grad():
                patch_preds = adapt_method.evaluate(inputs)

            if args.init_resize:
                reconstructed = aggregate_pred_patches(
                    patch_preds, patch_grid_shape, image_shapes,
                    args.patch_size, args.patch_stride
                )
            else:
                reconstructed = patch_preds

            pd = reconstructed[0].softmax(dim=0)   # [C, H, W]

            # class extension
            if args.class_extensions and loader.dataset.class_extensions is not None:
                ext_to_real = torch.Tensor(loader.dataset.extentions_to_real_class_idx).to(torch.int64).to(device)
                num_cls = max(ext_to_real) + 1
                one_hot = torch.nn.functional.one_hot(ext_to_real).T.view(num_cls, len(ext_to_real), 1, 1)
                pd = (pd.unsqueeze(0) * one_hot).max(1)[0]

            pd_label = pd.argmax(dim=0).cpu().numpy()   # [H, W]
            gt_label = original_gt[0].numpy()            # [H, W]

            # ── recover corrupted image from normalised tensor ────────────────
            # batch['img'] is a list of [3, H, W] tensors (CLIP normalised)
            CLIP_MEAN = torch.tensor([122.7709, 116.7460, 104.0937]).view(3, 1, 1)
            CLIP_STD  = torch.tensor([68.5005,  66.6322,  70.3232 ]).view(3, 1, 1)
            img_tensor = batch['img'][0].cpu().float()   # [3, H, W]
            raw_img_resized = (img_tensor * CLIP_STD + CLIP_MEAN).clamp(0, 255).byte()
            raw_img_resized = raw_img_resized.permute(1, 2, 0).numpy()  # [H, W, 3]

            # ── colourise masks ───────────────────────────────────────────────
            pred_colour = apply_palette(pd_label, palette)
            gt_colour   = apply_palette(gt_label, palette)

            # ── save ──────────────────────────────────────────────────────────
            Image.fromarray(raw_img_resized).save(os.path.join(out_dir, "image.png"))
            Image.fromarray(gt_colour).save(os.path.join(out_dir, "gt_mask.png"))
            Image.fromarray(pred_colour).save(os.path.join(out_dir, "pred_mask.png"))
            save_overlay(raw_img_resized, pred_colour, os.path.join(out_dir, "pred_overlay.png"),
                         alpha=args.overlay_alpha)
            save_overlay(raw_img_resized, gt_colour, os.path.join(out_dir, "gt_overlay.png"),
                         alpha=args.overlay_alpha)

            print(f"  [{img_idx}] saved → {out_dir}")

        del adapt_method
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
