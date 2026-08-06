"""Real source-model (NA-CLIP, no TTA) inference on the ACDC-snow image used on the
domain-shift slide, saving a prediction overlay. Shows how the source model degrades
on the target domain.

Output: figures/domain/acdc_snow_pred_overlay.png  (+ pred_mask.png)
"""
import os
import argparse
import numpy as np
import torch
from PIL import Image

from adapt import get_method
from utils import segmentation_datasets
from utils.misc import set_global_seeds, aggregate_pred_patches, custom_collate

TARGET = "GOPR0607_frame_000360"   # same image as the GT overlay
DATA_DIR = ".data/ACDC/"
INIT_RESIZE = [1120, 560]
PATCH, STRIDE = [224, 224], 112
OUT = "figures/domain"

PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32]], dtype=np.uint8)


def build_args():
    a = argparse.Namespace()
    a.method = "tent_continual"          # frozen NA-CLIP; we only call evaluate()
    a.ovss_type = "naclip"; a.ovss_backbone = "ViT-L/14"
    a.prompt_dir = "prompts.yaml"
    a.vision_outputs = list(range(-1, -19, -1))
    a.alpha_cls = 1.0; a.prompt_integration = "loss"
    a.lr = 1e-5; a.steps = 1
    a.classes = None; a.class_extensions = True
    a.adapt = False
    return a


def main():
    set_global_seeds(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = build_args()

    loader, org_classes = segmentation_datasets.prepare_data(
        "ACDCDataset", DATA_DIR, INIT_RESIZE, PATCH, STRIDE,
        corruption="snow", batch_size=1, num_workers=0, shuffle=False)
    ds = loader.dataset
    if args.class_extensions and ds.class_extensions is not None:
        args.classes = ds.class_extensions
    else:
        args.classes = org_classes

    # locate the target image
    idx = next(i for i in range(len(ds))
               if TARGET in os.path.basename(ds[i]["meta"]["img_path"]))
    print(f"target index = {idx}  ({os.path.basename(ds[idx]['meta']['img_path'])})")

    method = get_method(args, device)
    sample = ds[idx]
    batch = custom_collate([sample])
    inputs = batch["img_patches"].to(device)

    with torch.no_grad():
        patch_preds = method.evaluate(inputs)          # source model, NO adaptation
    recon = aggregate_pred_patches(patch_preds, batch["meta"]["patch_grid_shape"],
                                   batch["meta"]["img_shape"], PATCH, STRIDE)
    pd = recon[0].softmax(dim=0)

    # collapse class extensions back to the 19 real classes
    if ds.class_extensions is not None:
        ext = torch.tensor(ds.extentions_to_real_class_idx, dtype=torch.int64, device=device)
        ncls = int(ext.max()) + 1
        one_hot = torch.nn.functional.one_hot(ext).T.view(ncls, len(ext), 1, 1)
        pd = (pd.unsqueeze(0) * one_hot).max(1)[0]
    pred = pd.argmax(0).cpu().numpy()

    # colourise + overlay onto the full-res original image
    seg = PALETTE[pred]
    base = np.array(Image.open(ds[idx]["meta"]["img_path"]).convert("RGB"))
    seg_full = np.array(Image.fromarray(seg).resize(base.shape[1::-1], Image.NEAREST)).astype(np.float32)
    ov = (0.45 * base.astype(np.float32) + 0.55 * seg_full).clip(0, 255).astype(np.uint8)

    os.makedirs(OUT, exist_ok=True)
    Image.fromarray(seg).save(f"{OUT}/acdc_snow_pred_mask.png")
    Image.fromarray(ov).save(f"{OUT}/acdc_snow_pred_overlay.png")
    print(f"saved -> {OUT}/acdc_snow_pred_overlay.png")


if __name__ == "__main__":
    main()
