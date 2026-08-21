"""ACDC image with NA-CLIP (frozen, zero-shot) inference overlaid — for the talk.
No adaptation: just NA-CLIP + UAML evaluate on the frozen model.
Output: figures/acdc_naclip_overlay.png (+ side-by-side)
"""
import os
import argparse
import numpy as np
import torch
from PIL import Image

from adapt import get_method
from utils import segmentation_datasets
from utils.misc import aggregate_pred_patches, custom_collate

CONDITION = "night"          # fog / night / rain / snow
WANT = "GP010364_frame_000047"   # substring to match; else first sample
ALPHA = 0.55
device = "cuda" if torch.cuda.is_available() else "cpu"

PALETTE = [[128,64,128],[244,35,232],[70,70,70],[102,102,156],[190,153,153],
           [153,153,153],[250,170,30],[220,220,0],[107,142,35],[152,251,152],
           [70,130,180],[220,20,60],[255,0,0],[0,0,142],[0,0,70],
           [0,60,100],[0,80,100],[0,0,230],[119,11,32]]


def apply_palette(mask):
    out = np.zeros((*mask.shape, 3), np.uint8)
    for i, c in enumerate(PALETTE):
        out[mask == i] = c
    return out


loader, classes = segmentation_datasets.prepare_data(
    "ACDCDataset", ".data/ACDC/", [1120, 560], [224, 224], 112,
    corruption=CONDITION, batch_size=1, num_workers=0, shuffle=False)

# pick the sample
idx = 0
for i in range(len(loader.dataset)):
    if WANT in loader.dataset[i]["meta"]["img_path"]:
        idx = i; break
sample = loader.dataset[idx]
print("using:", sample["meta"]["img_path"])

# build the frozen NA-CLIP method (we only call evaluate -> no adaptation)
args = argparse.Namespace(
    method="deyo_mlmp_hmgate2_continual", ovss_type="naclip", ovss_backbone="ViT-L/14",
    prompt_dir="prompts.yaml", lr=5e-6, steps=1,
    vision_outputs=list(range(-1, -19, -1)), classes=classes, save_dir=None,
    runtime_calculation=False)
method = get_method(args, device)

batch = custom_collate([sample])
inputs = batch["img_patches"].to(device)
patch_grid_shape = batch["meta"]["patch_grid_shape"]
image_shapes = batch["meta"]["img_shape"]

with torch.no_grad():
    patch_preds = method.evaluate(inputs)            # NA-CLIP zero-shot, NO adapt
    reconstructed = aggregate_pred_patches(
        patch_preds, patch_grid_shape, image_shapes, [224, 224], 112)
    pd = reconstructed[0].softmax(dim=0)             # (C, H, W)
    # collapse class-extensions -> real classes
    if loader.dataset.class_extensions is not None:
        e = torch.tensor(loader.dataset.extentions_to_real_class_idx, dtype=torch.int64, device=device)
        ncls = int(e.max()) + 1
        oh = torch.nn.functional.one_hot(e).T.view(ncls, len(e), 1, 1)
        pd = (pd.unsqueeze(0) * oh).max(1)[0]
    pred = pd.argmax(dim=0).cpu().numpy()            # (H, W)

# recover the (resized) image from the normalised tensor
MEAN = torch.tensor([122.7709, 116.7460, 104.0937]).view(3, 1, 1)
STD = torch.tensor([68.5005, 66.6322, 70.3232]).view(3, 1, 1)
img = (batch["img"][0].cpu().float() * STD + MEAN).clamp(0, 255).byte()
img = img.permute(1, 2, 0).numpy()                   # (H, W, 3)

colour = apply_palette(pred)
overlay = ((1 - ALPHA) * img + ALPHA * colour).clip(0, 255).astype(np.uint8)

os.makedirs("figures", exist_ok=True)
Image.fromarray(overlay).save("figures/acdc_naclip_overlay.png")
sbs = np.concatenate([img, np.full((img.shape[0], 12, 3), 255, np.uint8), overlay], axis=1)
Image.fromarray(sbs).save("figures/acdc_naclip_sidebyside.png")
print("saved -> figures/acdc_naclip_overlay.png")
print("saved -> figures/acdc_naclip_sidebyside.png")
