"""Domain-shift slide (p2): one Cityscapes image + its GT, one ACDC-snow image + its GT.
Colourises the 19-class Cityscapes trainId labels; class 255 (ignore) -> black.
Outputs to figures/domain/.
"""
import os
import numpy as np
from PIL import Image

OUT = "figures/domain"
os.makedirs(OUT, exist_ok=True)

PALETTE = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32]], dtype=np.uint8)


def colourise(gt_path):
    lab = np.array(Image.open(gt_path))
    rgb = np.zeros((*lab.shape, 3), dtype=np.uint8)
    for c in range(19):
        rgb[lab == c] = PALETTE[c]
    # ignore label (255) stays black
    return Image.fromarray(rgb)


def overlay(img, gt_path, alpha=0.55):
    """Blend colourised GT onto the image; ignore label (255) keeps the raw image."""
    lab = np.array(Image.open(gt_path))
    base = np.array(img.resize(lab.shape[::-1])).astype(np.float32)
    seg = np.zeros((*lab.shape, 3), dtype=np.float32)
    valid = np.zeros(lab.shape, dtype=bool)
    for c in range(19):
        m = lab == c
        seg[m] = PALETTE[c]
        valid |= m
    out = base.copy()
    out[valid] = (1 - alpha) * base[valid] + alpha * seg[valid]
    return Image.fromarray(out.astype(np.uint8))


JOBS = [
    ("cityscapes",
     ".data/cityscapes/leftImg8bit/val/munster/munster_000132_000019_leftImg8bit.png",
     ".data/cityscapes/gtFine/val/munster/munster_000132_000019_gtFine_labelTrainIds.png"),
    ("acdc_snow",
     ".data/ACDC/rgb_anon/snow/val/GOPR0607/GOPR0607_frame_000360_rgb_anon.png",
     ".data/ACDC/gt/snow/val/GOPR0607/GOPR0607_frame_000360_gt_labelTrainIds.png"),
]

for tag, img_p, gt_p in JOBS:
    img = Image.open(img_p).convert("RGB")
    gt = colourise(gt_p)
    ov = overlay(img, gt_p)
    img.save(f"{OUT}/{tag}_image.png")
    gt.save(f"{OUT}/{tag}_gt.png")
    ov.save(f"{OUT}/{tag}_overlay.png")
    print(f"{tag}: {img.size}  ->  {tag}_image.png , {tag}_gt.png , {tag}_overlay.png")
