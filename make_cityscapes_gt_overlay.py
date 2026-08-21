"""Cityscapes image with GT segmentation overlaid — for the talk.
Output: figures/cityscapes_gt_overlay.png (+ side-by-side and legend)
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

IMG = ".data/cityscapes/leftImg8bit/val/munster/munster_000037_000019_leftImg8bit.png"
GT  = ".data/cityscapes/gtFine/val/munster/munster_000037_000019_gtFine_labelTrainIds.png"
ALPHA = 0.55

CLASSES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
           'traffic light', 'traffic sign', 'vegetation', 'terrain',
           'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
           'motorcycle', 'bicycle']
PALETTE = [[128,64,128],[244,35,232],[70,70,70],[102,102,156],[190,153,153],
           [153,153,153],[250,170,30],[220,220,0],[107,142,35],[152,251,152],
           [70,130,180],[220,20,60],[255,0,0],[0,0,142],[0,0,70],
           [0,60,100],[0,80,100],[0,0,230],[119,11,32]]

img = np.asarray(Image.open(IMG).convert("RGB"))
gt = np.asarray(Image.open(GT))                       # trainIds, 255 = ignore

colour = np.zeros_like(img)
valid = np.zeros(gt.shape, dtype=bool)
for idx, c in enumerate(PALETTE):
    m = gt == idx
    colour[m] = c
    valid |= m

# overlay only where GT is valid; keep original pixels where ignore(255)
blend = img.copy().astype(float)
blend[valid] = (1 - ALPHA) * img[valid] + ALPHA * colour[valid]
overlay = blend.clip(0, 255).astype(np.uint8)

Image.fromarray(overlay).save("figures/cityscapes_gt_overlay.png")

# side-by-side: original | overlay
h, w = img.shape[:2]
sbs = np.concatenate([img, np.full((h, 12, 3), 255, np.uint8), overlay], axis=1)
Image.fromarray(sbs).save("figures/cityscapes_gt_sidebyside.png")

# legend
cols, sw, pad, fs = 4, 22, 8, 15
rows = (len(CLASSES) + cols - 1) // cols
cw, ch = 190, sw + pad * 2
leg = Image.new("RGB", (cw * cols, ch * rows), "white")
d = ImageDraw.Draw(leg)
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", fs)
except Exception:
    font = ImageFont.load_default()
for i, (name, col) in enumerate(zip(CLASSES, PALETTE)):
    r, c = i // cols, i % cols
    x, y = c * cw + pad, r * ch + pad
    d.rectangle([x, y, x + sw, y + sw], fill=tuple(col), outline="black")
    d.text((x + sw + 8, y + 2), name, fill="black", font=font)
leg.save("figures/cityscapes_legend.png")

print("saved -> figures/cityscapes_gt_overlay.png")
print("saved -> figures/cityscapes_gt_sidebyside.png")
print("saved -> figures/cityscapes_legend.png")
