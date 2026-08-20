"""Make a patch-shuffled version of an image for the PLPD figure.
Replicates DeYO's _destroy_object (aug_type='patch', patch_len=4):
split into ps x ps = 4x4 = 16 blocks and randomly permute them.
"""
import numpy as np
from PIL import Image

SRC = ".data/ACDC/rgb_anon/night/test/GP010364/GP010364_frame_000047_rgb_anon.png"
PS = 4          # patch_len (4x4 = 16 blocks), same as the method default
SEED = 0

img = Image.open(SRC).convert("RGB")
W, H = img.size
a = np.asarray(img)                      # (H, W, 3)

# crop so H, W are divisible by PS (method does the same via resize)
H2, W2 = (H // PS) * PS, (W // PS) * PS
a = a[:H2, :W2]

# split into PS x PS blocks -> (PS, h, PS, w, 3) -> (PS*PS, h, w, 3)
h, w = H2 // PS, W2 // PS
blocks = a.reshape(PS, h, PS, w, 3).transpose(0, 2, 1, 3, 4).reshape(PS * PS, h, w, 3)

rng = np.random.default_rng(SEED)
perm = rng.permutation(PS * PS)
blocks = blocks[perm]

# reassemble
out = blocks.reshape(PS, PS, h, w, 3).transpose(0, 2, 1, 3, 4).reshape(H2, W2, 3)

Image.fromarray(out).save("figures/patch_shuffle_night.png")
img.crop((0, 0, W2, H2)).save("figures/patch_shuffle_night_orig.png")
print("saved -> figures/patch_shuffle_night.png (shuffled)")
print("saved -> figures/patch_shuffle_night_orig.png (original, same crop)")
print(f"grid: {PS}x{PS} = {PS*PS} blocks, block size {h}x{w}, perm={perm.tolist()}")
