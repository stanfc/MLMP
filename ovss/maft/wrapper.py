"""Wraps MAFT+ (Mask2Former-style, open-vocabulary, ConvNeXt-CLIP backbone,
arxiv 2408.00744, github.com/jiaosiyu1999/MAFT-Plus) behind the same
model(image, text, text_ensemble, vision_outputs=..., vision_out_type=...,
save_weights=...) -> (logits, _, _) interface that deyo_mlmp_adagate_continual.py
and friends expect from a plain-CLIP ovss_model (see ovss/clip/model.py).

Design decisions (per the 3-way call on this):
  1. UAML's multi-layer weighting is bypassed entirely. MAFT+'s ConvNeXt
     backbone doesn't have the same "N uniform transformer layers" structure
     ViT does (stage outputs differ in spatial/channel size), so there's no
     clean way to reuse the vision_outputs / adaptive_weighted_mean machinery.
     MAFT+ is treated as a black box: one full forward through its own
     mask-decoder + open-vocab classifier + semantic_inference gives a single
     dense (C, H, W) per-pixel logit map, which is all the DeYO/TENT-style
     losses actually need.
  2. Only the ConvNeXt backbone's LayerNorm/LayerNorm2d modules are meant to
     be adapted. Verified both `timm.layers.norm.LayerNorm2d` and
     `timm.layers.norm.LayerNorm` subclass `torch.nn.LayerNorm`, so the
     existing `isinstance(m, nn.LayerNorm)` collect_ln_params/set_ln_grads
     code (deyo_mlmp_adagate_continual.py) picks them up with zero changes --
     just point `.visual` at MAFT+'s backbone submodule.
  3. No changes to MLMP's own image preprocessing: its CLIP_MEAN/CLIP_STD
     (segmentation_datasets.py) are numerically identical to MAFT+'s own
     PIXEL_MEAN/PIXEL_STD (both are the standard 0-255-scale OpenAI-CLIP
     stats) -- so the incoming tensor is already normalized the way MAFT+
     expects. We reconstruct raw 0-255 pixels and feed them through MAFT+'s
     own unmodified forward() (which re-applies the identical normalization)
     rather than duplicating/risking a subtly different reimplementation.
"""
import os
import sys

import torch
import torch.nn as nn

_MAFT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '.external', 'MAFT-Plus'))
if _MAFT_ROOT not in sys.path:
    sys.path.insert(0, _MAFT_ROOT)

from detectron2.config import get_cfg  # noqa: E402
from detectron2.data import MetadataCatalog  # noqa: E402
from detectron2.modeling import build_model  # noqa: E402
from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
from detectron2.projects.deeplab import add_deeplab_config  # noqa: E402

# maft's dataset-registration submodules read a few label files via relative
# paths ('./maft/data/...'), assuming cwd == the MAFT-Plus repo root -- true
# only when running its own train_net.py directly. Chdir there just for this
# one import so those reads succeed, then restore our real cwd.
_cwd = os.getcwd()
os.chdir(_MAFT_ROOT)
try:
    from maft import add_maskformer2_config, add_fcclip_config  # noqa: E402
finally:
    os.chdir(_cwd)

CLIP_MEAN = torch.tensor([122.7709, 116.7460, 104.0937]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([68.5005, 66.6322, 70.3232]).view(1, 3, 1, 1)

_DATASET_NAME = "ovss_tta_custom_dataset"


class MAFTWrapper(nn.Module):
    def __init__(self, classes, checkpoint_path, device='cuda',
                 config_file=None):
        super().__init__()
        if config_file is None:
            config_file = os.path.join(
                _MAFT_ROOT, 'configs', 'semantic', 'train_semantic_large.yaml')

        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        add_fcclip_config(cfg)
        cfg.merge_from_file(config_file)
        cfg.DATASETS.TEST = (_DATASET_NAME,)
        cfg.MODEL.WEIGHTS = checkpoint_path
        cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
        cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
        cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
        cfg.freeze()

        MetadataCatalog.get(_DATASET_NAME).set(stuff_classes=list(classes))

        self.inner = build_model(cfg)
        DetectionCheckpointer(self.inner).load(checkpoint_path)
        self.inner.to(device)
        self.inner.eval()  # inference path (semantic_inference), not the
        # loss/matcher path -- TTA still backprops fine in eval mode since
        # autograd doesn't care, only BN/Dropout would (ConvNeXt has neither
        # here, just LayerNorm).
        self.device_ = device

        # no-op placeholders: DeYO+MLMP method code does
        # `self.model.transformer.requires_grad_(False)` etc. before wiring
        # up LayerNorm-only adaptation. MAFT+ has no equivalent submodules.
        self.transformer = nn.Identity()
        self.ln_final = nn.Identity()
        self.token_embedding = nn.Identity()
        # exposes the real nn.LayerNorm/LayerNorm2d modules for adaptation
        self.visual = self.inner.backbone

    def encode_text(self, texts):
        # MAFT+ builds its own text classifier internally from the class list
        # registered in MetadataCatalog at construction time; this is a stub
        # so generic `self.model.encode_text(...)` call sites don't crash.
        return torch.zeros(len(texts), 1, device=self.device_)

    def forward(self, image, text=None, text_ensemble=None, vision_outputs=None,
                interpolate=None, vision_out_type=None, save_weights=None,
                return_vanilla_cls=False, return_all_cls=False, **kw):
        device = image.device
        raw = image * CLIP_STD.to(device) + CLIP_MEAN.to(device)  # undo MLMP's normalize
        B, _, H, W = raw.shape
        batched_inputs = [{
            "image": raw[i],
            "height": H,
            "width": W,
            "file_name": f"tta_frame_{i}.png",
            "meta": {"dataname": _DATASET_NAME},
        } for i in range(B)]

        results = self.inner(batched_inputs)
        logits = torch.stack([r["sem_seg"] for r in results], dim=0)  # (B, C, H, W)
        # plain-CLIP's forward() returns an extra leading dim (one entry per
        # UAML output-layer combination mode); callers like `evaluate()` do
        # `logits = self.model(...)[0][0]`-style unwrapping (`logits[0]`) to
        # drop it. We only ever have one "layer" (decision 1: no multi-layer
        # weighting), so add that dim back purely to match the call site's
        # expected shape -- it is not a real ensemble.
        logits = logits.unsqueeze(0)
        if return_vanilla_cls or return_all_cls:
            # MAFT+ has no CLS-token/ILE concept (that's a plain-ViT idea);
            # this placeholder only matters if a caller sets alpha_cls > 0,
            # which we don't for MAFT+ runs (--alpha_cls 0.0).
            return logits, None, None, logits
        return logits, None, None
