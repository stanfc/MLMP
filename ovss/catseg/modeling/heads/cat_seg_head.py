# Copyright (c) Facebook, Inc. and its affiliates.
# Modified for standalone use in MLMP framework (no detectron2 dependency).
from einops import rearrange

from torch import nn
from torch.nn import functional as F

from ..transformer.cat_seg_predictor import CATSegPredictor


class CATSegHead(nn.Module):
    def __init__(
        self,
        num_classes: int,
        feature_resolution: list,
        predictor: CATSegPredictor,
        ignore_value: int = -1,
    ):
        super().__init__()
        self.ignore_value = ignore_value
        self.predictor = predictor
        self.num_classes = num_classes
        self.feature_resolution = feature_resolution

    def forward(self, features, guidance_features):
        """
        Args:
            features: CLIP dense features (B, HW+1, C) including CLS token
            guidance_features: dict of {'res3': ..., 'res4': ..., 'res5': ...}
        Returns:
            logits: (B, num_classes, H', W')
        """
        img_feat = rearrange(
            features[:, 1:, :],
            "b (h w) c -> b c h w",
            h=self.feature_resolution[0],
            w=self.feature_resolution[1],
        )
        return self.predictor(img_feat, guidance_features)
