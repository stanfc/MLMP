"""
CATSegWrapper: wraps the CAT-Seg model (CLIP + Aggregator + Decoder) so that
its forward() signature matches what the MLMP adaptation framework expects.

MLMP expects:
    model.forward(image, text_features, text_ensemble, interpolate, vision_outputs, ...)
    -> (logits, image_features, text_features)
    where logits shape = (#templates, B, #classes, H, W)

CAT-Seg internally:
    CLIP.encode_image(dense=True) -> (B, HW+1, C)
    Aggregator(img_feat, text, guidance) -> (B, #classes, H', W')

This wrapper bridges the two.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from ovss.catseg.modeling.heads.cat_seg_head import CATSegHead
from ovss.catseg.modeling.transformer.cat_seg_predictor import CATSegPredictor


class CATSegWrapper(nn.Module):
    """
    Wraps CAT-Seg into the MLMP model interface.

    Attributes:
        clip_model:  The CLIP ViT model (from CAT-Seg's third_party).
        sem_seg_head: CATSegHead containing the predictor (Aggregator).
        upsample1/2: ConvTranspose2d for intermediate CLIP layer upsampling.
        layer_indexes: Which CLIP transformer layers to hook for guidance.
    """

    def __init__(
        self,
        clip_pretrained: str,
        class_names: list,
        prompt_ensemble_type: str = "single",
        prompt_depth: int = 0,
        prompt_length: int = 0,
        # Aggregator params (defaults from vitb_384.yaml)
        text_guidance_dim: int = 512,
        text_guidance_proj_dim: int = 128,
        appearance_guidance_dim: int = 512,
        appearance_guidance_proj_dim: int = 128,
        decoder_dims: list = None,
        decoder_guidance_dims: list = None,
        decoder_guidance_proj_dims: list = None,
        num_heads: int = 4,
        num_layers: int = 2,
        hidden_dims: int = 128,
        pooling_sizes: tuple = (2, 2),
        feature_resolution: tuple = (24, 24),
        window_sizes: int = 12,
        attention_type: str = "linear",
        device: str = "cpu",
    ):
        super().__init__()

        if decoder_dims is None:
            decoder_dims = [64, 32]
        if decoder_guidance_dims is None:
            decoder_guidance_dims = [256, 128]
        if decoder_guidance_proj_dims is None:
            decoder_guidance_proj_dims = [32, 16]

        # Build predictor (contains CLIP model + Aggregator)
        predictor = CATSegPredictor(
            clip_pretrained=clip_pretrained,
            prompt_ensemble_type=prompt_ensemble_type,
            text_guidance_dim=text_guidance_dim,
            text_guidance_proj_dim=text_guidance_proj_dim,
            appearance_guidance_dim=appearance_guidance_dim,
            appearance_guidance_proj_dim=appearance_guidance_proj_dim,
            prompt_depth=prompt_depth,
            prompt_length=prompt_length,
            decoder_dims=decoder_dims,
            decoder_guidance_dims=decoder_guidance_dims,
            decoder_guidance_proj_dims=decoder_guidance_proj_dims,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dims=hidden_dims,
            pooling_sizes=pooling_sizes,
            feature_resolution=feature_resolution,
            window_sizes=window_sizes,
            attention_type=attention_type,
            class_names=class_names,
            device=device,
        )

        # Build head
        self.sem_seg_head = CATSegHead(
            num_classes=len(class_names),
            feature_resolution=list(feature_resolution),
            predictor=predictor,
        )

        # CLIP model reference (used for encode_image, encode_text)
        self.clip_model = predictor.clip_model

        # Upsampling layers for intermediate CLIP features (appearance guidance)
        self.proj_dim = 768 if clip_pretrained == "ViT-B/16" else 1024
        self.upsample1 = nn.ConvTranspose2d(self.proj_dim, 256, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(self.proj_dim, 128, kernel_size=4, stride=4)

        # Hook into CLIP intermediate layers for guidance features
        self.layer_indexes = [3, 7] if clip_pretrained == "ViT-B/16" else [7, 15]
        self.layers = []
        for l in self.layer_indexes:
            self.clip_model.visual.transformer.resblocks[l].register_forward_hook(
                lambda m, _, o: self.layers.append(o)
            )

        # CLIP normalization constants
        self.clip_pixel_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        self.clip_pixel_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

        # Input resolution for CLIP
        self.clip_resolution = (384, 384) if clip_pretrained == "ViT-B/16" else (336, 336)

        self.feature_resolution = feature_resolution

        # Compatibility with MLMP main.py which checks model.weights_track
        self.weights_track = []

        self.to(device)

    def forward(self, image, text_features=None, text_ensemble=False,
                interpolate=False, vision_outputs=(-1,),
                return_vanilla_cls=False, vision_out_type="mean",
                save_weights=False, **kwargs):
        """
        MLMP-compatible forward pass.

        Args:
            image: (B, 3, H, W) raw image tensor (already in [0, 1] range or
                   preprocessed by MLMP's dataloader).
            text_features: ignored (CAT-Seg manages its own text features).
            text_ensemble: ignored.
            interpolate: whether to upsample logits to input resolution.
            vision_outputs: ignored (CAT-Seg uses fixed intermediate layers).
            return_vanilla_cls: if True, returns a dummy cls logit for compatibility.
            vision_out_type: ignored.
            save_weights: ignored.

        Returns:
            logits: (#templates=1, B, #classes, H, W)
            image_features: (B, HW+1, C) CLIP dense features
            text_features: None (managed internally)
            [optional] vanilla_cls_logits if return_vanilla_cls
        """
        B = image.shape[0]

        # Normalize for CLIP
        clip_pixel_mean = self.clip_pixel_mean.to(image.device, image.dtype)
        clip_pixel_std = self.clip_pixel_std.to(image.device, image.dtype)

        clip_images = (image - clip_pixel_mean) / clip_pixel_std
        clip_images_resized = F.interpolate(
            clip_images, size=self.clip_resolution,
            mode='bilinear', align_corners=False,
        )

        # Clear hooks
        self.layers = []

        # CLIP encode_image (dense mode) -> (B, HW+1, C)
        clip_features = self.clip_model.encode_image(clip_images_resized, dense=True)

        # Build appearance guidance from intermediate layers + final features
        image_features = clip_features[:, 1:, :]
        res3 = rearrange(image_features, "B (H W) C -> B C H W",
                         H=self.feature_resolution[0])
        res4 = rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W",
                         H=self.feature_resolution[0])
        res5 = rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W",
                         H=self.feature_resolution[0])
        res4 = self.upsample1(res4)
        res5 = self.upsample2(res5)
        features = {'res5': res5, 'res4': res4, 'res3': res3}

        # Run segmentation head (predictor + aggregator)
        logits = self.sem_seg_head(clip_features, features)  # (B, #classes, H', W')

        # Upsample to input resolution if requested
        if interpolate:
            target_size = image.shape[-2:]
            logits = F.interpolate(logits, size=target_size, mode='bilinear',
                                   align_corners=False)

        # Add template dimension for MLMP compatibility: (1, B, #classes, H, W)
        logits = logits.unsqueeze(0)

        if return_vanilla_cls:
            # Dummy cls logit for compatibility
            dummy_cls = logits.mean(dim=(-2, -1))  # (1, B, #classes)
            return logits, clip_features, None, dummy_cls

        return logits, clip_features, None

    def encode_text(self, text):
        """Delegate to internal CLIP model."""
        return self.clip_model.encode_text(text)

    @property
    def visual(self):
        """Expose CLIP's visual encoder for LayerNorm parameter collection."""
        return self.clip_model.visual

    @property
    def transformer(self):
        """Expose CLIP's text transformer (frozen in TTA)."""
        return self.clip_model.transformer

    @property
    def ln_final(self):
        """Expose CLIP's final LayerNorm (frozen in TTA)."""
        return self.clip_model.ln_final

    @property
    def token_embedding(self):
        """Expose CLIP's token embedding (frozen in TTA)."""
        return self.clip_model.token_embedding

    @property
    def logit_scale(self):
        """Expose CLIP's logit scale."""
        return self.clip_model.logit_scale

    @property
    def aggregator(self):
        """Expose the Aggregator module for TTA parameter collection."""
        return self.sem_seg_head.predictor.transformer

    @property
    def patch_size(self):
        """Expose patch_size for compatibility."""
        return self.clip_model.visual.patch_size
