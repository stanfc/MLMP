"""CATSegWrapper -- exposes a CAT-Seg backbone through the same public
interface as ovss.clip.model.CLIP, so existing adapt methods need no
changes.

Stub at Phase 1: structure-only. Real forward, aggregator, and
checkpoint loading land in subsequent tasks.
"""
import torch
import torch.nn as nn

import ovss.clip as clip


class _AggregatorStub(nn.Module):
    """Placeholder aggregator used until Task 6 lands the real extracted code.

    Identity-pass-through over the class-channel dim. Frozen, eval mode.
    """
    def __init__(self):
        super().__init__()

    def forward(self, cost_volume, guidance):
        # cost_volume: (B*T, C, H, W) -- return as-is (no refinement)
        return cost_volume


class CATSegWrapper(nn.Module):
    """CAT-Seg backbone wrapper.

    Public interface matches ovss.clip.model.CLIP:
      - forward(x, text_x, text_ensemble, interpolate) -> (logits, image_features, text_features)
      - encode_text(tokens) -> (N, dim)
      - .visual, .transformer, .ln_final, .token_embedding, .logit_scale
        are direct references to the underlying CLIP sub-modules so that
        set_ln_grads / collect_ln_params traversal works without changes.

    The cost-aggregation transformer lives in self.aggregator (sibling of
    self.visual), so LayerNorm collection scoped to self.visual naturally
    skips it -- preserving the "CLIP backbone LN only" trainable invariant.
    """

    def __init__(self, backbone='ViT-L/14', device='cpu'):
        super().__init__()

        # ---- Base CLIP backbone (reused from ovss.clip) ----
        base_clip, _ = clip.load(backbone, device='cpu')
        base_clip.visual.set_params(arch='vanilla',
                                    attn_strategy='vanilla',
                                    gaussian_std=5.0)

        # ---- Expose CLIP sub-modules as direct attributes ----
        # NOTE: assigning nn.Modules as attrs registers them as submodules.
        # That's fine; set_ln_grads walks .modules() which still works.
        self.visual = base_clip.visual
        self.transformer = base_clip.transformer
        self.ln_final = base_clip.ln_final
        self.token_embedding = base_clip.token_embedding
        self.logit_scale = base_clip.logit_scale
        self.positional_embedding = base_clip.positional_embedding
        self.text_projection = base_clip.text_projection

        # ---- Aggregator stub (replaced in Task 6 with real extracted code) ----
        self.aggregator = _AggregatorStub()
        self.aggregator.requires_grad_(False)
        self.aggregator.eval()

        # ---- Move to target device ----
        self.to(device)
        self._device = device

    # Spatial size CAT-Seg was trained at. Forward resizes input to this.
    CATSEG_INPUT_SIZE = 384
    # Patch grid is CATSEG_INPUT_SIZE / patch_size (14 for ViT-L/14) -> 27.
    CATSEG_PATCH_GRID = 27

    def forward(self, image, text, text_ensemble=True,
                interpolate=False, **kwargs):
        """Forward pass with internal 224->384 resize + cost aggregation.

        Args:
            image: (B, 3, H_in, W_in) input tensor (typically 224x224).
            text:  (T, C, D) pre-encoded class text embeddings; or (C, D) -- promoted to T=1.
                   text_ensemble=True means caller already encoded these (matches CLIP).
            interpolate: if True, upsample logits to (H_in, W_in).
                         if False, return logits at aggregator grid resolution.

        Returns:
            logits:         (T, B, C, H_out, W_out)
            image_features: (B, S, D)  -- patch features (no CLS)
            text_features:  (T, C, D)
        """
        B = image.shape[0]
        H_in, W_in = image.shape[-2], image.shape[-1]

        # ---- Promote text shape to (T, C, D) ----
        if not text_ensemble:
            # Same convention as ovss.clip.model.CLIP -- caller passed tokens.
            text_features = self.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        else:
            text_features = text
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(0)   # (1, C, D)
        T, C, D = text_features.shape

        # ---- Step 1: resize 224 -> 384 ----
        if H_in != self.CATSEG_INPUT_SIZE or W_in != self.CATSEG_INPUT_SIZE:
            image_384 = nn.functional.interpolate(
                image,
                size=(self.CATSEG_INPUT_SIZE, self.CATSEG_INPUT_SIZE),
                mode='bilinear',
                align_corners=False,
            )
        else:
            image_384 = image

        # ---- Step 2: dense visual encoding (only path with grad) ----
        visual_feats = self._encode_image_dense(image_384)   # (B, S, D)
        S = visual_feats.shape[1]
        assert S == self.CATSEG_PATCH_GRID ** 2, (
            f"Expected {self.CATSEG_PATCH_GRID**2} patches, got {S}. "
            f"Check input size and ViT patch_size."
        )

        # ---- Step 3: cost-volume ----
        visual_n = visual_feats / visual_feats.norm(dim=-1, keepdim=True)   # (B, S, D)
        text_n   = text_features / text_features.norm(dim=-1, keepdim=True)  # (T, C, D)
        # einsum: per (template, batch, class) -- inner product over D, per-patch S
        cost = torch.einsum('bsd, tcd -> tbcs', visual_n, text_n)            # (T, B, C, S)
        H_grid = W_grid = self.CATSEG_PATCH_GRID
        cost = cost.reshape(T * B, C, H_grid, W_grid)                        # (T*B, C, H, W)

        # ---- Step 4: aggregator (frozen, eval) ----
        logits_27 = self.aggregator(cost_volume=cost, guidance=visual_n)     # (T*B, C, H, W)

        # ---- Step 5: output resize + reshape ----
        if interpolate:
            logits = nn.functional.interpolate(
                logits_27,
                size=(H_in, W_in),
                mode='bilinear',
                align_corners=False,
            )
            logits = logits.view(T, B, C, H_in, W_in)
        else:
            logits = logits_27.view(T, B, C, H_grid, W_grid)

        return logits, visual_feats, text_features

    def _encode_image_dense(self, image):
        """Run CLIP visual encoder and return dense patch tokens (no CLS).

        Mirrors VisionTransformer.forward(...) but skips the aggregation
        types in ovss/clip/model.py. We always want raw dense post-LN
        features for the cost-volume step.

        Returns:
            patches: (B, num_patches, embed_dim_or_proj)
        """
        v = self.visual
        B, _, H, W = image.shape
        x = v.conv1(image)                                       # (B, D, gH, gW)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (B, gH*gW, D)
        # Prepend CLS
        cls_tok = v.class_embedding.to(x.dtype) + \
                  torch.zeros(B, 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls_tok, x], dim=1)                       # (B, 1+gH*gW, D)
        # Positional embedding (interpolates if grid size differs from training)
        if x.shape[1] != v.positional_embedding.shape[0]:
            x = x + v.interpolate_pos_encoding(x, H, W).to(x.dtype)
        else:
            x = x + v.positional_embedding.to(x.dtype)
        x = v.ln_pre(x)
        # Transformer (vanilla forward -- we set arch=vanilla in __init__)
        x = x.permute(1, 0, 2)                                   # NLD -> LND
        for blk in v.transformer.resblocks:
            x = blk(x)
        x = x.permute(1, 0, 2)                                   # LND -> NLD
        x = v.ln_post(x)
        # Drop CLS; project to output_dim
        x = x[:, 1:, :]                                          # (B, num_patches, D)
        if v.proj is not None:
            x = x @ v.proj
        return x

    # ----- Text encode (delegates to underlying CLIP) -----
    def encode_text(self, text):
        """Re-implements ovss.clip.model.CLIP.encode_text but as a method on
        the wrapper, so callers can do wrapper.encode_text(tokens) just like
        on the base CLIP."""
        x = self.token_embedding(text)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x
