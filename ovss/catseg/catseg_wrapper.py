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

    # ----- Forward (stub -- real implementation in Task 4) -----
    def forward(self, image, text, text_ensemble=True,
                interpolate=False, **kwargs):
        raise NotImplementedError(
            "CATSegWrapper.forward is implemented in Task 4 of the plan."
        )

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
