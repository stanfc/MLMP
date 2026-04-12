import math
import torch
import torch.nn as nn
from functools import reduce
from operator import mul


class PromptVisualEncoder(nn.Module):
    '''
    NA-CLIP VisionTransformer with added prompts at the input layer.
    Mirrors PromptViT from DPCore but wraps VisionTransformer instead of timm ViT.
    '''
    def __init__(self, vit, num_prompts=1):
        super().__init__()
        self.vit = vit
        self.num_prompts = num_prompts
        self.prompt_dim = vit.conv1.out_channels  # width

        if num_prompts > 0:
            self.prompts = nn.Parameter(torch.zeros(1, num_prompts, self.prompt_dim))
            # zero init: keeps prompted features close to raw at the start so
            # the discrepancy loss baseline is meaningful before any optimization

    def reset(self):
        nn.init.zeros_(self.prompts.data)

    def prompt_injection(self, x):
        """Insert prompt tokens after CLS token, before patch tokens. x: (B, L, D) in NLD format."""
        if self.num_prompts > 0:
            prompts = self.prompts.to(device=x.device, dtype=x.dtype).expand(x.shape[0], -1, -1)
            x = torch.cat((
                x[:, :1, :],   # CLS token
                prompts,        # prompt tokens
                x[:, 1:, :]    # patch tokens
            ), dim=1)
        return x

    def forward_features(self, x):
        '''
        Forward with prompts injected after CLS token (before patch tokens).
        Used by DPCore's discrepancy loss. Returns (B, L, output_dim).
        Mirrors PromptViT.forward_features().
        '''
        x = x.type(self.vit.conv1.weight.dtype)
        vit = self.vit
        B, nc, w, h = x.shape
        n_patches = (w // vit.patch_size, h // vit.patch_size)

        x = vit.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # (B, grid^2, width)
        x = torch.cat([
            vit.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)  # (B, grid^2+1, width)

        if x.shape[1] != vit.positional_embedding.shape[0]:
            x = x + vit.interpolate_pos_encoding(x, w, h).to(x.dtype)
        else:
            x = x + vit.positional_embedding.to(x.dtype)

        # inject prompts (in NLD format)
        x = self.prompt_injection(x)

        x = vit.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        num_layers = len(vit.transformer.resblocks)
        last_layer_idx = num_layers - 1
        for idx, blk in enumerate(vit.transformer.resblocks):
            if idx != last_layer_idx:
                reduced = vit.custom_attn("vanilla", blk.attn, blk.ln_1(x), n_patches)
                x = x + reduced
                x = x + blk.mlp(blk.ln_2(x))
            else:
                # Remove prompt tokens before NA-CLIP attention (omega shape depends on n_patches)
                x_no_prompt = torch.cat([x[:1, :, :], x[1 + self.num_prompts:, :, :]], dim=0)
                reduced = vit.custom_attn(vit.attn_strategy, blk.attn, blk.ln_1(x_no_prompt), n_patches)
                final_x = x_no_prompt + reduced
                final_x = final_x + blk.mlp(blk.ln_2(final_x))
                # Match raw VisionTransformer.forward(): arch='reduced' returns reduced, 'vanilla' returns final_x
                x = reduced if vit.arch == 'reduced' else final_x

        x = x.permute(1, 0, 2)  # LND -> NLD
        return vit.ln_post(x) @ vit.proj

    def forward_raw_features(self, x):
        '''
        Forward without prompts. Mirrors PromptViT.forward_raw_features().
        Delegates directly to the wrapped VisionTransformer.
        '''
        return self.vit(x.type(self.vit.conv1.weight.dtype), output_layers=(-1,), out_type="mean")

    def forward(self, x, output_layers=(-1,), out_type="mean",
                return_vanilla_cls=False, weights=None):
        '''
        Full forward pass with prompts, matching VisionTransformer.forward() signature
        so that model(x, text_x, ...) still works via CLIP's encode_image path.
        Replicates VisionTransformer.forward() with prompt injection added.
        '''
        vit = self.vit
        B, nc, w, h = x.shape
        n_patches = (w // vit.patch_size, h // vit.patch_size)

        x = vit.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([
            vit.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x
        ], dim=1)

        if x.shape[1] != vit.positional_embedding.shape[0]:
            x = x + vit.interpolate_pos_encoding(x, w, h).to(x.dtype)
        else:
            x = x + vit.positional_embedding.to(x.dtype)

        # inject prompts
        x = self.prompt_injection(x)

        x = vit.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        num_layers = len(vit.transformer.resblocks)
        output_layers = tuple(num_layers + idx if idx < 0 else idx for idx in output_layers)
        last_layer_idx = max(output_layers)

        out_features = []
        for idx, blk in enumerate(vit.transformer.resblocks[:last_layer_idx + 1]):
            if idx != last_layer_idx:
                reduced = vit.custom_attn("vanilla", blk.attn, blk.ln_1(x), n_patches)
                x = x + reduced
                x = x + blk.mlp(blk.ln_2(x))
            else:
                # Remove prompt tokens before NA-CLIP attention (omega shape depends on n_patches)
                x_no_prompt = torch.cat([x[:1, :, :], x[1 + self.num_prompts:, :, :]], dim=0)
                reduced = vit.custom_attn(vit.attn_strategy, blk.attn, blk.ln_1(x_no_prompt), n_patches)
                final_x = x_no_prompt + reduced
                final_x = final_x + blk.mlp(blk.ln_2(final_x))
                if vit.attn_strategy != 'vanilla' and return_vanilla_cls:
                    vanilla_cls = blk(x_no_prompt)[0]

            if idx in output_layers:
                if vit.arch == 'reduced':
                    out_features.append(reduced)
                elif vit.arch == 'vanilla':
                    out_features.append(final_x)

        if out_type == "mean":
            if len(out_features) > 1:
                x = torch.mean(torch.stack(out_features), dim=0)
            else:
                x = out_features[0]
            x = x.permute(1, 0, 2)  # LND -> NLD
            if return_vanilla_cls:
                return vit.ln_post(x) @ vit.proj, vit.ln_post(vanilla_cls) @ vit.proj
            else:
                return vit.ln_post(x) @ vit.proj

        elif out_type == "all":
            out_features = [f.permute(1, 0, 2) for f in out_features]
            out_features = [vit.ln_post(f) @ vit.proj for f in out_features]
            out_features = torch.stack(out_features, dim=0)
            if return_vanilla_cls:
                return out_features, vit.ln_post(vanilla_cls) @ vit.proj
            else:
                return out_features

        elif out_type == "weighted_mean":
            x = torch.stack(out_features, dim=0)
            x = torch.sum(x * weights.unsqueeze(1).unsqueeze(-1), dim=0)
            x = x.permute(1, 0, 2)
            return vit.ln_post(x) @ vit.proj

    # Delegate all other VisionTransformer attributes/methods to self.vit
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.vit, name)
