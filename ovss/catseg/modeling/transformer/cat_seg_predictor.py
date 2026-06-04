# Copyright (c) Facebook, Inc. and its affiliates.
# Modified for standalone use in MLMP framework (no detectron2 dependency).
import torch

from torch import nn
from torch.nn import functional as F

from .model import Aggregator
from ovss.catseg.third_party import clip
from ovss.catseg.third_party import imagenet_templates

import numpy as np


class CATSegPredictor(nn.Module):
    def __init__(
        self,
        clip_pretrained: str,
        prompt_ensemble_type: str,
        text_guidance_dim: int,
        text_guidance_proj_dim: int,
        appearance_guidance_dim: int,
        appearance_guidance_proj_dim: int,
        prompt_depth: int,
        prompt_length: int,
        decoder_dims: list,
        decoder_guidance_dims: list,
        decoder_guidance_proj_dims: list,
        num_heads: int,
        num_layers: int,
        hidden_dims: int,
        pooling_sizes: tuple,
        feature_resolution: tuple,
        window_sizes: int,
        attention_type: str,
        class_names: list = None,
        device: str = 'cpu',
    ):
        super().__init__()

        self.tokenizer = None
        # Only support OpenAI CLIP models (ViT-B/16, ViT-L/14)
        clip_model, clip_preprocess = clip.load(
            clip_pretrained, device=device, jit=False,
            prompt_depth=prompt_depth, prompt_length=prompt_length,
        )

        self.prompt_ensemble_type = prompt_ensemble_type

        if self.prompt_ensemble_type == "imagenet_select":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES_SELECT
        elif self.prompt_ensemble_type == "imagenet":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES
        elif self.prompt_ensemble_type == "single":
            prompt_templates = ['A photo of a {} in the scene']
        else:
            raise NotImplementedError(f"Unknown prompt_ensemble_type: {prompt_ensemble_type}")

        self.prompt_templates = prompt_templates

        # Pre-compute text features if class_names provided
        self.class_names = class_names
        if class_names is not None:
            self.text_features_test = self.class_embeddings(
                class_names, prompt_templates, clip_model, device
            ).permute(1, 0, 2).float()
        else:
            self.text_features_test = None

        self.clip_model = clip_model.float()
        self.clip_preprocess = clip_preprocess

        self.transformer = Aggregator(
            text_guidance_dim=text_guidance_dim,
            text_guidance_proj_dim=text_guidance_proj_dim,
            appearance_guidance_dim=appearance_guidance_dim,
            appearance_guidance_proj_dim=appearance_guidance_proj_dim,
            decoder_dims=decoder_dims,
            decoder_guidance_dims=decoder_guidance_dims,
            decoder_guidance_proj_dims=decoder_guidance_proj_dims,
            num_layers=num_layers,
            nheads=num_heads,
            hidden_dim=hidden_dims,
            pooling_size=pooling_sizes,
            feature_resolution=feature_resolution,
            window_size=window_sizes,
            attention_type=attention_type,
            prompt_channel=len(prompt_templates),
        )

        self.tokens = None
        self.cache = None

    def forward(self, x, vis_guidance):
        """
        Args:
            x: image features (B, C, H, W)
            vis_guidance: dict of appearance guidance features {'res3': ..., 'res4': ..., 'res5': ...}
        Returns:
            logits: (B, num_classes, H', W')
        """
        vis = [vis_guidance[k] for k in vis_guidance.keys()][::-1]
        text = self.get_text_embeds(self.class_names, self.prompt_templates, self.clip_model)
        text = text.repeat(x.shape[0], 1, 1, 1)
        out = self.transformer(x, text, vis)
        return out

    @torch.no_grad()
    def class_embeddings(self, classnames, templates, clip_model, device='cpu'):
        zeroshot_weights = []
        for classname in classnames:
            if ', ' in classname:
                classname_splits = classname.split(', ')
                texts = []
                for template in templates:
                    for cls_split in classname_splits:
                        texts.append(template.format(cls_split))
            else:
                texts = [template.format(classname) for template in templates]
            texts = clip.tokenize(texts).to(device)
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            if len(templates) != class_embeddings.shape[0]:
                class_embeddings = class_embeddings.reshape(
                    len(templates), -1, class_embeddings.shape[-1]
                ).mean(dim=1)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            zeroshot_weights.append(class_embeddings)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
        return zeroshot_weights

    def get_text_embeds(self, classnames, templates, clip_model, prompt=None):
        if self.cache is not None and not self.training:
            return self.cache

        if self.tokens is None or prompt is not None:
            tokens = []
            for classname in classnames:
                if ', ' in classname:
                    classname_splits = classname.split(', ')
                    texts = [template.format(classname_splits[0]) for template in templates]
                else:
                    texts = [template.format(classname) for template in templates]
                texts = clip.tokenize(texts).to(next(clip_model.parameters()).device)
                tokens.append(texts)
            tokens = torch.stack(tokens, dim=0).squeeze(1)
            if prompt is None:
                self.tokens = tokens
        elif self.tokens is not None and prompt is None:
            tokens = self.tokens

        class_embeddings = clip_model.encode_text(tokens, prompt)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        class_embeddings = class_embeddings.unsqueeze(1)

        if not self.training:
            self.cache = class_embeddings

        return class_embeddings
