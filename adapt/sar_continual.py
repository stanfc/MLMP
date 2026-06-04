"""
SARContinual — Sharpness-Aware Reliable TTA (Niu et al., ICLR 2023 oral)
ported from https://github.com/mr-eggplant/SAR to the MLMP / NA-CLIP /
per-pixel-OVSS setting.

Three SAR mechanisms (paper §3):
  1. Reliable sample selection: keep only pixels with
     entropy < margin = e0_factor * log(C).
  2. Sharpness-Aware Minimization (SAM): two-step optimizer that
     first climbs to w + e(w) and then descends from there, so the
     final step lands in a flatter minimum that's less sensitive to
     test-time noise.
  3. Model recovery: an EMA of the post-climb loss; if it falls below
     reset_constant_em (=0.2 in paper), reload the source weights.

Compatibility with our CTTA hard rules:
  - No *per-sample* reset. The collapse-triggered recovery is the only
    reset path and fires from a global EMA, not per batch.
  - The continual_adapt() entry never calls self.reset() proactively;
    it is only invoked from inside perform_adaptation when the EMA
    threshold is crossed.

Deviations from upstream:
  - Base optimizer changed from SGD+momentum to Adam (MLMP convention).
    SAM wraps the chosen base optimizer; the algorithm is unchanged.
  - Reliable filter operates per-pixel (not per-sample) because we
    do dense prediction, not classification.
  - Top blocks excluded from adaptation: paper excludes the last 3 of
    12 ViT-Base blocks (blocks.9/10/11). For NA-CLIP ViT-L/14 (24
    blocks) we exclude the last 6 (resblocks.18..23) plus ln_post.

Hyperparameters (paper defaults, retained):
  - margin_e0_factor = 0.4  -> margin = 0.4 * log(num_classes)
  - reset_constant_em = 0.2 -> EMA threshold for model recovery
  - rho = 0.05              -> SAM neighborhood radius
"""
import time
import copy
import math
import re

import torch
import torch.nn as nn
import torch.optim as optim

from ovss import load_ovss
from utils.misc import print_clip_parameters, print_optimizer_parameters
from .sam import SAM

REFERENCE_PROMPT = 'a photo of a {}'


class SARContinual:
    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 margin_e0_factor=0.4,
                 reset_constant_em=0.2,
                 sar_rho=0.05,
                 top_block_exclude=6,
                 runtime_calculation=False, device='cpu'):
        """
        Args:
            margin_e0_factor: entropy filter margin = factor * log(C).
            reset_constant_em: EMA threshold below which we reset to source.
            sar_rho: SAM neighborhood radius (Eqn. 4 of paper).
            top_block_exclude: number of trailing transformer blocks whose
                LN params are NOT adapted (also excludes ln_post). For
                ViT-L/14 (24 blocks) the proportional match to the
                paper's ViT-Base setting is 6.
        """
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps
        self.runtime = runtime_calculation
        self.device = device

        self.margin = margin_e0_factor * math.log(len(classes))
        self.reset_constant_em = reset_constant_em
        self.sar_rho = sar_rho
        self.top_block_exclude = top_block_exclude
        self.ema = None
        self.total_batches = 0
        self.total_resets = 0

        if classes is None:
            raise ValueError("classes is required in __init__")
        self.classes = classes

        self.model, self.tokenize = load_ovss(ovss_type, ovss_backbone, device=device)

        self.prompt_templates = [REFERENCE_PROMPT]

        self.model.transformer.requires_grad_(False)
        self.model.ln_final.requires_grad_(False)
        self.model.token_embedding.requires_grad_(False)
        self.model.visual = self.set_ln_grads(
            self.model.visual, top_block_exclude=self.top_block_exclude)
        params, _ = self.collect_ln_params(
            self.model.visual, top_block_exclude=self.top_block_exclude)

        print_clip_parameters(self.model)
        print(f"+++ SAR-Continual: margin={self.margin:.3f} "
              f"(={margin_e0_factor}*log({len(classes)})), "
              f"reset_em={self.reset_constant_em}, rho={self.sar_rho}, "
              f"top_block_exclude={self.top_block_exclude} "
              f"-> trainable LN params: {len(params)}")

        # SAM wrapping Adam (deviation from paper's SGD+momentum; SAM
        # mechanism unchanged).
        self.optimizer = SAM(params, optim.Adam, rho=self.sar_rho,
                             lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer)

        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(
                self.classes, self.prompt_templates, average=False).squeeze()

        if self.runtime:
            self.adapt_times = []
            self.eval_times = []

    def adapt(self, x):
        return self.perform_adaptation(x)

    def continual_adapt(self, x):
        return self.perform_adaptation(x)

    @torch.no_grad()
    def evaluate(self, x):
        t1 = time.time()
        logits, _, _ = self.model(x, self.text_x, True, interpolate=True)
        logits = logits[0]
        if self.runtime:
            self.eval_times.append(time.time() - t1)
        return logits

    def reset(self):
        self.load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state)
        self.ema = None
        self.total_resets += 1

    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []
        for _ in range(self.steps):
            self.total_batches += 1

            # 1st forward + reliable-pixel filter
            logits, _, _ = self.model(x, self.text_x, True, interpolate=False)
            ent = self.softmax_entropy(logits)
            mask1 = ent < self.margin
            if mask1.sum() == 0:
                self.optimizer.zero_grad()
                continue
            loss1 = ent[mask1].mean()
            loss1.backward()
            self.optimizer.first_step(zero_grad=True)

            # 2nd forward at w + e(w), same filter
            logits2, _, _ = self.model(x, self.text_x, True, interpolate=False)
            ent2 = self.softmax_entropy(logits2)
            ent2_filtered = ent2[mask1]
            loss2_value = ent2_filtered.mean().detach()
            mask2 = ent2_filtered < self.margin
            if mask2.sum() == 0:
                self.optimizer.zero_grad()
                continue
            loss2 = ent2_filtered[mask2].mean()
            loss2.backward()
            self.optimizer.second_step(zero_grad=True)

            if not torch.isnan(loss2_value):
                self._update_ema(loss2_value.item())
            loss_report.append(loss2.item())

            if self.ema is not None and self.ema < self.reset_constant_em:
                print(f"[SAR] B{self.total_batches}: ema={self.ema:.4f} "
                      f"< {self.reset_constant_em} -> reset "
                      f"(total resets: {self.total_resets + 1})")
                self.reset()

        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

    def _update_ema(self, x):
        if self.ema is None:
            self.ema = x
        else:
            self.ema = 0.9 * self.ema + 0.1 * x

    def extract_text_embeddings(self, class_names, prompts, average=True):
        text_features = []
        for class_name in class_names:
            texts = [p.format(class_name) for p in prompts]
            texts = self.tokenize(texts).to(self.device)
            ce = self.model.encode_text(texts)
            ce = ce / ce.norm(dim=-1, keepdim=True)
            if average:
                avg = ce.mean(dim=0); avg = avg / avg.norm()
                ce = torch.cat([ce, avg.unsqueeze(0)], dim=0)
            text_features.append(ce)
        return torch.stack(text_features, dim=1).to(self.device)

    @staticmethod
    def _is_excluded(nm, top_block_exclude):
        if 'ln_post' in nm:
            return True
        m = re.search(r'resblocks\.(\d+)\.', nm)
        if m:
            blk = int(m.group(1))
            if blk >= (24 - top_block_exclude):
                return True
        return False

    @classmethod
    def set_ln_grads(cls, model, top_block_exclude=6):
        model.requires_grad_(False)
        for nm, m in model.named_modules():
            if isinstance(m, nn.LayerNorm):
                if cls._is_excluded(nm, top_block_exclude):
                    continue
                m.requires_grad_(True)
        return model

    @classmethod
    def collect_ln_params(cls, model, top_block_exclude=6):
        params, names = [], []
        for nm, m in model.named_modules():
            if isinstance(m, nn.LayerNorm):
                if cls._is_excluded(nm, top_block_exclude):
                    continue
                for np_, p in m.named_parameters():
                    if np_ in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"visual.{nm}.{np_}")
        return params, names

    @staticmethod
    def copy_model_and_optimizer(model, optimizer):
        return copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())

    @staticmethod
    def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)

    @staticmethod
    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        return -(x.softmax(-3) * x.log_softmax(-3)).sum(-3)
