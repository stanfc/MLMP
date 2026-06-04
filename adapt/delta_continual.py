"""
DELTA-Continual (DOT-only variant): TENT entropy minimization with Dynamic
Online re-weighTing to combat class imbalance / mode collapse.

Reference: Zhao et al., "DELTA: Degradation-Free Fully Test-Time Adaptation",
ICLR 2023.

DELTA originally has two mechanisms:
  1. DOT — class-aware per-pixel reweighting of the entropy loss
  2. TBR — test-time batch renormalization

NA-CLIP is LayerNorm-only (no BN), so TBR is inapplicable. This file implements
the DOT mechanism on top of the TENT-continual base.

Mechanism:
  - Maintain an EMA class-frequency tracker freq[c], initialised to uniform.
  - Each adapt step:
      * pseudo-label ĉ_i = argmax_c logits[c, i]      (no_grad)
      * batch_freq[c] = fraction of pixels with ĉ_i = c
      * freq <- momentum * freq + (1 - momentum) * batch_freq
      * w_i = freq[ĉ_i]^(-alpha)                     (per-pixel weight)
      * w_i <- w_i / mean(w_i)                       (renormalise: mean weight = 1)
      * L = mean_i (w_i * entropy_i)
  - Loss scale is preserved (mean-1 weighting), so lr does not need retuning.

  alpha = 1 → full inverse-frequency. alpha = 0 → degenerates to plain TENT.
  momentum = 0.9 → ~10-batch effective window for frequency tracking.
"""

import os
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class DELTAContinual:

    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 dot_momentum=0.9, dot_alpha=1.0,
                 prompt_dir=None,
                 save_dir=None,
                 runtime_calculation=False, device='cpu'):
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps

        self.dot_momentum = float(dot_momentum)
        self.dot_alpha = float(dot_alpha)
        if not (0.0 <= self.dot_momentum < 1.0):
            raise ValueError(f"dot_momentum must be in [0, 1), got {self.dot_momentum}")

        self.runtime = runtime_calculation
        self.device = device
        if classes is None:
            raise ValueError("classes is required in __init__")
        self.classes = classes
        self.num_classes = len(classes)

        # ---------- OVSS model ----------
        self.model, self.tokenize = load_ovss(ovss_type, ovss_backbone, device=device)

        # ---------- Prompts ----------
        if prompt_dir:
            self.prompt_templates = load_prompts_from_yaml(prompt_dir)
            print(f"Number of prompt templates: {len(self.prompt_templates)}")
        else:
            self.prompt_templates = [REFERENCE_PROMPT]

        # ---------- Freeze text encoder, enable LN grads ----------
        self.model.transformer.requires_grad_(False)
        self.model.ln_final.requires_grad_(False)
        self.model.token_embedding.requires_grad_(False)
        self.model.visual = self.set_ln_grads(self.model.visual)
        params, _ = self.collect_ln_params(self.model.visual)

        print_clip_parameters(self.model)
        print(f"+++ DELTA: dot_momentum={self.dot_momentum}, dot_alpha={self.dot_alpha}, "
              f"num_classes={self.num_classes}")

        # ---------- Optimizer ----------
        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        # ---------- Source state ----------
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer
        )

        # ---------- Text features ----------
        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(
                self.classes, self.prompt_templates, average=False
            ).squeeze()

        # ---------- DOT runtime state ----------
        # Uniform prior — first batch will shift it toward the actual distribution
        # because dot_momentum < 1.
        self.class_freq = torch.full((self.num_classes,),
                                     1.0 / self.num_classes,
                                     device=device)
        self.total_batches = 0

        # ---------- Optional per-batch log ----------
        self.log_path = None
        if save_dir is not None:
            try:
                os.makedirs(save_dir, exist_ok=True)
                self.log_path = os.path.join(save_dir, 'delta_log.txt')
                with open(self.log_path, 'w') as f:
                    f.write("# DELTA (DOT) log: per-batch entropy, class-freq stats, weighted loss\n")
                    f.write("total_batches,raw_entropy,weighted_loss,"
                            "max_freq,min_freq,max_weight,min_weight\n")
                print(f"+++ DELTA log -> {self.log_path}")
            except OSError as e:
                print(f"+++ DELTA log disabled ({save_dir}): {e}")
                self.log_path = None

        if self.runtime:
            self.adapt_times = []
            self.eval_times = []

    # ===========================================================
    # Public API
    # ===========================================================

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
            self.model, self.optimizer, self.model_state, self.optimizer_state
        )
        self.class_freq.fill_(1.0 / self.num_classes)

    # ===========================================================
    # Adaptation
    # ===========================================================

    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []

        raw_ent_log = float('nan')
        weighted_loss_log = float('nan')
        max_freq_log = float('nan')
        min_freq_log = float('nan')
        max_w_log = float('nan')
        min_w_log = float('nan')

        for _ in range(self.steps):
            logits, _, _ = self.model(x, self.text_x, True, interpolate=False)
            # logits: (T, B, C, w, h); single prompt template here so T=1
            ent_map = self.softmax_entropy(logits)             # (B, w, h)

            # ----- DOT: pseudo-label & class-freq EMA -----
            with torch.no_grad():
                # Use prompt-averaged logits for pseudo-label (consistent with eval pipeline)
                avg_logits = logits.mean(dim=0)                # (B, C, w, h)
                pseudo = avg_logits.argmax(dim=1)              # (B, w, h)

                # Batch class frequency
                batch_freq = torch.bincount(
                    pseudo.reshape(-1), minlength=self.num_classes
                ).to(self.class_freq.dtype)
                batch_freq = batch_freq / batch_freq.sum().clamp(min=1.0)

                # EMA update
                self.class_freq.mul_(self.dot_momentum).add_(
                    batch_freq * (1.0 - self.dot_momentum)
                )

                # Per-pixel weight: w_i = freq[ĉ_i]^(-alpha)
                freq_safe = self.class_freq.clamp(min=1e-4)
                pixel_w = freq_safe[pseudo].pow(-self.dot_alpha)   # (B, w, h)
                # Normalise so mean weight = 1 → preserves overall loss scale
                pixel_w = pixel_w / pixel_w.mean().clamp(min=1e-8)

                max_freq_log = float(self.class_freq.max().item())
                min_freq_log = float(self.class_freq.min().item())
                max_w_log = float(pixel_w.max().item())
                min_w_log = float(pixel_w.min().item())

            raw_ent = ent_map.mean()
            weighted_loss = (pixel_w * ent_map).mean()

            raw_ent_log = float(raw_ent.item())
            weighted_loss_log = float(weighted_loss.item())
            loss_report.append(weighted_loss.item())

            weighted_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.total_batches += 1
        if self.log_path is not None:
            try:
                with open(self.log_path, 'a') as f:
                    f.write(f"{self.total_batches},{raw_ent_log:.6f},"
                            f"{weighted_loss_log:.6f},"
                            f"{max_freq_log:.6f},{min_freq_log:.6e},"
                            f"{max_w_log:.6f},{min_w_log:.6e}\n")
            except OSError:
                pass

        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

    # ===========================================================
    # Loss (per-pixel entropy reduced over class dim and prompt T)
    # ===========================================================

    @staticmethod
    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, C, w, h) — sum over class, mean over T → (B, w, h)
        ent = -(x.softmax(-3) * x.log_softmax(-3)).sum(-3)     # (T, B, w, h)
        return ent.mean(dim=0)                                  # (B, w, h)

    # ===========================================================
    # Shared helpers
    # ===========================================================

    def extract_text_embeddings(self, class_names, prompts, average=True):
        text_features = []
        for class_name in class_names:
            texts = [p.format(class_name) for p in prompts]
            texts = self.tokenize(texts).to(self.device)
            class_embeddings = self.model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            if average:
                avg = class_embeddings.mean(dim=0)
                avg = avg / avg.norm()
                class_embeddings = torch.cat([class_embeddings, avg.unsqueeze(0)], dim=0)
            text_features.append(class_embeddings)
        return torch.stack(text_features, dim=1).to(self.device)

    @staticmethod
    def set_ln_grads(model):
        model.requires_grad_(False)
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)
        return model

    @staticmethod
    def collect_ln_params(model):
        params, names = [], []
        for nm, m in model.named_modules():
            if isinstance(m, nn.LayerNorm):
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
