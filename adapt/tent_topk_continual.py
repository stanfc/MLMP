"""
TENT-Top-K-Continual: TENT pixel-wise entropy, BUT only on the top-K%
most confident pixels per batch (CTTA, no reset).

Hypothesis: TENT averages entropy over ALL pixels including low-confidence
ones that are noisy and pull the optimizer in random directions. Restricting
the loss to high-confidence pixels concentrates gradient on samples the model
is already committed to (which are usually correct under CLIP). This trades
gradient diversity for gradient sharpness -- a more aggressive base loss
when paired with a downstream restoration gate (sig/div).

Loss:
  probs    = softmax(logits, dim=class)        # (B, C, h, w)
  max_p    = max(probs, dim=class)              # (B, h, w)
  threshold = topk_percentile(max_p, top_k_percent)
  mask     = max_p >= threshold                  # (B, h, w)
  ent      = -Σ probs * log probs                # (B, h, w)
  loss     = (ent * mask).sum() / mask.sum()

Default top_k_percent = 0.2 (matching CMA-Continual). Drops to plain TENT
when top_k_percent = 1.0.
"""

import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class TENTTopKContinual:
    """Pure TENT entropy on the top-K% confidence pixels only."""

    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 top_k_percent=0.2,
                 prompt_dir=None,
                 runtime_calculation=False, device='cpu'):
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps

        self.top_k_percent = float(top_k_percent)
        if not (0.0 < self.top_k_percent <= 1.0):
            raise ValueError(f"top_k_percent must be in (0, 1], got {self.top_k_percent}")

        self.runtime = runtime_calculation
        self.device = device

        if classes is None:
            raise ValueError("classes is required in __init__")
        self.classes = classes

        # ---------- OVSS model ----------
        self.model, self.tokenize = load_ovss(ovss_type, ovss_backbone, device=device)

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
        params, names = self.collect_ln_params(self.model.visual)
        self.named_ln_params = list(zip(names, params))

        print_clip_parameters(self.model)
        print(f"+++ TENT-Top-K-Continual: top_k_percent={self.top_k_percent}")

        # ---------- Optimizer ----------
        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer)

        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(
                self.classes, self.prompt_templates, average=False).squeeze()

        self.total_batches = 0
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

    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []
        for _ in range(self.steps):
            logits, _, _ = self.model(x, self.text_x, True, interpolate=False)
            # logits shape: (T=1, B, C, h, w)
            ent = self.softmax_entropy(logits)        # (T, B, h, w)
            ent0 = ent[0]                              # (B, h, w)

            with torch.no_grad():
                probs = logits[0].softmax(dim=1)       # (B, C, h, w)
                max_p = probs.max(dim=1).values        # (B, h, w)
                flat = max_p.reshape(-1)
                k = max(1, int(self.top_k_percent * flat.numel()))
                # threshold = the k-th largest value
                threshold = flat.topk(k, sorted=False).values.min()
                mask = (max_p >= threshold).to(ent0.dtype)   # (B, h, w)
                denom = mask.sum().clamp_min(1.0)

            loss = (ent0 * mask).sum() / denom
            loss_report.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.total_batches += 1
        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

    @staticmethod
    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        return -(x.softmax(-3) * x.log_softmax(-3)).sum(-3)

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
