"""
MLMP-TopK-MinPrompt-Continual: combines top-K confidence pixel mask
(spatial filtering) with min-prompt loss (prompt-view filtering).

Per-pixel loss = -log(min over T of probs_y), but applied ONLY to the
top-K% highest-confidence pixels (mask from ensemble probability).

No ILE (alpha_cls=0 by default).

Rationale:
  - TopK suppresses noisy gradients from low-confidence pixels (CMA-style)
  - MinPrompt forces ALL prompt views to agree with the consensus pseudo-label
  - Together: only confident pixels contribute, and the weakest prompt at
    each kept pixel is what gets gradient.
"""

import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class MLMPTopKMinPromptContinual:
    """MLMP min-prompt loss restricted to top-K% confidence pixels."""

    def __init__(self, ovss_type, ovss_backbone, lr, classes,
                 vision_outputs=(-1,), alpha_cls=0.0, steps=1,
                 top_k_percent=0.2,
                 prompt_dir='prompts.yaml',
                 runtime_calculation=False, device='cpu'):
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps
        self.vision_outputs = vision_outputs
        self.alpha_cls = alpha_cls
        self.top_k_percent = float(top_k_percent)
        if not (0.0 < self.top_k_percent <= 1.0):
            raise ValueError(f"top_k_percent must be in (0, 1], got {self.top_k_percent}")
        self.runtime = runtime_calculation
        self.device = device

        if classes is None:
            raise ValueError("classes is required in __init__")
        self.classes = classes

        self.model, self.tokenize = load_ovss(ovss_type, ovss_backbone, device=device)

        if prompt_dir:
            self.prompt_templates = load_prompts_from_yaml(prompt_dir)
            print(f"Number of prompt templates: {len(self.prompt_templates)}")
        else:
            self.prompt_templates = [REFERENCE_PROMPT]
            print("WARNING: MLMP-TopK-MinPrompt with 1 template degenerates.")

        self.model.transformer.requires_grad_(False)
        self.model.ln_final.requires_grad_(False)
        self.model.token_embedding.requires_grad_(False)
        self.model.visual = self.set_ln_grads(self.model.visual)
        params, _ = self.collect_ln_params(self.model.visual)

        print_clip_parameters(self.model)
        print(f"+++ MLMP-TopK-MinPrompt: T={len(self.prompt_templates)}, "
              f"top_k={self.top_k_percent}, alpha_cls={self.alpha_cls}, "
              f"vision_outputs={self.vision_outputs}")

        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer
        )

        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(
                self.classes, self.prompt_templates, average=True
            ).squeeze()

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
        logits, _, _ = self.model(
            x, self.text_x[-1], True,
            vision_outputs=self.vision_outputs,
            interpolate=True,
            vision_out_type="adaptive_weighted_mean",
            save_weights=True
        )
        logits = logits[0]
        if self.runtime:
            self.eval_times.append(time.time() - t1)
        return logits

    def reset(self):
        self.load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state
        )

    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []

        for _ in range(self.steps):
            logits, _, _, cls_logits = self.model(
                x, self.text_x[:-1], True,
                interpolate=False,
                vision_outputs=self.vision_outputs,
                return_vanilla_cls=True,
                vision_out_type="mean"
            )
            # logits: (T, B, C, h, w)
            probs = logits.softmax(dim=2)            # (T, B, C, h, w)

            with torch.no_grad():
                # MinPrompt step 1: ensemble pseudo-label
                ens_probs = probs.mean(dim=0)            # (B, C, h, w)
                pseudo = ens_probs.argmax(dim=1)          # (B, h, w)
                # TopK step: ensemble max confidence per pixel
                max_p = ens_probs.max(dim=1).values       # (B, h, w)
                flat = max_p.reshape(-1)
                k = max(1, int(self.top_k_percent * flat.numel()))
                threshold = flat.topk(k, sorted=False).values.min()
                mask = (max_p >= threshold).to(probs.dtype)  # (B, h, w)
                denom = mask.sum().clamp_min(1.0)

            # MinPrompt step 2: gather each prompt's prob of the pseudo class
            T_, B_, C_, H_, W_ = probs.shape
            pseudo_exp = pseudo.unsqueeze(0).unsqueeze(2).expand(T_, B_, 1, H_, W_)
            probs_y = probs.gather(2, pseudo_exp).squeeze(2)  # (T, B, h, w)
            min_p = probs_y.min(dim=0).values                 # (B, h, w)

            # Pixel-wise -log(min_p), masked by TopK
            loss_per_pixel = -torch.log(min_p.clamp_min(1e-12))  # (B, h, w)
            loss_pix = (loss_per_pixel * mask).sum() / denom

            if self.alpha_cls > 0:
                entropy_per_cls = self.softmax_entropy(cls_logits, dim=2)
                loss = loss_pix + self.alpha_cls * entropy_per_cls.mean()
            else:
                loss = loss_pix

            loss_report.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

    @staticmethod
    def softmax_entropy(x: torch.Tensor, dim: int = -3) -> torch.Tensor:
        return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)

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
