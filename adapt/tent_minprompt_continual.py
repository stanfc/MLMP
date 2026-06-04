"""
TENT-MinPrompt-Continual: multi-prompt min-margin loss (CTTA, no reset).

Uses the 7 prompt templates from prompts.yaml (or a custom yaml). For each
pixel, the pseudo-label comes from the ensemble (mean across prompts) of the
softmax distribution -- this is detached and treated as a fixed target.

Then for the loss we measure each prompt's probability OF THE PSEUDO-LABEL
and take the MINIMUM across prompts:

  probs    = softmax(logits, dim=class)         # (T, B, C, h, w)
  pseudo   = argmax(mean(probs, dim=template))  # (B, h, w)   -- detached
  probs_y  = gather probs at pseudo class       # (T, B, h, w)
  min_p    = probs_y.min(dim=template)          # (B, h, w)   -- weakest prompt
  loss     = -mean(log(min_p))

Interpretation: this minimizes the negative log of the WEAKEST prompt's
agreement with the consensus pseudo-label. It is strictly more aggressive
than TENT (which only minimizes one prompt's entropy) because:

  1. It rewards consensus across prompt views (multi-view consistency).
  2. It targets the weakest link, so a single dissenting prompt blocks the
     loss from going to zero -- forces all 7 to agree.
  3. Gradients flow only through the argmin prompt at each pixel, which is
     where the largest correction is needed.

This is a stronger pseudo-self-training than TENT, and pairs naturally with
the existing diversity-gate machinery (sig/div/smooth_anchor) for safety.
"""

import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class TENTMinPromptContinual:
    """Multi-prompt min-margin loss for CTTA."""

    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 prompt_dir=None,
                 runtime_calculation=False, device='cpu'):
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps

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
            # MinPrompt is meaningless with a single template; allow it but warn.
            self.prompt_templates = [REFERENCE_PROMPT]
            print("WARNING: TENT-MinPrompt with only 1 template degenerates to plain TENT-CE.")

        # ---------- Freeze text encoder, enable LN grads ----------
        self.model.transformer.requires_grad_(False)
        self.model.ln_final.requires_grad_(False)
        self.model.token_embedding.requires_grad_(False)
        self.model.visual = self.set_ln_grads(self.model.visual)
        params, names = self.collect_ln_params(self.model.visual)
        self.named_ln_params = list(zip(names, params))

        print_clip_parameters(self.model)
        print(f"+++ TENT-MinPrompt-Continual: T={len(self.prompt_templates)} prompts")

        # ---------- Optimizer ----------
        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer)

        with torch.no_grad():
            # extract_text_embeddings returns (T+1, C, D) when average=True
            # (last row is the average); for evaluate we use that average.
            self.text_x = self.extract_text_embeddings(
                self.classes, self.prompt_templates, average=True).squeeze()

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
        # evaluate uses the averaged-prompt embedding (last row)
        t1 = time.time()
        logits, _, _ = self.model(x, self.text_x[-1], True, interpolate=True)
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
            # Use all per-template embeddings (drop the last averaged one).
            logits, _, _ = self.model(
                x, self.text_x[:-1], True, interpolate=False)
            # logits shape: (T, B, C, h, w)
            probs = logits.softmax(dim=2)               # (T, B, C, h, w)

            with torch.no_grad():
                ens_probs = probs.mean(dim=0)            # (B, C, h, w)
                pseudo = ens_probs.argmax(dim=1)         # (B, h, w)

            T_, B_, C_, H_, W_ = probs.shape
            pseudo_exp = pseudo.unsqueeze(0).unsqueeze(2).expand(
                T_, B_, 1, H_, W_)
            probs_y = probs.gather(2, pseudo_exp).squeeze(2)  # (T, B, h, w)
            # Weakest prompt's confidence for the consensus label
            min_p = probs_y.min(dim=0).values            # (B, h, w)
            loss = -torch.log(min_p.clamp_min(1e-12)).mean()

            loss_report.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.total_batches += 1
        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

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
