"""
CLIPArTT-Continual: plain CLIPArTT self-distillation loss for CTTA
(no reset, no gate, no restoration).

This is the CLIPArTT analogue of `tent_continual.py`: same base loss as
the episodic adapt/clipartt.py, but the model state persists across the
entire stream -- no reset() between samples.

Loss (verbatim from adapt/clipartt.py):
  1. With no grad, compute per-pixel similarity to single-prompt class
     embeddings, take Top-K classes per pixel.
  2. For each pixel, build a multi-class prompt:
         "a photo of a A or B or C"      (K=3)
  3. Re-encode with this pixel-specific prompt to get logits, image
     features, text features.
  4. target = softmax((img_sim + txt_sim) / 2 / 0.01)
  5. loss = cross_entropy(logits, targets)

Only LayerNorm (gamma, beta) of the visual encoder is trained. Text
encoder is frozen.
"""

import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class CLIPArTTContinual:
    """Plain CLIPArTT base loss in the continual setting (no reset)."""

    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 clipartt_k=3,
                 prompt_dir=None,
                 runtime_calculation=False, device='cpu'):
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps
        self.k = int(clipartt_k)
        if self.k < 1:
            raise ValueError(f"clipartt_k must be >= 1, got {self.k}")

        self.runtime = runtime_calculation
        self.device = device

        if classes is None:
            raise ValueError("classes is required in __init__")
        self.classes = classes

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
        params, names = self.collect_ln_params(self.model.visual)
        self.named_ln_params = list(zip(names, params))

        print_clip_parameters(self.model)
        print(f"+++ CLIPArTT-Continual: clipartt_k={self.k} (no gate, no restoration)")

        # ---------- Optimizer ----------
        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        # ---------- Source state snapshot (kept only for reset()) ----------
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer
        )

        # ---------- Text features ----------
        with torch.no_grad():
            # evaluate uses averaged-prompt embedding (final row)
            self.text_x = self.extract_text_embeddings(
                self.classes, self.prompt_templates, average=True
            ).squeeze()
            # top-K selection uses single REFERENCE_PROMPT embedding, mirroring
            # episodic adapt/clipartt.py
            self.text_feat_topk = self.extract_text_embeddings(
                self.classes, [REFERENCE_PROMPT], average=False
            ).squeeze()

        self.total_batches = 0
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
        logits, _, _ = self.model(x, self.text_x[-1], True, interpolate=True)
        logits = logits[0]
        if self.runtime:
            self.eval_times.append(time.time() - t1)
        return logits

    def reset(self):
        self.load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state
        )

    # ===========================================================
    # Adaptation
    # ===========================================================

    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []
        for _ in range(self.steps):
            # --- Step 1: no-grad single-prompt similarity; Top-K per patch ---
            # Instance unit = patch (not pixel). See clipartt.py docstring.
            with torch.no_grad():
                similarity, _, _ = self.model(
                    x, self.text_feat_topk, True, interpolate=False
                )
                sim0 = similarity[0]                                  # (B, C, w, h)
                probs = sim0.softmax(dim=1)
                patch_marginal = probs.mean(dim=[2, 3])               # (B, C)
                _, pred_per_patch = patch_marginal.topk(self.k, dim=1)  # (B, K)

            # --- Step 2: one combined prompt per patch ---
            pred_inputs = torch.cat([
                self.tokenize(self.getprompt(self.k, c, self.classes))
                for c in pred_per_patch
            ]).to(self.device)                                        # (B, 77)

            # --- Step 3: grad forward with per-patch prompts ---
            # model.forward(): logits is per-pixel (CLS stripped); we don't use it.
            # We need patch-level (CLS token) features for the CLIPArTT loss.
            _, image_features, text_features = self.model(
                x, pred_inputs, False, interpolate=False
            )

            # --- Step 4: patch-level CLIPArTT self-distillation target ---
            # image_features shape: (B, tokens+1, D); CLS at index 0 is the
            # patch-level image embedding (already L2-normalised inside model).
            cls_feat = image_features[:, 0]                           # (B, D)
            cls_feat = cls_feat / cls_feat.norm(dim=-1, keepdim=True)
            text_features = text_features.squeeze()                   # (B, D)
            # patch_logits[i, j] = cosine(image_i CLS, text_j prompt)
            patch_logits = cls_feat @ text_features.t()               # (B, B)

            images_similarity = cls_feat @ cls_feat.t()               # (B, B)
            texts_similarity = text_features @ text_features.t()      # (B, B)
            targets = F.softmax(
                ((images_similarity + texts_similarity) / 2) / 0.01, dim=-1
            )
            loss = self.cross_entropy(patch_logits, targets, reduction='mean')

            loss_report.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.total_batches += 1
        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

    # ===========================================================
    # CLIPArTT helpers (verbatim from adapt/clipartt.py)
    # ===========================================================

    @staticmethod
    def cross_entropy(preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()

    @staticmethod
    def getprompt(K, c, classes):
        for k in range(K):
            if k == 0:
                text_prompt = "a photo of a " + classes[c[k]]
            else:
                text_prompt = text_prompt + " or " + classes[c[k]]
        return text_prompt

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
