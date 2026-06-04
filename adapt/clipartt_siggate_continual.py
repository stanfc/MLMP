"""
CLIPArTT-SigGate-Continual: CLIPArTT base loss + sigmoid-shaped diversity-gated
stochastic restoration (CTTA, no reset).

Same H_margin signal, Buffer-N machinery, and sigmoid H_margin->rst mapping as
TENTSigGateContinual. The only thing that changes is the base loss --
instead of pixel-wise entropy minimization (TENT), we use CLIPArTT's
self-distillation loss:

  1. With no grad, compute per-pixel similarity to single-prompt class
     embeddings, take Top-K classes per pixel.
  2. For each pixel, build a multi-class prompt:
         "a photo of a A or B or C"      (K=3)
  3. Re-encode with this pixel-specific prompt to get logits, image
     features, text features.
  4. Build the soft target from cosine similarity of image features and
     text features (averaged), with temperature 0.01.
  5. Loss = cross_entropy(logits, targets).

This is the same base loss as the existing episodic adapt/clipartt.py.
The gate logic is the same as tent_siggate_continual.py:

  rst(H) = max_rst * sigmoid(-k * (H - mid))
  mid    = (h_high + h_low) / 2
  k_auto = 6 / (h_high - h_low)

H_margin itself is computed from the FIRST (no-grad, single-prompt)
forward pass -- the per-pixel custom prompts vary in class count and
are not a stable signal for marginal-class entropy.
"""

import math
import os
import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class CLIPArTTSigGateContinual:
    """CLIPArTT base + sigmoid gate, continual (no reset)."""

    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 clipartt_k=3,
                 h_high=1.8, h_low=0.8,
                 max_rst=0.01,
                 sigmoid_k=None,
                 monitor_interval=50,
                 gate_log_path=None,
                 prompt_dir=None,
                 runtime_calculation=False, device='cpu'):
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps
        self.k = int(clipartt_k)
        if self.k < 1:
            raise ValueError(f"clipartt_k must be >= 1, got {self.k}")

        self.h_high = float(h_high)
        self.h_low = float(h_low)
        if self.h_low >= self.h_high:
            raise ValueError(f"h_low ({self.h_low}) must be < h_high ({self.h_high})")
        self.max_rst = float(max_rst)
        if not (0.0 <= self.max_rst <= 1.0):
            raise ValueError(f"max_rst must be in [0, 1], got {self.max_rst}")
        self.monitor_interval = int(monitor_interval)
        if self.monitor_interval < 1:
            raise ValueError(f"monitor_interval must be >= 1, got {self.monitor_interval}")

        if sigmoid_k is None or (isinstance(sigmoid_k, float) and sigmoid_k <= 0):
            self.sigmoid_k = 6.0 / (self.h_high - self.h_low)
            self._k_auto = True
        else:
            self.sigmoid_k = float(sigmoid_k)
            self._k_auto = False

        self.h_mid = 0.5 * (self.h_high + self.h_low)

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
        k_src = "auto" if self._k_auto else "user"
        print(f"+++ CLIPArTT-SigGate: clipartt_k={self.k}, "
              f"h_range=[{self.h_low}, {self.h_high}], mid={self.h_mid:.3f}, "
              f"k={self.sigmoid_k:.3f} ({k_src}), max_rst={self.max_rst}, "
              f"monitor_interval={self.monitor_interval}")
        for h_ref in [self.h_high, self.h_mid, self.h_low]:
            print(f"+++   H_margin={h_ref:.2f} -> rst={self._h_to_rst(h_ref):.5f}")
        print(f"+++ Initial rst=0 (buffer fills before first H_margin computation)")

        # ---------- Optimizer ----------
        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        # ---------- Source state snapshot ----------
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer
        )

        # ---------- Text features (for evaluate AND for top-K selection) ----------
        # evaluate uses the averaged prompt embedding (last entry); top-K
        # selection uses the simple REFERENCE_PROMPT embedding to mirror the
        # episodic CLIPArTT path exactly.
        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(
                self.classes, self.prompt_templates, average=True
            ).squeeze()
            # text_feat_topk: shape (1, num_classes, D), single template, no averaging
            self.text_feat_topk = self.extract_text_embeddings(
                self.classes, [REFERENCE_PROMPT], average=False
            ).squeeze()

        # ---------- Gate state ----------
        self.marginal_buf = []
        self.batch_count = 0
        self.total_batches = 0
        self.current_rst = 0.0
        self.last_h_margin = float('nan')

        # ---------- Per-batch gate logging (csv) ----------
        self.gate_log_path = gate_log_path
        if self.gate_log_path is not None:
            self._init_gate_log()

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
        # evaluate uses the averaged text embedding (final row of text_x)
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
            # --- Step 1: no-grad single-prompt similarity to get Top-K and H_margin ---
            # Instance unit = patch (not pixel). See clipartt.py docstring.
            with torch.no_grad():
                similarity, _, _ = self.model(
                    x, self.text_feat_topk, True, interpolate=False
                )
                sim0 = similarity[0]                                  # (B, C, w, h)
                # H_margin: marginal class distribution averaged over batch+spatial.
                # This is independent of patch/pixel loss granularity.
                probs = sim0.softmax(dim=1)
                batch_marginal = probs.mean(dim=[0, 2, 3]).detach().float().cpu()
                self.marginal_buf.append(batch_marginal)

                # Top-K per patch
                patch_marginal = probs.mean(dim=[2, 3])               # (B, C)
                _, pred_per_patch = patch_marginal.topk(self.k, dim=1)  # (B, K)

            # --- Step 2: one combined prompt per patch ---
            pred_inputs = torch.cat([
                self.tokenize(self.getprompt(self.k, c, self.classes))
                for c in pred_per_patch
            ]).to(self.device)                                        # (B, 77)

            # --- Step 3: grad forward with per-patch prompts ---
            _, image_features, text_features = self.model(
                x, pred_inputs, False, interpolate=False
            )

            # --- Step 4: patch-level CLIPArTT self-distillation target ---
            # CLS token (index 0) is the patch-level image embedding.
            cls_feat = image_features[:, 0]                           # (B, D)
            cls_feat = cls_feat / cls_feat.norm(dim=-1, keepdim=True)
            text_features = text_features.squeeze()                   # (B, D)
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
            if self.current_rst > 0.0:
                self._stochastic_restore_flat(self.current_rst)

        self.batch_count += 1
        self.total_batches += 1

        # Per-batch H_margin logging
        if self.gate_log_path is not None:
            with torch.no_grad():
                m = batch_marginal / batch_marginal.sum().clamp(min=1e-8)
                h_margin_batch = -(m * m.clamp(min=1e-12).log()).sum().item()
            self._append_gate_log(h_margin_batch)

        if self.batch_count >= self.monitor_interval:
            self._update_rst()
        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

    # ===========================================================
    # Sigmoid gate
    # ===========================================================

    def _h_to_rst(self, h_margin):
        z = self.sigmoid_k * (h_margin - self.h_mid)
        z = max(-50.0, min(50.0, z))
        sig = 1.0 / (1.0 + math.exp(z))
        return self.max_rst * sig

    @torch.no_grad()
    def _update_rst(self):
        if len(self.marginal_buf) == 0:
            self.batch_count = 0
            return
        agg = torch.stack(self.marginal_buf, dim=0).mean(dim=0)
        agg = agg / agg.sum().clamp(min=1e-8)
        h_margin = -(agg * agg.clamp(min=1e-12).log()).sum().item()

        new_rst = self._h_to_rst(h_margin)
        print(f"[SigGate-CA] B{self.total_batches}: "
              f"H_margin={h_margin:.3f}  rst={new_rst:.5f} "
              f"(prev {self.current_rst:.5f})")

        self.current_rst = new_rst
        self.last_h_margin = h_margin

        self.marginal_buf.clear()
        self.batch_count = 0

    @torch.no_grad()
    def _stochastic_restore_flat(self, rst):
        for name, p in self.named_ln_params:
            mask = (torch.rand(p.shape, device=p.device) < rst).to(p.dtype)
            src = self.model_state[name].to(p.device, dtype=p.dtype)
            p.data.mul_(1.0 - mask).add_(src * mask)

    # ===========================================================
    # Gate logging
    # ===========================================================

    def _init_gate_log(self):
        os.makedirs(os.path.dirname(self.gate_log_path) or '.', exist_ok=True)
        with open(self.gate_log_path, 'w') as f:
            f.write("total_batch,h_margin,rst\n")

    def _append_gate_log(self, h_margin_batch):
        with open(self.gate_log_path, 'a') as f:
            f.write(f"{self.total_batches},{h_margin_batch:.6f},"
                    f"{self.current_rst:.6f}\n")

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
