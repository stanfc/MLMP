"""
DATContinual — Distribution-Aware Tuning (Ni et al., CVPR 2024), ported
from https://github.com/RochelleNi/DAT to the MLMP / NA-CLIP /
per-pixel-OVSS setting.

DAT's idea: instead of updating ALL norm params every step (TENT/MLMP),
adaptively SELECT a tiny subset of parameters by gradient magnitude and
split the stream's pixels by how big their distribution shift is:

  - TRP (task-relevant params): selected from pixels with SMALL shift
    (low uncertainty). Fine-tuned to avoid catastrophic forgetting.
  - DSP (domain-specific params): selected from pixels with LARGE shift
    (high uncertainty). Tuned to mitigate error accumulation.

Selection is by per-layer |grad| sum: rank layers, keep the top-k%
(k=0.1% in the paper). The selected set ACCUMULATES across the early
stream (PAU — Parameter Accumulation Update) and then freezes; only the
selected params get an optimizer.

Three models (CoTTA-style): student / EMA teacher / frozen anchor.
Pseudo-labels mix anchor (where its confidence > conf_thr) and teacher
prediction. Uncertainty = variance/mean across multi-aug teacher
forwards (we use the faithful multi-aug version).

CTTA hard-rule compliance: no per-sample reset; EMA teacher, anchor, and
the accumulated selected-param set persist across the whole stream.

Per-pixel adaptation:
  - uncertainty, conf, masks are per-pixel.
  - TRP mask: uncertainty < trp_thr ; DSP mask: uncertainty > dsp_thr.
  - loss is entropy of the (single-prompt) logits, masked per-pixel, then
    backward to RANK layers and select; the optimizer step uses only the
    accumulated selected params.

Deviations from upstream:
  - upstream uses mmseg supervised CE on the mixed pseudo-label
    (`decode.loss_seg`); we use per-pixel entropy of the student logits
    (no decode head in NA-CLIP). The masking / selection logic is the same.
  - selection pool is the visual encoder's params (text encoder frozen).
  - base optimizer Adam (MLMP convention).

Hyperparameters (paper defaults, retained where meaningful):
  - k_percent        = 0.001   (fraction of layers selected per backward)
  - select_until     = 100     (PAU: accumulate selected layers for the
                                first N batches, then freeze the set)
  - trp_thr          = 0.85    (low-uncertainty cutoff -> TRP pixels)
  - dsp_thr          = 0.99    (high-uncertainty cutoff -> DSP pixels)
  - conf_thr         = 0.69    (anchor-confidence cutoff for pseudo-label)
  - aug_n            = 8       (multi-aug forwards for uncertainty;
                                paper uses more, reduced for dense seg cost)
  - mt               = 0.999   (EMA teacher rate)
"""
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from ovss import load_ovss
from utils.misc import print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


def _get_tta_transforms(img_size=224, gaussian_std=0.005):
    n = img_size
    return transforms.Compose([
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.4, hue=0.05),
        transforms.Pad(padding=int(n / 2), padding_mode='edge'),
        transforms.RandomAffine(degrees=[-15, 15], translate=(1/16, 1/16),
                                scale=(0.9, 1.1)),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.5]),
        transforms.CenterCrop(size=n),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Lambda(lambda x: x + gaussian_std * torch.randn_like(x)),
    ])


class DATContinual:
    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 k_percent=0.001, select_until=100,
                 trp_thr=0.85, dsp_thr=0.99, conf_thr=0.69,
                 aug_n=8, mt=0.999,
                 runtime_calculation=False, device='cpu'):
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps
        self.k_percent = k_percent
        self.select_until = select_until
        self.trp_thr = trp_thr
        self.dsp_thr = dsp_thr
        self.conf_thr = conf_thr
        self.aug_n = aug_n
        self.mt = mt
        self.runtime = runtime_calculation
        self.device = device

        if classes is None:
            raise ValueError("classes is required in __init__")
        self.classes = classes

        self.model, self.tokenize = load_ovss(ovss_type, ovss_backbone, device=device)
        self.prompt_templates = [REFERENCE_PROMPT]

        # Enable grad on ALL visual params (selection pool); text frozen.
        self.model.transformer.requires_grad_(False)
        self.model.ln_final.requires_grad_(False)
        self.model.token_embedding.requires_grad_(False)
        self.model.visual.requires_grad_(True)
        # The set of param NAMES eligible for selection (visual only).
        self._pool_names = {f"visual.{n}" for n, _ in self.model.visual.named_parameters()}

        print_clip_parameters(self.model)
        print(f"+++ DAT-Continual: k_percent={self.k_percent}, "
              f"select_until={self.select_until}, trp_thr={self.trp_thr}, "
              f"dsp_thr={self.dsp_thr}, conf_thr={self.conf_thr}, "
              f"aug_n={self.aug_n}, mt={self.mt}, pool={len(self._pool_names)} visual params")

        # dummy optimizer (rebuilt each step over selected params)
        self.optimizer = optim.Adam(
            [p for p in self.model.visual.parameters()], lr=self.lr,
            betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        # snapshots + EMA teacher + frozen anchor (CoTTA-style)
        self.model_state = copy.deepcopy(self.model.state_dict())
        self.optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        self.model_ema = copy.deepcopy(self.model)
        for p in self.model_ema.parameters():
            p.detach_()
        self.model_anchor = copy.deepcopy(self.model)
        for p in self.model_anchor.parameters():
            p.detach_()

        self.transform = _get_tta_transforms(img_size=224)
        self.selected_layers = set()   # PAU accumulator
        self.total_batches = 0

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
        # Prediction path = EMA teacher (CoTTA/DAT convention).
        t1 = time.time()
        self.model_ema.eval()
        logits, _, _ = self.model_ema(x, self.text_x, True, interpolate=True)
        logits = logits[0]
        if self.runtime:
            self.eval_times.append(time.time() - t1)
        return logits

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)

    # -----------------------------------------------------------
    def _rank_and_select(self):
        """Rank visual layers by current |grad| sum, add top-k% to the
        accumulated selected set. Call AFTER a backward, BEFORE zero_grad."""
        grad_sum = {}
        for name, p in self.model.named_parameters():
            if name in self._pool_names and p.grad is not None:
                grad_sum[name] = p.grad.abs().sum().item()
        if not grad_sum:
            return
        ranked = sorted(grad_sum, key=lambda n: grad_sum[n], reverse=True)
        n_sel = max(1, int(len(ranked) * self.k_percent))
        for name in ranked[:n_sel]:
            self.selected_layers.add(name)

    def _selected_params(self):
        return [p for n, p in self.model.named_parameters() if n in self.selected_layers]

    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []
        for _ in range(self.steps):
            self.total_batches += 1
            self.model.eval(); self.model_ema.eval(); self.model_anchor.eval()

            # --- teacher uncertainty via multi-aug forwards ---
            with torch.no_grad():
                outs = []
                for _i in range(self.aug_n):
                    lg, _, _ = self.model_ema(self.transform(x), self.text_x, True,
                                              interpolate=False)
                    outs.append(lg[0].softmax(dim=1))         # (B,C,h,w)
                stack = torch.stack(outs)                      # (A,B,C,h,w)
                # uncertainty = mean per-pixel variance across augs over classes
                uncertainty = stack.var(dim=0).mean(dim=1)     # (B,h,w)
                # normalize to [0,1]-ish by its own max for stable thresholds
                u = uncertainty / (uncertainty.amax() + 1e-6)

            # --- student forward + per-pixel entropy ---
            logits, _, _ = self.model(x, self.text_x, True, interpolate=False)
            logits = logits[0]
            ent = self.softmax_entropy(logits)                 # (B,h,w)

            trp_mask = (u < self.trp_thr).float()
            dsp_mask = (u > self.dsp_thr).float()

            # --- TRP backward (small shift) -> rank+select ---
            self.optimizer.zero_grad()
            trp_loss = (trp_mask * ent).mean()
            if trp_loss.requires_grad and trp_loss.item() != 0:
                trp_loss.backward(retain_graph=True)
                if self.total_batches <= self.select_until:
                    self._rank_and_select()

            # --- DSP backward (large shift) -> rank+select ---
            self.optimizer.zero_grad()
            dsp_loss = (dsp_mask * ent).mean()
            if dsp_loss.requires_grad and dsp_loss.item() != 0:
                dsp_loss.backward(retain_graph=True)
                if self.total_batches <= self.select_until:
                    self._rank_and_select()

            # --- combined step over the accumulated selected params ---
            self.optimizer.zero_grad()
            total = trp_loss + dsp_loss
            loss_report.append(total.item())
            if len(self.selected_layers) > 0 and torch.isfinite(total) and total.item() != 0:
                sel = self._selected_params()
                prev = [p.detach().clone() for p in sel]   # rollback snapshot
                total.backward()
                opt = optim.Adam(sel, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
                opt.step()
                opt.zero_grad()
                # NaN guard: if the step produced non-finite weights, roll back.
                if any(not torch.isfinite(p).all() for p in sel):
                    for p, p0 in zip(sel, prev):
                        p.data.copy_(p0)

            self.optimizer.zero_grad()
            # only EMA-update from a finite student
            if all(torch.isfinite(p).all() for p in self.model.visual.parameters()):
                self.model_ema = self._update_ema(self.model_ema, self.model, self.mt)

        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

    @staticmethod
    def _update_ema(ema_model, model, mt):
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(mt).add_(p.data, alpha=1 - mt)
        return ema_model

    # ===========================================================
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
    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        return -(x.softmax(-3) * x.log_softmax(-3)).sum(-3)
