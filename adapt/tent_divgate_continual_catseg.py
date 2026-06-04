"""
TENT-DivGate-Continual on CAT-Seg backbone.

Same gate machinery as tent_divgate_continual.py but adapted for CAT-Seg:
  - load_ovss called with ovss_type='catseg', requires classes + checkpoint path
  - forward(x, interpolate=...) signature instead of (x, text_x, True, ...)
  - LayerNorm parameters collected from BOTH self.model.clip_model.visual
    AND self.model.sem_seg_head (which contains the aggregator + decoder LNs)
  - No text encoder freezing logic — CAT-Seg's predictor manages text internally

Everything else (H_margin computation, monitor_interval Buffer-N aggregation,
3-tier mode → flat rst, gate logging) is identical to NA-CLIP DivGate.
"""

import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from ovss import load_ovss
from utils.misc import (load_prompts_from_yaml, print_clip_parameters,
                        print_optimizer_parameters)

REFERENCE_PROMPT = 'a photo of a {}'

MODE_AGGRESSIVE = 'aggressive'
MODE_CAUTIOUS = 'cautious'
MODE_BRAKE = 'brake'


class TENTDivGateContinualCatSeg:
    """TENT-DivGate-Continual with CAT-Seg backbone (CTTA, no reset)."""

    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 h_threshold=1.8, h_warning=1.5,
                 monitor_interval=50,
                 cautious_rst=0.005, brake_rst=0.02,
                 catseg_checkpoint=None,
                 prompt_dir=None,
                 runtime_calculation=False, device='cpu'):
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps

        self.h_threshold = float(h_threshold)
        self.h_warning = float(h_warning)
        if self.h_warning > self.h_threshold:
            raise ValueError(
                f"h_warning ({self.h_warning}) must be <= h_threshold ({self.h_threshold})")
        self.monitor_interval = int(monitor_interval)
        if self.monitor_interval < 1:
            raise ValueError(f"monitor_interval must be >= 1, got {self.monitor_interval}")
        self.cautious_rst = float(cautious_rst)
        self.brake_rst = float(brake_rst)
        for n, v in [('cautious_rst', self.cautious_rst), ('brake_rst', self.brake_rst)]:
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{n} must be in [0, 1], got {v}")

        self.runtime = runtime_calculation
        self.device = device

        if classes is None:
            raise ValueError("classes is required for CAT-Seg method")
        self.classes = classes
        self.catseg_checkpoint = catseg_checkpoint

        # ---------- OVSS model (CAT-Seg) ----------
        self.model, self.tokenize = load_ovss(
            ovss_type, ovss_backbone, device=device,
            classes=self.classes, catseg_checkpoint=self.catseg_checkpoint,
        )

        # ---------- Prompts (kept for future flexibility; CAT-Seg uses its own) ----------
        if prompt_dir:
            self.prompt_templates = load_prompts_from_yaml(prompt_dir)
            print(f"Number of prompt templates: {len(self.prompt_templates)}")
        else:
            self.prompt_templates = [REFERENCE_PROMPT]

        # ---------- Freeze everything, then enable LN grads ----------
        # CAT-Seg has CLIP (visual+text) and a sem_seg_head with aggregator+decoder.
        # We train LayerNorm parameters from BOTH visual and sem_seg_head.
        # Text-side LN (transformer/ln_final/token_embedding) stays frozen.
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.clip_model.visual = self.set_ln_grads(self.model.clip_model.visual)
        self.model.sem_seg_head = self.set_ln_grads(self.model.sem_seg_head)

        params, names = self.collect_ln_params(self.model)
        self.named_ln_params = list(zip(names, params))

        print_clip_parameters(self.model)
        print(f"+++ TENT-DivGate-CatSeg: thresholds=(warn {self.h_warning}, agg {self.h_threshold}), "
              f"monitor_interval={self.monitor_interval}, "
              f"rsts=(cautious {self.cautious_rst}, brake {self.brake_rst})")
        print(f"+++ Initial mode: {MODE_AGGRESSIVE} (rst=0); buffer fills before first re-evaluation")
        print(f"+++ Total LN parameters tracked: {len(self.named_ln_params)}")

        # ---------- Optimizer ----------
        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        # ---------- Source state snapshot ----------
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer)

        # ---------- Gate state ----------
        self.marginal_buf = []
        self.batch_count = 0
        self.total_batches = 0
        self.current_mode = MODE_AGGRESSIVE
        self.current_rst = 0.0

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
        # CAT-Seg forward: (logits_T_B_C_w_h, _, _)
        logits, _, _ = self.model(x, interpolate=True)
        logits = logits[0]  # drop prompt dim
        if self.runtime:
            self.eval_times.append(time.time() - t1)
        return logits

    def reset(self):
        self.load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state)

    # ===========================================================
    # Adaptation
    # ===========================================================

    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []
        for _ in range(self.steps):
            # CAT-Seg forward (no text_x argument)
            logits, _, _ = self.model(x, interpolate=False)

            # Buffer per-batch marginal class distribution for the gate.
            with torch.no_grad():
                probs = logits[0].softmax(dim=1)  # (B, C, w, h)
                self.marginal_buf.append(
                    probs.mean(dim=[0, 2, 3]).detach().float().cpu())

            loss = self.softmax_entropy(logits).mean()
            loss_report.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.current_rst > 0.0:
                self._stochastic_restore_flat(self.current_rst)
        self.batch_count += 1
        self.total_batches += 1
        if self.batch_count >= self.monitor_interval:
            self._update_mode()
        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

    # ===========================================================
    # Diversity gate (identical to TENTDivGateContinual)
    # ===========================================================

    @staticmethod
    def _pick_mode(h_margin, h_threshold, h_warning):
        if h_margin >= h_threshold:
            return MODE_AGGRESSIVE
        if h_margin >= h_warning:
            return MODE_CAUTIOUS
        return MODE_BRAKE

    def _mode_to_rst(self, mode):
        if mode == MODE_AGGRESSIVE:
            return 0.0
        if mode == MODE_CAUTIOUS:
            return self.cautious_rst
        if mode == MODE_BRAKE:
            return self.brake_rst
        raise ValueError(f"unknown mode {mode!r}")

    @torch.no_grad()
    def _update_mode(self):
        if len(self.marginal_buf) == 0:
            self.batch_count = 0
            return
        agg = torch.stack(self.marginal_buf, dim=0).mean(dim=0)
        agg = agg / agg.sum().clamp(min=1e-8)
        h_margin = -(agg * agg.clamp(min=1e-12).log()).sum().item()

        new_mode = self._pick_mode(h_margin, self.h_threshold, self.h_warning)
        if new_mode != self.current_mode:
            print(f"[DivGate-Cat] B{self.total_batches}: H_margin={h_margin:.3f}  "
                  f"{self.current_mode} -> {new_mode}")
        self.current_mode = new_mode
        self.current_rst = self._mode_to_rst(new_mode)

        self.marginal_buf.clear()
        self.batch_count = 0

    @torch.no_grad()
    def _stochastic_restore_flat(self, rst):
        for name, p in self.named_ln_params:
            mask = (torch.rand(p.shape, device=p.device) < rst).to(p.dtype)
            src = self.model_state[name].to(p.device, dtype=p.dtype)
            p.data.mul_(1.0 - mask).add_(src * mask)

    # ===========================================================
    # Loss (pure TENT, no Top-K mask)
    # ===========================================================

    @staticmethod
    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        # x shape: (T, B, C, w, h); class dim is -3
        return -(x.softmax(-3) * x.log_softmax(-3)).sum(-3)

    # ===========================================================
    # Helpers
    # ===========================================================

    @staticmethod
    def set_ln_grads(model):
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)
        return model

    @staticmethod
    def collect_ln_params(model):
        """Walk the entire model and grab every LayerNorm weight/bias whose
        requires_grad has been set to True."""
        params, names = [], []
        for nm, m in model.named_modules():
            if isinstance(m, nn.LayerNorm):
                for np_, p in m.named_parameters(recurse=False):
                    if np_ in ['weight', 'bias'] and p.requires_grad:
                        params.append(p)
                        names.append(f"{nm}.{np_}" if nm else np_)
        return params, names

    @staticmethod
    def copy_model_and_optimizer(model, optimizer):
        return copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())

    @staticmethod
    def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)
