"""
TENT-DivGate with hybrid anchor restoration.

Motivation:
  - In `tent_divgate_recent_anchor`, ALL restoration (cautious + brake) targets
    the N-batches-ago snapshot. But empirically, once H_margin slides below
    h_warning (1.5), the past N batches were ALSO contaminated -- so the
    recent anchor is no healthier than the current state. Restoration toward
    it is effectively a no-op (see entropy plot for recent_anchor_N300:
    H_margin stays pinned at 1.3-1.5 from R30 onward).

Idea:
  Split restoration by severity. Recent anchor is for *mild* drift; the
  source snapshot is the only credible anchor when drift is severe.

Mode -> restore target:
  - aggressive (H >= h_threshold)      : rst = 0           (no restore)
  - cautious   (h_warning <= H < h_thr) : rst = cautious_rst, anchor = RECENT (gentle correction)
  - brake      (H < h_warning)          : rst = brake_rst,    anchor = SOURCE (hard pullback)

Only the restore target switches between cautious and brake. The gate
triggering logic, H_margin computation, modes, thresholds, monitor_interval,
etc. are bit-identical to TENTDivGateContinual / TENTDivGateRecentAnchor.
"""

import time
import copy
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'

MODE_AGGRESSIVE = 'aggressive'
MODE_CAUTIOUS = 'cautious'
MODE_BRAKE = 'brake'


class TENTDivGateHybridAnchor:
    """TENT entropy + diversity gate, cautious->recent anchor, brake->source."""

    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 h_threshold=1.8, h_warning=1.5,
                 monitor_interval=50,
                 cautious_rst=0.005, brake_rst=0.02,
                 anchor_lag=300,
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

        self.anchor_lag = int(anchor_lag)
        if self.anchor_lag < 1:
            raise ValueError(f"anchor_lag must be >= 1, got {self.anchor_lag}")

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
        print(f"+++ TENT-DivGate-HybridAnchor: thresholds=(warn {self.h_warning}, agg {self.h_threshold}), "
              f"monitor_interval={self.monitor_interval}, "
              f"rsts=(cautious {self.cautious_rst}, brake {self.brake_rst})")
        print(f"+++ anchor_lag={self.anchor_lag}  |  "
              f"cautious -> recent anchor ({self.anchor_lag}B ago)  |  "
              f"brake -> SOURCE snapshot (frozen)")
        print(f"+++ Initial mode: {MODE_AGGRESSIVE} (rst=0); buffer fills before first re-evaluation")

        # ---------- Optimizer ----------
        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        # ---------- Source state snapshot (used for reset() AND brake restore) ----------
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer)
        # Pre-build a name -> tensor map of just the LN params, kept on CPU fp32
        # for clarity (fp16 conversion happens at restore time to match recent buf).
        self._source_ln_snapshot = {
            name: self.model_state[name].detach().clone()
            for name, _ in self.named_ln_params
        }

        # ---------- Recent anchor: rotating fp16-CPU LN snapshots ----------
        self._anchor_buf = deque(maxlen=self.anchor_lag + 1)
        self._anchor_buf.append(self._snapshot_ln_weights())

        # ---------- Text features ----------
        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(
                self.classes, self.prompt_templates, average=False).squeeze()

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
    # Snapshot helpers
    # ===========================================================

    @torch.no_grad()
    def _snapshot_ln_weights(self):
        return {
            name: p.detach().cpu().to(torch.float16).clone()
            for name, p in self.named_ln_params
        }

    @torch.no_grad()
    def _push_current_to_anchor_buf(self):
        self._anchor_buf.append(self._snapshot_ln_weights())

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
            self.model, self.optimizer, self.model_state, self.optimizer_state)
        self._anchor_buf.clear()
        self._anchor_buf.append(self._snapshot_ln_weights())

    # ===========================================================
    # Adaptation
    # ===========================================================

    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []

        # Snapshot CURRENT state BEFORE this batch's gradient step.
        self._push_current_to_anchor_buf()

        for _ in range(self.steps):
            logits, _, _ = self.model(x, self.text_x, True, interpolate=False)

            with torch.no_grad():
                probs = logits[0].softmax(dim=1)
                self.marginal_buf.append(
                    probs.mean(dim=[0, 2, 3]).detach().float().cpu())

            loss = self.softmax_entropy(logits).mean()
            loss_report.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.current_rst > 0.0:
                self._stochastic_restore(self.current_rst, self.current_mode)
        self.batch_count += 1
        self.total_batches += 1
        if self.batch_count >= self.monitor_interval:
            self._update_mode()
        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

    # ===========================================================
    # Diversity gate
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
            print(f"[DivGate-HA] B{self.total_batches}: H_margin={h_margin:.3f}  "
                  f"{self.current_mode} -> {new_mode}  "
                  f"(anchor_buf size: {len(self._anchor_buf)})")
        self.current_mode = new_mode
        self.current_rst = self._mode_to_rst(new_mode)

        self.marginal_buf.clear()
        self.batch_count = 0

    @torch.no_grad()
    def _stochastic_restore(self, rst, mode):
        """Hybrid restore: cautious -> recent anchor, brake -> source.

        Aggressive mode never reaches here because current_rst == 0.
        """
        if mode == MODE_CAUTIOUS:
            anchor_snapshot = self._anchor_buf[0]   # oldest in deque
        elif mode == MODE_BRAKE:
            anchor_snapshot = self._source_ln_snapshot
        else:
            return
        for name, p in self.named_ln_params:
            mask = (torch.rand(p.shape, device=p.device) < rst).to(p.dtype)
            src = anchor_snapshot[name].to(p.device, dtype=p.dtype)
            p.data.mul_(1.0 - mask).add_(src * mask)

    # ===========================================================
    # Loss (pure TENT, no Top-K mask)
    # ===========================================================

    @staticmethod
    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        return -(x.softmax(-3) * x.log_softmax(-3)).sum(-3)

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
