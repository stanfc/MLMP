"""
TENT-DivGate with sliding-window anchor.

Identical to TENTDivGateContinual except:
  - Restoration target is NOT the original source weights.
  - Instead we keep a deque of the last N LayerNorm-weight snapshots and
    restore toward the OLDEST one (i.e. the model state from N batches ago).
  - When the deque is shorter than N, we restore toward the oldest available
    snapshot (graceful fallback in early batches).

The hypothesis is that "N batches ago" might be a healthier anchor for
recent-state correction than the frozen source weights (which may be too
far away to provide useful gradient direction).

Only the restoration source changes. The DivGate triggering logic, modes,
H_margin computation, etc. are bit-identical to the original implementation.
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


class TENTDivGateRecentAnchor:
    """TENT entropy + diversity gate, restoration toward N-batches-ago anchor."""

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
        print(f"+++ TENT-DivGate-RecentAnchor: thresholds=(warn {self.h_warning}, agg {self.h_threshold}), "
              f"monitor_interval={self.monitor_interval}, "
              f"rsts=(cautious {self.cautious_rst}, brake {self.brake_rst})")
        print(f"+++ anchor_lag={self.anchor_lag} (restoration target = LN snapshot from "
              f"{self.anchor_lag} batches ago, fp16 CPU storage)")
        print(f"+++ Initial mode: {MODE_AGGRESSIVE} (rst=0); buffer fills before first re-evaluation")

        # ---------- Optimizer ----------
        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        # ---------- Source state snapshot (kept only for reset()) ----------
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer)

        # ---------- Anchor: rotating fp16-CPU LN snapshots ----------
        # maxlen=anchor_lag+1 so deque[0] is "anchor_lag batches ago" once
        # the buffer is full. We push the CURRENT state at the start of each
        # adapt() call (before the optimizer step), so deque[0] always
        # corresponds to a real past state.
        self._anchor_buf = deque(maxlen=self.anchor_lag + 1)
        # Seed with the source state so early batches still have a valid anchor.
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
        """Return a dict of fp16 CPU tensors capturing all trainable LN params."""
        return {
            name: p.detach().cpu().to(torch.float16).clone()
            for name, p in self.named_ln_params
        }

    @torch.no_grad()
    def _push_current_to_anchor_buf(self):
        """Append a fresh snapshot of the CURRENT LN weights to the deque.

        deque has maxlen=anchor_lag+1 so the oldest element naturally
        becomes the target as new ones are pushed; the deque internally
        drops old entries (no manual cleanup needed).
        """
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
        # Also reset anchor buffer: start fresh from source state.
        self._anchor_buf.clear()
        self._anchor_buf.append(self._snapshot_ln_weights())

    # ===========================================================
    # Adaptation
    # ===========================================================

    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []

        # Snapshot CURRENT state BEFORE this batch's gradient step.
        # This way deque[0] is the state from anchor_lag batches ago.
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
                self._stochastic_restore_to_recent_anchor(self.current_rst)
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
            print(f"[DivGate-RA] B{self.total_batches}: H_margin={h_margin:.3f}  "
                  f"{self.current_mode} -> {new_mode}  "
                  f"(anchor_buf size: {len(self._anchor_buf)})")
        self.current_mode = new_mode
        self.current_rst = self._mode_to_rst(new_mode)

        self.marginal_buf.clear()
        self.batch_count = 0

    @torch.no_grad()
    def _stochastic_restore_to_recent_anchor(self, rst):
        """Restore each LN param toward the OLDEST snapshot in the deque
        (i.e. state from anchor_lag batches ago if buffer is full, else
        the oldest available state — graceful fallback)."""
        anchor_snapshot = self._anchor_buf[0]
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
    # Shared helpers (verbatim from TENTDivGateContinual)
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
