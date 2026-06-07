"""
TENT-DivGate with smooth (continuous) anchor lag as a function of H_margin.

Motivation:
  hybrid_anchor used a discrete switch:
    cautious (1.5 <= H < 1.8) -> recent anchor (lag=300)
    brake    (H < 1.5)        -> source anchor (lag=infty)
  But the gap between "300 batches ago" and "frozen source" is huge --
  there's no reason the right anchor depth should jump discontinuously
  at H=1.5. We replace it with a smooth function of H_margin:

    lag(H) = lag_scale / (H - h_floor)        for h_floor < H < h_ceil
    lag(H) = infinity (-> source snapshot)    for H <= h_floor
    rst    = 0                                for H >= h_ceil

  With defaults lag_scale=90, h_floor=1.5, h_ceil=1.8:
    H=1.80 -> lag=300        (matches recent_anchor default)
    H=1.65 -> lag=600
    H=1.55 -> lag=1800
    H=1.51 -> lag=9000       (capped to max_lag=3000)
    H=1.50 -> lag=infty      (= source)

  Restoration rate is FIXED (single value, default 0.005). Only the
  anchor depth varies with H_margin -- "the worse the health, the deeper
  we reach back."

Buffer:
  We hold a deque of LN snapshots (fp16 CPU) of size max_lag+1. When
  lag(H) > max_lag we fall back to the frozen source snapshot, so we
  never need more than max_lag entries in memory. Default max_lag=3000
  -> ~0.9 GB CPU RAM for ViT-L/14 LN params.

Gate update cadence:
  Same buffer-N H_margin aggregation as TENTDivGateContinual --
  every monitor_interval batches we recompute H_margin and refresh lag.
"""

import time
import copy
import math
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class TENTDivGateSmoothAnchor:
    """TENT entropy + continuous H_margin->anchor_lag mapping."""

    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 h_ceil=1.8, h_floor=1.5,
                 lag_scale=90.0,
                 max_lag=3000,
                 rst=0.005,
                 monitor_interval=50,
                 marginal_buf_size=None,
                 prompt_dir=None,
                 runtime_calculation=False, device='cpu'):
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps

        self.h_ceil = float(h_ceil)
        self.h_floor = float(h_floor)
        if self.h_floor >= self.h_ceil:
            raise ValueError(f"h_floor ({self.h_floor}) must be < h_ceil ({self.h_ceil})")
        self.lag_scale = float(lag_scale)
        if self.lag_scale <= 0:
            raise ValueError(f"lag_scale must be > 0, got {self.lag_scale}")
        self.max_lag = int(max_lag)
        if self.max_lag < 1:
            raise ValueError(f"max_lag must be >= 1, got {self.max_lag}")
        self.rst = float(rst)
        if not (0.0 <= self.rst <= 1.0):
            raise ValueError(f"rst must be in [0, 1], got {self.rst}")
        self.monitor_interval = int(monitor_interval)
        if self.monitor_interval < 1:
            raise ValueError(f"monitor_interval must be >= 1, got {self.monitor_interval}")
        # marginal_buf_size DECOUPLES the H_margin averaging window from the
        # lag-update cadence (monitor_interval). None -> coupled legacy behaviour
        # (H averaged over a fresh, non-overlapping monitor_interval-batch window,
        # cleared after each update). If set, H is a rolling mean over the last
        # marginal_buf_size batches and the buffer is NOT cleared, so lag can be
        # refreshed every monitor_interval batches without shrinking the H window.
        self.marginal_buf_size = int(marginal_buf_size) if marginal_buf_size else None
        if self.marginal_buf_size is not None and self.marginal_buf_size < 1:
            raise ValueError(f"marginal_buf_size must be >= 1, got {self.marginal_buf_size}")

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
        print(f"+++ TENT-DivGate-SmoothAnchor: h_ceil={self.h_ceil}, h_floor={self.h_floor}, "
              f"lag_scale={self.lag_scale}, max_lag={self.max_lag}, rst={self.rst}, "
              f"monitor_interval={self.monitor_interval}, "
              f"marginal_buf_size={self.marginal_buf_size or 'coupled(=monitor_interval)'}")
        print(f"+++ lag(H) = {self.lag_scale} / (H - {self.h_floor})  "
              f"clamped to [1, {self.max_lag}]; H>={self.h_ceil} -> no restore; "
              f"H<={self.h_floor} -> source")
        # Print a few representative points so the dynamics are visible at run start
        for h in [self.h_ceil, 0.5*(self.h_ceil+self.h_floor), self.h_floor + 0.05,
                  self.h_floor + 0.01]:
            lg = self._lag_from_h(h)
            tag = "source" if lg is None else (f"lag={lg}" if lg <= self.max_lag else "source(>max)")
            print(f"      H={h:.3f} -> {tag}")

        # ---------- Optimizer ----------
        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        # ---------- Source state snapshot (kept for reset() AND deep-fallback) ----------
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer)
        self._source_ln_snapshot = {
            name: self.model_state[name].detach().clone()
            for name, _ in self.named_ln_params
        }

        # ---------- Rotating fp16-CPU snapshots ----------
        # deque holds up to max_lag+1 entries; index 0 is the OLDEST.
        # We push one snapshot per adapt() call BEFORE the gradient step.
        self._anchor_buf = deque(maxlen=self.max_lag + 1)
        self._anchor_buf.append(self._snapshot_ln_weights())

        # ---------- Text features ----------
        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(
                self.classes, self.prompt_templates, average=False).squeeze()

        # ---------- Gate state ----------
        # Decoupled mode: rolling window of fixed size; coupled mode: plain list.
        self.marginal_buf = (deque(maxlen=self.marginal_buf_size)
                             if self.marginal_buf_size else [])
        self.batch_count = 0
        self.total_batches = 0

        # `current_lag` semantics:
        #   None  -> use frozen source snapshot
        #   int>0 -> use deque[-lag-1] (or oldest if buffer shorter)
        #   0     -> no restore (rst effectively zero)
        self.current_lag = 0           # initial: aggressive (H assumed >= h_ceil)
        self.current_rst = 0.0

        if self.runtime:
            self.adapt_times = []
            self.eval_times = []

    # ===========================================================
    # Smooth mapping H_margin -> lag
    # ===========================================================

    def _lag_from_h(self, h):
        """Return desired lag for a given H_margin.

        Returns:
          0    -> no restore (H >= h_ceil)
          None -> use frozen source snapshot (H <= h_floor OR lag > max_lag)
          int  -> use anchor lag in [1, max_lag]
        """
        if h >= self.h_ceil:
            return 0
        if h <= self.h_floor:
            return None
        raw = self.lag_scale / (h - self.h_floor)
        if raw > self.max_lag:
            return None
        return max(1, int(round(raw)))

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
                self._stochastic_restore(self.current_rst, self.current_lag)

        self.batch_count += 1
        self.total_batches += 1
        if self.batch_count >= self.monitor_interval:
            self._update_lag()
        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

    # ===========================================================
    # Gate update
    # ===========================================================

    @torch.no_grad()
    def _update_lag(self):
        if len(self.marginal_buf) == 0:
            self.batch_count = 0
            return
        agg = torch.stack(list(self.marginal_buf), dim=0).mean(dim=0)
        agg = agg / agg.sum().clamp(min=1e-8)
        h_margin = -(agg * agg.clamp(min=1e-12).log()).sum().item()

        new_lag = self._lag_from_h(h_margin)
        if new_lag == 0:
            new_rst = 0.0
        else:
            new_rst = self.rst

        # Log only on meaningful changes:
        #   - turning restore on/off
        #   - lag transitioning to/from source fallback
        #   - lag changing by >25%
        def _label(lg):
            if lg is None: return "source"
            if lg == 0:    return "off"
            return f"lag={lg}"

        old = self.current_lag
        log_change = False
        if (old == 0) != (new_lag == 0):
            log_change = True
        elif (old is None) != (new_lag is None):
            log_change = True
        elif isinstance(old, int) and old > 0 and isinstance(new_lag, int) and new_lag > 0:
            if abs(new_lag - old) / max(old, 1) > 0.25:
                log_change = True

        if log_change:
            print(f"[DivGate-SA] B{self.total_batches}: H_margin={h_margin:.3f}  "
                  f"{_label(old)} -> {_label(new_lag)}  "
                  f"(buf={len(self._anchor_buf)})")

        self.current_lag = new_lag
        self.current_rst = new_rst

        # Coupled mode: clear so the next H uses a fresh non-overlapping window.
        # Decoupled mode: keep the rolling window (maxlen handles eviction).
        if self.marginal_buf_size is None:
            self.marginal_buf.clear()
        self.batch_count = 0

    # ===========================================================
    # Restoration
    # ===========================================================

    @torch.no_grad()
    def _stochastic_restore(self, rst, lag):
        """Restore each LN param toward the anchor selected by `lag`.

        lag is None: anchor = frozen source snapshot
        lag is int>0: anchor = deque entry `lag` batches ago.
                      If buffer is shorter than `lag`, use the oldest entry
                      (graceful fallback for early batches).
        """
        if lag is None:
            anchor_snapshot = self._source_ln_snapshot
        else:
            # deque index: -1 is newest (the one we pushed this batch).
            # Going back `lag` steps -> index -1-lag, but clamp to oldest.
            idx = -1 - lag
            if -idx > len(self._anchor_buf):
                anchor_snapshot = self._anchor_buf[0]   # oldest available
            else:
                anchor_snapshot = self._anchor_buf[idx]

        for name, p in self.named_ln_params:
            mask = (torch.rand(p.shape, device=p.device) < rst).to(p.dtype)
            src = anchor_snapshot[name].to(p.device, dtype=p.dtype)
            p.data.mul_(1.0 - mask).add_(src * mask)

    # ===========================================================
    # Loss (pure TENT)
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
