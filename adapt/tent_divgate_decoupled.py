"""
TENT-DivGate with decoupled marginal_buf_size and monitor_interval.

Original TENTDivGateContinual ties the two together via a single
`monitor_interval` value: every N batches the marginal buffer is
aggregated, H_margin computed, mode picked, then buffer cleared.

This variant separates them:
  - `marginal_buf_size`  : rolling window size for H_margin averaging
  - `monitor_interval`   : how often to re-evaluate mode (in batches)

`marginal_buf_size` controls *smoothness* of H_margin (large = stable
but slow to react, small = responsive but noisy).
`monitor_interval` controls *decision frequency* (large = mode locked
for many batches, small = mode can change rapidly).

This lets us isolate which of the two roles drives DivGate's behavior.
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


class TENTDivGateDecoupled:
    """TENT entropy + diversity gate with separate buffer / monitor sizes."""

    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 h_threshold=1.8, h_warning=1.5,
                 marginal_buf_size=50,
                 monitor_interval=50,
                 cautious_rst=0.005, brake_rst=0.02,
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

        self.marginal_buf_size = int(marginal_buf_size)
        self.monitor_interval = int(monitor_interval)
        if self.marginal_buf_size < 1 or self.monitor_interval < 1:
            raise ValueError("marginal_buf_size and monitor_interval must be >= 1")

        self.cautious_rst = float(cautious_rst)
        self.brake_rst = float(brake_rst)
        for n, v in [('cautious_rst', self.cautious_rst), ('brake_rst', self.brake_rst)]:
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{n} must be in [0, 1], got {v}")

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
        print(f"+++ TENT-DivGate-Decoupled: thresholds=(warn {self.h_warning}, agg {self.h_threshold}), "
              f"marginal_buf_size={self.marginal_buf_size}, "
              f"monitor_interval={self.monitor_interval}, "
              f"rsts=(cautious {self.cautious_rst}, brake {self.brake_rst})")
        print(f"+++ Initial mode: {MODE_AGGRESSIVE} (rst=0); "
              f"both buffer (>= {self.marginal_buf_size}) and interval "
              f"(>= {self.monitor_interval}) must trigger to update mode")

        # ---------- Optimizer ----------
        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer)

        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(
                self.classes, self.prompt_templates, average=False).squeeze()

        # ---------- Gate state ----------
        # marginal_buf is now a deque with maxlen = marginal_buf_size:
        # holds the latest N marginals, automatically drops the oldest.
        self.marginal_buf = deque(maxlen=self.marginal_buf_size)
        self.batches_since_last_update = 0   # counter toward monitor_interval
        self.total_batches = 0
        self.current_mode = MODE_AGGRESSIVE
        self.current_rst = 0.0

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
                self._stochastic_restore_flat(self.current_rst)
        self.batches_since_last_update += 1
        self.total_batches += 1

        # Update mode only when: (1) enough batches since last update, AND
        # (2) the buffer is full (so the H_margin average is over a full window).
        if (self.batches_since_last_update >= self.monitor_interval
                and len(self.marginal_buf) >= self.marginal_buf_size):
            self._update_mode()

        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

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
            self.batches_since_last_update = 0
            return
        # Average over all marginals currently in the rolling buffer.
        agg = torch.stack(list(self.marginal_buf), dim=0).mean(dim=0)
        agg = agg / agg.sum().clamp(min=1e-8)
        h_margin = -(agg * agg.clamp(min=1e-12).log()).sum().item()

        new_mode = self._pick_mode(h_margin, self.h_threshold, self.h_warning)
        if new_mode != self.current_mode:
            print(f"[DivGate-DEC] B{self.total_batches}: H_margin={h_margin:.3f}  "
                  f"{self.current_mode} -> {new_mode}  "
                  f"(buf={len(self.marginal_buf)}/{self.marginal_buf_size}, "
                  f"interval={self.monitor_interval})")
        self.current_mode = new_mode
        self.current_rst = self._mode_to_rst(new_mode)

        # IMPORTANT: do NOT clear marginal_buf here. The buffer is rolling
        # (deque maxlen), so we keep it as-is and just reset the interval
        # counter. The next update will average over the latest window.
        self.batches_since_last_update = 0

    @torch.no_grad()
    def _stochastic_restore_flat(self, rst):
        for name, p in self.named_ln_params:
            mask = (torch.rand(p.shape, device=p.device) < rst).to(p.dtype)
            src = self.model_state[name].to(p.device, dtype=p.dtype)
            p.data.mul_(1.0 - mask).add_(src * mask)

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
