import re
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'

MODE_AGGRESSIVE = 'aggressive'
MODE_CAUTIOUS = 'cautious'
MODE_BRAKE = 'brake'


class TENTDivGateLayeredContinual:
    """
    TENT-DivGate-Layered-Continual: pure TENT pixel-wise entropy +
    diversity-gated stochastic restoration, with a hard layer freeze.

    Identical to TENTDivGateContinual except that LayerNorm parameters
    in the late group ([late_cutoff, num_blocks) plus ln_post) are
    **completely frozen**: requires_grad=False, not handed to the
    optimizer, and skipped by the restoration step (no-op).

    Layer groups (regex on visual.<...>resblocks.<idx>... names):
        early:  blocks [0, early_cutoff)        + ln_pre
        mid:    blocks [early_cutoff, late_cutoff)
        late:   blocks [late_cutoff, num_blocks) + ln_post  (frozen)

    Default cutoffs match the layered ablation history:
        early_cutoff=8, late_cutoff=16  (24-block ViT-L/14)

    DivGate hyper-parameters retain TENT-DivGate defaults
    (h_threshold=1.8, h_warning=1.2, monitor_interval=50,
    cautious_rst=0.005, brake_rst=0.05). The cautious/brake rst is
    applied flatly to whichever LN params are still trainable
    (early + mid).
    """

    BLOCK_RE = re.compile(r'resblocks\.(\d+)\.')

    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 h_threshold=1.8, h_warning=1.2,
                 monitor_interval=50,
                 cautious_rst=0.005, brake_rst=0.05,
                 early_cutoff=8, late_cutoff=16,
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
                f"h_warning ({self.h_warning}) must be <= h_threshold ({self.h_threshold})"
            )
        self.monitor_interval = int(monitor_interval)
        if self.monitor_interval < 1:
            raise ValueError(f"monitor_interval must be >= 1, got {self.monitor_interval}")
        self.cautious_rst = float(cautious_rst)
        self.brake_rst = float(brake_rst)
        for n, v in [('cautious_rst', self.cautious_rst), ('brake_rst', self.brake_rst)]:
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{n} must be in [0, 1], got {v}")

        self.early_cutoff = int(early_cutoff)
        self.late_cutoff = int(late_cutoff)
        if self.early_cutoff < 0 or self.late_cutoff < self.early_cutoff:
            raise ValueError(
                f"need 0 <= early_cutoff <= late_cutoff, "
                f"got early={self.early_cutoff}, late={self.late_cutoff}"
            )

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

        # All LN params (named) for inspection / source snapshot lookup.
        all_params, all_names = self.collect_ln_params(self.model.visual)
        self.named_ln_params_all = list(zip(all_names, all_params))

        # Hard-freeze the late group (no gradients, not in optimizer,
        # not touched by restoration).
        trainable_params = []
        trainable_named = []
        frozen_count = 0
        for name, p in self.named_ln_params_all:
            if self._group_of(name) == 'late':
                p.requires_grad_(False)
                frozen_count += 1
            else:
                trainable_params.append(p)
                trainable_named.append((name, p))
        self.named_ln_params_trainable = trainable_named

        print_clip_parameters(self.model)
        self._print_layer_groups(frozen_count)
        print(f"+++ TENT-DivGate-Layered: thresholds=(warn {self.h_warning}, agg {self.h_threshold}), "
              f"monitor_interval={self.monitor_interval}, "
              f"rsts=(cautious {self.cautious_rst}, brake {self.brake_rst})")
        print(f"+++ Initial mode: {MODE_AGGRESSIVE} (rst=0); buffer fills before first re-evaluation")

        # ---------- Optimizer (only over trainable LN params) ----------
        self.optimizer = optim.Adam(trainable_params, lr=self.lr,
                                    betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        # ---------- Source state snapshot ----------
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer
        )

        # ---------- Text features ----------
        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(
                self.classes, self.prompt_templates, average=False
            ).squeeze()

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
        """Alias for adapt() -- matches main_continual.py protocol."""
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
            self.model, self.optimizer, self.model_state, self.optimizer_state
        )

    # ===========================================================
    # Adaptation
    # ===========================================================

    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []
        for _ in range(self.steps):
            logits, _, _ = self.model(x, self.text_x, True, interpolate=False)

            with torch.no_grad():
                probs = logits[0].softmax(dim=1)                           # (B, C, w, h)
                self.marginal_buf.append(
                    probs.mean(dim=[0, 2, 3]).detach().float().cpu()
                )

            loss = self.softmax_entropy(logits).mean()
            loss_report.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.current_rst > 0.0:
                self._stochastic_restore_trainable(self.current_rst)
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
            print(f"[DivGate-TL] B{self.total_batches}: H_margin={h_margin:.3f}  "
                  f"{self.current_mode} -> {new_mode}")
        self.current_mode = new_mode
        self.current_rst = self._mode_to_rst(new_mode)

        self.marginal_buf.clear()
        self.batch_count = 0

    @torch.no_grad()
    def _stochastic_restore_trainable(self, rst):
        # Only trainable (early + mid) params are touched.
        # Late group is frozen and never drifts, so restoration is a no-op there.
        for name, p in self.named_ln_params_trainable:
            mask = (torch.rand(p.shape, device=p.device) < rst).to(p.dtype)
            src = self.model_state[name].to(p.device, dtype=p.dtype)
            p.data.mul_(1.0 - mask).add_(src * mask)

    # ===========================================================
    # Layer grouping
    # ===========================================================

    def _group_of(self, name):
        if 'ln_pre' in name:
            return 'early'
        if 'ln_post' in name:
            return 'late'
        m = self.BLOCK_RE.search(name)
        if m is None:
            return 'mid'
        idx = int(m.group(1))
        if idx < self.early_cutoff:
            return 'early'
        if idx < self.late_cutoff:
            return 'mid'
        return 'late'

    def _print_layer_groups(self, frozen_count):
        groups = {'early': 0, 'mid': 0, 'late': 0}
        for name, _ in self.named_ln_params_all:
            groups[self._group_of(name)] += 1
        total = sum(groups.values())
        print(f"+++ Layered freeze: cutoffs=({self.early_cutoff}, {self.late_cutoff})")
        print(f"+++ LN params/group: early={groups['early']} (trainable), "
              f"mid={groups['mid']} (trainable), late={groups['late']} (FROZEN), "
              f"total={total}")
        if frozen_count != groups['late']:
            print(f"!!! sanity: frozen_count={frozen_count} != late group size {groups['late']}")

    # ===========================================================
    # Loss (pure TENT, no Top-K mask)
    # ===========================================================

    @staticmethod
    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        # x shape: (T, B, C, w, h); class dim is -3
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
