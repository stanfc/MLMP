import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class TENTContGateContinual:
    """
    TENT-ContGate-Continual: pure TENT pixel-wise entropy + continuous
    diversity-gated stochastic restoration (CTTA, no reset).

    Same H_margin signal as TENTDivGateContinual, but the restoration
    rate is a CONTINUOUS linear function of H_margin instead of three
    discrete modes:

        ratio = clamp((h_high - H_margin) / (h_high - h_low), 0, 1)
        rst   = ratio * max_rst

    so H_margin >= h_high          -> rst = 0          (healthy, no brake)
       H_margin = (h_high+h_low)/2 -> rst = 0.5*max_rst (mid)
       H_margin <= h_low           -> rst = max_rst    (dangerous, full brake)

    H_margin is recomputed every `monitor_interval` batches (same buffer
    machinery as DivGate); the rst value stays constant within each
    interval. This avoids per-batch noise in H_margin while removing the
    discrete jumps that cause the saw-tooth pattern in DivGate.
    """

    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 h_high=1.8, h_low=0.8,
                 max_rst=0.01,
                 monitor_interval=50,
                 prompt_dir=None,
                 runtime_calculation=False, device='cpu'):
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps

        self.h_high = float(h_high)
        self.h_low = float(h_low)
        if self.h_low >= self.h_high:
            raise ValueError(
                f"h_low ({self.h_low}) must be < h_high ({self.h_high})"
            )
        self.max_rst = float(max_rst)
        if not (0.0 <= self.max_rst <= 1.0):
            raise ValueError(f"max_rst must be in [0, 1], got {self.max_rst}")
        self.monitor_interval = int(monitor_interval)
        if self.monitor_interval < 1:
            raise ValueError(f"monitor_interval must be >= 1, got {self.monitor_interval}")

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
        print(f"+++ TENT-ContGate: h_range=[{self.h_low}, {self.h_high}], "
              f"max_rst={self.max_rst}, monitor_interval={self.monitor_interval}")
        print(f"+++ Initial rst=0 (buffer fills before first H_margin computation)")

        # ---------- Optimizer ----------
        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
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
        self.current_rst = 0.0
        self.last_h_margin = float('nan')

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
                self._stochastic_restore_flat(self.current_rst)
        self.batch_count += 1
        self.total_batches += 1
        if self.batch_count >= self.monitor_interval:
            self._update_rst()
        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

    # ===========================================================
    # Continuous gate
    # ===========================================================

    def _h_to_rst(self, h_margin):
        ratio = (self.h_high - h_margin) / (self.h_high - self.h_low)
        ratio = max(0.0, min(1.0, ratio))
        return ratio * self.max_rst

    @torch.no_grad()
    def _update_rst(self):
        if len(self.marginal_buf) == 0:
            self.batch_count = 0
            return
        agg = torch.stack(self.marginal_buf, dim=0).mean(dim=0)
        agg = agg / agg.sum().clamp(min=1e-8)
        h_margin = -(agg * agg.clamp(min=1e-12).log()).sum().item()

        new_rst = self._h_to_rst(h_margin)
        # Log every monitor window so we can see the H_margin / rst trajectory
        # without flooding stdout (only one line per `monitor_interval` batches).
        print(f"[ContGate-T] B{self.total_batches}: "
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
