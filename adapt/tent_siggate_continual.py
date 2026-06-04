import math
import os
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class TENTSigGateContinual:
    """
    TENT-SigGate-Continual: pure TENT pixel-wise entropy + sigmoid-shaped
    diversity-gated stochastic restoration (CTTA, no reset).

    Same H_margin signal and Buffer-N machinery as TENTDivGate /
    TENTContGate, but the H_margin -> rst mapping is a centred sigmoid
    instead of linear, giving plateaus at both extremes:

        mid  = (h_high + h_low) / 2
        rst  = max_rst * sigmoid(-k * (H_margin - mid))

    Default k is chosen so that H_high -> ~0.047*max_rst and
    H_low -> ~0.953*max_rst:

        k_auto = 6 / (h_high - h_low)

    Behaviour:
      - H >> mid  : rst ~ 0           (healthy, no brake)
      - H == mid  : rst = max_rst/2
      - H << mid  : rst ~ max_rst     (saturated brake)

    Compared to the linear ContGate, this damps very little when H is
    still healthy (e.g. 1.6+) and saturates earlier when H is clearly
    dangerous (e.g. 1.0-).
    """

    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
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

        # Auto-pick k so that the sigmoid is ~95% saturated at the anchors.
        # sigmoid(3) ~ 0.953 -> we want -k*(h_low - mid) = 3 -> k = 6/(h_high - h_low)
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
        print(f"+++ TENT-SigGate: h_range=[{self.h_low}, {self.h_high}], "
              f"mid={self.h_mid:.3f}, k={self.sigmoid_k:.3f} ({k_src}), "
              f"max_rst={self.max_rst}, monitor_interval={self.monitor_interval}")
        # Print the rst at a few reference H_margin values so it's easy
        # to sanity-check the shape from the log.
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

        # ---------- Per-batch gate logging (csv) ----------
        # Set externally via set_gate_log_path() before adaptation begins.
        # When set, every adapt batch appends `total_batch,h_margin_batch,
        # current_rst` to the file. h_margin_batch is computed from the
        # SAME marginal that was just pushed into self.marginal_buf, so it
        # is always available without extra forward passes.
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
                batch_marginal = probs.mean(dim=[0, 2, 3]).detach().float().cpu()
                self.marginal_buf.append(batch_marginal)

            loss = self.softmax_entropy(logits).mean()
            loss_report.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.current_rst > 0.0:
                self._stochastic_restore_flat(self.current_rst)
        self.batch_count += 1
        self.total_batches += 1

        # Per-batch H_margin logging (uses the marginal we just pushed).
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
        # numerically-stable sigmoid: 1 / (1 + exp(k*(H - mid)))
        z = self.sigmoid_k * (h_margin - self.h_mid)
        # clamp to avoid overflow in exp
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
        print(f"[SigGate-T] B{self.total_batches}: "
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
    # Loss (pure TENT, no Top-K mask)
    # ===========================================================

    @staticmethod
    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        return -(x.softmax(-3) * x.log_softmax(-3)).sum(-3)

    # ===========================================================
    # Shared helpers (verbatim from TENTContGateContinual)
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
