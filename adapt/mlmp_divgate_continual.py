import os
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


class MLMPDivGateContinual:
    """
    MLMP-DivGate-Continual: MLMP multi-prompt multi-level entropy loss + diversity-gated
    stochastic restoration (CTTA, no reset).

    Loss: identical to MLMPContinual (perform_adaptation) — pixel entropy averaged
    over T prompts plus optional ILE (CLS-token entropy weighted by alpha_cls).
    Supports both prompt_integration='loss' and 'text' modes.

    Gate: identical to TENTDivGateContinual / CMADivGateContinual. Every
    monitor_interval batches, marginal class entropy H_margin is computed from
    a buffer of per-batch marginals (averaged over prompts when applicable),
    and restoration probability is selected from three modes.
    """

    def __init__(self, ovss_type, ovss_backbone, lr, classes,
                 vision_outputs=(-1,), alpha_cls=0.0, steps=1,
                 prompt_dir='prompts.yaml', prompt_integration='loss',
                 h_threshold=1.8, h_warning=1.2,
                 monitor_interval=50,
                 cautious_rst=0.005, brake_rst=0.05,
                 save_dir=None,
                 runtime_calculation=False, device='cpu'):
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps
        self.vision_outputs = vision_outputs
        self.alpha_cls = alpha_cls
        self.prompt_dir = prompt_dir
        self.prompt_integration = prompt_integration

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

        self.runtime = runtime_calculation
        self.device = device

        if classes is None:
            raise ValueError("classes is required in __init__")
        self.classes = classes

        assert prompt_integration in ['loss', 'text'], \
            "prompt_integration must be 'loss' or 'text'"

        # ---------- OVSS model ----------
        self.model, self.tokenize = load_ovss(ovss_type, ovss_backbone, device=device)

        # ---------- Prompts ----------
        if prompt_dir:
            self.prompt_templates = load_prompts_from_yaml(prompt_dir)
            print(f"Number of prompt templates: {len(self.prompt_templates)}")
        else:
            self.prompt_templates = [REFERENCE_PROMPT]

        print(f"+++ Vision outputs (UAML layers): {self.vision_outputs}")

        # ---------- Freeze text encoder, enable LN grads in visual encoder ----------
        self.model.transformer.requires_grad_(False)
        self.model.ln_final.requires_grad_(False)
        self.model.token_embedding.requires_grad_(False)
        self.model.visual = self.set_ln_grads(self.model.visual)
        params, names = self.collect_ln_params(self.model.visual)
        self.named_ln_params = list(zip(names, params))

        print_clip_parameters(self.model)
        print(f"+++ MLMP-DivGate: thresholds=(warn {self.h_warning}, agg {self.h_threshold}), "
              f"monitor_interval={self.monitor_interval}, "
              f"rsts=(cautious {self.cautious_rst}, brake {self.brake_rst})")
        print(f"+++ Initial mode: {MODE_AGGRESSIVE} (rst=0); buffer fills before first re-evaluation")

        # ---------- Optimizer ----------
        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        # ---------- Source state snapshot ----------
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer
        )

        # ---------- Text embeddings: shape (T+1, C, D); avg at index [-1] ----------
        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(
                self.classes, self.prompt_templates, average=True
            ).squeeze()

        # ---------- Gate state ----------
        self.marginal_buf = []
        self.batch_count = 0
        self.total_batches = 0
        self.current_mode = MODE_AGGRESSIVE
        self.current_rst = 0.0

        # ---------- Optional per-monitor file log ----------
        self.log_path = None
        if save_dir is not None:
            try:
                os.makedirs(save_dir, exist_ok=True)
                self.log_path = os.path.join(save_dir, 'divgate_log.txt')
                with open(self.log_path, 'w') as f:
                    f.write("# DivGate log: per-monitor H_margin and mode\n")
                    f.write("total_batches,h_margin,mode\n")
                print(f"+++ DivGate log -> {self.log_path}")
            except OSError as e:
                print(f"+++ DivGate log disabled (could not open {save_dir}): {e}")
                self.log_path = None

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
        logits, _, _ = self.model(
            x, self.text_x[-1], True,
            vision_outputs=self.vision_outputs,
            interpolate=True,
            vision_out_type="adaptive_weighted_mean",
            save_weights=True
        )
        logits = logits[0]
        if self.runtime:
            self.eval_times.append(time.time() - t1)
        return logits

    def reset(self):
        self.load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state
        )

    # ===========================================================
    # Adaptation (MLMP loss + diversity gate)
    # ===========================================================

    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []

        for _ in range(self.steps):
            if self.prompt_integration == 'loss':
                logits, _, _, cls_logits = self.model(
                    x, self.text_x[:-1], True,
                    interpolate=False,
                    vision_outputs=self.vision_outputs,
                    return_vanilla_cls=True,
                    vision_out_type="mean"
                )
                # logits:     (T, B, C, h, w)
                # cls_logits: (T, B, C, 1, 1)
                entropy_per_pixel = self.softmax_entropy(logits)            # (T, B, h, w)
                entropy_per_cls = self.softmax_entropy(cls_logits, dim=2)   # (T, B, 1, 1)
                loss = entropy_per_pixel.mean() + self.alpha_cls * entropy_per_cls.mean()

                with torch.no_grad():
                    avg_logits = logits.mean(dim=0)  # (B, C, h, w)
                    probs = avg_logits.softmax(dim=1)
                    self.marginal_buf.append(
                        probs.mean(dim=[0, 2, 3]).detach().float().cpu()
                    )

            else:  # 'text' branch
                logits, _, _ = self.model(
                    x, self.text_x[-1], True,
                    interpolate=False,
                    vision_outputs=self.vision_outputs
                )
                loss = self.softmax_entropy(logits).mean()

                with torch.no_grad():
                    flat = logits if logits.dim() == 4 else logits.reshape(-1, *logits.shape[-3:])
                    probs = flat.softmax(dim=1)
                    self.marginal_buf.append(
                        probs.mean(dim=[0, 2, 3]).detach().float().cpu()
                    )

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
    # Diversity gate (identical to TENT/CMA DivGate variants)
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
            print(f"[DivGate-M] B{self.total_batches}: H_margin={h_margin:.3f}  "
                  f"{self.current_mode} -> {new_mode}")
        self.current_mode = new_mode
        self.current_rst = self._mode_to_rst(new_mode)

        if self.log_path is not None:
            try:
                with open(self.log_path, 'a') as f:
                    f.write(f"{self.total_batches},{h_margin:.6f},{self.current_mode}\n")
            except OSError:
                pass

        self.marginal_buf.clear()
        self.batch_count = 0

    @torch.no_grad()
    def _stochastic_restore_flat(self, rst):
        for name, p in self.named_ln_params:
            mask = (torch.rand(p.shape, device=p.device) < rst).to(p.dtype)
            src = self.model_state[name].to(p.device, dtype=p.dtype)
            p.data.mul_(1.0 - mask).add_(src * mask)

    # ===========================================================
    # Shared helpers (verbatim from MLMPContinual / TENTDivGateContinual)
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

    @staticmethod
    def softmax_entropy(x: torch.Tensor, dim=-3) -> torch.Tensor:
        return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)
