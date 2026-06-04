"""
SAR-Continual: Sharpness-Aware Reliable test-time adaptation for OVSS, no reset.

Three independent mechanisms over TENT base loss:
  1. Reliable sample filtering: skip samples with mean pixel entropy > e_margin.
  2. SAM optimizer step: perturb LN weights by rho*grad/||grad||, then take
     gradient at perturbed point as the real update direction. Flatter minima
     generalize better when the source-vs-target adaptation gap is small.
  3. Model recovery: if EMA of the (filtered) loss exceeds e_0 after warmup,
     reset all visual LN params to the source snapshot and clear the EMA.

Reference: Niu et al., "Towards Stable Test-Time Adaptation in Dynamic Wild
World", ICLR 2023. See docs/2026-05-17-sar-eata-v20-design.md.
"""

import os
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters
from .sam import SAM

REFERENCE_PROMPT = 'a photo of a {}'


class SARContinual:

    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 e_margin=1.198, sam_rho=0.05,
                 e_0=0.2, ema_factor=0.9, recovery_warmup=50,
                 prompt_dir=None,
                 save_dir=None,
                 runtime_calculation=False, device='cpu'):
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps

        self.e_margin = float(e_margin)
        self.sam_rho = float(sam_rho)
        self.e_0 = float(e_0)
        self.ema_factor = float(ema_factor)
        self.recovery_warmup = int(recovery_warmup)
        if not (0.0 <= self.ema_factor < 1.0):
            raise ValueError(f"ema_factor must be in [0, 1), got {self.ema_factor}")

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
        print(f"+++ SAR: e_margin={self.e_margin:.3f}, sam_rho={self.sam_rho}, "
              f"e_0={self.e_0}, ema_factor={self.ema_factor}, "
              f"recovery_warmup={self.recovery_warmup}")

        # ---------- SAM-wrapped Adam (LN params only) ----------
        self.optimizer = SAM(
            params, optim.Adam, rho=self.sam_rho,
            lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0,
        )
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

        # ---------- SAR runtime state ----------
        self.loss_ma = None        # EMA of filtered loss
        self.total_batches = 0

        # ---------- Optional per-batch log ----------
        self.log_path = None
        if save_dir is not None:
            try:
                os.makedirs(save_dir, exist_ok=True)
                self.log_path = os.path.join(save_dir, 'sar_log.txt')
                with open(self.log_path, 'w') as f:
                    f.write("# SAR log: per-batch entropy/filter/recovery\n")
                    f.write("total_batches,mean_entropy,was_filtered,loss_ma,was_reset\n")
                print(f"+++ SAR log -> {self.log_path}")
            except OSError as e:
                print(f"+++ SAR log disabled ({save_dir}): {e}")
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
        logits, _, _ = self.model(x, self.text_x, True, interpolate=True)
        logits = logits[0]
        if self.runtime:
            self.eval_times.append(time.time() - t1)
        return logits

    def reset(self):
        self.load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state
        )
        self.loss_ma = None

    # ===========================================================
    # Adaptation
    # ===========================================================

    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []

        was_filtered = False
        was_reset = False
        ent_mean_for_log = float('nan')

        for _ in range(self.steps):
            # First forward to evaluate reliability of this sample
            logits, _, _ = self.model(x, self.text_x, True, interpolate=False)
            ent_map = self.softmax_entropy(logits)                # (B, w, h) reduced over class
            ent_mean = ent_map.mean()
            ent_mean_for_log = ent_mean.item()

            # Filter (A): skip unreliable sample entirely
            if ent_mean.item() > self.e_margin:
                was_filtered = True
                self.optimizer.zero_grad()
                continue

            # Filter (A) passed → compute loss only on reliable pixels
            # (per-pixel entropy < e_margin within the kept sample).
            pixel_mask = ent_map < self.e_margin
            if pixel_mask.sum().item() == 0:
                was_filtered = True
                self.optimizer.zero_grad()
                continue
            loss = ent_map[pixel_mask].mean()
            loss_report.append(loss.item())

            # ----- SAM step 1: ascend to perturbed weights -----
            loss.backward()
            self.optimizer.first_step(zero_grad=True)

            # ----- SAM step 2: gradient at perturbed point -----
            logits2, _, _ = self.model(x, self.text_x, True, interpolate=False)
            ent_map2 = self.softmax_entropy(logits2)
            pixel_mask2 = ent_map2 < self.e_margin
            if pixel_mask2.sum().item() == 0:
                # Edge case: perturbation made all pixels unreliable. Skip update
                # but restore weights from the perturbed position.
                self.optimizer.second_step(zero_grad=True)
                continue
            loss2 = ent_map2[pixel_mask2].mean()
            loss2.backward()
            self.optimizer.second_step(zero_grad=True)

            # ----- Update loss EMA (use the second-step loss) -----
            current_loss = loss2.item()
            if self.loss_ma is None:
                self.loss_ma = current_loss
            else:
                self.loss_ma = (self.ema_factor * self.loss_ma
                                + (1.0 - self.ema_factor) * current_loss)

            # ----- Filter (C): model recovery -----
            if (self.total_batches >= self.recovery_warmup
                    and self.loss_ma is not None
                    and self.loss_ma > self.e_0):
                self._model_recovery()
                was_reset = True

        self.total_batches += 1
        if self.log_path is not None:
            try:
                with open(self.log_path, 'a') as f:
                    f.write(f"{self.total_batches},{ent_mean_for_log:.6f},"
                            f"{int(was_filtered)},"
                            f"{self.loss_ma if self.loss_ma is not None else float('nan'):.6f},"
                            f"{int(was_reset)}\n")
            except OSError:
                pass

        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

    # ===========================================================
    # Model recovery
    # ===========================================================

    @torch.no_grad()
    def _model_recovery(self):
        for name, p in self.named_ln_params:
            src = self.model_state[name].to(p.device, dtype=p.dtype)
            p.data.copy_(src)
        self.loss_ma = None
        print(f"[SAR] B{self.total_batches}: model recovery triggered (loss_ma exceeded e_0={self.e_0})")

    # ===========================================================
    # Loss (per-pixel entropy, dim=-3 over classes)
    # ===========================================================

    @staticmethod
    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, C, w, h) — sum over class dim, reduce T by mean
        ent = -(x.softmax(-3) * x.log_softmax(-3)).sum(-3)   # (T, B, w, h)
        return ent.mean(dim=0)                                 # (B, w, h)

    # ===========================================================
    # Shared helpers (mirroring TENTDivGateContinual)
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
