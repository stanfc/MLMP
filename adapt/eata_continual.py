"""
EATA-Continual: Efficient anti-forgetting test-time adaptation for OVSS, no reset.

Three mechanisms over TENT base loss:
  1. Pre-stream Fisher: one forward pass over the clean source split before
     adaptation begins; per-LN-parameter Fisher F_i = E[(d log p / d theta)^2].
  2. Reliable + non-redundant sample filtering at adapt time:
     - Reliable: mean pixel entropy < e_margin.
     - Non-redundant: cosine(current sample's mean-pixel logit, running EMA
       of recent samples' logits) < 1 - d_margin.
  3. EWC loss: L = L_TENT + fisher_alpha * sum_i F_i * (theta_i - theta_i^src)^2

Reference: Niu et al., "Efficient Test-Time Model Adaptation without
Forgetting", ICML 2022. See docs/2026-05-17-sar-eata-v20-design.md.
"""

import os
import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class EATAContinual:

    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 e_margin=1.198, d_margin=0.05,
                 fisher_alpha=2000.0, fisher_size=2000,
                 prompt_dir=None,
                 save_dir=None,
                 runtime_calculation=False, device='cpu'):
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps

        self.e_margin = float(e_margin)
        self.d_margin = float(d_margin)
        self.fisher_alpha = float(fisher_alpha)
        self.fisher_size = int(fisher_size)

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
        print(f"+++ EATA: e_margin={self.e_margin:.3f}, d_margin={self.d_margin}, "
              f"fisher_alpha={self.fisher_alpha}, fisher_size={self.fisher_size}")

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

        # ---------- EATA runtime state ----------
        self.fisher = None              # dict[str, Tensor], populated by obtain_src_fisher
        self.src_params = None          # dict[str, Tensor], snapshot at Fisher computation time
        self.current_logit_ema = None   # running mean of per-sample mean-logit (normalised)
        self.logit_ema_decay = 0.9
        self.total_batches = 0

        # ---------- Optional per-batch log ----------
        self.log_path = None
        if save_dir is not None:
            try:
                os.makedirs(save_dir, exist_ok=True)
                self.log_path = os.path.join(save_dir, 'eata_log.txt')
                with open(self.log_path, 'w') as f:
                    f.write("# EATA log: per-batch filtering and loss components\n")
                    f.write("total_batches,mean_entropy,reliable,non_redundant,"
                            "tent_loss,ewc_loss\n")
                print(f"+++ EATA log -> {self.log_path}")
            except OSError as e:
                print(f"+++ EATA log disabled ({save_dir}): {e}")
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
        self.current_logit_ema = None

    # ===========================================================
    # Pre-stream Fisher computation (called once before adaptation)
    # ===========================================================

    def obtain_src_fisher(self, src_loader):
        """Compute per-LN-param Fisher diagonal from clean source data.

        Iterates up to fisher_size samples. For each: forward, build pseudo-
        label (argmax over class), CE loss against it, backward; accumulate
        grad^2 per param. After: normalise by n_samples; snapshot LN weights
        as src_params (the EWC anchor).
        """
        device = self.device
        print(f"\n[EATA] Computing source Fisher on up to {self.fisher_size} samples ...")

        # Track running grad^2 accumulators per LN param
        fisher_accum = {name: torch.zeros_like(p, device=device)
                        for name, p in self.named_ln_params}
        n_seen = 0

        # Ensure grads can flow into LN
        for _, p in self.named_ln_params:
            p.requires_grad_(True)

        for batch in src_loader:
            # MMSeg-style loader yields dicts with 'img_patches' key.
            if isinstance(batch, dict):
                x = batch['img_patches']
            elif isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device)

            self.optimizer.zero_grad()
            logits, _, _ = self.model(x, self.text_x, True, interpolate=False)
            # logits: (T, B, C, w, h); use first prompt template (single)
            l0 = logits[0]                                          # (B, C, w, h)
            with torch.no_grad():
                pseudo = l0.argmax(dim=1)                            # (B, w, h)
            ce = F.cross_entropy(l0, pseudo, reduction='mean')
            ce.backward()

            with torch.no_grad():
                for name, p in self.named_ln_params:
                    if p.grad is not None:
                        fisher_accum[name].add_(p.grad.detach().pow(2))

            n_seen += x.size(0)
            if n_seen >= self.fisher_size:
                break

        self.optimizer.zero_grad()
        if n_seen == 0:
            raise RuntimeError("EATA: source loader produced no samples.")

        self.fisher = {name: (acc / float(n_seen)).detach()
                       for name, acc in fisher_accum.items()}
        self.src_params = {name: p.detach().clone()
                           for name, p in self.named_ln_params}

        total = sum(v.numel() for v in self.fisher.values())
        mean_f = sum(v.sum().item() for v in self.fisher.values()) / total
        print(f"[EATA] Fisher done: {n_seen} samples, {total} params, mean F={mean_f:.3e}")

    # ===========================================================
    # Adaptation
    # ===========================================================

    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []

        reliable = False
        non_redundant = False
        ent_mean_for_log = float('nan')
        tent_loss_for_log = float('nan')
        ewc_loss_for_log = float('nan')

        for _ in range(self.steps):
            logits, _, _ = self.model(x, self.text_x, True, interpolate=False)
            ent_map = self.softmax_entropy(logits)                  # (B, w, h)
            ent_mean = ent_map.mean()
            ent_mean_for_log = ent_mean.item()

            # ----- Reliable filter -----
            if ent_mean.item() > self.e_margin:
                self.optimizer.zero_grad()
                continue
            reliable = True

            # ----- Non-redundant filter (cosine vs running EMA of mean-logits) -----
            with torch.no_grad():
                l0 = logits[0]                                       # (B, C, w, h)
                # Mean over batch + spatial → (C,)
                mean_logit = l0.mean(dim=[0, 2, 3])
                mean_logit_n = mean_logit / mean_logit.norm().clamp(min=1e-8)

                if self.current_logit_ema is None:
                    cos_sim = 0.0      # first sample is always non-redundant
                else:
                    cos_sim = (mean_logit_n
                               * self.current_logit_ema).sum().item()
                if cos_sim < 1.0 - self.d_margin:
                    non_redundant = True

            if not non_redundant:
                # Update EMA with what we saw (even if filtered) so future
                # comparisons reflect current stream.
                self._update_logit_ema(mean_logit_n)
                self.optimizer.zero_grad()
                continue

            # ----- TENT loss + EWC penalty -----
            tent_loss = ent_map.mean()
            ewc_loss = self._ewc_penalty() if self.fisher is not None else 0.0
            loss = tent_loss + self.fisher_alpha * ewc_loss

            tent_loss_for_log = float(tent_loss.item())
            ewc_loss_for_log = float(ewc_loss.item()) if torch.is_tensor(ewc_loss) else float(ewc_loss)
            loss_report.append(loss.item())

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Update logit EMA with the (reliable + non-redundant) sample
            with torch.no_grad():
                self._update_logit_ema(mean_logit_n)

        self.total_batches += 1
        if self.log_path is not None:
            try:
                with open(self.log_path, 'a') as f:
                    f.write(f"{self.total_batches},{ent_mean_for_log:.6f},"
                            f"{int(reliable)},{int(non_redundant)},"
                            f"{tent_loss_for_log:.6f},{ewc_loss_for_log:.6e}\n")
            except OSError:
                pass

        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

    # ===========================================================
    # EATA helpers
    # ===========================================================

    @torch.no_grad()
    def _update_logit_ema(self, mean_logit_n):
        if self.current_logit_ema is None:
            self.current_logit_ema = mean_logit_n.detach().clone()
        else:
            self.current_logit_ema.mul_(self.logit_ema_decay).add_(
                mean_logit_n.detach() * (1.0 - self.logit_ema_decay)
            )
            self.current_logit_ema.div_(self.current_logit_ema.norm().clamp(min=1e-8))

    def _ewc_penalty(self):
        """Sum_i F_i * (theta_i - theta_i^src)^2 over all LN params."""
        total = 0.0
        for name, p in self.named_ln_params:
            f = self.fisher.get(name)
            src = self.src_params.get(name)
            if f is None or src is None:
                continue
            total = total + (f * (p - src.to(p)).pow(2)).sum()
        return total

    # ===========================================================
    # Loss (per-pixel entropy reduced over class dim)
    # ===========================================================

    @staticmethod
    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        ent = -(x.softmax(-3) * x.log_softmax(-3)).sum(-3)   # (T, B, w, h)
        return ent.mean(dim=0)                                # (B, w, h)

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
