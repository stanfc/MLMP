import re
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class CMALayeredContinual:
    """
    CMA-Layered-Continual: Cross-Modal Alignment loss + layer-stratified
    stochastic restoration toward source weights (CTTA, no reset).

    Same forward and CMA loss as CMAContinual. After each gradient step
    every visual-encoder LayerNorm parameter is stochastically restored
    toward its source value with a probability that depends on which
    transformer block the parameter belongs to:

        early ([0, early_cutoff) + ln_pre)         -> early_rst (default 0.001)
        mid   ([early_cutoff, late_cutoff))         -> mid_rst   (default 0.01)
        late  ([late_cutoff, num_blocks) + ln_post) -> late_rst  (default 0.05)

    Setting any rate to 0 disables restoration for that group; setting it
    to 1 freezes those parameters. With all three rates at 0 the method
    reduces exactly to CMAContinual.

    See docs/cma_layered_continual_spec.md for the full design.
    """

    BLOCK_RE = re.compile(r'resblocks\.(\d+)\.')

    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 top_k_percent=0.2,
                 early_rst=0.001, mid_rst=0.01, late_rst=0.05,
                 early_cutoff=8, late_cutoff=16,
                 prompt_dir=None,
                 runtime_calculation=False, device='cpu'):
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps
        self.top_k_percent = float(top_k_percent)
        if not (0.0 < self.top_k_percent <= 1.0):
            raise ValueError(
                f"top_k_percent must be in (0, 1], got {self.top_k_percent}"
            )

        self.early_rst = float(early_rst)
        self.mid_rst = float(mid_rst)
        self.late_rst = float(late_rst)
        for n, v in [('early_rst', self.early_rst),
                     ('mid_rst', self.mid_rst),
                     ('late_rst', self.late_rst)]:
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
        params, names = self.collect_ln_params(self.model.visual)
        self.named_ln_params = list(zip(names, params))

        print_clip_parameters(self.model)
        self._print_layer_groups()

        # ---------- Optimizer ----------
        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        # ---------- Source state snapshot (used by layered restoration) ----------
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer
        )

        # ---------- Text features ----------
        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(
                self.classes, self.prompt_templates, average=False
            ).squeeze()

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
            logits, image_features, text_features = self.model(
                x, self.text_x, True, interpolate=False
            )
            loss = self.cma_loss(logits, image_features, text_features)
            loss_report.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self._stochastic_layered_restore()
        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

    def cma_loss(self, logits, image_features, text_features):
        """Identical to CMAContinual.cma_loss -- see docs/cma_continual_spec.md §2."""
        avg_logits = logits.mean(dim=0)
        avg_text = text_features.mean(dim=0)
        avg_text = avg_text / avg_text.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        with torch.no_grad():
            probs = avg_logits.softmax(dim=1)
            confidence, pred_cls = probs.max(dim=1)
            n_total = confidence.numel()
            k = max(1, int(round(n_total * self.top_k_percent)))
            threshold = torch.topk(confidence.flatten(), k, sorted=True).values[-1]
            mask = confidence >= threshold

        B, _, w, h = avg_logits.shape
        patch_feats = image_features[:, 1:, :]
        D = patch_feats.shape[-1]
        vis_feat = patch_feats.reshape(B, w, h, D)

        text_target = avg_text[pred_cls]
        cos_sim = (vis_feat * text_target).sum(dim=-1)
        return -cos_sim[mask].mean()

    # ===========================================================
    # Layer-stratified restoration
    # ===========================================================

    def _get_restoration_rate(self, name):
        if 'ln_pre' in name:
            return self.early_rst
        if 'ln_post' in name:
            return self.late_rst
        m = self.BLOCK_RE.search(name)
        if m is None:
            return 0.0
        idx = int(m.group(1))
        if idx < self.early_cutoff:
            return self.early_rst
        if idx < self.late_cutoff:
            return self.mid_rst
        return self.late_rst

    @torch.no_grad()
    def _stochastic_layered_restore(self):
        for name, p in self.named_ln_params:
            rst = self._get_restoration_rate(name)
            if rst <= 0.0:
                continue
            mask = (torch.rand(p.shape, device=p.device) < rst).to(p.dtype)
            src = self.model_state[name].to(p.device, dtype=p.dtype)
            p.data.mul_(1.0 - mask).add_(src * mask)

    def _print_layer_groups(self):
        groups = {'early': 0, 'mid': 0, 'late': 0}
        for name, _ in self.named_ln_params:
            if 'ln_pre' in name:
                groups['early'] += 1
                continue
            if 'ln_post' in name:
                groups['late'] += 1
                continue
            m = self.BLOCK_RE.search(name)
            if m is None:
                continue
            idx = int(m.group(1))
            if idx < self.early_cutoff:
                groups['early'] += 1
            elif idx < self.late_cutoff:
                groups['mid'] += 1
            else:
                groups['late'] += 1
        total = sum(groups.values())
        print(f"+++ CMA-Layered: cutoffs=({self.early_cutoff}, {self.late_cutoff}), "
              f"rsts=({self.early_rst}, {self.mid_rst}, {self.late_rst})")
        print(f"+++ LN params/group: early={groups['early']}, "
              f"mid={groups['mid']}, late={groups['late']}, total={total}")

    # ===========================================================
    # Shared helpers (verbatim from CMAContinual / TENTContinual)
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
