"""
TRIBEContinual: port of TRIBE (Towards Real-World Test-Time Adaptation:
Tri-Net Self-Training with Balanced Normalization, AAAI 2024,
https://arxiv.org/abs/2309.14949, github.com/Gorilla-Lab-SCUT/TRIBE ->
vendored at /home/stanfc/TTA-on-OVSS/tribe/).

*** Faithful port, by explicit user choice, of a method whose headline
mechanism (Balanced BatchNorm) has no LayerNorm equivalent -- see the
"Deviations" section below for exactly what had to be reinterpreted and
why, mirroring the disclosure precedent set by adapt/vida_continual.py.
The user was asked to pick between a Tri-Net-only reinterpretation, an
outright skip, and this faithful-anyway port; they chose this one,
accepting it is the least "natural" fit of the 5 baselines added this
session. ***

Why TRIBE doesn't transfer cleanly (see core/adapter/TRIBE.py +
core/utils/bn_layers.py for the original):
  TRIBE's core contribution, "Balanced BN", replaces each BatchNorm2d/1d
  with a layer that keeps a *per-pseudo-class* running (mean, var) per
  channel, and normalizes using the *unweighted average of the per-class
  means/vars* rather than the natural (class-frequency-weighted) batch
  statistic -- this is literally what makes it robust to class-imbalanced
  batches. `configure_model` in the original explicitly `raise`s for any
  normalization layer that isn't BatchNorm1d/2d. CLIP ViT has no
  BatchNorm at all: LayerNorm normalizes per-token over the embedding
  dim, with no batch dimension in its statistics to be "imbalanced" in
  the first place.

  However: dense OVSS turns out to have its own, different source of
  "local class imbalance" that Balanced-BN's mechanism *does* meaningfully
  address -- a single 224x224 tile's ~256 patch tokens are themselves a
  class-imbalanced population (e.g. mostly "road"/"sky", a handful of
  "pole"/"rider" tokens), which is exactly the kind of skew the paper's
  per-class running statistic was designed to counteract, just at the
  token level instead of the (whole-image) batch level. That's the
  reinterpretation used here: pseudo-label *every patch token* (from the
  model's own dense prediction) and run the paper's exact per-class
  running-statistic update (see BalancedNorm._update_statistics, ported
  from core/utils/balanced_bn_pyv.py's update_statistics_1d_v5) over that
  per-token population, instead of over a population of separate images.

Core mechanism (see core/adapter/TRIBE.py's `update_model`):
  1. A BalancedNorm layer (bolted on *after* CLIP's own final LayerNorm
     `ln_post`, on the L2-normalize-able image embedding -- see below for
     why this insertion point, not "every LayerNorm") -- trainable
     weight/bias, plus per-class running (mean, var) updated online as
     pseudo-labeled tokens stream in, normalizing by the *class-averaged*
     global statistic rather than the raw population statistic.
  2. Three "networks": student (`self.model`, standard weights), an
     auxiliary/clean-view network, and a frozen source anchor
     (`self.model_src`). In the original, "model" and "aux_model" literally
     *share the same weight tensors* (confirmed by reading TRIBE.py's
     `__init__`: `set_named_submodule(aux_model, name, param)` re-points
     aux_model's parameters at model's own tensors) -- `update_ema_variables`
     is defined but never actually called anywhere in TRIBE.py. The two
     "networks" differ only in (a) which augmentation view they see and
     (b) which BalancedNorm running-statistics profile they update. This
     port implements that faithfully as *one* shared model with *two*
     named statistics profiles on the same BalancedNorm instance
     ('clean' / 'student') rather than two separately-deep-copied model
     objects -- mathematically identical to the original, more direct to
     implement. `self.model_src` (the third network) *is* a genuine
     separate frozen deepcopy, matching the original exactly.
  3. Per adapt step: (a) a no-grad "clean" forward bootstraps a per-token
     pseudo-label from the model's own current prediction; (b) a
     grad-enabled "clean" forward (updates the 'clean' BN stats with that
     label) produces the entropy mask and the CE target / MSE-regression
     target; (c) a grad-enabled "student" forward on a strongly-augmented
     view (updates the 'student' BN stats with the *same* label) is
     pushed toward the clean view's pseudo-label via masked cross-entropy;
     (d) a no-grad forward through the frozen source-anchor network
     regularizes the clean view's output toward its own (never-adapted)
     prediction via masked MSE. Only entropy-reliable pixels
     (entropy < H0 * log(num_classes)) contribute to either loss term.

Deviations from the original (documented, not the paper's own choices):
  - BalancedNorm is inserted at exactly one point: on the L2-normalizable
    image embedding CLIP produces right before the cosine-similarity-
    with-text step (same insertion point adapt/spa_continual.py's
    predictor uses) -- the closest analog to "the final BatchNorm before
    a ResNet's FC classifier" in the original backbone, not a
    per-transformer-block replacement (there is nothing residing at every
    block to replace -- see above). The CLS token is left untouched by
    BalancedNorm (it has no per-pixel pseudo-label); only the patch
    tokens are pseudo-labeled and normalized through it.
  - Per-*token* pseudo-labels (from the model's own dense prediction)
    replace the original's per-*image* label -- see above for why this is
    arguably a *better*-motivated fit for TRIBE's mechanism here, not a
    weaker one.
  - "Tri-Net" implemented as one shared model + two named BalancedNorm
    statistics profiles instead of two deep-copied model objects (see
    point 2 above) -- confirmed mathematically identical to the original
    given `update_ema_variables` is dead code there.
  - Dense (B, C, H, W) logits/entropy replace (B, C) throughout; the
    reliable-pixel mask (entropy < H0 * log(num_classes)) is per-pixel.
  - Adam `eps=1e-4` instead of PyTorch's fp32-tuned default `1e-8` (see
    adapt/vida_continual.py's docstring for why); BalancedNorm's own
    running-statistics buffers are kept in fp32 for the same underflow/
    overflow reasons documented there and in adapt/come_continual.py.
"""
import math
import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ovss import load_ovss
from adapt.cotta import get_tta_transforms
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class BalancedNorm(nn.Module):
    """Per-class running (mean, var) over an embedding dim, normalizing by
    the *class-averaged* (not population) global statistic. Ported from
    core/utils/balanced_bn_pyv.py's update_statistics_1d_v5 + BalancedBNV5/
    BalancedRobustBN1dV5 (see module docstring for the dim/insertion-point
    deviations). Maintains two independent statistics profiles ('clean',
    'student') selected via `self.view`, matching the original's two
    weight-sharing "networks"."""

    def __init__(self, dim, num_classes, momentum=0.005, gamma=0.0,
                 device='cpu', dtype=torch.float16):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.momentum = momentum
        self.gamma = gamma
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(dim, device=device, dtype=dtype))
        for view in ('clean', 'student'):
            self.register_buffer(f'{view}_local_mean', torch.zeros(num_classes, dim, dtype=torch.float32, device=device))
            self.register_buffer(f'{view}_local_var', torch.ones(num_classes, dim, dtype=torch.float32, device=device))
        self.view = 'clean'
        self.label = None  # (N,) long, set externally right before forward()

    def forward(self, x):
        # x: (N, D), fp16 (or whatever the model's dtype is)
        local_mean = getattr(self, f'{self.view}_local_mean')
        local_var = getattr(self, f'{self.view}_local_var')
        if self.label is not None and self.training:
            self._update_statistics(local_mean, local_var, x.detach().float(), self.label)
            self.label = None
        global_mean = local_mean.mean(0)
        global_var = local_var.mean(0) + local_mean.var(0)
        xn = (x - global_mean.to(x.dtype)) / torch.sqrt(global_var.to(x.dtype) + self.eps)
        return xn * self.weight + self.bias

    @torch.no_grad()
    def _update_statistics(self, local_mean, local_var, data, label):
        """Verbatim formula from update_statistics_1d_v5, specialized to
        L=1 (each token is already a single (D,) sample -- no trailing
        spatial dim to pre-average, unlike the original's per-image conv
        feature map)."""
        l = label.unique()
        l_mapping = torch.arange(l.size(0), device=l.device)
        l_reverse = l.new_full((self.num_classes,), -1)
        l_reverse[l] = l_mapping
        label_local = l_reverse[label]                              # (N,)
        lm = local_mean[label]                                      # (N, D)

        label_num = torch.zeros(l.size(0), device=label.device, dtype=torch.long)
        label_num.scatter_add_(0, label_local, torch.ones_like(label_local))
        mask = label_num > (1.0 / self.momentum)
        m = torch.where(mask, 1.0 / label_num.float().clamp(min=1),
                         torch.full_like(label_num, self.momentum, dtype=torch.float32))

        delta = data - lm                                           # (N, D)
        delta_k = torch.zeros(l.size(0), self.dim, dtype=torch.float32, device=data.device)
        delta_k.scatter_add_(0, label_local.unsqueeze(-1).expand(-1, self.dim), delta)
        delta_k *= m.view(-1, 1)
        local_mean[l] = (1.0 - self.gamma) * delta_k + local_mean[l]
        local_mean += self.gamma * delta_k.mean(0, keepdim=True)

        delta_sq_k = torch.zeros(l.size(0), self.dim, dtype=torch.float32, device=data.device)
        delta_sq_k.scatter_add_(0, label_local.unsqueeze(-1).expand(-1, self.dim), delta.pow(2))
        local_var[l] = local_var[l] + (1.0 - self.gamma) * (
            m.view(-1, 1) * (delta_sq_k - label_num.view(-1, 1).float() * local_var[l]) - delta_k.pow(2))
        var_gap = (m.view(-1, 1) * delta_sq_k - delta_k.pow(2)).mean(0, keepdim=True)
        local_var += self.gamma * (var_gap - (m * label_num.float()).mean() * local_var)


class TRIBEContinual:
    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 catseg_checkpoint=None, maft_checkpoint=None,
                 prompt_dir=None, runtime_calculation=False, device='cpu',
                 eta=0.005, gamma=0.0, lam=0.5, h0=0.05):
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps
        self.runtime = runtime_calculation
        self.device = device
        self.lam = float(lam)
        self.h0 = float(h0)

        if classes is None:
            raise ValueError("classes is required in __init__")
        self.classes = classes
        self.num_classes = len(classes)

        self.model, self.tokenize = load_ovss(ovss_type, ovss_backbone, device=device,
                                              classes=self.classes, catseg_checkpoint=catseg_checkpoint,
                                              maft_checkpoint=maft_checkpoint)

        if prompt_dir:
            self.prompt_templates = load_prompts_from_yaml(prompt_dir)
        else:
            self.prompt_templates = [REFERENCE_PROMPT]

        self.model.transformer.requires_grad_(False)
        self.model.ln_final.requires_grad_(False)
        self.model.token_embedding.requires_grad_(False)
        self.model.visual = self.set_ln_grads(self.model.visual)
        ln_params, _ = self.collect_ln_params(self.model.visual)

        embed_dim = self.model.visual.output_dim
        self.balanced_norm = BalancedNorm(embed_dim, self.num_classes, momentum=eta, gamma=gamma,
                                           device=device, dtype=self.model.dtype)

        print_clip_parameters(self.model)

        self.optimizer = optim.Adam(list(ln_params) + list(self.balanced_norm.parameters()),
                                     lr=self.lr, betas=(0.9, 0.999), eps=1e-4, weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer)

        # Frozen source anchor: a genuine separate deepcopy (matches original).
        self.model_src = copy.deepcopy(self.model)
        self.model_src.requires_grad_(False)
        self.balanced_norm_src = BalancedNorm(embed_dim, self.num_classes, momentum=eta, gamma=gamma,
                                               device=device, dtype=self.model.dtype)
        self.balanced_norm_src.load_state_dict(self.balanced_norm.state_dict())
        self.balanced_norm_src.requires_grad_(False)

        self.transform = get_tta_transforms(img_size=224)

        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(
                self.classes, self.prompt_templates, average=False
            ).squeeze()

        if self.runtime:
            self.adapt_times = []
            self.eval_times = []

    # ===========================================================
    # Pre-stream BalancedNorm calibration (called once before adaptation)
    # ===========================================================

    def obtain_src_bn_stats(self, src_loader, max_samples=200):
        """Calibrate BalancedNorm's (mean, var) from frozen source-model
        patch-token embeddings, before any adaptation happens.

        Bug found on a real 150-round run (TRIBE showed no early-gain
        phase at all -- monotonic decline from Round 1, unlike every other
        entropy/consistency-style method in this repo which improves for
        some number of rounds before drifting): the original's
        `BalancedBNV5.__init__` seeds *every* class's local_mean/local_var
        from the wrapped BatchNorm layer's own *pretrained* running
        statistics (`deepcopy(bn_layer.running_mean)`, see
        core/utils/bn_layers.py) -- it never starts from a cold (0, 1)
        baseline. CLIP ViT has no pretrained BatchNorm to copy from (that's
        the whole reason BalancedNorm had to be bolted on fresh here), so
        this port was left initializing at (mean=0, var=1) -- a badly wrong
        baseline for CLIP's actual patch-embedding distribution, which
        distorts every class's normalized tokens (including common ones
        like "road"/"sky") until enough per-class updates accumulate to
        correct it. This computes the equivalent empirical statistic from
        a small source-proxy pass instead, matching the paper's own
        "start from a real, calibrated statistic" property.
        """
        device = self.device
        feats = []
        n_seen = 0
        with torch.no_grad():
            for batch in src_loader:
                x = batch['img_patches'] if isinstance(batch, dict) else batch[0]
                x = x.to(device)
                _, feat, _ = self.model(x, self.text_x, True, interpolate=False)
                feats.append(feat[:, 1:, :].reshape(-1, feat.shape[-1]).float())
                n_seen += x.size(0)
                if n_seen >= max_samples:
                    break
        all_feats = torch.cat(feats, dim=0)  # (N, D)
        mean = all_feats.mean(dim=0)
        var = all_feats.var(dim=0, unbiased=False)
        for bn in (self.balanced_norm, self.balanced_norm_src):
            for view in ('clean', 'student'):
                getattr(bn, f'{view}_local_mean').copy_(mean.unsqueeze(0).expand(self.num_classes, -1))
                getattr(bn, f'{view}_local_var').copy_(var.unsqueeze(0).expand(self.num_classes, -1))
        print(f"[TRIBE] BalancedNorm calibrated from {n_seen} source samples "
              f"(mean norm={mean.norm().item():.3f}, var mean={var.mean().item():.3f})")

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
        self.balanced_norm.eval()
        logits, _, _ = self._forward_with_bn(self.model, self.balanced_norm, x,
                                              view='clean', interpolate=True)
        logits = logits[0]
        if self.runtime:
            self.eval_times.append(time.time() - t1)
        return logits

    def reset(self):
        self.load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state)

    def state_dict(self):
        """Adapt-state checkpoint for mid-stream resume: every tensor in the
        optimizer (LN + balanced_norm weight/bias) plus its state, plus
        balanced_norm's full state_dict (its running-mean/var buffers are not
        gradient-tracked and would otherwise be silently lost on resume)."""
        opt_params = [p for g in self.optimizer.param_groups for p in g['params']]
        return {
            'params': [p.detach().cpu().clone() for p in opt_params],
            'optimizer': self.optimizer.state_dict(),
            'balanced_norm': {k: v.detach().cpu().clone()
                              for k, v in self.balanced_norm.state_dict().items()},
        }

    def load_state_dict(self, sd):
        opt_params = [p for g in self.optimizer.param_groups for p in g['params']]
        with torch.no_grad():
            for p, saved in zip(opt_params, sd['params']):
                p.data.copy_(saved.to(device=p.device, dtype=p.dtype))
        self.optimizer.load_state_dict(sd['optimizer'])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)
        bn_sd = {k: v.to(self.device) for k, v in sd['balanced_norm'].items()}
        self.balanced_norm.load_state_dict(bn_sd, strict=True)

    # ===========================================================
    # Adaptation
    # ===========================================================

    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []
        for _ in range(self.steps):
            # ---- Pass 1: bootstrap pseudo-label (eval mode: no stats update) ----
            self.balanced_norm.eval()
            with torch.no_grad():
                logits0, _, _ = self._forward_with_bn(self.model, self.balanced_norm, x, view='clean')
            p_l = logits0[0].argmax(dim=1)          # (B, gh, gw)
            p_l_flat = p_l.reshape(-1)

            # ---- Pass 2: clean-view forward WITH grad, updates 'clean' BN stats ----
            self.balanced_norm.train()
            clean_logits, _, _ = self._forward_with_bn(
                self.model, self.balanced_norm, x, view='clean', update_label=p_l_flat)
            clean_l0 = clean_logits[0]

            entropy = self.softmax_entropy(clean_l0)
            mask = entropy < self.h0 * math.log(self.num_classes)

            if mask.sum() == 0:
                loss = clean_l0.sum() * 0.0
                loss_report.append(loss.item())
                self.optimizer.zero_grad()
                continue

            # ---- Pass 3: student forward on strongly-augmented view, 'student' BN stats ----
            aug_x = self.transform(x)
            stu_logits, _, _ = self._forward_with_bn(
                self.model, self.balanced_norm, aug_x, view='student', update_label=p_l_flat)
            stu_l0 = stu_logits[0]

            # ---- Pass 4: frozen source anchor (no grad, stats never updated after init) ----
            self.balanced_norm_src.eval()
            with torch.no_grad():
                src_logits, _, _ = self._forward_with_bn(self.model_src, self.balanced_norm_src, x, view='clean')
            src_l0 = src_logits[0]

            l_sup = F.cross_entropy(stu_l0.permute(0, 2, 3, 1)[mask],
                                     clean_l0.argmax(dim=1)[mask])
            l_reg = self.lam * F.mse_loss(clean_l0.permute(0, 2, 3, 1)[mask],
                                           src_l0.permute(0, 2, 3, 1)[mask])
            loss = l_sup + l_reg

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_report.append(loss.item())

        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

    def _forward_with_bn(self, model, balanced_norm, x, view=None,
                          update_label=None, interpolate=False):
        """One CLIP forward + external BalancedNorm applied to the patch
        tokens (CLS left untouched, see module docstring), then logits
        recomputed from the BN-adjusted, renormalized features -- same
        two-step pattern adapt/spa_continual.py's predictor branch uses."""
        _, feat, text_features = model(x, self.text_x, True, interpolate=False)
        B, tok, D = feat.shape
        cls_tok, patch_tok = feat[:, :1, :], feat[:, 1:, :]
        if view is not None:
            balanced_norm.view = view
        if update_label is not None:
            balanced_norm.label = update_label
        patch_bn = balanced_norm(patch_tok.reshape(-1, D)).reshape(B, tok - 1, D)
        feat_bn = torch.cat([cls_tok, patch_bn], dim=1)
        feat_bn = feat_bn / feat_bn.norm(dim=-1, keepdim=True)
        logits = self._logits_from_image_features(feat_bn, text_features, x.shape, interpolate)
        return logits, feat_bn, text_features

    def _logits_from_image_features(self, image_features, text_features, image_shape, interpolate=False):
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * torch.einsum('bsd,tcd->tbsc', image_features, text_features)
        logits = logits[:, :, 1:]  # drop CLS
        patch_size = self.model.visual.patch_size
        w, h = image_shape[-2] // patch_size, image_shape[-1] // patch_size
        T, B, S, C = logits.shape
        logits = logits.permute(0, 1, 3, 2).reshape(T, B, C, w, h)
        if interpolate:
            logits = logits.reshape(-1, C, w, h)
            logits = F.interpolate(logits, size=image_shape[-2:], mode='bilinear', align_corners=False)
            logits = logits.view(T, B, C, image_shape[-2], image_shape[-1])
        return logits

    @staticmethod
    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        return -(x.softmax(1) * x.log_softmax(1)).sum(1)

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
