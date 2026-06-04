"""
EATAContinual — Efficient Anti-forgetting Test-time Adaptation
(Niu et al., ICML 2022). Ported from SAR/eata.py (same author group)
to the MLMP / NA-CLIP / per-pixel-OVSS setting.

EATA's three mechanisms (paper §3):
  1. Reliable sample selection: keep only low-entropy pixels
     (entropy < e_margin). Same idea as SAR's filter.
  2. Non-redundant sample selection: maintain an EMA of the predicted
     probability vector (current_model_probs); drop pixels whose
     softmax is too cosine-similar to it (|cos| < d_margin keeps it,
     i.e. only diverse predictions contribute). This avoids wasting
     gradient on near-duplicate samples.
  3. Fisher anti-forgetting regularizer (the EATA-specific part):
     L += fisher_alpha * Σ_n F_n * (θ_n − θ_src,n)^2 where F_n is the
     Fisher importance of parameter n, estimated on a SOURCE proxy pass
     BEFORE the stream. This is elastic-weight-consolidation: keep
     source-important params near their source values.

The reweighted entropy loss is the same family as SAR/DeYO:
  coeff = 1/exp(entropy − e_margin), loss = (coeff * entropy).mean().

Compatibility with our CTTA hard rules:
  - No per-sample reset. State (EMA prob vector) persists for the whole
    stream. Fisher + θ_src are fixed after the pre-stream pass.

Per-pixel adaptation:
  - entropy / cosine filters operate per-pixel (logits (1,B,C,h,w)).
  - current_model_probs is a (C,) EMA of the mean softmax over kept pixels.

Pre-stream Fisher pass:
  - compute_fishers(loader) mirrors SAR/main.py: forward source-proxy
    patches, take pseudo-label CE, accumulate grad^2 as Fisher, and
    snapshot θ_src. Wired in main_continual.py like DPCore's
    obtain_src_stat. For ACDC (no clean source) the first condition is
    the proxy.

Deviations from upstream:
  - Base optimizer Adam (MLMP convention) instead of SGD.
  - LN-only adaptation (NA-CLIP has no BatchNorm).

Hyperparameters (paper defaults, retained):
  - e_margin = 0.4 * log(C)   (reliable-entropy threshold; paper uses
    log(1000)/2-1 for 1000-class ImageNet; we use the class-count-scaled
    form consistent with SAR/DeYO here)
  - d_margin = 0.05           (cosine redundancy threshold)
  - fisher_alpha = 2000.0     (EWC trade-off)
  - fisher_size = 200         (source patches for Fisher estimate)
"""
import time
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ovss import load_ovss
from utils.misc import print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class EATAContinual:
    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 e_margin_factor=0.4,
                 d_margin=0.05,
                 fisher_alpha=2000.0,
                 fisher_size=200,
                 runtime_calculation=False, device='cpu'):
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps
        self.runtime = runtime_calculation
        self.device = device

        if classes is None:
            raise ValueError("classes is required in __init__")
        self.classes = classes
        self.num_class = len(classes)

        self.e_margin = e_margin_factor * math.log(self.num_class)
        self.d_margin = d_margin
        self.fisher_alpha = fisher_alpha
        self.fisher_size = fisher_size

        self.current_model_probs = None   # (C,) EMA of kept-pixel softmax
        self.fishers = None               # {param_name: [F, theta_src]}
        self.total_batches = 0

        self.model, self.tokenize = load_ovss(ovss_type, ovss_backbone, device=device)
        self.prompt_templates = [REFERENCE_PROMPT]

        self.model.transformer.requires_grad_(False)
        self.model.ln_final.requires_grad_(False)
        self.model.token_embedding.requires_grad_(False)
        self.model.visual = self.set_ln_grads(self.model.visual)
        params, names = self.collect_ln_params(self.model.visual)
        self._ln_param_names = set(names)

        print_clip_parameters(self.model)
        print(f"+++ EATA-Continual: e_margin={self.e_margin:.3f} "
              f"(={e_margin_factor}*log({self.num_class})), d_margin={self.d_margin}, "
              f"fisher_alpha={self.fisher_alpha}, fisher_size={self.fisher_size} "
              f"-> trainable LN params: {len(params)}")

        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer)

        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(
                self.classes, self.prompt_templates, average=False).squeeze()

        if self.runtime:
            self.adapt_times = []
            self.eval_times = []

    # -----------------------------------------------------------
    # Pre-stream Fisher pass (called once by main_continual.py)
    # -----------------------------------------------------------
    @torch.enable_grad()
    def compute_fishers(self, src_loader):
        """Estimate Fisher importance of each trainable LN param on a
        source-proxy stream, and snapshot the source weights. Mirrors
        SAR/main.py's Fisher loop (pseudo-label CE -> grad^2)."""
        print(f"[EATA] Computing Fisher information (size={self.fisher_size}) ...")
        # Only the visual LN params we actually adapt — exclude any other
        # trainable scalar (e.g. logit_scale) so EWC anchors only LN.
        named_train = {n: p for n, p in self.model.named_parameters()
                       if p.requires_grad and n in self._ln_param_names}
        fishers = {}
        seen = 0
        iter_ = 0
        for batch in src_loader:
            # prepare_data loaders yield dicts keyed 'img_patches' (see
            # main_continual.py main loop). Fall back gracefully otherwise.
            if isinstance(batch, dict):
                images = batch['img_patches']
            else:
                images = batch[0]
            images = images.to(self.device)
            iter_ += 1
            logits, _, _ = self.model(images, self.text_x, True, interpolate=False)
            logits = logits[0]                          # (B, C, h, w)
            pseudo = logits.argmax(dim=1)               # (B, h, w)
            loss = F.cross_entropy(logits, pseudo)
            loss.backward()
            for name, p in named_train.items():
                if p.grad is not None:
                    g2 = p.grad.data.clone().detach() ** 2
                    if name in fishers:
                        fishers[name][0] += g2
                    else:
                        fishers[name] = [g2, p.data.clone().detach()]
            self.optimizer.zero_grad()
            seen += images.shape[0]
            if seen >= self.fisher_size:
                break
        # normalize by number of iterations
        for name in fishers:
            fishers[name][0] /= max(iter_, 1)
        self.fishers = fishers
        print(f"[EATA] Fisher done over {iter_} batches / {seen} patches.")

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
        self.current_model_probs = None

    # -----------------------------------------------------------
    # Adaptation
    # -----------------------------------------------------------
    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []
        for _ in range(self.steps):
            self.total_batches += 1

            logits, _, _ = self.model(x, self.text_x, True, interpolate=False)
            logits = logits[0]                          # (B, C, h, w)
            ent = self.softmax_entropy(logits)          # (B, h, w)

            # --- filter 1: reliable (low-entropy) pixels ---
            mask1 = ent < self.e_margin
            if mask1.sum() == 0:
                self.optimizer.zero_grad(); continue

            probs = logits.softmax(dim=1)               # (B, C, h, w)

            # --- filter 2: non-redundant (diverse) — PER-IMAGE ---
            # EATA's redundancy filter is sample-level: it compares each
            # SAMPLE's prediction vector to a running mean and drops near-
            # duplicates. For dense seg the natural "sample" is one patch,
            # so we summarise each patch by its reliable-pixel mean softmax
            # and keep patches whose cosine to the EMA is below 1-d_margin
            # (i.e. NOT near-identical to what we've already seen). Applying
            # the original per-pixel cosine here would drop ~everything,
            # since all pixels of one image are highly collinear with the
            # global mean (their cosine never approaches d_margin=0.05).
            B = logits.shape[0]
            keep_imgs = []
            patch_means = []
            for b in range(B):
                m = mask1[b]
                if m.sum() == 0:
                    continue
                pmean = probs[b].permute(1, 2, 0)[m].mean(0)   # (C,)
                if self.current_model_probs is not None:
                    cos = F.cosine_similarity(
                        self.current_model_probs.unsqueeze(0),
                        pmean.unsqueeze(0), dim=1).item()
                    if cos > (1.0 - self.d_margin):
                        continue  # too similar -> redundant, skip
                keep_imgs.append(b)
                patch_means.append(pmean.detach())

            if len(keep_imgs) == 0:
                self.optimizer.zero_grad(); continue

            keep_mask = torch.zeros_like(mask1)
            for b in keep_imgs:
                keep_mask[b] = mask1[b]
            ent_kept = ent[keep_mask]                          # (Nkept,)

            self.current_model_probs = self._update_probs(
                self.current_model_probs, torch.stack(patch_means))

            # --- reweighted entropy loss ---
            coeff = 1.0 / torch.exp(ent_kept.clone().detach() - self.e_margin)
            loss = (ent_kept * coeff).mean()

            # --- Fisher anti-forgetting regularizer ---
            if self.fishers is not None:
                ewc = 0.0
                for name, p in self.model.named_parameters():
                    if name in self.fishers:
                        F_n, theta_src = self.fishers[name]
                        ewc = ewc + (F_n * (p - theta_src) ** 2).sum()
                loss = loss + self.fisher_alpha * ewc

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_report.append(loss.item())

        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

    @staticmethod
    def _update_probs(current, new_probs):
        if new_probs.size(0) == 0:
            return current
        with torch.no_grad():
            m = new_probs.mean(0)
            if current is None:
                return m
            return 0.9 * current + 0.1 * m

    # ===========================================================
    # Helpers
    # ===========================================================
    def extract_text_embeddings(self, class_names, prompts, average=True):
        text_features = []
        for class_name in class_names:
            texts = [p.format(class_name) for p in prompts]
            texts = self.tokenize(texts).to(self.device)
            ce = self.model.encode_text(texts)
            ce = ce / ce.norm(dim=-1, keepdim=True)
            if average:
                avg = ce.mean(dim=0); avg = avg / avg.norm()
                ce = torch.cat([ce, avg.unsqueeze(0)], dim=0)
            text_features.append(ce)
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
    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        return -(x.softmax(-3) * x.log_softmax(-3)).sum(-3)
