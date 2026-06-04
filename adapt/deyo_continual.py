"""
DeYOContinual — "Entropy is not Enough for Test-Time Adaptation: From the
Perspective of Disentangled Factors" (Lee et al., ICLR 2024 Spotlight).
Ported from https://github.com/Jhyun17/DeYO to the MLMP / NA-CLIP /
per-pixel-OVSS setting.

DeYO's central claim is that entropy alone is not a reliable confidence
signal — it can be high for samples where the prediction is driven by
spurious factors (background colour, texture, context) rather than the
actual object. The fix is a two-stage filter:

  Filter 1 (entropy):  ent  < deyo_margin   = 0.5 * log(C)
  Filter 2 (PLPD):     plpd > plpd_threshold = 0.2

PLPD (Pseudo-Label Probability Difference) is the difference between
P(c1 | x) and P(c1 | x') where x' is an "object-destructive"
transformation (we use the paper's default: patch shuffle, dividing
the image into 4x4 = 16 tiles and randomly permuting them). If the
model is using object structure to predict c1, shuffling the patches
destroys that signal and P(c1 | x') drops a lot -> large plpd ->
reliable. If the model is shortcut-ing on texture/colour, plpd is
small -> unreliable.

After filtering, the surviving entropies are optionally REWEIGHTED:
  coeff = reweight_ent * 1/exp(ent - margin_e0)
        + reweight_plpd * 1/exp(-plpd)
which up-weights samples that are simultaneously low-entropy AND
high-plpd.

The final loss is the reweighted mean entropy.

Compatibility with our CTTA hard rules:
  - DeYO has no reset path of its own. continual_adapt() never calls
    reset() proactively. State persists across the whole stream.

Per-pixel adaptation:
  - All scalars become per-pixel: ent shape (1, B, h, w), plpd same.
  - cls1 from argmax along the class axis (-3).
  - Filters are boolean masks combined with logical_and.

Deviations from upstream:
  - Base optimizer is Adam (MLMP convention) instead of SGD.
  - Top blocks excluded: paper excludes 3/12 ViT-Base blocks; we
    exclude proportionally (last 6 of 24) on ViT-L/14 + ln_post.

Hyperparameters (paper's wild-setting defaults, retained):
  - deyo_margin_factor = 0.5
  - margin_e0_factor   = 0.4
  - plpd_threshold     = 0.2
  - aug_type           = 'patch'
  - patch_len          = 4
  - reweight_ent       = True
  - reweight_plpd      = True
"""
import time
import copy
import math
import re

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from einops import rearrange

from ovss import load_ovss
from utils.misc import print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class DeYOContinual:
    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 deyo_margin_factor=0.5,
                 deyo_margin_e0_factor=0.4,
                 plpd_threshold=0.2,
                 aug_type='patch',
                 patch_len=4,
                 reweight_ent=True,
                 reweight_plpd=True,
                 top_block_exclude=6,
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

        self.deyo_margin = deyo_margin_factor * math.log(len(classes))
        self.margin_e0 = deyo_margin_e0_factor * math.log(len(classes))
        self.plpd_threshold = plpd_threshold
        self.aug_type = aug_type
        self.patch_len = patch_len
        self.reweight_ent = bool(reweight_ent)
        self.reweight_plpd = bool(reweight_plpd)
        self.top_block_exclude = top_block_exclude
        self.total_batches = 0

        self.model, self.tokenize = load_ovss(ovss_type, ovss_backbone, device=device)

        self.prompt_templates = [REFERENCE_PROMPT]

        self.model.transformer.requires_grad_(False)
        self.model.ln_final.requires_grad_(False)
        self.model.token_embedding.requires_grad_(False)
        self.model.visual = self.set_ln_grads(
            self.model.visual, top_block_exclude=self.top_block_exclude)
        params, _ = self.collect_ln_params(
            self.model.visual, top_block_exclude=self.top_block_exclude)

        print_clip_parameters(self.model)
        print(f"+++ DeYO-Continual: deyo_margin={self.deyo_margin:.3f} "
              f"(={deyo_margin_factor}*log({len(classes)})), "
              f"margin_e0={self.margin_e0:.3f}, "
              f"plpd_thr={self.plpd_threshold}, "
              f"aug={self.aug_type}/{self.patch_len}, "
              f"reweight ent/plpd={int(self.reweight_ent)}/{int(self.reweight_plpd)}, "
              f"top_block_exclude={self.top_block_exclude} "
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

    # -----------------------------------------------------------
    # Object-destructive transform (paper's Section 3.2)
    # -----------------------------------------------------------
    def _destroy_object(self, x):
        """Return x' where the object structure is destroyed.

        Defaults to patch shuffle (paper's main setting). Operates on the
        adapter-level batch of patches (B*N_patches, 3, H, W) since by
        this point main_continual.py has already cropped to 224-px patches.
        """
        if self.aug_type == 'pixel':
            x2 = rearrange(x, 'b c h w -> b c (h w)')
            perm = torch.randperm(x2.shape[-1], device=x.device)
            x2 = x2[:, :, perm]
            return rearrange(x2, 'b c (h w) -> b c h w', h=x.shape[-2], w=x.shape[-1])

        if self.aug_type == 'occ':
            # Replace a small centre window with the per-channel mean.
            B, C, H, W = x.shape
            occ = max(16, H // 8)
            r = (H - occ) // 2
            c = (W - occ) // 2
            x2 = x.clone()
            mean = x2.view(B, C, -1).mean(dim=2).unsqueeze(-1).unsqueeze(-1)
            x2[:, :, r:r+occ, c:c+occ] = mean.expand(-1, -1, occ, occ)
            return x2

        # default 'patch': split into patch_len x patch_len tiles, shuffle
        H, W = x.shape[-2], x.shape[-1]
        ps = self.patch_len
        H2 = (H // ps) * ps
        W2 = (W // ps) * ps
        if (H2, W2) != (H, W):
            x = T.functional.resize(x, [H2, W2], antialias=True)
        x2 = rearrange(x, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=ps, ps2=ps)
        # independent permutation per sample
        perm = torch.argsort(torch.rand(x2.shape[0], x2.shape[1], device=x.device), dim=-1)
        x2 = x2[torch.arange(x2.shape[0], device=x.device).unsqueeze(-1), perm]
        x2 = rearrange(x2, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=ps, ps2=ps)
        if (H2, W2) != (H, W):
            x2 = T.functional.resize(x2, [H, W], antialias=True)
        return x2

    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []
        for _ in range(self.steps):
            self.total_batches += 1

            # --- original forward ---
            logits, _, _ = self.model(x, self.text_x, True, interpolate=False)
            ent = self.softmax_entropy(logits)  # (N_prompts=1, B, h, w)
            mask_ent = ent < self.deyo_margin
            if mask_ent.sum() == 0:
                self.optimizer.zero_grad()
                continue

            # --- destroyed-object forward (no grad) ---
            with torch.no_grad():
                x_prime = self._destroy_object(x)
                logits_prime, _, _ = self.model(x_prime, self.text_x, True, interpolate=False)

            prob = logits.softmax(dim=-3)            # (1, B, C, h, w)
            prob_prime = logits_prime.softmax(dim=-3)
            cls1 = prob.argmax(dim=-3, keepdim=True) # (1, B, 1, h, w)
            p_orig = prob.gather(dim=-3, index=cls1).squeeze(-3)         # (1, B, h, w)
            p_prime = prob_prime.gather(dim=-3, index=cls1).squeeze(-3)
            plpd = (p_orig - p_prime).detach()

            mask_plpd = plpd > self.plpd_threshold
            final_mask = mask_ent & mask_plpd
            if final_mask.sum() == 0:
                self.optimizer.zero_grad()
                continue

            ent_kept = ent[final_mask]               # 1-D
            plpd_kept = plpd[final_mask]

            if self.reweight_ent or self.reweight_plpd:
                ent_det = ent_kept.detach()
                coeff = 0.0
                if self.reweight_ent:
                    coeff = coeff + 1.0 / torch.exp(ent_det - self.margin_e0)
                if self.reweight_plpd:
                    coeff = coeff + 1.0 / torch.exp(-1.0 * plpd_kept)
                loss = (ent_kept * coeff).mean()
            else:
                loss = ent_kept.mean()

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_report.append(loss.item())

        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

    # ===========================================================
    # Helpers (same shape as TENTContinual / SARContinual)
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
    def _is_excluded(nm, top_block_exclude):
        if 'ln_post' in nm:
            return True
        m = re.search(r'resblocks\.(\d+)\.', nm)
        if m:
            blk = int(m.group(1))
            if blk >= (24 - top_block_exclude):
                return True
        return False

    @classmethod
    def set_ln_grads(cls, model, top_block_exclude=6):
        model.requires_grad_(False)
        for nm, m in model.named_modules():
            if isinstance(m, nn.LayerNorm):
                if cls._is_excluded(nm, top_block_exclude):
                    continue
                m.requires_grad_(True)
        return model

    @classmethod
    def collect_ln_params(cls, model, top_block_exclude=6):
        params, names = [], []
        for nm, m in model.named_modules():
            if isinstance(m, nn.LayerNorm):
                if cls._is_excluded(nm, top_block_exclude):
                    continue
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
