"""
DeYOUAMLContinual — DeYO (ICLR 2024) adapt loss, but with MLMP-style
MULTI-LAYER adaptation (18-layer mean fusion during adapt) and UAML
EVALUATION (18-layer entropy-weighted fusion at inference).

This is the SINGLE-PROMPT variant: it keeps DeYO's single reference
prompt ('a photo of a {}', T=1) and only swaps the layer machinery.
Use it to isolate "does multi-layer adapt + UAML eval help DeYO?" with
no prompt-count confound. The multi-prompt (T=7) sibling is
deyo_mlmp_continual.py.

What changes vs deyo_continual.py:
  - adapt forward: vision_outputs=(-1..-18), vision_out_type="mean"
    -> logits shape (T, B, C, h, w) with T=1 here.
  - the destroyed-object (x') forward also goes multi-layer.
  - evaluate(): UAML adaptive_weighted_mean over 18 layers (copied from
    mlmp_continual.evaluate).
  - entropy / PLPD / filters / reweighting are unchanged in spirit, just
    carry the extra leading T axis. Class axis is dim=-3 throughout.

DeYO recap (see deyo_continual.py for the full writeup):
  Filter 1: entropy < deyo_margin = 0.5*log(C)
  Filter 2: plpd > plpd_threshold (PLPD = P(c1|x) - P(c1|x'), x'=patch shuffle)
  loss    : reweighted mean entropy of surviving pixels

Everything else (LN-only adaptation, no reset, top_block_exclude) is the
same as deyo_continual.
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


class DeYOUAMLContinual:
    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 vision_outputs=tuple(range(-1, -19, -1)),
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
        self.vision_outputs = vision_outputs
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
        print(f"+++ DeYO-UAML (single-prompt): deyo_margin={self.deyo_margin:.3f}, "
              f"margin_e0={self.margin_e0:.3f}, plpd_thr={self.plpd_threshold}, "
              f"aug={self.aug_type}/{self.patch_len}, "
              f"UAML layers={len(self.vision_outputs)}, "
              f"top_block_exclude={self.top_block_exclude} -> LN params: {len(params)}")

        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer)

        # average=True -> text_x shape (T+1, C, D); [-1] is the averaged prompt.
        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(
                self.classes, self.prompt_templates, average=True).squeeze()

        if self.runtime:
            self.adapt_times = []
            self.eval_times = []

    def adapt(self, x):
        return self.perform_adaptation(x)

    def continual_adapt(self, x):
        return self.perform_adaptation(x)

    @torch.no_grad()
    def evaluate(self, x):
        # MLMP UAML: 18-layer entropy-weighted fusion, single averaged prompt.
        t1 = time.time()
        logits, _, _ = self.model(
            x, self.text_x[-1], True,
            vision_outputs=self.vision_outputs,
            interpolate=True,
            vision_out_type="adaptive_weighted_mean",
            save_weights=True)
        logits = logits[0]
        if self.runtime:
            self.eval_times.append(time.time() - t1)
        return logits

    def reset(self):
        self.load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state)

    # -- multi-layer forward used for both x and x' during adapt --
    def _adapt_forward(self, x):
        # text_x[-1] (averaged single prompt) keeps T=1; mean fusion over layers.
        logits, _, _ = self.model(
            x, self.text_x[-1], True,
            interpolate=False,
            vision_outputs=self.vision_outputs,
            vision_out_type="mean")
        return logits  # (T=1, B, C, h, w)

    def _destroy_object(self, x):
        if self.aug_type == 'pixel':
            x2 = rearrange(x, 'b c h w -> b c (h w)')
            perm = torch.randperm(x2.shape[-1], device=x.device)
            return rearrange(x2[:, :, perm], 'b c (h w) -> b c h w',
                             h=x.shape[-2], w=x.shape[-1])
        if self.aug_type == 'occ':
            B, C, H, W = x.shape
            occ = max(16, H // 8); r = (H - occ) // 2; c = (W - occ) // 2
            x2 = x.clone()
            mean = x2.view(B, C, -1).mean(dim=2).unsqueeze(-1).unsqueeze(-1)
            x2[:, :, r:r+occ, c:c+occ] = mean.expand(-1, -1, occ, occ)
            return x2
        # patch shuffle
        H, W = x.shape[-2], x.shape[-1]; ps = self.patch_len
        H2, W2 = (H // ps) * ps, (W // ps) * ps
        if (H2, W2) != (H, W):
            x = T.functional.resize(x, [H2, W2], antialias=True)
        x2 = rearrange(x, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=ps, ps2=ps)
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

            logits = self._adapt_forward(x)            # (T, B, C, h, w)
            ent = self.softmax_entropy(logits)         # (T, B, h, w)
            mask_ent = ent < self.deyo_margin
            if mask_ent.sum() == 0:
                self.optimizer.zero_grad(); continue

            with torch.no_grad():
                x_prime = self._destroy_object(x)
                logits_prime = self._adapt_forward(x_prime)

            prob = logits.softmax(dim=-3)              # (T, B, C, h, w)
            prob_prime = logits_prime.softmax(dim=-3)
            cls1 = prob.argmax(dim=-3, keepdim=True)   # (T, B, 1, h, w)
            p_orig = prob.gather(dim=-3, index=cls1).squeeze(-3)    # (T, B, h, w)
            p_prime = prob_prime.gather(dim=-3, index=cls1).squeeze(-3)
            plpd = (p_orig - p_prime).detach()

            mask_plpd = plpd > self.plpd_threshold
            final_mask = mask_ent & mask_plpd
            if final_mask.sum() == 0:
                self.optimizer.zero_grad(); continue

            ent_kept = ent[final_mask]
            plpd_kept = plpd[final_mask]
            if self.reweight_ent or self.reweight_plpd:
                ent_det = ent_kept.detach(); coeff = 0.0
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
    def _is_excluded(nm, top_block_exclude):
        if 'ln_post' in nm:
            return True
        m = re.search(r'resblocks\.(\d+)\.', nm)
        if m and int(m.group(1)) >= (24 - top_block_exclude):
            return True
        return False

    @classmethod
    def set_ln_grads(cls, model, top_block_exclude=6):
        model.requires_grad_(False)
        for nm, m in model.named_modules():
            if isinstance(m, nn.LayerNorm) and not cls._is_excluded(nm, top_block_exclude):
                m.requires_grad_(True)
        return model

    @classmethod
    def collect_ln_params(cls, model, top_block_exclude=6):
        params, names = [], []
        for nm, m in model.named_modules():
            if isinstance(m, nn.LayerNorm) and not cls._is_excluded(nm, top_block_exclude):
                for np_, p in m.named_parameters():
                    if np_ in ['weight', 'bias']:
                        params.append(p); names.append(f"visual.{nm}.{np_}")
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
