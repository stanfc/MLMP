"""
DeYOMLMPGradRatioContinual — proposed method. Same best-state anchor as
GradAnchor (method 2), but the TRIGGER is changed from the (noisy) grad slope to
the ratio of EMA-smoothed grad_norm to its running minimum.

Why: every slope-triggered variant (gradslope, gradanchor, lagadapt, hybrid)
failed because the *derivative* (slope) amplifies grad_norm's noise — on VOC20
the noise slope (up to ~0.5) is larger than ACDC's real collapse slope (~0.08),
so no slope threshold can separate "real degradation" from "noise", and slope
flickers positive during the climb -> false restores that suppress the climb.

Fix: judge degradation by a *level* comparison of a *smoothed* quantity:
    g_ema = EMA(grad_norm)                      # smooth out per-batch noise
    g_min = running minimum of g_ema            # the best state's grad level
            -> snapshot LN as the anchor on each new minimum (best state)
    r     = g_ema / g_min                       # relative rise above the best (self-normalizing)

    r <= trigger_ratio  -> rst = 0   (climb: grad still hugging its min -> NO restore -> free climb)
    r >  trigger_ratio  -> rst = clip(rst_gain * (r - trigger_ratio), 0, max_rst)
                           restore toward the best-state anchor.

This fixes all prior failure modes at once:
  * noise immunity   — EMA + ratio (not derivative); a transient spike reverts
                       and never sustains r above threshold.
  * climb preserved  — during the climb r ~ 1, so rst = 0 (unlike the slope
                       trigger which fired on noise and flattened the climb).
  * holds fast collapse — r rises and stays up; rst ramps with r; restore goes
                       toward the best state (deep), not a shallow lag.
  * dataset-agnostic — r is relative to each run's own best, so one threshold
                       (r ~ 1.1) works on slow (VOC20) and fast (ACDC) alike.

Memory: one anchor snapshot (~148KB).
gate_log.csv columns: total_batches, grad_norm, grad_ema, g_min, ratio, rst.
CTTA hard-rule compliant: no per-sample reset; partial stochastic restore only.
"""
import os
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
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class DeYOMLMPGradRatioContinual:
    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 vision_outputs=tuple(range(-1, -19, -1)),
                 prompt_dir='prompts.yaml',
                 deyo_margin_factor=0.5,
                 deyo_margin_e0_factor=0.4,
                 plpd_threshold=0.2,
                 aug_type='patch',
                 patch_len=4,
                 reweight_ent=True,
                 reweight_plpd=True,
                 top_block_exclude=6,
                 # --- grad-ratio gate ---
                 grad_ema_alpha=0.99,
                 trigger_ratio=1.1,
                 rst_gain=0.15,
                 max_rst=0.1,
                 monitor_interval=50,
                 save_dir=None,
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

        # grad-ratio gate state
        self.grad_ema_alpha = float(grad_ema_alpha)
        self.trigger_ratio = float(trigger_ratio)
        self.rst_gain = float(rst_gain)
        self.max_rst = float(max_rst)
        self.monitor_interval = int(monitor_interval)
        self.grad_ema = None
        self.g_min = float('inf')
        self.batch_count = 0
        self.total_batches = 0
        self.current_rst = 0.0

        self.model, self.tokenize = load_ovss(ovss_type, ovss_backbone, device=device)
        if prompt_dir:
            self.prompt_templates = load_prompts_from_yaml(prompt_dir)
        else:
            self.prompt_templates = [REFERENCE_PROMPT]

        self.model.transformer.requires_grad_(False)
        self.model.ln_final.requires_grad_(False)
        self.model.token_embedding.requires_grad_(False)
        self.model.visual = self.set_ln_grads(
            self.model.visual, top_block_exclude=self.top_block_exclude)
        params, names = self.collect_ln_params(
            self.model.visual, top_block_exclude=self.top_block_exclude)
        self.named_ln_params = list(zip(names, params))

        print_clip_parameters(self.model)
        print(f"+++ DeYO-MLMP-GradRatio (T={len(self.prompt_templates)}): "
              f"deyo_margin={self.deyo_margin:.3f}, plpd_thr={self.plpd_threshold}, "
              f"UAML={len(self.vision_outputs)} | "
              f"ema_alpha={self.grad_ema_alpha}, trigger_ratio={self.trigger_ratio}, "
              f"rst_gain={self.rst_gain}, max_rst={self.max_rst}, "
              f"monitor={self.monitor_interval} -> LN params: {len(params)}")

        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer)
        self.anchor = self._snapshot_ln_weights()   # best-state snapshot

        self.gate_log_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self.gate_log_path = os.path.join(save_dir, "gate_log.csv")
            with open(self.gate_log_path, 'w') as _f:
                _f.write("total_batches,grad_norm,grad_ema,g_min,ratio,rst\n")

        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(
                self.classes, self.prompt_templates, average=True).squeeze()

        if self.runtime:
            self.adapt_times = []
            self.eval_times = []

    @torch.no_grad()
    def _snapshot_ln_weights(self):
        return {name: p.detach().cpu().to(torch.float16).clone()
                for name, p in self.named_ln_params}

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
            save_weights=True)
        logits = logits[0]
        if self.runtime:
            self.eval_times.append(time.time() - t1)
        return logits

    def reset(self):
        self.load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state)
        self.anchor = self._snapshot_ln_weights()
        self.grad_ema = None
        self.g_min = float('inf')

    def _adapt_forward(self, x):
        logits, _, _ = self.model(
            x, self.text_x[:-1], True,
            interpolate=False,
            vision_outputs=self.vision_outputs,
            vision_out_type="mean")
        return logits  # (T, B, C, h, w)

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
            logits = self._adapt_forward(x)            # (T, B, C, h, w)
            ent = self.softmax_entropy(logits)         # (T, B, h, w)

            mask_ent = ent < self.deyo_margin
            if mask_ent.sum() > 0:
                with torch.no_grad():
                    x_prime = self._destroy_object(x)
                    logits_prime = self._adapt_forward(x_prime)
                prob = logits.softmax(dim=-3)
                prob_prime = logits_prime.softmax(dim=-3)
                cls1 = prob.argmax(dim=-3, keepdim=True)
                p_orig = prob.gather(dim=-3, index=cls1).squeeze(-3)
                p_prime = prob_prime.gather(dim=-3, index=cls1).squeeze(-3)
                plpd = (p_orig - p_prime).detach()
                final_mask = mask_ent & (plpd > self.plpd_threshold)
                if final_mask.sum() > 0:
                    ent_kept = ent[final_mask]; plpd_kept = plpd[final_mask]
                    if self.reweight_ent or self.reweight_plpd:
                        ent_det = ent_kept.detach(); coeff = 0.0
                        if self.reweight_ent:
                            coeff = coeff + 1.0 / torch.exp(ent_det - self.margin_e0)
                        if self.reweight_plpd:
                            coeff = coeff + 1.0 / torch.exp(-1.0 * plpd_kept)
                        loss = (ent_kept * coeff).mean()
                    else:
                        loss = ent_kept.mean()
                    loss_report.append(loss.item())
                    loss.backward()
                    with torch.no_grad():
                        gn = 0.0
                        for _n, p in self.named_ln_params:
                            if p.grad is not None:
                                gn += float(p.grad.detach().float().pow(2).sum())
                        gn = gn ** 0.5
                        # EMA of grad_norm (noise-smoothed gate signal)
                        if self.grad_ema is None:
                            self.grad_ema = gn
                        else:
                            self.grad_ema = (self.grad_ema_alpha * self.grad_ema
                                             + (1.0 - self.grad_ema_alpha) * gn)
                        self._last_gn = gn
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.current_rst > 0.0:
                        self._stochastic_restore(self.current_rst)

            self.batch_count += 1
            self.total_batches += 1
            if self.batch_count >= self.monitor_interval:
                self._update_gate()

        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

    @torch.no_grad()
    def _update_gate(self):
        if self.grad_ema is None:
            self.batch_count = 0
            return
        g = self.grad_ema
        # update best-state anchor on a new EMA-grad minimum
        if g < self.g_min:
            self.g_min = g
            self.anchor = self._snapshot_ln_weights()
        r = g / max(self.g_min, 1e-8)
        if r <= self.trigger_ratio:
            rst = 0.0
        else:
            rst = min(max(self.rst_gain * (r - self.trigger_ratio), 0.0), self.max_rst)

        print(f"[DeYO-MLMP-GradRatio] B{self.total_batches}: "
              f"grad_ema={g:.3f} g_min={self.g_min:.3f} r={r:.3f} rst={rst:.5f}")
        if self.gate_log_path is not None:
            with open(self.gate_log_path, 'a') as _f:
                _f.write(f"{self.total_batches},{getattr(self,'_last_gn',float('nan')):.6f},"
                         f"{g:.6f},{self.g_min:.6f},{r:.6f},{rst:.6f}\n")
        self.current_rst = rst
        self.batch_count = 0

    @torch.no_grad()
    def _stochastic_restore(self, rst):
        for name, p in self.named_ln_params:
            mask = (torch.rand(p.shape, device=p.device) < rst).to(p.dtype)
            src = self.anchor[name].to(p.device, dtype=p.dtype)
            p.data.mul_(1.0 - mask).add_(src * mask)

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
