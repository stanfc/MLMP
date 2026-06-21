"""
DeYOMLMPCompositeGateContinual — DeYO+MLMP with a COMPOSITE degradation gate
that uses two signals, each for what it is good at (see
docs/2026-06-18-contribution.md §7):

  * mean_conf (mean max-softmax)  — monotone in time on all 3 datasets, so it
    is the TRIGGER (timing): while confidence is still low the model is still
    climbing toward its peak, so do not restore; once confidence rises past
    `conf_ceil` we are past the peak and the gate turns on.

  * grad_norm (||LN-grad||)       — U-shaped, bottoms near the peak then rises
    as degradation deepens. It is self-normalising if read RELATIVE to its
    value at the trigger, so it sets the BRAKE DEPTH (intensity): rst grows as
    grad_norm climbs above its trigger baseline.

Gate (evaluated every `monitor_interval` batches on window means):
    c = mean_conf window,  g = grad_norm window
    if c < conf_ceil:                       # still climbing
        rst = 0
    else:                                   # past peak
        g_trigger := g at the first crossing (captured once)
        rst = base_rst * clamp(g / g_trigger, 1.0, grad_mult_max)
After each optimizer step, a fraction `rst` of visual LN params are
stochastically restored toward source (DivGate-style flat restore).

Rationale: a single H_margin threshold cannot mark VOC20's peak (uniform
degradation keeps the marginal diverse). mean_conf is monotone there too, so
its threshold DOES fire near the peak; grad_norm then scales the brake by how
badly the run is degrading — gentle on VOC20 (small grad rise), hard on
ACDC/Cityscapes (large grad rise during collapse).

Per-dataset conf_ceil guidance (mean_conf at the observed mIoU peak):
    ACDC:       conf_ceil ~0.85   (peak conf 0.848 @R34)
    Cityscapes: conf_ceil ~0.72   (peak conf 0.707 @R15)
    VOC20:      conf_ceil ~0.74   (peak conf 0.732 @R59)
    base_rst=0.005, grad_mult_max=4.0 (start tuning here)

CTTA hard-rule compliance: no per-sample reset; restore is partial/stochastic.
gate_log.csv columns: total_batches, mean_conf, grad_norm, g_trigger, rst.
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


class DeYOMLMPCompositeGateContinual:
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
                 # --- composite gate ---
                 conf_ceil=0.80,
                 base_rst=0.005,
                 grad_mult_max=4.0,
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

        self.gate_log_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self.gate_log_path = os.path.join(save_dir, "gate_log.csv")
            with open(self.gate_log_path, 'w') as _f:
                _f.write("total_batches,mean_conf,grad_norm,g_trigger,rst\n")

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

        # composite-gate state
        self.conf_ceil = float(conf_ceil)
        self.base_rst = float(base_rst)
        self.grad_mult_max = float(grad_mult_max)
        self.monitor_interval = int(monitor_interval)
        self.conf_buf = []          # per-batch mean_conf
        self.grad_buf = []          # per-batch grad_norm (steps that updated)
        self.g_trigger = None       # grad_norm captured at first conf crossing
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
        print(f"+++ DeYO-MLMP-CompositeGate (T={len(self.prompt_templates)}): "
              f"deyo_margin={self.deyo_margin:.3f}, plpd_thr={self.plpd_threshold}, "
              f"UAML layers={len(self.vision_outputs)} | "
              f"conf_ceil={self.conf_ceil}, base_rst={self.base_rst}, "
              f"grad_mult_max={self.grad_mult_max}, monitor={self.monitor_interval} "
              f"-> LN params: {len(params)}")

        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer)

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

            # gate signal 1: ensemble mean max-softmax (confidence)
            with torch.no_grad():
                probs_ens = logits.softmax(dim=-3).mean(dim=0)   # (B,C,h,w)
                self.conf_buf.append(float(probs_ens.max(dim=1).values.mean()))

            mask_ent = ent < self.deyo_margin
            do_step = mask_ent.sum() > 0
            if do_step:
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
                    loss_report.append(loss.item())
                    loss.backward()
                    # gate signal 2: LN-gradient norm (read before zero_grad)
                    with torch.no_grad():
                        gn = 0.0
                        for _n, p in self.named_ln_params:
                            if p.grad is not None:
                                gn += float(p.grad.detach().float().pow(2).sum())
                        self.grad_buf.append(gn ** 0.5)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.current_rst > 0.0:
                        self._stochastic_restore_flat(self.current_rst)

            self.batch_count += 1
            self.total_batches += 1
            if self.batch_count >= self.monitor_interval:
                self._update_gate()

        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

    # -------- composite-gate machinery --------
    @torch.no_grad()
    def _update_gate(self):
        if len(self.conf_buf) == 0:
            self.batch_count = 0
            return
        c = float(sum(self.conf_buf) / len(self.conf_buf))
        g = float(sum(self.grad_buf) / len(self.grad_buf)) if self.grad_buf else float('nan')

        if c < self.conf_ceil or not (g == g):   # still climbing, or no grad yet
            rst = 0.0
        else:
            if self.g_trigger is None:
                self.g_trigger = g                # capture baseline at first crossing
            ratio = g / max(self.g_trigger, 1e-8)
            ratio = min(max(ratio, 1.0), self.grad_mult_max)
            rst = self.base_rst * ratio

        print(f"[DeYO-MLMP-Composite] B{self.total_batches}: "
              f"conf={c:.3f} grad={g:.3f} g_trig="
              f"{self.g_trigger if self.g_trigger is not None else float('nan')} rst={rst:.5f}")
        if self.gate_log_path is not None:
            with open(self.gate_log_path, 'a') as _f:
                _f.write(f"{self.total_batches},{c:.6f},{g:.6f},"
                         f"{self.g_trigger if self.g_trigger is not None else float('nan')},"
                         f"{rst:.6f}\n")
        self.current_rst = rst
        self.conf_buf.clear()
        self.grad_buf.clear()
        self.batch_count = 0

    @torch.no_grad()
    def _stochastic_restore_flat(self, rst):
        for name, p in self.named_ln_params:
            mask = (torch.rand(p.shape, device=p.device) < rst).to(p.dtype)
            src = self.model_state[name].to(p.device, dtype=p.dtype)
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
