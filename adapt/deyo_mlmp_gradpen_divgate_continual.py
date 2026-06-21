"""
DeYOMLMPGradPenDivGateContinual — DeYO+MLMP+DivGate with an explicit
gradient-norm penalty L' = L + lambda * ||grad_theta L||^2 (double backward).

Identical to deyo_mlmp_divgate_continual except the loss/backward block: when
grad_pen_lambda > 0 we add a penalty on the L2 norm of the LN-param gradient
(the signal 學長 found highly anti-correlated with mIoU, docs/2026-06-18-contribution.md
§7) so the optimiser actively avoids high-grad_norm states. lambda == 0 falls
back to the exact divgate path (bit-identical control). H_margin DivGate is
unchanged. New log: gradpen_log.csv (total_batches, grad_norm, penalty, lambda).

DeYOMLMPDivGateContinual — DeYO (ICLR 2024) adapt loss in the full MLMP
machinery (multi-prompt + multi-layer + UAML eval), PLUS the Diversity
Gate (DivGate) stochastic-restoration safety net.

Base = deyo_mlmp_continual:
  - T=7 prompts, 18-layer mean fusion during adapt, UAML eval.
  - DeYO two-stage pixel filter (entropy < deyo_margin, PLPD >
    plpd_threshold) + reweighted entropy loss.

Added = DivGate (same machinery as tent_divgate / mlmp_topk_divgate):
  - Every adapt step, the ensemble (prompt-averaged) marginal class
    distribution probs.mean over (B,h,w) is buffered.
  - Every `monitor_interval` batches, H_margin = entropy of the mean
    buffered marginal is computed. Mode is picked:
        H_margin >= h_threshold              -> aggressive (rst = 0)
        h_warning <= H_margin < h_threshold  -> cautious   (rst = cautious_rst)
        H_margin <  h_warning                -> brake      (rst = brake_rst)
  - After each optimizer step, a fraction `rst` of visual LN params are
    stochastically restored toward source — the anti-collapse net.

Why combine: DeYO+MLMP has the highest peak of any baseline (V20 78.7,
ACDC 33.5) but, like all entropy methods, eventually drifts down. DivGate
adds a marginal-diversity-triggered restore so that once the marginal
starts collapsing toward 1-2 classes (H_margin drops) the gate brakes.
The goal is "DeYO+MLMP peak, but it does not collapse".

h_margin logging: main_continual.py already writes per-batch h_margin to
entropy_log.csv for EVERY method (computed from evaluate logits). This
class ADDITIONALLY prints the gate's own decision H_margin to stdout on
every mode transition (`[DeYO-MLMP-DivGate] ...`). Both are available.

Per-dataset threshold guidance (from observed early-round h_margin, max
= log(C) ~ 2.94 for 19 cls / ~3.0 for 20 cls):
  ACDC:       h_threshold=2.0, h_warning=1.7
  V20:        h_threshold=1.9, h_warning=1.5
  Cityscapes: h_threshold=2.1, h_warning=1.8
  rst: cautious_rst=0.005, brake_rst=0.02 (validated on TENT-DivGate)

CTTA hard-rule compliance: no per-sample reset; gate restore is partial
and stochastic, never a full reset.
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
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'
MODE_AGGRESSIVE = 'aggressive'
MODE_CAUTIOUS = 'cautious'
MODE_BRAKE = 'brake'


class DeYOMLMPGradPenDivGateContinual:
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
                 # --- DivGate ---
                 h_threshold=2.0, h_warning=1.7,
                 monitor_interval=50,
                 cautious_rst=0.005, brake_rst=0.02,
                 # --- gradient-norm penalty ---
                 grad_pen_lambda=0.0, grad_pen_form='linear',
                 grad_pen_mode='raw', grad_pen_ema_decay=0.99,
                 grad_clip=0.0,
                 save_dir=None,
                 runtime_calculation=False, device='cpu'):
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps
        self.vision_outputs = vision_outputs
        self.runtime = runtime_calculation
        self.device = device

        # Gate-decision log: every _update_mode() appends the H_margin the
        # GATE actually used (adapt-time ensemble marginal, window-averaged),
        # which differs from main_continual's evaluate-based entropy_log.csv.
        import os
        self.grad_pen_lambda = float(grad_pen_lambda)
        self.grad_pen_form = str(grad_pen_form)
        self.grad_pen_mode = str(grad_pen_mode)
        self.grad_pen_ema_decay = float(grad_pen_ema_decay)
        self.grad_clip = float(grad_clip)
        self.g_ref = None          # healthy grad_norm baseline (excess/ema modes)
        self.gradn_buf = []        # per-step grad_norm within the current window
        self.gate_log_path = None
        self.gradpen_log_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self.gate_log_path = os.path.join(save_dir, "gate_log.csv")
            with open(self.gate_log_path, 'w') as _f:
                _f.write("total_batches,h_margin,mode,rst\n")
            self.gradpen_log_path = os.path.join(save_dir, "gradpen_log.csv")
            with open(self.gradpen_log_path, 'w') as _f:
                _f.write("total_batches,grad_norm,penalty,lambda\n")

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

        # DivGate state
        self.h_threshold = float(h_threshold)
        self.h_warning = float(h_warning)
        if self.h_warning > self.h_threshold:
            raise ValueError("h_warning > h_threshold")
        self.monitor_interval = int(monitor_interval)
        self.cautious_rst = float(cautious_rst)
        self.brake_rst = float(brake_rst)
        self.marginal_buf = []
        self.batch_count = 0
        self.total_batches = 0
        self.current_mode = MODE_AGGRESSIVE
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
        print(f"+++ DeYO-MLMP-DivGate (T={len(self.prompt_templates)}): "
              f"deyo_margin={self.deyo_margin:.3f}, plpd_thr={self.plpd_threshold}, "
              f"UAML layers={len(self.vision_outputs)} | "
              f"gate thr=(warn {self.h_warning}, agg {self.h_threshold}), "
              f"rst=(cau {self.cautious_rst}, brake {self.brake_rst}), "
              f"monitor={self.monitor_interval} -> LN params: {len(params)}")

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

            # gate marginal: prompt-averaged class distribution
            with torch.no_grad():
                probs_ens = logits.softmax(dim=-3).mean(dim=0)   # (B,C,h,w)
                self.marginal_buf.append(
                    probs_ens.mean(dim=[0, 2, 3]).detach().float().cpu())

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
                    ln_params = [p for _, p in self.named_ln_params]
                    if self.grad_pen_lambda > 0.0:
                        # explicit grad-norm penalty (double backward, fp32)
                        g = torch.autograd.grad(loss, ln_params, create_graph=True)
                        grad_sq = sum((gi.float() ** 2).sum() for gi in g)
                        gnorm = grad_sq.clamp(min=1e-12).sqrt()
                        if self.grad_pen_mode == 'raw':
                            quantity = gnorm
                        elif self.g_ref is None:
                            quantity = gnorm * 0.0           # first window: no penalty
                        else:
                            quantity = (gnorm - self.g_ref).clamp(min=0.0)  # penalise rise
                        penalty = quantity ** 2 if self.grad_pen_form == 'sq' else quantity
                        total = loss + self.grad_pen_lambda * penalty
                        total.backward()
                        if self.grad_clip > 0.0:
                            torch.nn.utils.clip_grad_norm_(ln_params, self.grad_clip)
                        grad_norm_raw = float(grad_sq.detach().sqrt())
                        penalty_val = float(penalty.detach())
                    else:
                        # lambda == 0: exact divgate path (bit-identical control)
                        loss.backward()
                        with torch.no_grad():
                            gsq = sum((p.grad.float() ** 2).sum()
                                      for _, p in self.named_ln_params
                                      if p.grad is not None)
                        grad_norm_raw = float(gsq.sqrt())
                        penalty_val = 0.0
                    self._log_gradpen(grad_norm_raw, penalty_val)
                    if self.grad_pen_mode != 'raw':
                        self.gradn_buf.append(grad_norm_raw)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # gate stochastic restore after the step
                    if self.current_rst > 0.0:
                        self._stochastic_restore_flat(self.current_rst)

            self.batch_count += 1
            self.total_batches += 1
            if self.batch_count >= self.monitor_interval:
                if self.grad_pen_mode != 'raw' and self.gradn_buf:
                    gw = sum(self.gradn_buf) / len(self.gradn_buf)
                    if self.grad_pen_mode == 'excess':
                        self.g_ref = gw if self.g_ref is None else min(self.g_ref, gw)
                    elif self.grad_pen_mode == 'ema':
                        self.g_ref = gw if self.g_ref is None else \
                            self.grad_pen_ema_decay * self.g_ref + (1 - self.grad_pen_ema_decay) * gw
                    self.gradn_buf = []
                self._update_mode()

        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

    def _log_gradpen(self, grad_norm, penalty):
        if self.gradpen_log_path is None:
            return
        with open(self.gradpen_log_path, 'a') as _f:
            _f.write(f"{self.total_batches},{grad_norm:.6f},"
                     f"{penalty:.6f},{self.grad_pen_lambda:.6g}\n")

    # -------- DivGate machinery --------
    @staticmethod
    def _pick_mode(h, h_threshold, h_warning):
        if h >= h_threshold: return MODE_AGGRESSIVE
        if h >= h_warning:   return MODE_CAUTIOUS
        return MODE_BRAKE

    def _mode_to_rst(self, mode):
        if mode == MODE_AGGRESSIVE: return 0.0
        if mode == MODE_CAUTIOUS:   return self.cautious_rst
        if mode == MODE_BRAKE:      return self.brake_rst
        raise ValueError(mode)

    @torch.no_grad()
    def _update_mode(self):
        if len(self.marginal_buf) == 0:
            self.batch_count = 0; return
        agg = torch.stack(self.marginal_buf, dim=0).mean(dim=0)
        agg = agg / agg.sum().clamp(min=1e-8)
        h = -(agg * agg.clamp(min=1e-12).log()).sum().item()
        new_mode = self._pick_mode(h, self.h_threshold, self.h_warning)
        rst = self._mode_to_rst(new_mode)
        # always log the gate's decision H_margin (not just transitions)
        tag = "->" + new_mode if new_mode != self.current_mode else ""
        print(f"[DeYO-MLMP-DivGate] B{self.total_batches}: H_margin={h:.3f} "
              f"mode={new_mode}{tag} rst={rst}")
        # persist to disk so the gate decision is analyzable after the run
        if self.gate_log_path is not None:
            with open(self.gate_log_path, 'a') as _f:
                _f.write(f"{self.total_batches},{h:.6f},{new_mode},{rst}\n")
        self.current_mode = new_mode
        self.current_rst = rst
        self.marginal_buf.clear()
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
