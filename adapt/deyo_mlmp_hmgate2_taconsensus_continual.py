"""
DeYOMLMPHMGate2Continual — HMGate with a PERMANENT best-state anchor for deep restore.

Diagnosis fixed here: the original HMGate stores per-window LN snapshots in a deque
of fixed length (max_windows=2000 windows = 100k batches). On light runs (sub100,
500 batch/round) that covers 200 rounds, so the healthy peak snapshot is always
reachable and the curve looks flat. But on heavy runs (VOC20 full+15corr, 21735
batch/round) the deque only covers ~4.6 rounds: the grad-min (peak) snapshot is
EVICTED by ~R13, after which deep restore falls back to _win_buf[0] (the oldest
still-held, already-degraded window). The reachable anchor then slides forward and
downward -> a residual ~-0.06/round decline (delay, not prevent).

Fix: keep a dedicated `best_snapshot` that is updated ONLY when a new grad-norm
minimum is found, and is NEVER evicted. In the COLLAPSE regime (deep restore),
restore toward this permanent best instead of the rolling deque, so the true
healthy peak is reachable no matter how many iterations have passed. The HEALTHY
regime still uses the lag-deque shallow restore (so the slow climb isn't
suppressed). Everything else (DeYO+MLMP loss, H_margin regime switch, grad slope
trigger, base_rst stochastic restore) is identical to deyo_mlmp_hmgate_continual.

gate_log.csv columns: total_batches, grad_norm, grad_slope, h_margin, collapse,
windows_since_min, lag, rst, deep.
CTTA hard-rule compliant: no per-sample reset; partial stochastic restore only.
"""
import os
import time
import copy
import math
import re
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from einops import rearrange

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class DeYOMLMPHMGate2TAConsensusContinual:
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
                 lambda_align=0.1, top_k_align=0.2,
                 # --- adaptive-lag gate (lag in WINDOW units) ---
                 slope_window=10,
                 slope_deadzone=0.002,
                 lag_gain=1500.0,
                 base_rst=0.01,
                 max_windows=2000,
                 # two-regime cap switched by H_MARGIN (not slope): collapse
                 # regime (windowed H_margin < h_drop_ratio * running-max H) ->
                 # DEEP adaptive cap (= grad-min distance), holds fast collapse
                 # (ACDC/Cityscapes). Healthy/uniform regime (H stays high, VOC20)
                 # -> SHALLOW cap, never suppresses the slow climb. H_margin is a
                 # clean regime switch where grad-magnitude (slope/ratio) is not:
                 # VOC20 noise ratio ~1.25 exceeds ACDC collapse ratio ~1.24, but
                 # H_margin drops ONLY on real marginal collapse.
                 h_drop_ratio=0.9,
                 maxlag_shallow=6,
                 monitor_interval=50,
                 save_dir=None,
                 runtime_calculation=False, device='cpu'):
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.lambda_align = float(lambda_align)
        self.top_k_align = float(top_k_align)
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

        # adaptive-lag gate state
        self.slope_window = int(slope_window)
        self.slope_deadzone = float(slope_deadzone)
        self.lag_gain = float(lag_gain)          # window units
        self.base_rst = float(base_rst)
        self.max_windows = int(max_windows)
        self.h_drop_ratio = float(h_drop_ratio)
        self.maxlag_shallow = int(maxlag_shallow)
        self.monitor_interval = int(monitor_interval)
        # H_margin regime state
        self.marginal_buf = []
        self.h_max = 0.0
        self.grad_buf = []          # per-batch grad_norm within current window
        self.grad_hist = []         # per-window mean grad_norm (for the slope)
        self.g_min = float('inf')
        self.windows_since_min = 0
        self.batch_count = 0
        self.total_batches = 0
        self.current_lag = 0        # in windows
        self.current_rst = 0.0
        self.current_deep = False   # deep (toward permanent best) vs shallow (lag-deque)
        self.best_snapshot = None   # permanent best-state LN snapshot (never evicted)

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
        print(f"+++ DeYO-MLMP-HMGate2 permanent-best-anchor (T={len(self.prompt_templates)}): "
              f"deyo_margin={self.deyo_margin:.3f}, plpd_thr={self.plpd_threshold}, "
              f"UAML={len(self.vision_outputs)} | "
              f"slope_window={self.slope_window}, deadzone={self.slope_deadzone}, "
              f"lag_gain={self.lag_gain}(win), base_rst={self.base_rst}, "
              f"max_windows={self.max_windows}, monitor={self.monitor_interval} "
              f"-> LN params: {len(params)}")

        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer)
        # per-window snapshot deque (lag measured in windows)
        self._win_buf = deque(maxlen=self.max_windows + 1)
        self._win_buf.append(self._snapshot_ln_weights())
        # permanent best-state anchor (deep restore target; never evicted)
        self.best_snapshot = self._win_buf[-1]

        self.gate_log_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self.gate_log_path = os.path.join(save_dir, "gate_log.csv")
            with open(self.gate_log_path, 'w') as _f:
                _f.write("total_batches,grad_norm,grad_slope,h_margin,collapse,"
                         "windows_since_min,lag,rst,deep\n")

        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(
                self.classes, self.prompt_templates, average=True).squeeze()

        if self.runtime:
            self.adapt_times = []
            self.eval_times = []

    # ---------- slope ----------
    @staticmethod
    def _slope(ys):
        n = len(ys)
        if n < 2:
            return 0.0
        xm = (n - 1) / 2.0
        ym = sum(ys) / n
        num = sum((i - xm) * (ys[i] - ym) for i in range(n))
        den = sum((i - xm) ** 2 for i in range(n))
        return num / den if den > 0 else 0.0

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
        self._win_buf.clear()
        self._win_buf.append(self._snapshot_ln_weights())
        self.best_snapshot = self._win_buf[-1]
        self.current_deep = False
        self.g_min = float('inf')
        self.windows_since_min = 0
        self.grad_buf.clear()
        self.grad_hist.clear()
        self.marginal_buf.clear()
        self.h_max = 0.0

    def _adapt_forward(self, x):
        logits, _, _ = self.model(
            x, self.text_x[:-1], True,
            interpolate=False,
            vision_outputs=self.vision_outputs,
            vision_out_type="mean")
        return logits  # (T, B, C, h, w)

    def _align_loss(self, x):
        # single-layer forward; per-prompt logits + per-pixel visual features + text
        logits, image_features, text_features = self.model(
            x, self.text_x[:-1], True, interpolate=False)   # logits (T,B,C,w,h)
        avg_logits = logits.mean(dim=0)                       # (B,C,w,h)
        avg_text = text_features.mean(dim=0)                  # (C,D)
        avg_text = avg_text / avg_text.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        with torch.no_grad():
            probs = avg_logits.softmax(dim=1)
            conf, pred = probs.max(dim=1)                     # (B,w,h)
            # cross-prompt CONSENSUS: keep only pixels where ALL T prompts agree
            # on the class. This is a RELIABILITY filter distinct from entropy
            # (entropy uses the mean logit; here the 7 text views must concur),
            # so aligning to the agreed class's frozen text anchor is a genuine
            # correctness signal, not a redundant re-statement of the mean logit.
            per_pred = logits.argmax(dim=2)                   # (T,B,w,h)
            consensus = (per_pred == pred.unsqueeze(0)).all(dim=0)   # (B,w,h)
            k = max(1, int(round(conf.numel() * self.top_k_align)))
            thr = torch.topk(conf.flatten(), k, sorted=True).values[-1]
            mask = (conf >= thr) & consensus
        if mask.sum() == 0:
            return avg_logits.sum() * 0.0                     # no consensus pixels
        B, _, w, h = avg_logits.shape
        D = image_features.shape[-1]
        vis = image_features[:, 1:, :].reshape(B, w, h, D)    # (B,w,h,D)
        tgt = avg_text[pred]                                  # (B,w,h,D)
        cos = (vis * tgt).sum(dim=-1)                         # (B,w,h)
        return -cos[mask].mean()

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

            # H_margin regime signal: ensemble (prompt-averaged) marginal
            with torch.no_grad():
                probs_ens = logits.softmax(dim=-3).mean(dim=0)   # (B,C,h,w)
                self.marginal_buf.append(
                    probs_ens.mean(dim=[0, 2, 3]).detach().float().cpu())

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
                    # --- text-alignment auxiliary (CMA-style cosine to the
                    # frozen text anchor of the pseudo-label; rewards CORRECTNESS,
                    # not just confidence). Dedicated single-layer forward. ---
                    if self.lambda_align > 0.0:
                        loss = loss + self.lambda_align * self._align_loss(x)
                    loss_report.append(loss.item())
                    loss.backward()
                    with torch.no_grad():
                        gn = 0.0
                        for _n, p in self.named_ln_params:
                            if p.grad is not None:
                                gn += float(p.grad.detach().float().pow(2).sum())
                        self.grad_buf.append(gn ** 0.5)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.current_rst > 0.0 and self.current_lag > 0:
                        self._stochastic_restore(self.current_rst, self.current_lag)

            self.batch_count += 1
            self.total_batches += 1
            if self.batch_count >= self.monitor_interval:
                self._update_gate()

        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

    @torch.no_grad()
    def _update_gate(self):
        if len(self.grad_buf) == 0:
            self.batch_count = 0
            return
        # snapshot this window's LN state (lag is measured in windows)
        self._win_buf.append(self._snapshot_ln_weights())

        g = float(sum(self.grad_buf) / len(self.grad_buf))
        self.grad_hist.append(g)
        if len(self.grad_hist) > self.slope_window:
            self.grad_hist = self.grad_hist[-self.slope_window:]

        # track distance (in windows) back to the best (grad-min) state
        if g < self.g_min:
            self.g_min = g
            self.windows_since_min = 0          # the best state is THIS window
            self.best_snapshot = self._win_buf[-1]   # pin the permanent best anchor
        else:
            self.windows_since_min += 1

        # H_margin (windowed ensemble marginal entropy) -> regime switch
        if len(self.marginal_buf) > 0:
            agg = torch.stack(self.marginal_buf, dim=0).mean(dim=0)
            agg = agg / agg.sum().clamp(min=1e-8)
            h_margin = float(-(agg * agg.clamp(min=1e-12).log()).sum())
        else:
            h_margin = self.h_max
        self.marginal_buf.clear()
        self.h_max = max(self.h_max, h_margin)
        # collapse regime when H_margin has dropped below a fraction of its peak
        collapse_regime = h_margin < self.h_drop_ratio * self.h_max

        slope = self._slope(self.grad_hist)
        deep = False
        if slope <= self.slope_deadzone or self.windows_since_min < 1:
            lag, rst = 0, 0.0
        elif collapse_regime:
            # DEEP: restore toward the PERMANENT best anchor (never evicted), so the
            # true healthy peak is reachable however many iterations have passed.
            # lag is irrelevant here (anchor is best_snapshot, not the deque).
            deep = True
            lag, rst = self.windows_since_min, self.base_rst
        else:
            # SHALLOW: lag-deque restore so the slow climb (VOC20) is never suppressed.
            cap = min(self.maxlag_shallow, self.windows_since_min)
            lag = max(1, min(int(round(self.lag_gain * slope)), cap))
            rst = self.base_rst

        print(f"[DeYO-MLMP-HMGate2] B{self.total_batches}: grad={g:.3f} "
              f"slope={slope:+.4f} H={h_margin:.3f}/{self.h_max:.3f} "
              f"{'COLLAPSE' if collapse_regime else 'healthy'} "
              f"since_min={self.windows_since_min} lag={lag}w rst={rst:.4f} "
              f"{'DEEP->best' if deep else 'shallow'}")
        if self.gate_log_path is not None:
            with open(self.gate_log_path, 'a') as _f:
                _f.write(f"{self.total_batches},{g:.6f},{slope:.6f},"
                         f"{h_margin:.6f},{int(collapse_regime)},"
                         f"{self.windows_since_min},{lag},{rst:.6f},{int(deep)}\n")
        self.current_lag = lag
        self.current_rst = rst
        self.current_deep = deep
        self.grad_buf.clear()
        self.batch_count = 0

    @torch.no_grad()
    def _stochastic_restore(self, rst, lag):
        if self.current_deep:
            # deep restore -> permanent best anchor (never evicted)
            anchor = self.best_snapshot
        else:
            idx = -1 - lag
            anchor = self._win_buf[0] if -idx > len(self._win_buf) else self._win_buf[idx]
        for name, p in self.named_ln_params:
            mask = (torch.rand(p.shape, device=p.device) < rst).to(p.dtype)
            src = anchor[name].to(p.device, dtype=p.dtype)
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
