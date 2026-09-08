"""
DeYOMLMPAdaGateContinual — GDG-PA (hmgate2) with SELF-CALIBRATING trigger and lag.

Motivation (measured on save/*/hmgate2_prompt_S0_baseline/gate_log.csv):
Two of hmgate2's gate hyperparameters are nominally tunable but empirically INERT,
and both are expressed in absolute grad-norm units, so neither transfers across
datasets by design -- only by accident.

  (A) slope_deadzone=0.002 is meant to reject noise-level grad-norm rises. But the
      slope noise band is +-0.03 (ACDC) / +-0.35 (VOC20), i.e. 15x / 150x larger,
      so the "deadzone" degenerates to `slope > 0`: firing rate 0.463 / 0.507
      vs P(slope>0) ~ 0.5. It filters nothing, and its meaning differs 10x between
      the two datasets.

  (B) lag = round(lag_gain * slope) with lag_gain=1500 is meant to scale restore
      depth continuously with degradation speed. But lag saturates at the cap
      (maxlag_shallow=6) whenever slope > 6/1500 = 0.004, which holds for 93.2%
      (ACDC) / 99.7% (VOC20) of active windows. The realised lag histogram is
      binary: ACDC {0:322, 6:162}, VOC20 {0:741, 6:709}. To actually spread lag
      over 1..6 you would need lag_gain ~= 153 (ACDC) vs ~= 17 (VOC20) -- a 9x
      dataset-specific gap, i.e. no single absolute gain can work.

Fix -- both decisions are re-expressed as UNITLESS quantities derived from the
stream's own statistics:

  (A) trend_stat: the raw OLS slope is normalised before thresholding.
        'mad'   z = slope / (1.4826 * MAD(recent slopes))   <- robust z-score
        'tstat' z = slope / SE(slope)                        <- OLS t-statistic
        'rel'   z = slope / grad_norm                        <- fractional growth
        'abs'   z = slope                                    <- hmgate2 behaviour
      With 'mad' the two datasets' z distributions almost coincide (p75 0.635 vs
      0.656, p90 1.125 vs 1.114), so ONE unitless trend_thr gives the same firing
      rate on both (thr=0.5 -> 0.305 / 0.311). `trend_thr` is then a genuine
      signal-to-noise deadzone, not a scale-bound constant.

  (B) lag_mode: restore depth is a FRACTION of the available budget.
        'ecdf' u = ECDF of z among recent ACTIVE z  -> lag = ceil(cap * u)
               rank-based, hence invariant to any monotone rescaling of the trend
               statistic and free of tunables entirely (lag_gain is deleted).
        'sat'  u = clamp(z / lag_sat, 0, 1)         -> lag = ceil(cap * u)
        'gain' lag = round(lag_gain * slope)         <- hmgate2 behaviour
      Offline replay of the S0_baseline stream gives a genuinely graded lag under
      'ecdf': ACDC {1:34, 2:31, 3:31, 4:37, 5:28, 6:26} instead of {0, 6}.

Setting trend_stat='abs', trend_thr=slope_deadzone, lag_mode='gain' reproduces
deyo_mlmp_hmgate2_continual exactly (no extra RNG is consumed), which is the
`ctrl` arm of the sweep.

Everything else -- DeYO+MLMP loss, H_margin regime switch, permanent best-state
anchor, base_rst stochastic restore -- is unchanged from hmgate2.

gate_log.csv columns: total_batches, grad_norm, grad_slope, z, u, h_margin,
collapse, windows_since_min, lag, rst, deep.
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

# z assigned when the normalising scale is degenerate (zero spread) but the slope
# is non-zero: treat as an unambiguously strong trend.
_Z_DEGENERATE = 10.0
# minimum history before the MAD scale / ECDF rank are considered meaningful
_MAD_MIN_HIST = 5
_ECDF_MIN_HIST = 10


class DeYOMLMPAdaGateContinual:
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
                 # --- self-calibrating trigger (A) and lag (B) ---
                 # trend_stat: how the raw OLS slope is normalised before the
                 #   deadzone test. 'abs' + lag_mode='gain' == hmgate2.
                 trend_stat='mad',
                 trend_thr=0.5,
                 trend_hist=50,
                 # lag_mode: how restore depth is chosen inside the budget cap.
                 lag_mode='ecdf',
                 lag_sat=1.5,
                 # shallow_cap_mode: (C) how far back SHALLOW restore may reach.
                 #   'fixed'   cap = min(maxlag_shallow, windows_since_min)  <- hmgate2/AdaGate A/B
                 #   'growing' cap = min(len(win_buf)-1, windows_since_min)  <- maxlag_shallow deleted;
                 #             reach grows with time-since-best, so a slow post-peak drift
                 #             (never entering collapse_regime, e.g. VOC20) still gets pulled
                 #             back toward its own best state instead of being capped at a few
                 #             windows forever. No-op whenever windows_since_min <= maxlag_shallow
                 #             (i.e. identical to 'fixed' during the healthy climb).
                 #   'growing_scaled'  cap = min(buf, ceil(windows_since_min * u)), u = today's
                 #             ecdf severity rank (0..1, already computed for lag_mode=ecdf).
                 #             'growing' grows purely with elapsed time regardless of how mild
                 #             today's trend is; this scales that growth down unless the current
                 #             window's z is itself unusually severe, so a still-improving run
                 #             whose grad-norm merely bottomed out early isn't over-restored.
                 #   'growing_hmargin' cap = min(buf, windows_since_h_peak) -- same idea as
                 #             'growing' but the distance is measured since H_margin's own
                 #             running peak, not since grad_norm's minimum. grad_norm and true
                 #             performance are not always in step; H_margin tracked collapse
                 #             more consistently across datasets in practice.
                 #   'growing_hmargin_scaled' both of the above combined.
                 shallow_cap_mode='fixed',
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

        # adaptive-lag gate state
        self.slope_window = int(slope_window)
        self.slope_deadzone = float(slope_deadzone)
        self.lag_gain = float(lag_gain)          # window units
        self.base_rst = float(base_rst)
        self.max_windows = int(max_windows)
        self.h_drop_ratio = float(h_drop_ratio)
        self.maxlag_shallow = int(maxlag_shallow)
        self.monitor_interval = int(monitor_interval)
        if trend_stat not in ('abs', 'rel', 'mad', 'tstat'):
            raise ValueError(f"trend_stat must be abs|rel|mad|tstat, got {trend_stat}")
        if lag_mode not in ('gain', 'sat', 'ecdf'):
            raise ValueError(f"lag_mode must be gain|sat|ecdf, got {lag_mode}")
        if shallow_cap_mode not in ('fixed', 'growing', 'growing_scaled',
                                    'growing_hmargin', 'growing_hmargin_scaled'):
            raise ValueError(f"unknown shallow_cap_mode: {shallow_cap_mode}")
        self.shallow_cap_mode = shallow_cap_mode
        self.trend_stat = trend_stat
        # 'abs' keeps hmgate2's semantics, where the deadzone IS slope_deadzone
        self.trend_thr = float(slope_deadzone) if trend_stat == 'abs' else float(trend_thr)
        self.trend_hist = int(trend_hist)
        self.lag_mode = lag_mode
        self.lag_sat = float(lag_sat)
        # slope history (for the MAD scale) and active-z history (for the ECDF rank)
        self.slope_buf = deque(maxlen=self.trend_hist)
        self.active_z_buf = deque(maxlen=self.trend_hist)
        # H_margin regime state
        self.marginal_buf = []
        self.h_max = 0.0
        self.windows_since_h_peak = 0   # (C, hmargin variants) windows since h_margin == h_max
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
        lag_desc = (f"gain={self.lag_gain}" if self.lag_mode == 'gain'
                    else (f"sat={self.lag_sat}" if self.lag_mode == 'sat' else "rank-based"))
        print(f"+++ DeYO-MLMP-AdaGate self-calibrating (T={len(self.prompt_templates)}): "
              f"deyo_margin={self.deyo_margin:.3f}, plpd_thr={self.plpd_threshold}, "
              f"UAML={len(self.vision_outputs)} | "
              f"trend_stat={self.trend_stat}, trend_thr={self.trend_thr}, "
              f"trend_hist={self.trend_hist}, lag_mode={self.lag_mode}({lag_desc}), "
              f"slope_window={self.slope_window}, base_rst={self.base_rst}, "
              f"maxlag_shallow={self.maxlag_shallow}, shallow_cap_mode={self.shallow_cap_mode}, "
              f"h_drop_ratio={self.h_drop_ratio}, "
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
                _f.write("total_batches,grad_norm,grad_slope,z,u,h_margin,collapse,"
                         "windows_since_min,lag,rst,deep\n")

        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(
                self.classes, self.prompt_templates, average=True).squeeze()

        if self.runtime:
            self.adapt_times = []
            self.eval_times = []

    # ---------- trend statistics ----------
    @staticmethod
    def _slope(ys):
        """OLS slope of ys against 0..n-1 (identical to hmgate2's _slope)."""
        n = len(ys)
        if n < 2:
            return 0.0
        xm = (n - 1) / 2.0
        ym = sum(ys) / n
        num = sum((i - xm) * (ys[i] - ym) for i in range(n))
        den = sum((i - xm) ** 2 for i in range(n))
        return num / den if den > 0 else 0.0

    @classmethod
    def _slope_tstat(cls, ys):
        """OLS slope together with its t-statistic slope / SE(slope)."""
        n = len(ys)
        b = cls._slope(ys)
        if n < 3:
            return b, 0.0
        xm = (n - 1) / 2.0
        ym = sum(ys) / n
        Sxx = sum((i - xm) ** 2 for i in range(n))
        if Sxx <= 0:
            return b, 0.0
        resid = [ys[i] - (ym + b * (i - xm)) for i in range(n)]
        se = math.sqrt(sum(r * r for r in resid) / (n - 2)) / math.sqrt(Sxx)
        if se <= 1e-12:
            return b, (0.0 if abs(b) < 1e-12 else math.copysign(_Z_DEGENERATE, b))
        return b, b / se

    @staticmethod
    def _median(xs):
        s = sorted(xs)
        n = len(s)
        return 0.5 * (s[(n - 1) // 2] + s[n // 2])

    def _trend_z(self, slope, tstat, g):
        """Normalise the raw slope into a unitless trend strength z.

        The slope buffer is appended by the caller AFTER this call so that z is
        always scored against the PRECEDING windows, never against itself.
        """
        if self.trend_stat == 'abs':
            return slope
        if self.trend_stat == 'tstat':
            return tstat
        if self.trend_stat == 'rel':
            return slope / g if g > 1e-9 else 0.0
        # 'mad': robust z-score of the slope against its own recent spread
        if len(self.slope_buf) < _MAD_MIN_HIST:
            scale = 0.0
        else:
            med = self._median(self.slope_buf)
            scale = 1.4826 * self._median([abs(x - med) for x in self.slope_buf])
        if scale > 1e-9:
            return slope / scale
        return 0.0 if abs(slope) < 1e-12 else math.copysign(_Z_DEGENERATE, slope)

    def _lag_fraction(self, z):
        """Fraction u in [0, 1] of the lag budget to use for this window."""
        if self.lag_mode == 'sat':
            return min(1.0, max(0.0, z / self.lag_sat)) if self.lag_sat > 1e-9 else 1.0
        # 'ecdf': rank of z among recent ACTIVE z. Rank-based, so invariant to any
        # monotone rescaling of the trend statistic and free of tunables.
        if len(self.active_z_buf) < _ECDF_MIN_HIST:
            return 0.5
        return sum(1 for x in self.active_z_buf if x < z) / len(self.active_z_buf)

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
        self.windows_since_h_peak = 0
        self.grad_buf.clear()
        self.grad_hist.clear()
        self.marginal_buf.clear()
        self.slope_buf.clear()
        self.active_z_buf.clear()
        self.h_max = 0.0

    def state_dict(self):
        """Full adapt-state checkpoint: learnable LN params, optimizer, and all
        gate bookkeeping needed to resume mid-stream with bit-for-bit continuation
        (not just the same hyperparameters -- the actual trend/lag/collapse state)."""
        return {
            'ln_params': {n: p.detach().cpu().clone() for n, p in self.named_ln_params},
            'optimizer': self.optimizer.state_dict(),
            'slope_buf': list(self.slope_buf),
            'active_z_buf': list(self.active_z_buf),
            'marginal_buf': [t.clone() for t in self.marginal_buf],
            'h_max': self.h_max,
            'windows_since_h_peak': self.windows_since_h_peak,
            'grad_buf': list(self.grad_buf),
            'grad_hist': list(self.grad_hist),
            'g_min': self.g_min,
            'windows_since_min': self.windows_since_min,
            'batch_count': self.batch_count,
            'total_batches': self.total_batches,
            'current_lag': self.current_lag,
            'current_rst': self.current_rst,
            'current_deep': self.current_deep,
            'best_snapshot': (dict(self.best_snapshot)
                              if self.best_snapshot is not None else None),
            '_win_buf': [dict(w) for w in self._win_buf],
        }

    def load_state_dict(self, sd):
        with torch.no_grad():
            for n, p in self.named_ln_params:
                p.data.copy_(sd['ln_params'][n].to(device=p.device, dtype=p.dtype))
        self.optimizer.load_state_dict(sd['optimizer'])
        # optimizer.state_dict() was captured from GPU params but the checkpoint is
        # loaded on CPU (see load_checkpoint's map_location='cpu'); load_state_dict()
        # does not move state tensors to match the param device, so Adam's
        # exp_avg/exp_avg_sq must be moved back explicitly or the next step() fails.
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)
        self.slope_buf = deque(sd['slope_buf'], maxlen=self.trend_hist)
        self.active_z_buf = deque(sd['active_z_buf'], maxlen=self.trend_hist)
        self.marginal_buf = list(sd['marginal_buf'])
        self.h_max = sd['h_max']
        self.windows_since_h_peak = sd['windows_since_h_peak']
        self.grad_buf = list(sd['grad_buf'])
        self.grad_hist = list(sd['grad_hist'])
        self.g_min = sd['g_min']
        self.windows_since_min = sd['windows_since_min']
        self.batch_count = sd['batch_count']
        self.total_batches = sd['total_batches']
        self.current_lag = sd['current_lag']
        self.current_rst = sd['current_rst']
        self.current_deep = sd['current_deep']
        self.best_snapshot = sd['best_snapshot']
        self._win_buf = deque((dict(w) for w in sd['_win_buf']),
                              maxlen=self.max_windows + 1)

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
        if h_margin >= self.h_max:
            self.windows_since_h_peak = 0
        else:
            self.windows_since_h_peak += 1
        self.h_max = max(self.h_max, h_margin)
        # collapse regime when H_margin has dropped below a fraction of its peak
        collapse_regime = h_margin < self.h_drop_ratio * self.h_max

        slope, tstat = self._slope_tstat(self.grad_hist)
        # (A) unitless trend strength, scored against the PRECEDING windows
        z = self._trend_z(slope, tstat, g)
        self.slope_buf.append(slope)

        deep = False
        u = 0.0
        if z <= self.trend_thr or self.windows_since_min < 1:
            lag, rst = 0, 0.0
        elif collapse_regime:
            # DEEP: restore toward the PERMANENT best anchor (never evicted), so the
            # true healthy peak is reachable however many iterations have passed.
            # lag is irrelevant here (anchor is best_snapshot, not the deque).
            deep = True
            lag, rst = self.windows_since_min, self.base_rst
            self.active_z_buf.append(z)
        else:
            # SHALLOW: lag-deque restore so the slow climb (VOC20) is never suppressed.
            # u_sev: today's ecdf severity rank (0..1), computed once and reused both for
            # scaling the cap (growing_scaled/growing_hmargin_scaled) and, further down, for
            # picking lag within whatever cap results (lag_mode='ecdf', unchanged from before).
            u_sev = self._lag_fraction(z)
            if self.shallow_cap_mode == 'growing':
                # (C) reach grows with time-since-best (bounded only by buffer length),
                # instead of being permanently stuck at maxlag_shallow once
                # windows_since_min exceeds it -- targets the post-peak tail decline on
                # datasets whose H_margin never drops enough to trigger collapse_regime.
                cap = min(len(self._win_buf) - 1, self.windows_since_min)
            elif self.shallow_cap_mode == 'growing_scaled':
                # (C1) same reach-grows-over-time idea, but scaled by how severe TODAY's
                # trend is, not just elapsed time -- a still-improving run whose grad-norm
                # merely bottomed out early accrues reach slowly (u_sev usually mid-range);
                # a genuinely worsening run (u_sev repeatedly high) accrues it fast.
                cap = min(len(self._win_buf) - 1, math.ceil(self.windows_since_min * u_sev))
            elif self.shallow_cap_mode == 'growing_hmargin':
                # (C2) distance measured since H_margin's own running peak instead of since
                # grad_norm's minimum -- grad_norm and true performance are not always in
                # step; H_margin tracked collapse more consistently across datasets.
                cap = min(len(self._win_buf) - 1, self.windows_since_h_peak)
            elif self.shallow_cap_mode == 'growing_hmargin_scaled':
                # (C1+C2) combined.
                cap = min(len(self._win_buf) - 1, math.ceil(self.windows_since_h_peak * u_sev))
            else:
                cap = min(self.maxlag_shallow, self.windows_since_min)
            if self.lag_mode == 'gain':
                lag = max(1, min(int(round(self.lag_gain * slope)), cap))
            else:
                # (B) restore depth as a FRACTION of the available budget
                u = u_sev
                lag = max(1, math.ceil(cap * u))
            self.active_z_buf.append(z)
            rst = self.base_rst

        print(f"[DeYO-MLMP-AdaGate] B{self.total_batches}: grad={g:.3f} "
              f"slope={slope:+.4f} z({self.trend_stat})={z:+.3f}/{self.trend_thr:.3f} "
              f"H={h_margin:.3f}/{self.h_max:.3f} "
              f"{'COLLAPSE' if collapse_regime else 'healthy'} "
              f"since_min={self.windows_since_min} since_hpeak={self.windows_since_h_peak} "
              f"u={u:.2f} lag={lag}w rst={rst:.4f} "
              f"{'DEEP->best' if deep else 'shallow'}")
        if self.gate_log_path is not None:
            with open(self.gate_log_path, 'a') as _f:
                _f.write(f"{self.total_batches},{g:.6f},{slope:.6f},{z:.6f},{u:.6f},"
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
    def _num_visual_blocks(model, default=24):
        """Depth of the visual transformer, so `top_block_exclude` means the same
        FRACTION of the encoder on every backbone (ViT-L/14 has 24 blocks,
        ViT-B/16 and ViT-B/32 have 12).  Falls back to 24, which keeps every
        historical ViT-L/14 run bit-identical."""
        try:
            return len(model.transformer.resblocks)
        except AttributeError:
            return default

    @staticmethod
    def _is_excluded(nm, top_block_exclude, num_blocks=24):
        if 'ln_post' in nm:
            return True
        m = re.search(r'resblocks\.(\d+)\.', nm)
        if m and int(m.group(1)) >= (num_blocks - top_block_exclude):
            return True
        return False

    @classmethod
    def set_ln_grads(cls, model, top_block_exclude=6):
        nb = cls._num_visual_blocks(model)
        model.requires_grad_(False)
        for nm, m in model.named_modules():
            if isinstance(m, nn.LayerNorm) and not cls._is_excluded(nm, top_block_exclude, nb):
                m.requires_grad_(True)
        return model

    @classmethod
    def collect_ln_params(cls, model, top_block_exclude=6):
        nb = cls._num_visual_blocks(model)
        params, names = [], []
        for nm, m in model.named_modules():
            if isinstance(m, nn.LayerNorm) and not cls._is_excluded(nm, top_block_exclude, nb):
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
