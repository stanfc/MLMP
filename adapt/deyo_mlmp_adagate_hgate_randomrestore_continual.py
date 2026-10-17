"""
DeYOMLMPAdaGateHGateRandomRestoreContinual -- ablation: does it matter THAT
the restore timing is informed by the grad-norm trend, or would firing at
the same overall rate but at RANDOM windows work just as well?

Subclass of the locked CAGA-hgate method. Only `_update_gate` is overridden:
the trigger condition `z <= self.trend_thr` is replaced by an independent
coin flip `random.random() < fire_prob`, where `fire_prob` defaults to
0.296 -- the empirically measured fraction of windows the real (grad-norm-
gated) CAGA-hgate run actually restores on ACDC (361/1218 windows in
save/ACDCDataset/adagate_hgate/gate_log.csv), so this is a same-restore-
budget, uninformed-timing control. grad_norm/z is still computed and logged
(and still used for SHALLOW lag severity, since that is a magnitude question,
not a timing question) -- only the yes/no trigger decision is randomized.
DEEP-vs-SHALLOW target selection (h_margin/collapse_regime) is unchanged.

Separate method name/save_dir -- does not touch CAGA's own results.
"""
import math
import random

import torch

from .deyo_mlmp_adagate_hgate_continual import DeYOMLMPAdaGateHGateContinual


class DeYOMLMPAdaGateHGateRandomRestoreContinual(DeYOMLMPAdaGateHGateContinual):
    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 catseg_checkpoint=None,
                 maft_checkpoint=None,
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
                 slope_window=10,
                 slope_deadzone=0.002,
                 lag_gain=1500.0,
                 base_rst=0.01,
                 max_windows=2000,
                 h_drop_ratio=0.9,
                 h_max_mode='lifetime',
                 h_max_decay=0.98,
                 maxlag_shallow=6,
                 monitor_interval=50,
                 trend_stat='mad',
                 trend_thr=0.5,
                 trend_hist=50,
                 lag_mode='ecdf',
                 lag_sat=1.5,
                 shallow_cap_mode='fixed',
                 anchor_h_min_ratio=0.95,
                 # --- random-restore ablation only ---
                 fire_prob=0.296,
                 random_seed=0,
                 save_dir=None,
                 runtime_calculation=False, device='cpu'):
        self.fire_prob = float(fire_prob)
        self._rng = random.Random(random_seed)
        super().__init__(
            ovss_type=ovss_type, ovss_backbone=ovss_backbone, lr=lr, classes=classes,
            steps=steps, catseg_checkpoint=catseg_checkpoint, maft_checkpoint=maft_checkpoint,
            vision_outputs=vision_outputs, prompt_dir=prompt_dir,
            deyo_margin_factor=deyo_margin_factor, deyo_margin_e0_factor=deyo_margin_e0_factor,
            plpd_threshold=plpd_threshold, aug_type=aug_type, patch_len=patch_len,
            reweight_ent=reweight_ent, reweight_plpd=reweight_plpd,
            top_block_exclude=top_block_exclude, slope_window=slope_window,
            slope_deadzone=slope_deadzone, lag_gain=lag_gain, base_rst=base_rst,
            max_windows=max_windows, h_drop_ratio=h_drop_ratio, h_max_mode=h_max_mode,
            h_max_decay=h_max_decay, maxlag_shallow=maxlag_shallow,
            monitor_interval=monitor_interval, trend_stat=trend_stat, trend_thr=trend_thr,
            trend_hist=trend_hist, lag_mode=lag_mode, lag_sat=lag_sat,
            shallow_cap_mode=shallow_cap_mode, anchor_h_min_ratio=anchor_h_min_ratio,
            save_dir=save_dir, runtime_calculation=runtime_calculation, device=device)
        print(f"+++ CAGA-hgate-RANDOM: restore trigger is an independent "
              f"coin flip p={self.fire_prob} (grad-norm trend ignored for timing)")

    @torch.no_grad()
    def _update_gate(self):
        if len(self.grad_buf) == 0:
            self.batch_count = 0
            return
        self._win_buf.append(self._snapshot_ln_weights())

        g = float(sum(self.grad_buf) / len(self.grad_buf))
        self.grad_hist.append(g)
        if len(self.grad_hist) > self.slope_window:
            self.grad_hist = self.grad_hist[-self.slope_window:]

        if len(self.marginal_buf) > 0:
            agg = torch.stack(self.marginal_buf, dim=0).mean(dim=0)
            agg = agg / agg.sum().clamp(min=1e-8)
            h_margin = float(-(agg * agg.clamp(min=1e-12).log()).sum())
        else:
            h_margin = self.h_max
        self.marginal_buf.clear()

        h_healthy = (h_margin >= self.anchor_h_min_ratio * self.h_max) if self.h_max > 0 else True
        if g < self.g_min and h_healthy:
            self.g_min = g
            self.windows_since_min = 0
            self.best_snapshot = self._win_buf[-1]
        else:
            self.windows_since_min += 1

        if h_margin >= self.h_max:
            self.windows_since_h_peak = 0
        else:
            self.windows_since_h_peak += 1
        if self.h_max_mode == 'decaying':
            self.h_max = max(h_margin, self.h_max_decay * self.h_max)
        else:
            self.h_max = max(self.h_max, h_margin)
        collapse_regime = h_margin < self.h_drop_ratio * self.h_max

        slope, tstat = self._slope_tstat(self.grad_hist)
        z = self._trend_z(slope, tstat, g)
        self.slope_buf.append(slope)

        deep = False
        u = 0.0
        fired = self._rng.random() < self.fire_prob
        # --- ABLATION: trigger is a coin flip, not the grad-norm-trend gate ---
        if (not fired) or self.windows_since_min < 1:
            lag, rst = 0, 0.0
        elif collapse_regime:
            deep = True
            lag, rst = self.windows_since_min, self.base_rst
            self.active_z_buf.append(z)
        else:
            u_sev = self._lag_fraction(z)
            if self.shallow_cap_mode == 'growing':
                cap = min(len(self._win_buf) - 1, self.windows_since_min)
            elif self.shallow_cap_mode == 'growing_scaled':
                cap = min(len(self._win_buf) - 1, math.ceil(self.windows_since_min * u_sev))
            elif self.shallow_cap_mode == 'growing_hmargin':
                cap = min(len(self._win_buf) - 1, self.windows_since_h_peak)
            elif self.shallow_cap_mode == 'growing_hmargin_scaled':
                cap = min(len(self._win_buf) - 1, math.ceil(self.windows_since_h_peak * u_sev))
            else:
                cap = min(self.maxlag_shallow, self.windows_since_min)
            if self.lag_mode == 'gain':
                lag = max(1, min(int(round(self.lag_gain * slope)), cap))
            else:
                u = u_sev
                lag = max(1, math.ceil(cap * u))
            self.active_z_buf.append(z)
            rst = self.base_rst

        print(f"[AdaGate-hgate-RANDOM] B{self.total_batches}: grad={g:.3f} "
              f"slope={slope:+.4f} z({self.trend_stat})={z:+.3f} fired={int(fired)}(p={self.fire_prob}) "
              f"H={h_margin:.3f}/{self.h_max:.3f} h_healthy={int(h_healthy)} "
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
