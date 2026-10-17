"""
DeYOMLMPAdaGateGradNormOnlyContinual -- ablation: what if the gate used ONLY
grad_norm, with h_margin removed from every decision it currently makes?

Subclass of the plain (non-hgate) CAGA (deyo_mlmp_adagate_continual). In the
parent, grad_norm (via its OLS-trend z) decides WHEN to consider a restore,
but h_margin still decides (a) DEEP vs SHALLOW (collapse_regime = h_margin
dropped below h_drop_ratio * running-max), and (b) restore intensity is a
flat `base_rst` regardless of how bad grad_norm actually is. This variant
removes h_margin from both:
  (a) DEEP is disabled outright -- every trigger is SHALLOW (user's explicit
      choice: fixed SHALLOW, not a grad-norm-derived DEEP/SHALLOW split).
  (b) restore intensity is graded by grad_norm's own severity instead of a
      flat rate: severity = current windowed grad_norm / best-ever grad_norm
      (g_min), rst = base_rst * clamp(severity, 1.0, grad_severity_cap).
      A window right at the best-ever grad_norm gets the base rate; a window
      whose grad_norm has drifted far above the best-ever minimum gets
      restored harder, capped at grad_severity_cap x base_rst.
Trigger (whether to fire this window) and lag depth (how far back SHALLOW
reaches) are UNCHANGED from the parent -- both already grad_norm-derived
(the z-score trigger, and the ECDF lag-severity u_sev computed from the same
z), so nothing needs to change there. Anchor acceptance (g < g_min) was
already grad_norm-only in the parent (the h-margin health check is CAGA-
hgate's addition, not present here) -- also unchanged.

Separate method name/save_dir -- does not touch CAGA's own results.
"""
import math

import torch

from .deyo_mlmp_adagate_continual import DeYOMLMPAdaGateContinual


class DeYOMLMPAdaGateGradNormOnlyContinual(DeYOMLMPAdaGateContinual):
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
                 # --- grad-norm-only ablation ---
                 grad_severity_cap=3.0,
                 save_dir=None,
                 runtime_calculation=False, device='cpu'):
        self.grad_severity_cap = float(grad_severity_cap)
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
            shallow_cap_mode=shallow_cap_mode, save_dir=save_dir,
            runtime_calculation=runtime_calculation, device=device)
        print(f"+++ CAGA-gradnormonly: DEEP disabled (always SHALLOW), restore "
              f"intensity graded by grad_norm severity (cap={self.grad_severity_cap}x "
              f"base_rst); h_margin plays no role anywhere in this gate.")

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

        if g < self.g_min:
            self.g_min = g
            self.windows_since_min = 0
            self.best_snapshot = self._win_buf[-1]
        else:
            self.windows_since_min += 1

        # h_margin is still computed (for logging/comparability only) but
        # NEVER used below: no collapse_regime, no DEEP branch, no intensity
        # grading from it.
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
        if self.h_max_mode == 'decaying':
            self.h_max = max(h_margin, self.h_max_decay * self.h_max)
        else:
            self.h_max = max(self.h_max, h_margin)

        slope, tstat = self._slope_tstat(self.grad_hist)
        z = self._trend_z(slope, tstat, g)
        self.slope_buf.append(slope)

        u = 0.0
        severity = g / max(self.g_min, 1e-8)
        if z <= self.trend_thr or self.windows_since_min < 1:
            lag, rst = 0, 0.0
        else:
            # ABLATION: always SHALLOW (no DEEP branch at all)
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
            # ABLATION: intensity graded by grad_norm severity, not flat base_rst
            rst = self.base_rst * min(max(severity, 1.0), self.grad_severity_cap)

        print(f"[AdaGate-GRADNORMONLY] B{self.total_batches}: grad={g:.3f} "
              f"severity={severity:.2f} slope={slope:+.4f} z({self.trend_stat})={z:+.3f}/{self.trend_thr:.3f} "
              f"(h_margin={h_margin:.3f}, unused) since_min={self.windows_since_min} "
              f"u={u:.2f} lag={lag}w rst={rst:.4f} shallow-only")
        if self.gate_log_path is not None:
            with open(self.gate_log_path, 'a') as _f:
                _f.write(f"{self.total_batches},{g:.6f},{slope:.6f},{z:.6f},{u:.6f},"
                         f"{h_margin:.6f},0,"
                         f"{self.windows_since_min},{lag},{rst:.6f},0\n")
        self.current_lag = lag
        self.current_rst = rst
        self.current_deep = False
        self.grad_buf.clear()
        self.batch_count = 0
