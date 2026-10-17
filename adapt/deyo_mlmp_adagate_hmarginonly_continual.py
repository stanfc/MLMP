"""
DeYOMLMPAdaGateHMarginOnlyContinual -- ablation: what if the gate used ONLY
h_margin, with grad_norm removed from every decision it currently makes?

Subclass of the plain (non-hgate) CAGA (deyo_mlmp_adagate_continual). Mirrors
deyo_mlmp_adagate_gradnormonly_continual.py's design (same user choice: fixed
SHALLOW, intensity graded by the signal's own severity) but with the roles of
grad_norm and h_margin swapped everywhere:
  - TRIGGER: fires whenever h_margin has dropped below h_drop_ratio *
    running-max h_margin (this is exactly the parent's `collapse_regime`
    condition -- here it drives the trigger itself, not just a DEEP-vs-
    SHALLOW split inside an already-grad_norm-triggered branch). grad_norm's
    OLS-trend z plays no role in deciding whether to fire.
  - ANCHOR: pinned when h_margin sets a new running-max (mirrors the parent's
    "g < g_min" rule, with h_margin's OWN peak replacing grad_norm's own
    minimum) -- NOT when grad_norm is lowest. This is the h_margin analogue
    of g_min/windows_since_min: h_peak/windows_since_h_peak (the latter
    already existed in the parent for bookkeeping; here it drives lag depth).
  - INTENSITY: graded by h_margin severity, i.e. how far below the running
    peak the current window is: severity = (h_max - h_margin) / (h_max *
    (1 - h_drop_ratio)), so severity = 1.0 exactly at the trigger threshold
    and grows as h_margin keeps falling. rst = base_rst *
    clamp(severity, 1.0, h_severity_cap).
  - LAG (SHALLOW reach): cap = min(len(win_buf)-1, ceil(windows_since_h_peak
    * u)) where u = clamp(severity, 0, 1) -- the h_margin analogue of
    shallow_cap_mode='growing_scaled', keyed on windows_since_h_peak instead
    of windows_since_min.
  - DEEP is disabled outright (always SHALLOW), matching the grad-norm-only
    sibling ablation.

grad_norm is still computed (for logging/comparability only) but never used.

Separate method name/save_dir -- does not touch CAGA's own results.
"""
import math

import torch

from .deyo_mlmp_adagate_continual import DeYOMLMPAdaGateContinual


class DeYOMLMPAdaGateHMarginOnlyContinual(DeYOMLMPAdaGateContinual):
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
                 # --- h_margin-only ablation ---
                 h_severity_cap=3.0,
                 save_dir=None,
                 runtime_calculation=False, device='cpu'):
        self.h_severity_cap = float(h_severity_cap)
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
        print(f"+++ CAGA-hmarginonly: DEEP disabled (always SHALLOW), trigger/anchor/"
              f"intensity/lag all h_margin-derived (cap={self.h_severity_cap}x base_rst); "
              f"grad_norm plays no role anywhere in this gate.")

    @torch.no_grad()
    def _update_gate(self):
        if len(self.grad_buf) == 0:
            self.batch_count = 0
            return
        self._win_buf.append(self._snapshot_ln_weights())

        # grad_norm is still computed for logging/comparability only.
        g = float(sum(self.grad_buf) / len(self.grad_buf))
        self.grad_hist.append(g)
        if len(self.grad_hist) > self.slope_window:
            self.grad_hist = self.grad_hist[-self.slope_window:]
        if g < self.g_min:
            self.g_min = g

        if len(self.marginal_buf) > 0:
            agg = torch.stack(self.marginal_buf, dim=0).mean(dim=0)
            agg = agg / agg.sum().clamp(min=1e-8)
            h_margin = float(-(agg * agg.clamp(min=1e-12).log()).sum())
        else:
            h_margin = self.h_max
        self.marginal_buf.clear()

        # ANCHOR: pinned on a new h_margin PEAK (h_margin analogue of g<g_min).
        if h_margin >= self.h_max:
            self.windows_since_h_peak = 0
            self.best_snapshot = self._win_buf[-1]
        else:
            self.windows_since_h_peak += 1
        if self.h_max_mode == 'decaying':
            self.h_max = max(h_margin, self.h_max_decay * self.h_max)
        else:
            self.h_max = max(self.h_max, h_margin)

        # windows_since_min is kept in sync so any downstream code/logging
        # that still reads it (e.g. base class helpers) sees something
        # sane; it plays no role in THIS gate's own decisions below.
        self.windows_since_min = self.windows_since_h_peak

        slope, tstat = self._slope_tstat(self.grad_hist)
        z = self._trend_z(slope, tstat, g)  # computed for logging only, unused below
        self.slope_buf.append(slope)

        threshold_gap = max(1.0 - self.h_drop_ratio, 1e-6)
        severity = (self.h_max - h_margin) / max(self.h_max, 1e-8) / threshold_gap
        fired = severity > 1.0

        u = 0.0
        if (not fired) or self.windows_since_h_peak < 1:
            lag, rst = 0, 0.0
        else:
            # ABLATION: always SHALLOW, lag keyed on windows_since_h_peak
            u = min(max(severity, 0.0), 1.0)
            cap = min(len(self._win_buf) - 1, math.ceil(self.windows_since_h_peak * u))
            lag = max(1, math.ceil(cap * u))
            rst = self.base_rst * min(max(severity, 1.0), self.h_severity_cap)

        print(f"[AdaGate-HMARGINONLY] B{self.total_batches}: H={h_margin:.3f}/{self.h_max:.3f} "
              f"severity={severity:.2f} fired={int(fired)} (grad={g:.3f}, z={z:+.3f}, unused) "
              f"since_hpeak={self.windows_since_h_peak} u={u:.2f} lag={lag}w rst={rst:.4f} shallow-only")
        if self.gate_log_path is not None:
            with open(self.gate_log_path, 'a') as _f:
                _f.write(f"{self.total_batches},{g:.6f},{slope:.6f},{z:.6f},{u:.6f},"
                         f"{h_margin:.6f},{int(fired)},"
                         f"{self.windows_since_h_peak},{lag},{rst:.6f},0\n")
        self.current_lag = lag
        self.current_rst = rst
        self.current_deep = False
        self.grad_buf.clear()
        self.batch_count = 0
