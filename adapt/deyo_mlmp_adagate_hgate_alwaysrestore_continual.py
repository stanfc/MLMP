"""
DeYOMLMPAdaGateHGateAlwaysRestoreContinual -- ablation: is the grad-norm-trend
TRIGGER (the `z <= trend_thr` gate) actually doing anything, or would
restoring on every monitor window work just as well?

Subclass of the locked CAGA-hgate method (deyo_mlmp_adagate_hgate_continual).
Only `_update_gate` is overridden: the trigger condition
`if z <= self.trend_thr or self.windows_since_min < 1: lag, rst = 0, 0.0`
is replaced with an UNCONDITIONAL fire every window (still skipped for one
window right after a fresh anchor is pinned, since restoring toward the
anchor you just became makes no sense). DEEP-vs-SHALLOW target selection
(based on h_margin/collapse_regime), the anchor-health hgate check, and the
lag/severity machinery are all otherwise identical to the parent -- this
isolates exactly one thing: does the grad-norm-trend *timing* signal matter,
holding everything else (restore probability, target choice) fixed?

Separate method name/save_dir -- does not touch CAGA's own results.
"""
import math

import torch

from .deyo_mlmp_adagate_hgate_continual import DeYOMLMPAdaGateHGateContinual


class DeYOMLMPAdaGateHGateAlwaysRestoreContinual(DeYOMLMPAdaGateHGateContinual):

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
        # --- ABLATION: no grad-norm-trend gate -- always fire except right
        # after pinning a fresh anchor (windows_since_min < 1) ---
        if self.windows_since_min < 1:
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

        print(f"[AdaGate-hgate-ALWAYS] B{self.total_batches}: grad={g:.3f} "
              f"slope={slope:+.4f} z({self.trend_stat})={z:+.3f} (trigger DISABLED) "
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
