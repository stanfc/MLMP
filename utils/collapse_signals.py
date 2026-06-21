"""
Collapse / degradation signal monitor for CTTA.

Goal: find signals beyond H_margin that correlate with mIoU degradation —
in particular ones sensitive to VOC20's *uniform degradation* (where the class
marginal stays diverse so H_margin is blind), not only the *collapse* type
(ACDC / Cityscapes, where the marginal collapses to 1-2 classes).

All signals are computed per batch from the pre-adapt evaluate logits
`patch_preds` (shape (N, C, H, W)); one signal (ln_param_drift) is read from the
adapt method's trainable params vs its frozen source snapshot.

Usage (in main_continual.py, opt-in via --log_signals):
    mon = SignalMonitor(adapt_method)
    row = mon.update(patch_preds, round_num, condition, batch_idx)
    # row is an OrderedDict {signal_name: float}; write to csv.

The monitor keeps cross-round memory:
  * ref_marginal[condition]  — class marginal at the first encounter of a
    condition (round 1); kl_marg_ref measures drift away from it.
  * prev_hist[(condition, batch_idx)] — last round's predicted-class histogram
    for this exact image; pred_hist_drift is its L1 change (a uniform-drift cue).
"""
from collections import OrderedDict

import numpy as np
import torch

# Signal column order (also the CSV header order after the id columns).
SIGNAL_NAMES = [
    "h_margin",          # A1 marginal entropy (baseline; collapse-sensitive)
    "max_marginal",      # A2 dominant-class share
    "top2_marginal",     # A3 top-2 class mass
    "n_active_classes",  # A4 # classes with marginal > 1%
    "marginal_gini",     # A5 Gini of the class marginal
    "kl_marg_ref",       # A6 KL(marginal || round-1 marginal)  <- drift
    "h_pixel_mean",      # B1 mean per-pixel entropy
    "mean_conf",         # B2 mean max-softmax
    "frac_conf_high",    # B3 frac pixels max-prob > 0.9
    "frac_conf_low",     # B4 frac pixels max-prob < 0.5
    "mean_logit_gap",    # B5 mean(top1 - top2 logit)
    "logit_std",         # B6 mean over pixels of std-over-classes of logits
    "pixel_ent_std",     # B7 std of the per-pixel entropy map
    "pred_hist_drift",   # C1 L1 drift of this image's pred-class histogram vs last round
    "ln_param_drift",    # D1 ||theta_LN - theta_LN^source||_2
    "conn_components",   # E7 mean # connected components per patch (spatial fragmentation)
    # --- method-internal diag (read from adapt_method.diag; zero extra forward) ---
    "prompt_disagree",   # E1 disagreement across the 7 prompts (adapt logits)
    "layer_disagree",    # E2 disagreement across the 18 UAML layers (diagnose forward)
    "plpd_mean",         # E3 mean DeYO PLPD (patch-shuffle disagreement)
    "feat_norm",         # E4 mean visual-feature L2 norm (diagnose forward)
    "feat_text_align",   # E5 mean best-class feature-text cosine (diagnose forward)
    "grad_norm",         # E6 L2 norm of the LN-param gradients this step
    "filter_pass_rate",  # bonus: fraction of (prompt,pixel) surviving DeYO's dual filter
]

# Signals sourced from adapt_method.diag rather than from patch_preds.
_DIAG_SIGNALS = ["prompt_disagree", "layer_disagree", "plpd_mean",
                 "feat_norm", "feat_text_align", "grad_norm", "filter_pass_rate"]


@torch.no_grad()
def _marginal_signals(probs):
    """probs: (N, C, H, W) softmax. Returns dict of marginal-distribution signals."""
    C = probs.shape[1]
    marg = probs.mean(dim=(0, 2, 3))
    marg = marg / marg.sum().clamp(min=1e-8)
    h_margin = float(-(marg * marg.clamp(min=1e-12).log()).sum())
    sorted_marg, _ = torch.sort(marg, descending=True)
    max_marginal = float(sorted_marg[0])
    top2_marginal = float(sorted_marg[:2].sum())
    n_active = float((marg > 0.01).sum())
    # Gini of the marginal (sorted-ascending formula).
    asc, _ = torch.sort(marg)
    idx = torch.arange(1, C + 1, device=marg.device, dtype=marg.dtype)
    gini = float((2.0 * (idx * asc).sum()) / (C * asc.sum().clamp(min=1e-12)) - (C + 1.0) / C)
    return marg, OrderedDict(
        h_margin=h_margin,
        max_marginal=max_marginal,
        top2_marginal=top2_marginal,
        n_active_classes=n_active,
        marginal_gini=gini,
    )


@torch.no_grad()
def _pixel_signals(logits, probs):
    """logits/probs: (N, C, H, W). Returns dict of per-pixel confidence signals."""
    maxp = probs.max(dim=1).values                       # (N, H, W)
    mean_conf = float(maxp.mean())
    frac_conf_high = float((maxp > 0.9).float().mean())
    frac_conf_low = float((maxp < 0.5).float().mean())
    ent_map = -(probs * probs.clamp(min=1e-12).log()).sum(dim=1)   # (N, H, W)
    h_pixel_mean = float(ent_map.mean())
    pixel_ent_std = float(ent_map.std())
    top2 = torch.topk(logits, k=2, dim=1).values         # (N, 2, H, W)
    mean_logit_gap = float((top2[:, 0] - top2[:, 1]).mean())
    logit_std = float(logits.std(dim=1).mean())
    return OrderedDict(
        h_pixel_mean=h_pixel_mean,
        mean_conf=mean_conf,
        frac_conf_high=frac_conf_high,
        frac_conf_low=frac_conf_low,
        mean_logit_gap=mean_logit_gap,
        logit_std=logit_std,
        pixel_ent_std=pixel_ent_std,
    )


class SignalMonitor:
    def __init__(self, adapt_method):
        self.m = adapt_method
        self.ref_marginal = {}                  # condition -> marginal (C,)
        self.prev_hist = {}                     # (condition, batch_idx) -> hist (C,)
        # Snapshot source trainable params (the LN params being adapted).
        self.src = {}
        model = getattr(adapt_method, "model", None)
        state = getattr(adapt_method, "model_state", None)
        if model is not None and state is not None:
            for name, p in model.named_parameters():
                if p.requires_grad and name in state:
                    self.src[name] = state[name].detach().float().cpu().clone()

    @torch.no_grad()
    def _param_drift(self):
        if not self.src:
            return float("nan")
        total = 0.0
        for name, p in self.m.model.named_parameters():
            if name in self.src:
                total += float(((p.detach().float().cpu() - self.src[name]) ** 2).sum())
        return total ** 0.5

    @torch.no_grad()
    def update(self, patch_preds, round_num, condition, batch_idx):
        logits = patch_preds.float()
        probs = logits.softmax(dim=1)
        C = probs.shape[1]

        marg, msig = _marginal_signals(probs)
        psig = _pixel_signals(logits, probs)

        # A6: KL(current marginal || round-1 marginal for this condition).
        if condition not in self.ref_marginal:
            self.ref_marginal[condition] = marg.clone()
        ref = self.ref_marginal[condition]
        kl_marg_ref = float((marg * (marg.clamp(min=1e-12).log()
                                     - ref.clamp(min=1e-12).log())).sum())

        # C1: predicted-class histogram drift vs same image last round.
        pred = probs.argmax(dim=1).reshape(-1)
        hist = torch.bincount(pred, minlength=C).float()
        hist = hist / hist.sum().clamp(min=1e-8)
        key = (condition, batch_idx)
        if key in self.prev_hist:
            pred_hist_drift = float(0.5 * (hist - self.prev_hist[key]).abs().sum())
        else:
            pred_hist_drift = float("nan")
        self.prev_hist[key] = hist

        row = OrderedDict()
        row.update(msig)
        row["kl_marg_ref"] = kl_marg_ref
        row.update(psig)
        row["pred_hist_drift"] = pred_hist_drift
        row["ln_param_drift"] = self._param_drift()
        row["conn_components"] = self._conn_components(probs)
        # Method-internal diagnostics (populated by adapt_method during this batch).
        diag = getattr(self.m, "diag", {}) or {}
        for k in _DIAG_SIGNALS:
            row[k] = float(diag.get(k, float("nan")))
        # Reorder to SIGNAL_NAMES for a stable CSV layout.
        return OrderedDict((k, row.get(k, float("nan"))) for k in SIGNAL_NAMES)

    @torch.no_grad()
    def _conn_components(self, probs, max_patches=8):
        """Mean number of connected components across classes, per patch, per
        1k pixels (spatial-fragmentation proxy). Sampled over <=max_patches to
        bound cost; argmax map only, no model access."""
        try:
            from scipy import ndimage
        except ImportError:
            return float("nan")
        argmax = probs.argmax(dim=1)                 # (N, H, W)
        N = argmax.shape[0]
        sel = range(min(N, max_patches))
        counts = []
        for i in sel:
            lbl = argmax[i].cpu().numpy()
            ncomp = 0
            for c in np.unique(lbl):
                _, n = ndimage.label(lbl == c)
                ncomp += n
            counts.append(1000.0 * ncomp / lbl.size)
        return float(np.mean(counts)) if counts else float("nan")
