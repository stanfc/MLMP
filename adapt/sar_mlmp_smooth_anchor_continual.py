"""
SARMLMPSmoothAnchorContinual — SAR (reliable filter + SAM) loss in full MLMP
machinery (multi-prompt + multi-layer UAML eval), with a SMOOTH-ANCHOR
diversity gate instead of a discrete 3-tier DivGate.

Combination of three existing methods:
  - SAR (adapt/sar_continual.py): sample-level reliable filter (skip if mean
    pixel entropy > e_margin), pixel-level reliable filter (loss only on
    entropy < e_margin pixels), and the SAM optimizer (two forward+backward
    passes per step -> flat minima). SAR's hard loss-EMA model recovery is
    DROPPED; the smooth anchor's "H <= h_floor -> restore toward source" is the
    gentler replacement (same choice as sar_divgate_continual).
  - MLMP / UAML (adapt/mlmp_continual.py): 7 prompt templates (loss-averaged)
    and 18-layer feature fusion. Adapt forward uses vision_out_type="mean";
    evaluate uses "adaptive_weighted_mean" (entropy-weighted, sharpened).
  - SmoothAnchor (adapt/tent_divgate_smooth_anchor.py): continuous
    lag(H) = lag_scale / (H - h_floor) mapping from marginal-diversity entropy
    to restoration anchor depth, over a rotating fp16-CPU LN-snapshot buffer.
    rst is fixed; only the anchor depth moves. "The worse the marginal health
    (lower H), the deeper back we reach for the restore anchor."

The --uaml_in_adapt flag toggles whether the multi-layer fusion drives the
adapt loss (default True) or only the evaluate path (False -> single-layer
adapt, halving the per-step SAM cost; evaluate still uses 18-layer fusion).

Gate H_margin here is the ADAPT-time 7-prompt ensemble (sits ~0.9 above
evaluate-based diversity; healthy ~3.0), so thresholds are anchored near 3.0,
NOT the 1.5-1.8 of the single-prompt TENT original.

CTTA hard-rule compliance: no per-sample reset; anchors are partial stochastic
restores toward past/source snapshots, never a full reset.

See docs/sar_mlmp_smooth_anchor_continual_spec.md for the full design.
"""

import os
import time
import copy
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters
from .sam import SAM

REFERENCE_PROMPT = 'a photo of a {}'


class SARMLMPSmoothAnchorContinual:

    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 vision_outputs=tuple(range(-1, -19, -1)),
                 prompt_dir='prompts.yaml',
                 alpha_cls=0.0,
                 uaml_in_adapt=True,
                 # --- SAR ---
                 e_margin=1.8, sam_rho=0.05,
                 # --- SmoothAnchor ---
                 h_ceil=2.9, h_floor=2.2,
                 lag_scale=150.0, max_lag=3000, rst=0.005,
                 monitor_interval=50,
                 save_dir=None,
                 runtime_calculation=False, device='cpu'):
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps
        self.vision_outputs = vision_outputs
        self.alpha_cls = float(alpha_cls)
        self.uaml_in_adapt = bool(uaml_in_adapt)
        self.runtime = runtime_calculation
        self.device = device

        if classes is None:
            raise ValueError("classes is required in __init__")
        self.classes = classes

        # SAR knobs
        self.e_margin = float(e_margin)
        self.sam_rho = float(sam_rho)

        # SmoothAnchor knobs
        self.h_ceil = float(h_ceil)
        self.h_floor = float(h_floor)
        if self.h_floor >= self.h_ceil:
            raise ValueError(f"h_floor ({self.h_floor}) must be < h_ceil ({self.h_ceil})")
        self.lag_scale = float(lag_scale)
        if self.lag_scale <= 0:
            raise ValueError(f"lag_scale must be > 0, got {self.lag_scale}")
        self.max_lag = int(max_lag)
        if self.max_lag < 1:
            raise ValueError(f"max_lag must be >= 1, got {self.max_lag}")
        self.rst = float(rst)
        if not (0.0 <= self.rst <= 1.0):
            raise ValueError(f"rst must be in [0, 1], got {self.rst}")
        self.monitor_interval = int(monitor_interval)
        if self.monitor_interval < 1:
            raise ValueError(f"monitor_interval must be >= 1, got {self.monitor_interval}")

        # ---------- OVSS model ----------
        self.model, self.tokenize = load_ovss(ovss_type, ovss_backbone, device=device)

        # ---------- Prompts ----------
        if prompt_dir:
            self.prompt_templates = load_prompts_from_yaml(prompt_dir)
            print(f"Number of prompt templates: {len(self.prompt_templates)}")
        else:
            self.prompt_templates = [REFERENCE_PROMPT]

        # ---------- Freeze text encoder, enable ALL visual LN grads (SAR-faithful) ----------
        self.model.transformer.requires_grad_(False)
        self.model.ln_final.requires_grad_(False)
        self.model.token_embedding.requires_grad_(False)
        self.model.visual = self.set_ln_grads(self.model.visual)
        params, names = self.collect_ln_params(self.model.visual)
        self.named_ln_params = list(zip(names, params))

        print_clip_parameters(self.model)
        print(f"+++ SAR-MLMP-SmoothAnchor (T={len(self.prompt_templates)}): "
              f"e_margin={self.e_margin:.3f}, sam_rho={self.sam_rho}, "
              f"alpha_cls={self.alpha_cls}, uaml_in_adapt={self.uaml_in_adapt}, "
              f"UAML={len(self.vision_outputs)} layers")
        print(f"+++ lag(H) = {self.lag_scale} / (H - {self.h_floor})  "
              f"clamped to [1, {self.max_lag}]; H>={self.h_ceil} -> no restore; "
              f"H<={self.h_floor} -> source; monitor={self.monitor_interval} "
              f"-> LN params: {len(params)}")
        for h in [self.h_ceil, 0.5 * (self.h_ceil + self.h_floor), self.h_floor + 0.05]:
            lg = self._lag_from_h(h)
            tag = "source" if lg is None else (f"lag={lg}" if lg else "off")
            print(f"    H={h:.3f} -> {tag}")

        # ---------- SAM-wrapped Adam (LN params only) ----------
        self.optimizer = SAM(
            params, optim.Adam, rho=self.sam_rho,
            lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0,
        )
        print_optimizer_parameters(self.optimizer, self.model)

        # ---------- Source snapshot (for reset() and the source-fallback anchor) ----------
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer)
        self._source_ln_snapshot = {
            name: self.model_state[name].detach().clone()
            for name, _ in self.named_ln_params
        }

        # ---------- Rotating fp16-CPU LN snapshots ----------
        self._anchor_buf = deque(maxlen=self.max_lag + 1)
        self._anchor_buf.append(self._snapshot_ln_weights())

        # ---------- Text features (T per-template + averaged at index [-1]) ----------
        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(
                self.classes, self.prompt_templates, average=True).squeeze()

        # ---------- Gate state ----------
        self.marginal_buf = []
        self.batch_count = 0
        self.total_batches = 0
        self.current_lag = 0          # 0 -> no restore, None -> source, int -> lagged anchor
        self.current_rst = 0.0

        # ---------- Optional logs ----------
        self.gate_log_path = None
        self.sar_log_path = None
        if save_dir:
            try:
                os.makedirs(save_dir, exist_ok=True)
                self.gate_log_path = os.path.join(save_dir, "gate_log.csv")
                with open(self.gate_log_path, 'w') as f:
                    f.write("total_batches,h_margin,lag,rst\n")
                self.sar_log_path = os.path.join(save_dir, "sar_log.txt")
                with open(self.sar_log_path, 'w') as f:
                    f.write("# SAR-MLMP-SA log: per-batch reliability diagnostics\n")
                    f.write("total_batches,mean_entropy,was_filtered,was_reset\n")
                print(f"+++ gate log -> {self.gate_log_path}; sar log -> {self.sar_log_path}")
            except OSError as e:
                print(f"+++ logs disabled ({save_dir}): {e}")
                self.gate_log_path = None
                self.sar_log_path = None

        if self.runtime:
            self.adapt_times = []
            self.eval_times = []

    # ===========================================================
    # Smooth mapping H_margin -> lag
    # ===========================================================

    def _lag_from_h(self, h):
        """0 -> no restore (H >= h_ceil); None -> source (H <= h_floor or lag>max_lag);
        int -> anchor lag in [1, max_lag]."""
        if h >= self.h_ceil:
            return 0
        if h <= self.h_floor:
            return None
        raw = self.lag_scale / (h - self.h_floor)
        if raw > self.max_lag:
            return None
        return max(1, int(round(raw)))

    # ===========================================================
    # Snapshot helpers
    # ===========================================================

    @torch.no_grad()
    def _snapshot_ln_weights(self):
        return {name: p.detach().cpu().to(torch.float16).clone()
                for name, p in self.named_ln_params}

    @torch.no_grad()
    def _push_current_to_anchor_buf(self):
        self._anchor_buf.append(self._snapshot_ln_weights())

    # ===========================================================
    # Public API
    # ===========================================================

    def adapt(self, x):
        return self.perform_adaptation(x)

    def continual_adapt(self, x):
        return self.perform_adaptation(x)

    @torch.no_grad()
    def evaluate(self, x):
        # MLMP UAML: 18-layer entropy-weighted fusion, single averaged prompt.
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
        self._anchor_buf.clear()
        self._anchor_buf.append(self._snapshot_ln_weights())

    # ===========================================================
    # Adaptation
    # ===========================================================

    def _adapt_forward(self, x):
        """Multi-prompt forward; multi-layer mean fusion iff uaml_in_adapt.
        Returns (logits (T,B,C,h,w), cls_logits (T,B,C,1,1) or None)."""
        vo = self.vision_outputs if self.uaml_in_adapt else (-1,)
        if self.alpha_cls > 0:
            logits, _, _, cls = self.model(
                x, self.text_x[:-1], True, interpolate=False,
                vision_outputs=vo, return_vanilla_cls=True, vision_out_type="mean")
            return logits, cls
        logits, _, _ = self.model(
            x, self.text_x[:-1], True, interpolate=False,
            vision_outputs=vo, vision_out_type="mean")
        return logits, None

    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []

        # Snapshot CURRENT LN state BEFORE this batch's gradient step.
        self._push_current_to_anchor_buf()

        was_filtered = True
        ent_mean_for_log = float('nan')

        for _ in range(self.steps):
            # ----- forward 1: real entry state (reliability + gate signal) -----
            logits, cls = self._adapt_forward(x)            # (T,B,C,h,w)
            ent_map = self.softmax_entropy(logits)          # (B,h,w)
            ent_mean = ent_map.mean()
            ent_mean_for_log = ent_mean.item()

            # ----- gate signal: pushed every batch, even if filtered -----
            with torch.no_grad():
                probs_ens = logits.softmax(dim=-3).mean(dim=0)   # (B,C,h,w)
                self.marginal_buf.append(
                    probs_ens.mean(dim=[0, 2, 3]).detach().float().cpu())

            # ----- SAR sample-level reliable filter -----
            if ent_mean.item() <= self.e_margin:
                pixel_mask = ent_map < self.e_margin
                if pixel_mask.sum().item() > 0:
                    was_filtered = False
                    loss = ent_map[pixel_mask].mean()
                    if cls is not None:
                        loss = loss + self.alpha_cls * self._cls_entropy(cls).mean()

                    # ----- SAM step 1: ascend to perturbed weights -----
                    loss.backward()
                    self.optimizer.first_step(zero_grad=True)

                    # ----- SAM step 2: gradient at perturbed point -----
                    logits2, cls2 = self._adapt_forward(x)
                    ent_map2 = self.softmax_entropy(logits2)
                    pixel_mask2 = ent_map2 < self.e_margin
                    if pixel_mask2.sum().item() > 0:
                        loss2 = ent_map2[pixel_mask2].mean()
                        if cls2 is not None:
                            loss2 = loss2 + self.alpha_cls * self._cls_entropy(cls2).mean()
                        loss2.backward()
                        self.optimizer.second_step(zero_grad=True)
                        loss_report.append(loss2.item())
                        if self.current_rst > 0.0:
                            self._stochastic_restore(self.current_rst, self.current_lag)
                    else:
                        # Perturbation made all pixels unreliable: undo perturbation,
                        # take no real update, no restore.
                        self.optimizer.second_step(zero_grad=True)
                else:
                    self.optimizer.zero_grad()
            else:
                self.optimizer.zero_grad()

            self.batch_count += 1
            self.total_batches += 1
            if self.batch_count >= self.monitor_interval:
                self._update_lag()

        if self.sar_log_path is not None:
            try:
                with open(self.sar_log_path, 'a') as f:
                    f.write(f"{self.total_batches},{ent_mean_for_log:.6f},"
                            f"{int(was_filtered)},0\n")
            except OSError:
                pass

        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

    # ===========================================================
    # Gate update
    # ===========================================================

    @torch.no_grad()
    def _update_lag(self):
        if len(self.marginal_buf) == 0:
            self.batch_count = 0
            return
        agg = torch.stack(self.marginal_buf, dim=0).mean(dim=0)
        agg = agg / agg.sum().clamp(min=1e-8)
        h = -(agg * agg.clamp(min=1e-12).log()).sum().item()

        new_lag = self._lag_from_h(h)
        new_rst = 0.0 if new_lag == 0 else self.rst
        lag_str = "source" if new_lag is None else ("off" if new_lag == 0 else str(new_lag))

        print(f"[SAR-MLMP-SA] B{self.total_batches}: H_margin={h:.3f} "
              f"lag={lag_str} rst={new_rst}")
        if self.gate_log_path is not None:
            try:
                with open(self.gate_log_path, 'a') as f:
                    f.write(f"{self.total_batches},{h:.6f},{lag_str},{new_rst}\n")
            except OSError:
                pass

        self.current_lag = new_lag
        self.current_rst = new_rst
        self.marginal_buf.clear()
        self.batch_count = 0

    # ===========================================================
    # Restoration
    # ===========================================================

    @torch.no_grad()
    def _stochastic_restore(self, rst, lag):
        if lag is None:
            anchor = self._source_ln_snapshot
        else:
            idx = -1 - lag
            anchor = self._anchor_buf[0] if -idx > len(self._anchor_buf) else self._anchor_buf[idx]
        for name, p in self.named_ln_params:
            mask = (torch.rand(p.shape, device=p.device) < rst).to(p.dtype)
            src = anchor[name].to(p.device, dtype=p.dtype)
            p.data.mul_(1.0 - mask).add_(src * mask)

    # ===========================================================
    # Loss helpers
    # ===========================================================

    @staticmethod
    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        # x: (T, B, C, h, w) -- entropy over class dim, mean over prompt dim T.
        ent = -(x.softmax(-3) * x.log_softmax(-3)).sum(-3)   # (T, B, h, w)
        return ent.mean(dim=0)                                # (B, h, w)

    @staticmethod
    def _cls_entropy(cls_logits: torch.Tensor) -> torch.Tensor:
        # cls_logits: (T, B, C, 1, 1) -- entropy over class dim, mean over T.
        ent = -(cls_logits.softmax(-3) * cls_logits.log_softmax(-3)).sum(-3)  # (T,B,1,1)
        return ent.mean(dim=0)                                                 # (B,1,1)

    # ===========================================================
    # Shared helpers (mirroring SARContinual)
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
    def set_ln_grads(model):
        model.requires_grad_(False)
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)
        return model

    @staticmethod
    def collect_ln_params(model):
        params, names = [], []
        for nm, m in model.named_modules():
            if isinstance(m, nn.LayerNorm):
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
