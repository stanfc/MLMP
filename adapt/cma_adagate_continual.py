"""
CMAAdaGateContinual -- the AdaGate restoration gate on a NON-ENTROPY objective.

This is the scientifically decisive arm of the plug-and-play claim. TENT,
DeYO+MLMP and MLMP are all entropy-family objectives, so together they only show
the gate works across entropy *variants*. CMA's loss is cross-modal cosine
alignment to the frozen text embeddings -- no entropy term anywhere:

    L = -mean_{i in top-K% confident pixels} cos(v_i, t_{c_i})

If the same gate, at the same unitless threshold, also keeps this objective from
collapsing, the claim upgrades from "works across entropy variants" to "works
regardless of the loss family".

Precedent: `cma_divgate_continual` (the OLDER absolute-threshold gate) already
took CMA from 0.84 to 19.32 at R150, so the objective is rescuable. Honest
expectation: CMA's own ceiling is low (peak 27.40), so a gated CMA will likely
sit BELOW the no-adaptation floor of 23.34. That is still a valid data point for
the scoped claim ("the gate prevents collapse"), and it is the cleanest evidence
that the gate is orthogonal to how good the base objective is -- the gate decides
whether the method falls, the objective decides how high it can climb. Report it
as such; do not hide it.

Base objective and parameter set follow `adapt/cma_continual.py` exactly.
`_is_excluded` is overridden so `top_block_exclude <= 0` excludes NOTHING, which
reproduces `cma_continual.set_ln_grads`. Launch with `--top_block_exclude 0` and
`save/ACDCDataset/cma_continual_k_0.5` (R150 = 0.84) is a valid no-gate control.

`TOP_K_PERCENT` is a class constant, not a CLI flag: `get_method()` inspects the
constructor signature, so this subclass deliberately does not override `__init__`.
"""
import time

import torch

from .deyo_mlmp_adagate_continual import DeYOMLMPAdaGateContinual


class CMAAdaGateContinual(DeYOMLMPAdaGateContinual):

    TOP_K_PERCENT = 0.5      # matches save/ACDCDataset/cma_continual_k_0.5 (the
                             # best of the k sweep: peak 27.40, and a full 150R)

    @staticmethod
    def _is_excluded(nm, top_block_exclude, num_blocks=24):
        if top_block_exclude <= 0:
            return False
        return DeYOMLMPAdaGateContinual._is_excluded(nm, top_block_exclude, num_blocks)

    def _cma_text(self):
        """CMA passes the full table (templates + appended averaged embedding),
        exactly as cma_continual does."""
        return self.text_x

    def cma_loss(self, logits, image_features, text_features):
        """Verbatim from adapt/cma_continual.py -- see that file's docstring."""
        avg_logits = logits.mean(dim=0)                 # (B, C, w, h)
        avg_text = text_features.mean(dim=0)            # (C, D)
        avg_text = avg_text / avg_text.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        with torch.no_grad():
            probs = avg_logits.softmax(dim=1)
            confidence, pred_cls = probs.max(dim=1)
            n_total = confidence.numel()
            k = max(1, int(round(n_total * self.TOP_K_PERCENT)))
            threshold = torch.topk(confidence.flatten(), k, sorted=True).values[-1]
            mask = confidence >= threshold

        B, _, w, h = avg_logits.shape
        patch_feats = image_features[:, 1:, :]          # drop CLS
        D = patch_feats.shape[-1]
        vis_feat = patch_feats.reshape(B, w, h, D)
        text_target = avg_text[pred_cls]
        cos_sim = (vis_feat * text_target).sum(dim=-1)
        return -cos_sim[mask].mean(), avg_logits

    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []
        for _ in range(self.steps):
            logits, image_features, text_features = self.model(
                x, self._cma_text(), True, interpolate=False)

            loss, avg_logits = self.cma_loss(logits, image_features, text_features)

            # H_margin regime signal, from the prompt-averaged logits (cma_divgate's
            # convention)
            with torch.no_grad():
                self.marginal_buf.append(
                    avg_logits.softmax(dim=1).mean(dim=[0, 2, 3]).detach().float().cpu())

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
