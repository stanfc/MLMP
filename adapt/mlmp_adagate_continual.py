"""
MLMPAdaGateContinual -- the AdaGate restoration gate on the MLMP-continual objective.

Third base objective for the plug-and-play claim (after DeYO+MLMP and plain TENT).
The gate is inherited byte-for-byte from `DeYOMLMPAdaGateContinual`; only the base
objective and the trainable-parameter set change.

Base objective follows `adapt/mlmp_continual.py` (`prompt_integration='loss'`):
    L = mean pixel entropy over T prompt templates  +  alpha_cls * mean CLS entropy
with the 18-layer mean-fused forward. No DeYO entropy margin, no PLPD, no
reweighting.

**Parameter set matches the published baseline exactly.** `_is_excluded` is
overridden so that `top_block_exclude <= 0` excludes NOTHING (not even `ln_post`),
reproducing `mlmp_continual.set_ln_grads`, which makes every LayerNorm of the
visual encoder trainable. Launch with `--top_block_exclude 0` and the existing
`save/ACDCDataset/mlmp_continual_round_150_step_1` run (R150 = 1.53) is a valid
no-gate control -- no second arm needs to be run.

`alpha_cls` and `prompt_integration` are class constants rather than CLI flags:
`get_method()` inspects the constructor signature to decide what to pass, so this
subclass deliberately does not override `__init__`.
"""
import time

import torch

from .deyo_mlmp_adagate_continual import DeYOMLMPAdaGateContinual


class MLMPAdaGateContinual(DeYOMLMPAdaGateContinual):

    ALPHA_CLS = 1.0          # adapt/mlmp_continual.py default (ILE weight)

    @staticmethod
    def softmax_entropy(x: torch.Tensor, dim=-3) -> torch.Tensor:
        # mlmp_continual's signature: the CLS term needs dim=2, while the parent's
        # version hard-codes dim=-3
        return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)

    @staticmethod
    def _is_excluded(nm, top_block_exclude, num_blocks=24):
        # top_block_exclude <= 0 -> train every visual LayerNorm, exactly like
        # mlmp_continual, so the published run is a drop-in no-gate control.
        if top_block_exclude <= 0:
            return False
        return DeYOMLMPAdaGateContinual._is_excluded(nm, top_block_exclude, num_blocks)

    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []
        for _ in range(self.steps):
            logits, _, _, cls_logits = self.model(
                x, self.text_x[:-1], True,
                interpolate=False,
                vision_outputs=self.vision_outputs,
                return_vanilla_cls=True,
                vision_out_type="mean")
            # logits: (T, B, C, h, w)   cls_logits: (T, B, C, 1, 1)

            with torch.no_grad():
                probs_ens = logits.softmax(dim=-3).mean(dim=0)      # (B, C, h, w)
                self.marginal_buf.append(
                    probs_ens.mean(dim=[0, 2, 3]).detach().float().cpu())

            loss = (self.softmax_entropy(logits).mean()
                    + self.ALPHA_CLS * self.softmax_entropy(cls_logits, dim=2).mean())
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
