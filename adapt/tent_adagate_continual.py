"""
TENTAdaGateContinual -- the AdaGate restoration gate on a TENT base objective.

Purpose: the plug-and-play claim. `deyo_mlmp_adagate_continual` demonstrates the
self-calibrating gate on ONE base objective (DeYO filter + PLPD reweighting on top
of the MLMP multi-prompt / 18-layer fused forward). This subclass keeps the gate
byte-for-byte and swaps the base objective for plain TENT, so the only difference
from `tent_continual` is the gate:

    tent_continual          : loss.backward() -> optimizer.step()
    tent_adagate_continual  : loss.backward() -> optimizer.step() -> gate/restore

Everything the gate needs is inherited unchanged from DeYOMLMPAdaGateContinual:
the MAD-z trend trigger, the ECDF lag, the H_margin regime switch, the permanent
best-state anchor, `_update_gate()`, `_stochastic_restore()` and the gate log.

Base objective follows `adapt/tent_continual.py` exactly:
  - text embeddings built with average=False (all prompt templates, no averaged one)
  - single-level forward (no `vision_outputs`), i.e. last layer only
  - pixel-wise softmax entropy over ALL pixels; no DeYO entropy margin, no PLPD
    mask, no entropy/PLPD reweighting
  - evaluate() uses logits[0] with interpolate=True

The DeYO-specific constructor arguments (`deyo_margin_factor`, `plpd_threshold`,
`aug_type`, `reweight_*`, `vision_outputs`, ...) are still accepted so the launcher
can pass one shared flag block, but the loss never reads them.
"""
import time

import torch

from .deyo_mlmp_adagate_continual import DeYOMLMPAdaGateContinual


class TENTAdaGateContinual(DeYOMLMPAdaGateContinual):

    # NOTE: no __init__ override. `get_method()` inspects the constructor
    # signature to decide which argparse values to pass, so a (*args, **kwargs)
    # forwarder breaks dispatch and copying the parent's long signature would
    # drift from it. The one thing that must differ -- the text table -- is built
    # lazily instead.

    @staticmethod
    def _is_excluded(nm, top_block_exclude, num_blocks=24):
        # top_block_exclude <= 0 -> train every visual LayerNorm, exactly like
        # tent_continual, so the published run is a drop-in no-gate control.
        # (The finished top_block_exclude=6 pair is unaffected: this only changes
        # behaviour at <= 0.)
        if top_block_exclude <= 0:
            return False
        return DeYOMLMPAdaGateContinual._is_excluded(nm, top_block_exclude, num_blocks)

    def _tent_text(self):
        """TENT convention: all prompt templates, no appended averaged embedding.

        The parent's __init__ built an average=True table; this replaces it on
        first use, leaving the parent file untouched.
        """
        if getattr(self, "_tent_text_x", None) is None:
            with torch.no_grad():
                self._tent_text_x = self.extract_text_embeddings(
                    self.classes, self.prompt_templates, average=False).squeeze()
            print("[TENT-AdaGate] base objective = plain TENT "
                  "(single-level, all-pixel entropy, no DeYO filter); gate unchanged")
        return self._tent_text_x

    @torch.no_grad()
    def evaluate(self, x):
        t1 = time.time()
        logits, _, _ = self.model(x, self._tent_text(), True, interpolate=True)
        logits = logits[0]
        if self.runtime:
            self.eval_times.append(time.time() - t1)
        return logits

    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []
        for _ in range(self.steps):
            logits, _, _ = self.model(x, self._tent_text(), True, interpolate=False)

            # H_margin regime signal. tent_continual's convention is the first
            # prompt template (`logits[0]`), matching tent_divgate_continual.
            with torch.no_grad():
                probs = logits[0].softmax(dim=-3)          # (B, C, h, w)
                self.marginal_buf.append(
                    probs.mean(dim=[0, 2, 3]).detach().float().cpu())

            loss = self.softmax_entropy(logits).mean()
            loss_report.append(loss.item())
            loss.backward()

            # grad-norm over exactly the LN params we train -- the gate's trigger
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
