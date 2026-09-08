"""
SARAdaGateContinual -- the AdaGate restoration gate on SAR's objective.

Plug-and-play arm #4, and the one with a DIFFERENT failure mode. SAR (ICLR 2023)
does not collapse on ACDC: it oscillates. `save/ACDCDataset/sar_continual_weather`
peaks 33.30 @ R67 -- the second-highest peak of any baseline -- but dips at ~R50
and ~R100 and ends at 25.31 (mean 30.32). So this arm does not ask "can the gate
stop a collapse"; it asks "can the gate remove the oscillation of a method that
already claims stability".

⚠️ This arm REPLACES rather than ADDS a mechanism, unlike the TENT / MLMP / DELTA
arms. SAR ships its own recovery rule (Filter C: when the EMA of the loss falls
below `e_0`, hard-reset every LN param to source). Running that alongside AdaGate
would put two controllers on the same parameters. Following the design already
used in `adapt/sar_divgate_continual.py`, SAR's Filter C is disabled here and
AdaGate's graded restore takes its place. Everything else in SAR is untouched:
SAM (rho = 0.05), the sample-level reliability filter, and the per-pixel entropy
filter at `e_margin`.

State it that way in the paper -- "the gate replaces SAR's hand-designed hard
recovery" -- not as a pure one-flag addition.

Parameter set: `_is_excluded` is overridden so `--top_block_exclude 0` trains all
100 visual LN params, matching `sar_continual.set_ln_grads`, so the published run
is a valid reference.

`E_MARGIN` / `SAM_RHO` are class attributes rather than CLI flags (`get_method()`
inspects the constructor signature, so this subclass does not override
`__init__`); their values are those of the published run: `--e_margin 1.8
--sam_rho 0.05`.
"""
import time

import torch
import torch.optim as optim

from .deyo_mlmp_adagate_continual import DeYOMLMPAdaGateContinual
from .sam import SAM


class SARAdaGateContinual(DeYOMLMPAdaGateContinual):

    E_MARGIN = 1.8       # save/ACDCDataset/sar_continual_weather/cmd.sh
    SAM_RHO = 0.05

    @staticmethod
    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        # sar_continual's convention: (T,B,C,w,h) -> sum over class, mean over T
        ent = -(x.softmax(-3) * x.log_softmax(-3)).sum(-3)
        return ent.mean(dim=0)

    @staticmethod
    def _is_excluded(nm, top_block_exclude, num_blocks=24):
        if top_block_exclude <= 0:
            return False
        return DeYOMLMPAdaGateContinual._is_excluded(nm, top_block_exclude, num_blocks)

    def _sar_setup(self):
        """Lazily swap in SAR's text table and SAM optimizer.

        The parent built a plain Adam over the same parameter list; SAM wraps Adam
        as its base optimizer over that identical list, so nothing else changes.
        """
        if getattr(self, "_sar_text_x", None) is None:
            with torch.no_grad():
                self._sar_text_x = self.extract_text_embeddings(
                    self.classes, self.prompt_templates, average=False).squeeze()
            params = [p for _n, p in self.named_ln_params]
            lr = self.optimizer.param_groups[0]['lr']
            self.optimizer = SAM(params, optim.Adam, rho=self.SAM_RHO, lr=lr)
            print(f"[SAR-AdaGate] base objective = SAR (SAM rho={self.SAM_RHO}, "
                  f"e_margin={self.E_MARGIN}); SAR's own hard recovery is DISABLED, "
                  f"AdaGate's graded restore replaces it")
        return self._sar_text_x

    @torch.no_grad()
    def evaluate(self, x):
        t1 = time.time()
        logits, _, _ = self.model(x, self._sar_setup(), True, interpolate=True)
        logits = logits[0]
        if self.runtime:
            self.eval_times.append(time.time() - t1)
        return logits

    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []
        text = self._sar_setup()
        for _ in range(self.steps):
            logits, _, _ = self.model(x, text, True, interpolate=False)
            ent_map = self.softmax_entropy(logits)               # (B, w, h)

            with torch.no_grad():
                self.marginal_buf.append(
                    logits.mean(dim=0).softmax(dim=1).mean(dim=[0, 2, 3])
                    .detach().float().cpu())

            # SAR filter (A): skip the whole sample if it is unreliable
            if ent_map.mean().item() > self.E_MARGIN:
                self.optimizer.zero_grad()
                self._tick_gate()
                continue
            pixel_mask = ent_map < self.E_MARGIN
            if pixel_mask.sum().item() == 0:
                self.optimizer.zero_grad()
                self._tick_gate()
                continue

            loss = ent_map[pixel_mask].mean()
            loss_report.append(loss.item())

            # ---- SAM step 1: ascend to the perturbed weights ----
            loss.backward()
            self.optimizer.first_step(zero_grad=True)

            # ---- SAM step 2: gradient at the perturbed point ----
            logits2, _, _ = self.model(x, text, True, interpolate=False)
            ent_map2 = self.softmax_entropy(logits2)
            pixel_mask2 = ent_map2 < self.E_MARGIN
            if pixel_mask2.sum().item() == 0:
                self.optimizer.second_step(zero_grad=True)
                self._tick_gate()
                continue
            loss2 = ent_map2[pixel_mask2].mean()
            loss2.backward()

            # the gate's trigger reads the second-step gradient, i.e. the one that
            # actually updates the weights
            with torch.no_grad():
                gn = 0.0
                for _n, p in self.named_ln_params:
                    if p.grad is not None:
                        gn += float(p.grad.detach().float().pow(2).sum())
                self.grad_buf.append(gn ** 0.5)

            self.optimizer.second_step(zero_grad=True)
            if self.current_rst > 0.0 and self.current_lag > 0:
                self._stochastic_restore(self.current_rst, self.current_lag)

            self._tick_gate()

        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

    def _tick_gate(self):
        """Advance the gate clock. Factored out because SAR's filters have three
        `continue` paths that must still count as elapsed batches, otherwise the
        monitor window would stretch whenever samples are rejected."""
        self.batch_count += 1
        self.total_batches += 1
        if self.batch_count >= self.monitor_interval:
            self._update_gate()
