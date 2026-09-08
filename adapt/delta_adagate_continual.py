"""
DELTAAdaGateContinual -- the AdaGate restoration gate on DELTA's objective.

Plug-and-play arm #3 (after DeYO+MLMP and TENT). DELTA (ICLR 2023) is
entropy minimisation reweighted by an EMA of the pseudo-label class frequency
("DOT"): rare predicted classes get up-weighted, so the loss resists the
class-collapse that plain entropy minimisation drifts into.

    w_i = freq[c_hat_i]^(-alpha),  normalised to mean 1
    L   = mean_i w_i * H(p_i)

DELTA already contains a hand-designed anti-collapse device, and it still fails:
`save/ACDCDataset/delta_continual_alpha_1.0_mom_0.9` peaks 26.51@R12 and drops
below 15 mIoU at R95, ending at 12.24. That makes it a useful arm -- the gate is
not merely substituting for a missing diversity term.

Everything about the gate is inherited from `DeYOMLMPAdaGateContinual`. The base
objective and the trainable-parameter set follow `adapt/delta_continual.py`:
`_is_excluded` is overridden so `--top_block_exclude 0` trains all 100 visual
LayerNorm params, matching `delta_continual.set_ln_grads`, which makes the
published run a drop-in no-gate control.

DOT constants are class attributes rather than CLI flags (`get_method()` inspects
the constructor signature, so this subclass does not override `__init__`); the
values are those of the published run: `--dot_momentum 0.9 --dot_alpha 1.0`.
"""
import time

import torch

from .deyo_mlmp_adagate_continual import DeYOMLMPAdaGateContinual


class DELTAAdaGateContinual(DeYOMLMPAdaGateContinual):

    DOT_MOMENTUM = 0.9
    DOT_ALPHA = 1.0

    @staticmethod
    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        # delta_continual's convention: (T,B,C,w,h) -> sum over class, mean over T
        ent = -(x.softmax(-3) * x.log_softmax(-3)).sum(-3)
        return ent.mean(dim=0)

    @staticmethod
    def _is_excluded(nm, top_block_exclude, num_blocks=24):
        if top_block_exclude <= 0:
            return False
        return DeYOMLMPAdaGateContinual._is_excluded(nm, top_block_exclude, num_blocks)

    def _delta_text(self):
        """DELTA builds the table with average=False (all templates, no averaged
        embedding appended)."""
        if getattr(self, "_delta_text_x", None) is None:
            with torch.no_grad():
                self._delta_text_x = self.extract_text_embeddings(
                    self.classes, self.prompt_templates, average=False).squeeze()
            self._class_freq = torch.full(
                (len(self.classes),), 1.0 / len(self.classes),
                dtype=torch.float32, device=self.device)
            print(f"[DELTA-AdaGate] base objective = DELTA (DOT-reweighted entropy, "
                  f"momentum={self.DOT_MOMENTUM}, alpha={self.DOT_ALPHA}); gate unchanged")
        return self._delta_text_x

    @torch.no_grad()
    def evaluate(self, x):
        t1 = time.time()
        logits, _, _ = self.model(x, self._delta_text(), True, interpolate=True)
        logits = logits[0]
        if self.runtime:
            self.eval_times.append(time.time() - t1)
        return logits

    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []
        for _ in range(self.steps):
            logits, _, _ = self.model(x, self._delta_text(), True, interpolate=False)
            ent_map = self.softmax_entropy(logits)               # (B, w, h)

            with torch.no_grad():
                avg_logits = logits.mean(dim=0)                  # (B, C, w, h)
                pseudo = avg_logits.argmax(dim=1)                # (B, w, h)

                # H_margin regime signal, from the same prompt-averaged logits
                self.marginal_buf.append(
                    avg_logits.softmax(dim=1).mean(dim=[0, 2, 3]).detach().float().cpu())

                # ---- DOT: EMA of the pseudo-label class frequency ----
                batch_freq = torch.bincount(
                    pseudo.reshape(-1), minlength=len(self.classes)
                ).to(self._class_freq.dtype)
                batch_freq = batch_freq / batch_freq.sum().clamp(min=1.0)
                self._class_freq.mul_(self.DOT_MOMENTUM).add_(
                    batch_freq * (1.0 - self.DOT_MOMENTUM))

                pixel_w = self._class_freq.clamp(min=1e-4)[pseudo].pow(-self.DOT_ALPHA)
                pixel_w = pixel_w / pixel_w.mean().clamp(min=1e-8)

            loss = (pixel_w * ent_map).mean()
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
