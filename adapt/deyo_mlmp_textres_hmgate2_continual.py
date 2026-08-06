"""
DeYOMLMPTextResHMGate2Continual — GDG-PA (deyo_mlmp_hmgate2_continual) with a
LEARNABLE TEXT-EMBEDDING RESIDUAL plus an ORTHOGONALITY regularizer.

Motivation. GDG-PA's text side is completely frozen: `text_x` is encoded once in
__init__ and never moves, so the classifier lives forever in the span of the N
hand-written templates. Two prior levers inside that span both came back near-null
(prompt CONTENT sweep S0-S8: stock wins; prompt-axis entropy WEIGHTING: +0.34 at
best). This method is the first that lets the class vectors LEAVE that span.

  * RESIDUAL (text_res_lr): a zero-initialized parameter t_hat of shape (C, D) is
    added to every template's class embedding and renormalized,
        t_c <- (t_base_c + t_hat_c) / ||t_base_c + t_hat_c||.
    Gradients stop at t_hat -- the text ENCODER is never re-run, so the extra cost
    per step is a (T+1, C, D) elementwise op (negligible). This is the DPE / TPS
    "residual tuning" parameterization, not TPT's soft-token one (which would need
    C*T text-encoder forwards with graph retained every single step).

  * ORTHOGONALITY (lambda_orth): C-TPT / O-TPT observe that test-time prompt tuning
    drives class text features together, and that penalizing that collapse helps.
    Our own collapse pathology is the same disease on the probability side (marginal
    entropy H_margin falls as predictions pile onto 1-2 classes). The penalty is
        L_orth = || T T^T - I ||_F^2 / (C*(C-1)),
    i.e. mean squared off-diagonal cosine between (unit-norm) class vectors. It is
    a PREVENTIVE geometric constraint, complementary to the gate's REACTIVE H_margin
    detection. Note L_orth is constant (zero-gradient) unless the residual exists,
    so lambda_orth only does anything when text_res_lr > 0.

Interaction with the gate, by design:
  - t_hat is EXCLUDED from the grad_norm sum, so the gate's slope signal keeps
    exactly the same meaning/scale as in GDG-PA (comparisons stay apples-to-apples).
  - t_hat IS included in the window snapshots, so shallow/deep stochastic restore
    pulls the text residual back too. It never drifts unsupervised.
  - L_orth has no path to the LN parameters, so it cannot perturb the gate signal.

Degradation: text_res_lr == 0 -> no parameter is created, no param group is added,
_text_eff() returns the frozen base embedding -> BIT-IDENTICAL to GDG-PA. That is
the control arm of the sweep.

CTTA hard-rule compliant: no per-sample reset; partial stochastic restore only.
"""
import time

import torch

from .deyo_mlmp_hmgate2_continual import DeYOMLMPHMGate2Continual


class DeYOMLMPTextResHMGate2Continual(DeYOMLMPHMGate2Continual):
    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
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
                 maxlag_shallow=6,
                 monitor_interval=50,
                 # --- learnable text residual + orthogonality (NEW) ---
                 text_res_lr=0.0,
                 lambda_orth=0.0,
                 text_res_max_norm=0.5,
                 save_dir=None,
                 runtime_calculation=False, device='cpu'):
        # must exist before super().__init__ -- it snapshots weights during setup
        self.text_res = None
        self.text_res_lr = float(text_res_lr)
        self.lambda_orth = float(lambda_orth)
        self.text_res_max_norm = float(text_res_max_norm)

        super().__init__(
            ovss_type, ovss_backbone, lr, classes, steps=steps,
            vision_outputs=vision_outputs, prompt_dir=prompt_dir,
            deyo_margin_factor=deyo_margin_factor,
            deyo_margin_e0_factor=deyo_margin_e0_factor,
            plpd_threshold=plpd_threshold, aug_type=aug_type, patch_len=patch_len,
            reweight_ent=reweight_ent, reweight_plpd=reweight_plpd,
            top_block_exclude=top_block_exclude, slope_window=slope_window,
            slope_deadzone=slope_deadzone, lag_gain=lag_gain, base_rst=base_rst,
            max_windows=max_windows, h_drop_ratio=h_drop_ratio,
            maxlag_shallow=maxlag_shallow, monitor_interval=monitor_interval,
            save_dir=save_dir, runtime_calculation=runtime_calculation, device=device)

        # self.text_x is (T+1, C, D): T templates then their renormalized average.
        self.text_base = self.text_x.detach()
        self.num_classes = self.text_base.shape[1]
        self._eye = torch.eye(self.num_classes, device=self.device)

        if self.text_res_lr > 0.0:
            # MUST be fp32: CLIP's text embeddings are fp16, and Adam on an fp16
            # parameter NaNs out immediately (eps=1e-8 and v=grad^2 both underflow
            # the fp16 subnormal range). The residual is cast to the text dtype only
            # when it is composed in _text_eff().
            self.text_res = torch.zeros(
                self.text_base.shape[1], self.text_base.shape[2],
                device=self.device, dtype=torch.float32, requires_grad=True)
            self.optimizer.add_param_group(
                {'params': [self.text_res], 'lr': self.text_res_lr})
            # re-take the baselines so they include the (zero) residual and the
            # 2-group optimizer, otherwise reset() would mismatch.
            self.optimizer_state = self.copy_model_and_optimizer(
                self.model, self.optimizer)[1]
            self._win_buf.clear()
            self._win_buf.append(self._snapshot_ln_weights())
            self.best_snapshot = self._win_buf[-1]

        # diagnostics: does the TEXT side actually degrade over a 150-round stream?
        # mean_offdiag_cos rising = class vectors collapsing together (the text-side
        # analogue of the H_margin drop); res_norm = how far the residual has drifted.
        self.text_log_path = None
        if self.gate_log_path is not None and self.text_res is not None:
            self.text_log_path = self.gate_log_path.replace("gate_log.csv", "text_log.csv")
            with open(self.text_log_path, 'w') as _f:
                _f.write("total_batches,res_norm_mean,res_norm_max,mean_offdiag_cos,"
                         "max_offdiag_cos,orth_loss\n")

        print(f"+++ TextRes: text_res_lr={self.text_res_lr} "
              f"lambda_orth={self.lambda_orth} max_norm={self.text_res_max_norm} "
              f"-> {'ACTIVE (C=%d)' % self.num_classes if self.text_res is not None else 'DISABLED (== GDG-PA)'}")

    # ---------- effective text embeddings ----------
    def _text_eff_f32(self):
        """Unit-norm class embeddings in fp32 (the numerically safe representation)."""
        t = self.text_base.float() + self.text_res.unsqueeze(0)   # (T+1, C, D)
        return t / t.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    def _text_eff(self):
        """What the backbone consumes -- cast back to CLIP's text dtype (fp16)."""
        if self.text_res is None:
            return self.text_base
        return self._text_eff_f32().to(self.text_base.dtype)

    def _adapt_forward(self, x):
        logits, _, _ = self.model(
            x, self._text_eff()[:-1], True,
            interpolate=False,
            vision_outputs=self.vision_outputs,
            vision_out_type="mean")
        return logits  # (T, B, C, h, w)

    @torch.no_grad()
    def evaluate(self, x):
        t1 = time.time()
        logits, _, _ = self.model(
            x, self._text_eff()[-1], True,
            vision_outputs=self.vision_outputs,
            interpolate=True,
            vision_out_type="adaptive_weighted_mean",
            save_weights=True)
        out = logits[0]
        if self.runtime:
            self.eval_times.append(time.time() - t1)
        return out

    def _orth_loss(self):
        """Mean squared off-diagonal cosine between the (unit-norm) class vectors.

        Uses the averaged-template row, i.e. the vector that eval actually classifies
        with. Has no gradient path to the LN parameters, so the gate signal is
        unaffected; and no gradient at all when the residual is disabled.
        """
        T = self._text_eff_f32()[-1]                  # (C, D), unit norm, fp32
        G = T @ T.t()                                 # (C, C)
        off = G - self._eye
        return off.pow(2).sum() / (self.num_classes * (self.num_classes - 1))

    @torch.no_grad()
    def _clip_text_res(self):
        if self.text_res is None or self.text_res_max_norm <= 0.0:
            return
        n = self.text_res.norm(dim=-1, keepdim=True)
        scale = (self.text_res_max_norm / n.clamp(min=1e-8)).clamp(max=1.0)
        self.text_res.mul_(scale)

    # ---------- snapshot / restore also cover the residual ----------
    @torch.no_grad()
    def _snapshot_ln_weights(self):
        snap = super()._snapshot_ln_weights()
        if self.text_res is not None:
            # fp32, unlike the LN entries: residual values are ~1e-5 and would lose
            # most of their precision in fp16.
            snap['__text_res__'] = self.text_res.detach().cpu().clone()
        return snap

    @torch.no_grad()
    def _stochastic_restore(self, rst, lag):
        super()._stochastic_restore(rst, lag)
        if self.text_res is None:
            return
        if self.current_deep:
            anchor = self.best_snapshot
        else:
            idx = -1 - lag
            anchor = self._win_buf[0] if -idx > len(self._win_buf) else self._win_buf[idx]
        src = anchor.get('__text_res__')
        if src is None:
            return
        mask = (torch.rand(self.text_res.shape, device=self.text_res.device)
                < rst).to(self.text_res.dtype)
        src = src.to(self.text_res.device, dtype=self.text_res.dtype)
        self.text_res.data.mul_(1.0 - mask).add_(src * mask)

    @torch.no_grad()
    def _update_gate(self):
        super()._update_gate()
        if self.text_log_path is None:
            return
        T = self._text_eff_f32()[-1]
        G = (T @ T.t()) - self._eye
        n = self.text_res.norm(dim=-1)
        C = self.num_classes
        with open(self.text_log_path, 'a') as _f:
            _f.write(f"{self.total_batches},{float(n.mean()):.6e},{float(n.max()):.6e},"
                     f"{float(G.abs().sum() / (C * (C - 1))):.6f},"
                     f"{float(G.abs().max()):.6f},"
                     f"{float(G.pow(2).sum() / (C * (C - 1))):.6f}\n")

    # ---------- adapt: DeYO loss + orthogonality ----------
    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []
        for _ in range(self.steps):
            logits = self._adapt_forward(x)            # (T, B, C, h, w)
            ent = self.softmax_entropy(logits)         # (T, B, h, w)

            with torch.no_grad():
                probs_ens = logits.softmax(dim=-3).mean(dim=0)   # (B,C,h,w)
                self.marginal_buf.append(
                    probs_ens.mean(dim=[0, 2, 3]).detach().float().cpu())

            mask_ent = ent < self.deyo_margin
            if mask_ent.sum() > 0:
                with torch.no_grad():
                    x_prime = self._destroy_object(x)
                    logits_prime = self._adapt_forward(x_prime)
                prob = logits.softmax(dim=-3)
                prob_prime = logits_prime.softmax(dim=-3)
                cls1 = prob.argmax(dim=-3, keepdim=True)
                p_orig = prob.gather(dim=-3, index=cls1).squeeze(-3)
                p_prime = prob_prime.gather(dim=-3, index=cls1).squeeze(-3)
                plpd = (p_orig - p_prime).detach()
                final_mask = mask_ent & (plpd > self.plpd_threshold)
                if final_mask.sum() > 0:
                    ent_kept = ent[final_mask]; plpd_kept = plpd[final_mask]
                    if self.reweight_ent or self.reweight_plpd:
                        ent_det = ent_kept.detach(); coeff = 0.0
                        if self.reweight_ent:
                            coeff = coeff + 1.0 / torch.exp(ent_det - self.margin_e0)
                        if self.reweight_plpd:
                            coeff = coeff + 1.0 / torch.exp(-1.0 * plpd_kept)
                        loss = (ent_kept * coeff).mean()
                    else:
                        loss = ent_kept.mean()
                    if self.text_res is not None and self.lambda_orth > 0.0:
                        loss = loss + self.lambda_orth * self._orth_loss()
                    loss_report.append(loss.item())
                    loss.backward()
                    with torch.no_grad():
                        # LN params only -- keeps the gate signal identical to GDG-PA
                        gn = 0.0
                        for _n, p in self.named_ln_params:
                            if p.grad is not None:
                                gn += float(p.grad.detach().float().pow(2).sum())
                        self.grad_buf.append(gn ** 0.5)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self._clip_text_res()
                    if self.current_rst > 0.0 and self.current_lag > 0:
                        self._stochastic_restore(self.current_rst, self.current_lag)

            self.batch_count += 1
            self.total_batches += 1
            if self.batch_count >= self.monitor_interval:
                self._update_gate()

        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report
