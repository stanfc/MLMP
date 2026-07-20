"""
DeYOMLMPPromptWHMGate2Continual — GDG-PA (deyo_mlmp_hmgate2_continual) with
ENTROPY-WEIGHTED PROMPT AGGREGATION on the prompt (text-template) axis.

MLMP's UAML already weights the LAYER axis by entropy (alpha^l ~ exp(-beta*h^l)).
The prompt axis, by contrast, is aggregated UNIFORMLY: the adapt loss pools every
template's kept pixels into one mean, and eval uses the pre-averaged text embedding.
This method extends the same entropy-weighting idea to the prompt axis, at two
INDEPENDENT injection points so their effects can be attributed separately:

  * ADAPT side (prompt_weight_beta):  per template t, reliability weight
        w^t = softmax(-beta_adapt * h^t),   h^t = mean softmax-entropy of template t
    (detached). The DeYO loss becomes a w^t-weighted mean over templates instead of
    a uniform pool. beta_adapt = 0 -> uniform weights -> BIT-IDENTICAL to GDG-PA.

  * EVAL side (eval_prompt_mode='ent_weight', eval_prompt_beta): the classifier
    becomes an entropy-weighted ensemble over the T templates' softmax outputs,
        p = sum_t softmax(-beta_eval * h^t) * softmax(logits_t),
    instead of the single pre-averaged-embedding classifier. eval_prompt_mode=
    'avg_embed' (default) keeps GDG-PA's original eval exactly.

Why this is NOT the confirmation-bias trap that collapsed CMA: the weight selects
across TEXT TEMPLATES (which phrasing is reliable), never across CLASSES. The class
pseudo-label loop is untouched; low-entropy templates are simply trusted more. GDG-PA's
gate provides the usual stability backstop.

Everything else (DeYO+MLMP loss, H_margin regime gate, permanent best-state anchor,
stochastic restore) is inherited unchanged from DeYOMLMPHMGate2Continual.
"""
import time
import torch

from .deyo_mlmp_hmgate2_continual import DeYOMLMPHMGate2Continual


class DeYOMLMPPromptWHMGate2Continual(DeYOMLMPHMGate2Continual):
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
                 # --- prompt-axis entropy weighting (NEW) ---
                 prompt_weight_beta=0.0,
                 eval_prompt_mode='avg_embed',
                 eval_prompt_beta=1.0,
                 save_dir=None,
                 runtime_calculation=False, device='cpu'):
        self.prompt_weight_beta = float(prompt_weight_beta)
        self.eval_prompt_mode = str(eval_prompt_mode)
        self.eval_prompt_beta = float(eval_prompt_beta)
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
        print(f"+++ PromptW: beta_adapt={self.prompt_weight_beta} "
              f"eval_mode={self.eval_prompt_mode} beta_eval={self.eval_prompt_beta}")

    # ---------- eval: optional entropy-weighted prompt ensemble ----------
    @torch.no_grad()
    def evaluate(self, x):
        t1 = time.time()
        if self.eval_prompt_mode != 'ent_weight':
            logits, _, _ = self.model(
                x, self.text_x[-1], True,
                vision_outputs=self.vision_outputs, interpolate=True,
                vision_out_type="adaptive_weighted_mean", save_weights=True)
            out = logits[0]
        else:
            logits, _, _ = self.model(          # (T, B, C, h, w)
                x, self.text_x[:-1], True,
                vision_outputs=self.vision_outputs, interpolate=True,
                vision_out_type="adaptive_weighted_mean", save_weights=False)
            ent = self.softmax_entropy(logits)                      # (T,B,h,w)
            h_t = ent.mean(dim=[1, 2, 3])                           # (T,)
            w_t = torch.softmax(-self.eval_prompt_beta * h_t, dim=0)
            prob = logits.softmax(dim=-3)                           # (T,B,C,h,w)
            comb = (prob * w_t.view(-1, 1, 1, 1, 1)).sum(dim=0)     # (B,C,h,w)
            # return log-prob so the downstream softmax over C recovers the weighted prob
            out = comb.clamp(min=1e-8).log()
        if self.runtime:
            self.eval_times.append(time.time() - t1)
        return out

    # ---------- adapt: w^t-weighted DeYO loss over the prompt axis ----------
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

                    # per-template reliability weight (prompt-axis entropy weighting).
                    # beta_adapt=0 -> uniform -> weighted mean == plain mean (identical).
                    if self.prompt_weight_beta != 0.0:
                        with torch.no_grad():
                            h_t = ent.detach().mean(dim=[1, 2, 3])           # (T,)
                            w_t = torch.softmax(-self.prompt_weight_beta * h_t, dim=0)
                        w_kept = w_t.view(-1, 1, 1, 1).expand_as(ent)[final_mask]
                    else:
                        w_kept = None

                    if self.reweight_ent or self.reweight_plpd:
                        ent_det = ent_kept.detach(); coeff = 0.0
                        if self.reweight_ent:
                            coeff = coeff + 1.0 / torch.exp(ent_det - self.margin_e0)
                        if self.reweight_plpd:
                            coeff = coeff + 1.0 / torch.exp(-1.0 * plpd_kept)
                        terms = ent_kept * coeff
                    else:
                        terms = ent_kept

                    if w_kept is not None:
                        loss = (terms * w_kept).sum() / w_kept.sum().clamp(min=1e-8)
                    else:
                        loss = terms.mean()

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
