"""
MGPDeYOMLMPContinual - MGP (anonymous submission, repo tta-373C,
methods/MGP/proposal.py: RobustMGP / RobustSubspaceTracker) ported onto our
DeYO+MLMP base so it is directly comparable to GDG-PA.

MGP is an anti-saturation GRADIENT-PROJECTION method for long-horizon TTA.
After loss.backward(), for every tracked LayerNorm parameter it removes the
component of the gradient that lies in a maintained subspace B:
    g <- g - B (B^T g)
so the update can no longer keep pushing along the dominant historical
gradient directions. B is re-distilled every DISTILL_FREQ batches by SVD of
a buffer of past (already-projected) gradients, keeping the eigen-directions
above a Marchenko-Pastur noise threshold (rank <= MAX_RANK), then merged with
the previous basis by "inertial fusion" (only directions whose residual norm
exceeds RESIDUAL_NORM_THRESHOLD are appended).

Upstream defaults (conf.py): DISTILL_FREQ=100, BUFFER_SIZE=32, MAX_RANK=32,
RESIDUAL_NORM_THRESHOLD=0.75, COLLECT_FREQ=40.

PORT NOTES (ours, not the paper):
  - upstream tracks every BN/LN/GN parameter of a classification backbone;
    here we track exactly the visual-encoder LN params we train
    (top_block_exclude=6 -> 74 params on NA-CLIP ViT-L/14).
  - upstream LOSS_TYPE defaults to 'eta'; it also ships a 'deyo' branch. We
    keep OUR DeYO+MLMP loss (7 prompts x 18-layer mean fusion) unchanged so
    the only difference vs the DeYO+MLMP baseline is the gradient projection.
  - upstream also keeps an EMA "trusted gradient direction"; in the released
    code the per-sample alignments it feeds are never consumed, so it is dead
    code. We keep the EMA (cheap) but likewise do not gate on it.
  - no reset() is called in the stream (continual convention).
"""
import time
import copy
import math
import re

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from einops import rearrange

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'



class _MGPSubspaceTracker:
    """Faithful port of RobustSubspaceTracker (methods/MGP/proposal.py)."""

    def __init__(self, named_params, buffer_size=32, max_rank=32,
                 residual_thr=0.75, min_rank=1):
        self.named = named_params                     # list[(name, Parameter)]
        self.buffer_size = int(buffer_size)
        self.max_rank = int(max_rank)
        self.residual_thr = float(residual_thr)
        self.min_rank = int(min_rank)
        self.buffers = {n: [] for n, _ in named_params}
        self.bases = {n: None for n, _ in named_params}

    @torch.no_grad()
    def add_trusted_gradient(self):
        for n, p in self.named:
            if p.grad is None:
                continue
            self.buffers[n].append(p.grad.detach().view(-1).float().cpu())
            while len(self.buffers[n]) > self.buffer_size:
                self.buffers[n].pop(0)

    @staticmethod
    def _mp_threshold(eigenvalues, n, d):
        gamma = d / n
        mp_upper_edge = (1.0 + math.sqrt(gamma)) ** 2
        n_noise = max(len(eigenvalues) // 2, 1)
        sigma_sq = eigenvalues[-n_noise:].median()
        return sigma_sq * mp_upper_edge

    def _inertial_fusion(self, B_old, B_new):
        B_new_residual = B_new - B_old @ (B_old.T @ B_new)
        residual_norms = torch.norm(B_new_residual, dim=0)
        significant = residual_norms > self.residual_thr
        if significant.sum() > 0:
            B_novel = B_new_residual[:, significant]
            B_novel = B_novel / (torch.norm(B_novel, dim=0, keepdim=True) + 1e-8)
            B_combined = torch.cat([B_old, B_novel], dim=1)
        else:
            B_combined = B_old
        if B_combined.shape[1] > self.max_rank:
            B_combined = B_combined[:, :self.max_rank]
        return B_combined

    @torch.no_grad()
    def distill(self, device):
        for n, _ in self.named:
            buf = self.buffers.get(n, [])
            if len(buf) < max(2, self.min_rank + 1):
                continue
            G = torch.stack(buf, dim=0).to(device)
            n_, d = G.shape
            G = G - G.mean(dim=0, keepdim=True)
            _, S, Vh = torch.linalg.svd(G, full_matrices=False)
            eig = S ** 2
            thr = self._mp_threshold(eig, n_, d)
            r_new = max(self.min_rank, min(int((eig > thr).sum().item()), self.max_rank))
            B_new = Vh[:r_new, :].T
            B_old = self.bases.get(n, None)
            B_comb = B_new if B_old is None else self._inertial_fusion(B_old.to(device), B_new)
            Q, _ = torch.linalg.qr(B_comb)
            self.bases[n] = Q.cpu()

    @torch.no_grad()
    def project_gradient(self):
        """g <- g - B (B^T g): strip the historical-subspace component."""
        for n, p in self.named:
            if p.grad is None:
                continue
            B = self.bases.get(n, None)
            if B is None:
                continue
            B = B.to(p.grad.device, dtype=torch.float32)
            g = p.grad.detach().view(-1).float()
            g_perp = g - B @ (B.T @ g)
            p.grad.copy_(g_perp.view_as(p.grad).to(p.grad.dtype))


class MGPDeYOMLMPContinual:
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
                 mgp_distill_freq=100, mgp_buffer_size=32,
                 mgp_max_rank=32, mgp_residual_thr=0.75,
                 mgp_collect_freq=40,
                 runtime_calculation=False, device='cpu'):
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps
        self.vision_outputs = vision_outputs
        self.runtime = runtime_calculation
        self.device = device

        if classes is None:
            raise ValueError("classes is required in __init__")
        self.classes = classes

        self.deyo_margin = deyo_margin_factor * math.log(len(classes))
        self.margin_e0 = deyo_margin_e0_factor * math.log(len(classes))
        self.plpd_threshold = plpd_threshold
        self.aug_type = aug_type
        self.patch_len = patch_len
        self.reweight_ent = bool(reweight_ent)
        self.reweight_plpd = bool(reweight_plpd)
        self.top_block_exclude = top_block_exclude
        self.total_batches = 0

        self.model, self.tokenize = load_ovss(ovss_type, ovss_backbone, device=device)

        if prompt_dir:
            self.prompt_templates = load_prompts_from_yaml(prompt_dir)
            print(f"Number of prompt templates: {len(self.prompt_templates)}")
        else:
            self.prompt_templates = [REFERENCE_PROMPT]

        self.model.transformer.requires_grad_(False)
        self.model.ln_final.requires_grad_(False)
        self.model.token_embedding.requires_grad_(False)
        self.model.visual = self.set_ln_grads(
            self.model.visual, top_block_exclude=self.top_block_exclude)
        params, _names = self.collect_ln_params(
            self.model.visual, top_block_exclude=self.top_block_exclude)
        self.named_ln_params = list(zip(_names, params))
        # --- MGP gradient-projection state ---
        self.mgp_distill_freq = int(mgp_distill_freq)
        self.mgp_collect_freq = int(mgp_collect_freq)
        self.mgp = _MGPSubspaceTracker(
            self.named_ln_params, buffer_size=mgp_buffer_size,
            max_rank=mgp_max_rank, residual_thr=mgp_residual_thr)
        self.mgp_dir_ema = None

        print_clip_parameters(self.model)
        print(f"+++ DeYO-MLMP (multi-prompt T={len(self.prompt_templates)}): "
              f"deyo_margin={self.deyo_margin:.3f}, margin_e0={self.margin_e0:.3f}, "
              f"plpd_thr={self.plpd_threshold}, aug={self.aug_type}/{self.patch_len}, "
              f"UAML layers={len(self.vision_outputs)}, "
              f"top_block_exclude={self.top_block_exclude} -> LN params: {len(params)}")

        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer)

        # text_x shape (T+1, C, D): [:-1] = T per-template, [-1] = averaged.
        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(
                self.classes, self.prompt_templates, average=True).squeeze()

        if self.runtime:
            self.adapt_times = []
            self.eval_times = []

        # --- Diagnostics (opt-in; set collect_diag=True from main_continual) ---
        # When on, perform_adaptation() stashes per-batch signals derived from
        # tensors it ALREADY computes (zero extra forward, zero RNG), and
        # diagnose() does ONE no_grad RNG-free forward for per-layer/feature
        # signals. The adaptation math is untouched either way.
        self.collect_diag = False
        self.diag = {}

    def adapt(self, x):
        return self.perform_adaptation(x)

    def continual_adapt(self, x):
        return self.perform_adaptation(x)

    @torch.no_grad()
    def evaluate(self, x):
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

    @torch.no_grad()
    def _mgp_update_direction(self):
        gs = [p.grad.detach().view(-1).float()
              for _n, p in self.named_ln_params if p.grad is not None]
        if not gs:
            return
        g = torch.cat(gs)
        g = g / (g.norm() + 1e-8)
        if self.mgp_dir_ema is None:
            self.mgp_dir_ema = g
        else:
            self.mgp_dir_ema = 0.9 * self.mgp_dir_ema + 0.1 * g
            self.mgp_dir_ema /= (self.mgp_dir_ema.norm() + 1e-8)

    def reset(self):
        self.load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state)

    def _adapt_forward(self, x):
        # text_x[:-1] -> T=7 per-template; mean fusion over 18 layers.
        logits, _, _ = self.model(
            x, self.text_x[:-1], True,
            interpolate=False,
            vision_outputs=self.vision_outputs,
            vision_out_type="mean")
        return logits  # (T, B, C, h, w)

    def _destroy_object(self, x):
        if self.aug_type == 'pixel':
            x2 = rearrange(x, 'b c h w -> b c (h w)')
            perm = torch.randperm(x2.shape[-1], device=x.device)
            return rearrange(x2[:, :, perm], 'b c (h w) -> b c h w',
                             h=x.shape[-2], w=x.shape[-1])
        if self.aug_type == 'occ':
            B, C, H, W = x.shape
            occ = max(16, H // 8); r = (H - occ) // 2; c = (W - occ) // 2
            x2 = x.clone()
            mean = x2.view(B, C, -1).mean(dim=2).unsqueeze(-1).unsqueeze(-1)
            x2[:, :, r:r+occ, c:c+occ] = mean.expand(-1, -1, occ, occ)
            return x2
        H, W = x.shape[-2], x.shape[-1]; ps = self.patch_len
        H2, W2 = (H // ps) * ps, (W // ps) * ps
        if (H2, W2) != (H, W):
            x = T.functional.resize(x, [H2, W2], antialias=True)
        x2 = rearrange(x, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=ps, ps2=ps)
        perm = torch.argsort(torch.rand(x2.shape[0], x2.shape[1], device=x.device), dim=-1)
        x2 = x2[torch.arange(x2.shape[0], device=x.device).unsqueeze(-1), perm]
        x2 = rearrange(x2, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=ps, ps2=ps)
        if (H2, W2) != (H, W):
            x2 = T.functional.resize(x2, [H, W], antialias=True)
        return x2

    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []
        for _ in range(self.steps):
            self.total_batches += 1

            logits = self._adapt_forward(x)            # (T, B, C, h, w)
            ent = self.softmax_entropy(logits)         # (T, B, h, w)
            mask_ent = ent < self.deyo_margin
            if mask_ent.sum() == 0:
                if self.collect_diag:
                    self.diag.update(prompt_disagree=self._prompt_disagree(logits),
                                     plpd_mean=float('nan'), grad_norm=0.0,
                                     filter_pass_rate=0.0)
                self.optimizer.zero_grad(); continue

            with torch.no_grad():
                x_prime = self._destroy_object(x)
                logits_prime = self._adapt_forward(x_prime)

            prob = logits.softmax(dim=-3)
            prob_prime = logits_prime.softmax(dim=-3)
            cls1 = prob.argmax(dim=-3, keepdim=True)
            p_orig = prob.gather(dim=-3, index=cls1).squeeze(-3)
            p_prime = prob_prime.gather(dim=-3, index=cls1).squeeze(-3)
            plpd = (p_orig - p_prime).detach()

            mask_plpd = plpd > self.plpd_threshold
            final_mask = mask_ent & mask_plpd
            if final_mask.sum() == 0:
                if self.collect_diag:
                    self.diag.update(prompt_disagree=self._prompt_disagree(logits),
                                     plpd_mean=float(plpd.mean()), grad_norm=0.0,
                                     filter_pass_rate=0.0)
                self.optimizer.zero_grad(); continue

            ent_kept = ent[final_mask]
            plpd_kept = plpd[final_mask]
            if self.reweight_ent or self.reweight_plpd:
                ent_det = ent_kept.detach(); coeff = 0.0
                if self.reweight_ent:
                    coeff = coeff + 1.0 / torch.exp(ent_det - self.margin_e0)
                if self.reweight_plpd:
                    coeff = coeff + 1.0 / torch.exp(-1.0 * plpd_kept)
                loss = (ent_kept * coeff).mean()
            else:
                loss = ent_kept.mean()

            loss.backward()
            # --- MGP: strip the historical-subspace component of the grad ---
            self.mgp.project_gradient()
            if self.total_batches % self.mgp_collect_freq == 0:
                self.mgp.add_trusted_gradient()
                self._mgp_update_direction()
            if self.collect_diag:
                gn = 0.0
                for p in self.optimizer.param_groups[0]['params']:
                    if p.grad is not None:
                        gn += float(p.grad.detach().float().pow(2).sum())
                self.diag.update(
                    prompt_disagree=self._prompt_disagree(logits),
                    plpd_mean=float(plpd.mean()),
                    grad_norm=gn ** 0.5,
                    filter_pass_rate=float(final_mask.float().mean()),
                )
            self.optimizer.step()
            self.optimizer.zero_grad()
            if (self.total_batches % self.mgp_distill_freq == 0
                    and self.total_batches > 0):
                self.mgp.distill(self.device)
            loss_report.append(loss.item())

        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

    # ===================== Diagnostics (side-effect-free) =====================

    @staticmethod
    @torch.no_grad()
    def _disagree(preds, n_views, num_classes):
        """preds: (n_views, ...) integer argmax maps. Returns mean prediction
        disagreement = 1 - (per-pixel majority-vote fraction), in [0, 1)."""
        oh = torch.nn.functional.one_hot(preds, num_classes).float()  # (n_views, ..., C)
        votes = oh.sum(dim=0)                                          # (..., C)
        agree = votes.max(dim=-1).values / float(n_views)             # (...,)
        return float((1.0 - agree).mean())

    @torch.no_grad()
    def _prompt_disagree(self, logits):
        """logits: (T, B, C, h, w) per-prompt. Disagreement across the T prompts."""
        T, B, C = logits.shape[0], logits.shape[1], logits.shape[2]
        preds = logits.argmax(dim=2)                                  # (T, B, h, w)
        return self._disagree(preds, T, C)

    @torch.no_grad()
    def diagnose(self, x):
        """ONE no_grad, RNG-free forward (out_type='all') for per-layer/feature
        signals. Stores into self.diag. Safe: the model forward has no dropout /
        no RNG (dropout_p=0), so inserting this does not perturb the adaptation
        trajectory (verified by with/without --log_signals bit-identical mIoU)."""
        if not self.collect_diag:
            return
        feats = self.model.encode_image(
            x, self.vision_outputs, out_type="all")      # (L, B, tokens, D)
        feats = feats[:, :, 1:, :]                        # drop CLS token
        L, B, N, D = feats.shape
        feat_norm = float(feats.norm(dim=-1).mean())
        fn = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        text = self.text_x[-1]                            # (C, D) averaged prompt
        text = text / text.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        cos = fn @ text.t()                               # (L, B, N, C)
        feat_text_align = float(cos.max(dim=-1).values.mean())
        preds = cos.argmax(dim=-1)                        # (L, B, N)
        layer_disagree = self._disagree(preds, L, text.shape[0])
        self.diag.update(feat_norm=feat_norm,
                         feat_text_align=feat_text_align,
                         layer_disagree=layer_disagree)

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
    def _is_excluded(nm, top_block_exclude):
        if 'ln_post' in nm:
            return True
        m = re.search(r'resblocks\.(\d+)\.', nm)
        if m and int(m.group(1)) >= (24 - top_block_exclude):
            return True
        return False

    @classmethod
    def set_ln_grads(cls, model, top_block_exclude=6):
        model.requires_grad_(False)
        for nm, m in model.named_modules():
            if isinstance(m, nn.LayerNorm) and not cls._is_excluded(nm, top_block_exclude):
                m.requires_grad_(True)
        return model

    @classmethod
    def collect_ln_params(cls, model, top_block_exclude=6):
        params, names = [], []
        for nm, m in model.named_modules():
            if isinstance(m, nn.LayerNorm) and not cls._is_excluded(nm, top_block_exclude):
                for np_, p in m.named_parameters():
                    if np_ in ['weight', 'bias']:
                        params.append(p); names.append(f"visual.{nm}.{np_}")
        return params, names

    @staticmethod
    def copy_model_and_optimizer(model, optimizer):
        return copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())

    @staticmethod
    def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)

    @staticmethod
    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        return -(x.softmax(-3) * x.log_softmax(-3)).sum(-3)
