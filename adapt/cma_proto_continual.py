import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class CMAProtoContinual:
    """
    CMA-Proto-Continual: Cross-Modal Alignment + Prototype Memory Bank (CTTA).

    Combined loss with three cosine-alignment terms sharing the same Top-K%
    confident pixel mask:

        L_total = lambda_cma * L_CMA(t_c)
                + lambda_src * L_src(p_src_c)     # frozen source prototype
                + lambda_tgt * L_tgt(p_tgt_c)     # EMA target prototype

    The source prototypes are computed once from a source proxy (ACDC fog) with
    pseudo-labels passed through a conservative multi-stage filter:
        1. confidence >= src_conf_threshold
        2. all 7 prompt templates agree on the predicted class
    Classes with zero confident pixels fall back to the text embedding.

    The source prototype is NEVER updated during the stream -- it is the
    external anchor that breaks the confirmation-bias loop observed in
    CMA-continual. Target prototype is EMA-updated under no_grad.

    See docs/cma_proto_continual_spec.md for the full design.
    """

    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 top_k_percent=0.2,
                 lambda_cma=1.0, lambda_src=1.0, lambda_tgt=0.5,
                 ema_alpha=0.999,
                 src_conf_threshold=0.5, src_max_samples=5000,
                 prompt_dir=None,
                 runtime_calculation=False, device='cpu'):
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps
        self.top_k_percent = float(top_k_percent)
        if not (0.0 < self.top_k_percent <= 1.0):
            raise ValueError(
                f"top_k_percent must be in (0, 1], got {self.top_k_percent}"
            )
        self.lambda_cma = float(lambda_cma)
        self.lambda_src = float(lambda_src)
        self.lambda_tgt = float(lambda_tgt)
        self.ema_alpha = float(ema_alpha)
        if not (0.0 <= self.ema_alpha < 1.0):
            raise ValueError(
                f"ema_alpha must be in [0, 1), got {self.ema_alpha}"
            )
        self.src_conf_threshold = float(src_conf_threshold)
        self.src_max_samples = int(src_max_samples)
        self.runtime = runtime_calculation
        self.device = device

        if classes is None:
            raise ValueError("classes is required in __init__")
        self.classes = classes
        self.num_classes = len(classes)

        # ---------- OVSS model ----------
        self.model, self.tokenize = load_ovss(ovss_type, ovss_backbone, device=device)

        # ---------- Prompts ----------
        if prompt_dir:
            self.prompt_templates = load_prompts_from_yaml(prompt_dir)
            print(f"Number of prompt templates: {len(self.prompt_templates)}")
        else:
            self.prompt_templates = [REFERENCE_PROMPT]

        # ---------- Freeze text encoder, enable LN grads ----------
        self.model.transformer.requires_grad_(False)
        self.model.ln_final.requires_grad_(False)
        self.model.token_embedding.requires_grad_(False)
        self.model.visual = self.set_ln_grads(self.model.visual)
        params, _ = self.collect_ln_params(self.model.visual)

        print_clip_parameters(self.model)

        # ---------- Optimizer ----------
        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        # ---------- Source state snapshot ----------
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer
        )

        # ---------- Text features ----------
        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(
                self.classes, self.prompt_templates, average=False
            ).squeeze()
            # Precompute prompt-averaged, re-normalized text embeddings for
            # per-class fallback. text_x may be (T, C, D) or (C, D) after the
            # squeeze if T==1 — normalize handling explicitly.
            tx = self.text_x if self.text_x.dim() == 3 else self.text_x.unsqueeze(0)
            avg_text = tx.mean(dim=0)                                       # (C, D)
            self.avg_text = avg_text / avg_text.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # ---------- Prototype buffers (populated by obtain_src_prototypes) ----------
        self.p_src = None   # (C, D), frozen after init
        self.p_tgt = None   # (C, D), EMA-updated during stream

        if self.runtime:
            self.adapt_times = []
            self.eval_times = []

    # ===========================================================
    # Public API
    # ===========================================================

    def adapt(self, x):
        return self.perform_adaptation(x)

    def continual_adapt(self, x):
        """Alias for adapt() -- matches main_continual.py protocol."""
        return self.perform_adaptation(x)

    @torch.no_grad()
    def evaluate(self, x):
        t1 = time.time()
        logits, _, _ = self.model(x, self.text_x, True, interpolate=True)
        logits = logits[0]
        if self.runtime:
            self.eval_times.append(time.time() - t1)
        return logits

    def reset(self):
        self.load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state
        )

    # ===========================================================
    # Source prototype initialization (called once before stream)
    # ===========================================================

    @torch.no_grad()
    def obtain_src_prototypes(self, data_loader):
        """
        Compute per-class source visual prototypes from a source proxy loader.

        Uses pseudo-labels filtered by confidence AND cross-prompt agreement.
        Classes with zero confident pixels fall back to text embedding.

        After this call:
            self.p_src  -- (C, D), unit-normalized, frozen for the rest of training
            self.p_tgt  -- (C, D), initialized as a copy of p_src for EMA updates
        """
        C = self.num_classes
        D = self.avg_text.shape[-1]
        feat_sum   = torch.zeros(C, D, device=self.device)
        feat_count = torch.zeros(C, device=self.device)
        seen = 0

        print(f"\n[CMA-Proto] Computing source prototypes (max {self.src_max_samples} images) ...")

        for data in data_loader:
            images = data['img_patches'].to(self.device)     # (B, Np, C, H, W) or flattened

            # Some datasets return (B, Np, 3, H, W); flatten to (B*Np, 3, H, W) like DPCore does.
            if images.dim() == 5:
                b_, np_, c_, h_, w_ = images.shape
                images = images.reshape(b_ * np_, c_, h_, w_)

            logits, image_features, text_features = self.model(
                images, self.text_x, True, interpolate=False
            )
            # logits         : (T, B, C, w, h)
            # image_features : (B, S, D), S = w*h + 1, CLS at index 0
            # text_features  : (T, C, D)

            B, _, w, h = logits.shape[1], logits.shape[2], logits.shape[3], logits.shape[4]

            # Per-prompt argmax → cross-prompt agreement mask
            per_prompt_pred = logits.argmax(dim=2)                      # (T, B, w, h)
            agreement_mask  = (per_prompt_pred == per_prompt_pred[0:1]).all(dim=0)  # (B, w, h)

            # Prompt-averaged prediction and confidence
            avg_logits = logits.mean(dim=0)                             # (B, C, w, h)
            probs = avg_logits.softmax(dim=1)
            confidence, pred_cls = probs.max(dim=1)                     # (B, w, h)

            # Combined filter
            mask = (confidence >= self.src_conf_threshold) & agreement_mask   # (B, w, h)

            # Drop CLS, reshape patch features to (B, w, h, D)
            patch_feats = image_features[:, 1:, :].reshape(B, w, h, D)

            # Accumulate per class (cast to float32 for numerically stable summation
            # of many vectors; CLIP model runs in fp16 on CUDA).
            if mask.any():
                masked_feats = patch_feats[mask].float()                # (N_valid, D)
                masked_cls   = pred_cls[mask]                           # (N_valid,)
                # scatter-add per class
                feat_sum.index_add_(0, masked_cls, masked_feats)
                feat_count.index_add_(
                    0, masked_cls,
                    torch.ones_like(masked_cls, dtype=feat_count.dtype)
                )

            seen += B
            if seen >= self.src_max_samples:
                break

        # Finalize prototypes with text-embedding fallback. Accumulate in fp32
        # for stability, then cast to model dtype for downstream compatibility.
        target_dtype = self.avg_text.dtype
        p_src = torch.zeros(C, D, device=self.device)   # fp32 staging
        fallback = 0
        pixel_counts = []
        for c in range(C):
            cnt = int(feat_count[c].item())
            pixel_counts.append((self.classes[c], cnt))
            if cnt > 0:
                mu = feat_sum[c] / feat_count[c]
                p_src[c] = mu / mu.norm().clamp(min=1e-8)
            else:
                p_src[c] = self.avg_text[c].to(p_src.dtype)
                fallback += 1

        self.p_src = p_src.to(target_dtype)
        self.p_tgt = self.p_src.clone()

        # Report
        pixel_counts.sort(key=lambda kv: -kv[1])
        top3 = ", ".join(f"{n}({k})" for n, k in pixel_counts[:3])
        zero = [n for n, k in pixel_counts if k == 0]
        print(f"[CMA-Proto] Source images used: {seen}")
        print(f"[CMA-Proto] Top-3 classes by confident-pixel count: {top3}")
        print(f"[CMA-Proto] Fallback classes (used text embedding): {fallback}/{C}"
              + (f" — {zero}" if zero else ""))

    # ===========================================================
    # Adaptation
    # ===========================================================

    def perform_adaptation(self, x):
        if self.p_src is None or self.p_tgt is None:
            raise RuntimeError(
                "Source prototypes not initialized. "
                "Call obtain_src_prototypes(data_loader) before adaptation."
            )

        t1 = time.time()
        loss_report = []
        for _ in range(self.steps):
            logits, image_features, text_features = self.model(
                x, self.text_x, True, interpolate=False
            )
            loss, parts = self.cma_proto_loss(logits, image_features, text_features)
            loss_report.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

    def cma_proto_loss(self, logits, image_features, text_features):
        """
        Compute the combined three-term loss.

        Returns:
            loss (scalar Tensor)
            parts (tuple of floats): (L_CMA, L_src, L_tgt) -- raw components
        """
        # (1) Prompt-averaged targets
        avg_logits = logits.mean(dim=0)                                      # (B, C, w, h)
        avg_text = text_features.mean(dim=0)
        avg_text = avg_text / avg_text.norm(dim=-1, keepdim=True).clamp(min=1e-8)   # (C, D)

        # (2) Pseudo-label and confidence (no_grad: used only to build mask/targets)
        with torch.no_grad():
            probs = avg_logits.softmax(dim=1)
            confidence, pred_cls = probs.max(dim=1)                          # (B, w, h)

            # (3) Top-K% confidence mask
            n_total = confidence.numel()
            k = max(1, int(round(n_total * self.top_k_percent)))
            threshold = torch.topk(confidence.flatten(), k, sorted=True).values[-1]
            mask = confidence >= threshold                                   # (B, w, h)

        # (4) Drop CLS, reshape patch features
        B, _, w, h = avg_logits.shape
        patch_feats = image_features[:, 1:, :]                               # (B, w*h, D)
        D = patch_feats.shape[-1]
        vis_feat = patch_feats.reshape(B, w, h, D)                           # (B, w, h, D)

        # (5) Gather per-pixel class-specific targets
        t_text = avg_text[pred_cls]                                          # (B, w, h, D)
        t_src = self.p_src[pred_cls]                                         # (B, w, h, D)
        t_tgt = self.p_tgt[pred_cls]                                         # (B, w, h, D)

        # (6) Cosine similarities (all vectors unit-normalized → dot product)
        cos_cma = (vis_feat * t_text).sum(dim=-1)                            # (B, w, h)
        cos_src = (vis_feat * t_src).sum(dim=-1)
        cos_tgt = (vis_feat * t_tgt).sum(dim=-1)

        # (7) Mask and combine
        L_cma = -cos_cma[mask].mean()
        L_src = -cos_src[mask].mean()
        L_tgt = -cos_tgt[mask].mean()
        loss = (self.lambda_cma * L_cma
                + self.lambda_src * L_src
                + self.lambda_tgt * L_tgt)

        # (8) EMA update for target prototypes (no_grad)
        with torch.no_grad():
            self._update_target_prototypes(vis_feat.detach(), pred_cls, mask)

        return loss, (L_cma.item(), L_src.item(), L_tgt.item())

    @torch.no_grad()
    def _update_target_prototypes(self, vis_feat, pred_cls, mask):
        """
        EMA update per-class target prototypes using masked pixels only.

        Classes absent in this batch are not updated.
        """
        alpha = self.ema_alpha
        if not mask.any():
            return
        masked_feats = vis_feat[mask].float()                                # (N, D) fp32
        masked_cls = pred_cls[mask]                                          # (N,)
        tgt_dtype = self.p_tgt.dtype
        # Per-class mean of the masked pixels
        for c in masked_cls.unique().tolist():
            cm = masked_cls == c
            mu_c = masked_feats[cm].mean(dim=0)
            mu_c = mu_c / mu_c.norm().clamp(min=1e-8)
            # EMA in fp32 to avoid fp16 underflow when (1-alpha) is tiny
            prev = self.p_tgt[c].float()
            new = alpha * prev + (1.0 - alpha) * mu_c
            new = new / new.norm().clamp(min=1e-8)
            self.p_tgt[c] = new.to(tgt_dtype)

    # ===========================================================
    # Shared helpers (identical to TENTContinual / CMAContinual)
    # ===========================================================

    def extract_text_embeddings(self, class_names, prompts, average=True):
        text_features = []
        for class_name in class_names:
            texts = [p.format(class_name) for p in prompts]
            texts = self.tokenize(texts).to(self.device)
            class_embeddings = self.model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            if average:
                avg = class_embeddings.mean(dim=0)
                avg = avg / avg.norm()
                class_embeddings = torch.cat([class_embeddings, avg.unsqueeze(0)], dim=0)
            text_features.append(class_embeddings)
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
