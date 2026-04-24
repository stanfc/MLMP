import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class CMAContinual:
    """
    CMA-Continual: Cross-Modal Alignment TTA without episodic reset.

    Replaces the entropy-minimization objective of TENT with a cross-modal
    alignment loss that pulls each confident pixel's visual feature toward
    the frozen text embedding of its predicted class:

        L_CMA = -mean_{i in S_conf} cos(v_i, t_{c_i})

    where S_conf is the set of pixels whose prediction confidence falls in
    the top-K% of the current batch.

    Anti-collapse intuition: text embeddings are fixed and geometrically
    diverse, so there is no single low-energy degenerate state. Each pixel
    is pulled toward a class-specific direction rather than rewarded for
    raw confidence.

    See docs/cma_continual_spec.md for the full design.
    """

    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 top_k_percent=0.2, prompt_dir=None,
                 runtime_calculation=False, device='cpu'):
        """
        Args:
            ovss_type: OVSS model identifier (e.g., 'naclip')
            ovss_backbone: Backbone name (e.g., 'ViT-L/14')
            lr: Learning rate for the LayerNorm optimizer
            classes: List of class names
            steps: Gradient steps per sample (default 1 for online CTTA)
            top_k_percent: Fraction of highest-confidence pixels per batch
                           that contribute to the CMA loss (default 0.2)
            prompt_dir: Path to YAML with prompt templates (None -> single template)
            runtime_calculation: Record per-sample timing if True
            device: Compute device
        """
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps
        self.top_k_percent = float(top_k_percent)
        if not (0.0 < self.top_k_percent <= 1.0):
            raise ValueError(
                f"top_k_percent must be in (0, 1], got {self.top_k_percent}"
            )
        self.runtime = runtime_calculation
        self.device = device

        if classes is None:
            raise ValueError("classes is required in __init__")
        self.classes = classes

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

        # ---------- Source state (for episodic reset if needed) ----------
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer
        )

        # ---------- Text features ----------
        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(
                self.classes, self.prompt_templates, average=False
            ).squeeze()

        if self.runtime:
            self.adapt_times = []
            self.eval_times = []

    # ===========================================================
    # Public API
    # ===========================================================

    def adapt(self, x):
        """CMA continual adaptation -- no reset, model state persists."""
        return self.perform_adaptation(x)

    def continual_adapt(self, x):
        """Alias for adapt() -- compatible with main_continual.py protocol."""
        return self.perform_adaptation(x)

    @torch.no_grad()
    def evaluate(self, x):
        """Standard single-level evaluation (matches TENT/MLMP eval)."""
        t1 = time.time()
        logits, _, _ = self.model(x, self.text_x, True, interpolate=True)
        logits = logits[0]
        if self.runtime:
            self.eval_times.append(time.time() - t1)
        return logits

    def reset(self):
        """Reset to source state (episodic mode only). NOT used during CTTA."""
        self.load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state
        )

    # ===========================================================
    # Adaptation
    # ===========================================================

    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []
        for _ in range(self.steps):
            # Forward pass returns:
            #   logits         : (T, B, C, w, h)   -- T prompt templates
            #   image_features : (B, S, D)         -- S = w*h + 1, CLS at index 0
            #   text_features  : (T, C, D)
            logits, image_features, text_features = self.model(
                x, self.text_x, True, interpolate=False
            )
            loss = self.cma_loss(logits, image_features, text_features)
            loss_report.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

    def cma_loss(self, logits, image_features, text_features):
        """
        Cross-Modal Alignment loss.

        Args:
            logits         : (T, B, C, w, h)
            image_features : (B, S, D), L2-normalized, CLS at index 0
            text_features  : (T, C, D), L2-normalized

        Returns:
            scalar loss = -mean_{i in S_conf} cos(v_i, t_{c_i})
        """
        # (1) Average over prompt templates for stable pseudo-labels and target
        avg_logits = logits.mean(dim=0)            # (B, C, w, h)
        avg_text = text_features.mean(dim=0)       # (C, D)
        # Re-normalize the averaged text vector (mean of unit vectors is no longer unit-norm)
        avg_text = avg_text / avg_text.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # (2) Per-pixel pseudo-label and confidence (no_grad: pseudo-labels are targets)
        with torch.no_grad():
            probs = avg_logits.softmax(dim=1)
            confidence, pred_cls = probs.max(dim=1)   # (B, w, h), (B, w, h)

            # (3) Top-K% confidence mask within this batch
            n_total = confidence.numel()
            k = max(1, int(round(n_total * self.top_k_percent)))
            threshold = torch.topk(confidence.flatten(), k, sorted=True).values[-1]
            mask = confidence >= threshold            # (B, w, h)

        # (4) Drop CLS, reshape patch features to spatial grid
        B, _, w, h = avg_logits.shape
        patch_feats = image_features[:, 1:, :]        # (B, w*h, D)
        D = patch_feats.shape[-1]
        vis_feat = patch_feats.reshape(B, w, h, D)    # (B, w, h, D)

        # (5) Lookup text target per pixel
        text_target = avg_text[pred_cls]              # (B, w, h, D)

        # (6) Cosine similarity (both unit-norm -> dot product)
        cos_sim = (vis_feat * text_target).sum(dim=-1)   # (B, w, h)

        # (7) Mean over the confident-pixel mask, negate to minimize
        return -cos_sim[mask].mean()

    # ===========================================================
    # Shared helpers (identical to TENTContinual)
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
