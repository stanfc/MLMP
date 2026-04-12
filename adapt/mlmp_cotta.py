import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class MLMPCoTTA:
    """
    MLMP-CoTTA: Multi-Level Multi-Prompt adaptation with CoTTA's continual mechanisms.

    Combines the adaptation signal quality of MLMP with CoTTA's long-term stability:
      - MLMP's uncertainty-aware multi-level feature fusion (UAML) for the student loss
      - MLMP's multi-prompt loss-level integration (T prompts, averaged gradient)
      - MLMP's image-level CLS entropy (ILE) term
      - CoTTA's EMA teacher for stable pseudo-labels
      - CoTTA's augmentation-averaged pseudo-labels with confidence gating
      - CoTTA's stochastic restoration for anti-forgetting

    Pseudo-label generation uses the teacher's evaluation-mode forward pass
    (adaptive_weighted_mean fusion, averaged prompt), which is the most accurate
    and stable prediction available. The student is then trained to match the teacher
    using MLMP's multi-level multi-prompt loss structure.

    Only visual encoder LayerNorm (γ, β) parameters are updated.
    """

    def __init__(self, ovss_type, ovss_backbone, lr, classes,
                 vision_outputs=(-1,), alpha_cls=0.0, steps=1,
                 prompt_dir='prompts.yaml', prompt_integration='loss',
                 ema_alpha=0.999, restoration_p=0.01, conf_threshold=0.1,
                 n_augmentations=8, runtime_calculation=False, device='cpu'):
        """
        Args:
            vision_outputs: Tuple of layer indices for UAML (e.g., tuple(range(-1,-19,-1)))
            alpha_cls: Weight for image-level CLS entropy term (MLMP ILE)
            steps: Gradient steps per sample — use 1 for online CTTA
            prompt_dir: Path to YAML with prompt templates ('prompts.yaml')
            prompt_integration: 'loss' (MLMP default) or 'text'
            ema_alpha: EMA smoothing factor for teacher (default 0.999)
            restoration_p: Stochastic restoration probability (default 0.01)
            conf_threshold: Source confidence threshold for augmentation gating
            n_augmentations: Number of augmented teacher views when conf < threshold
        """
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps
        self.vision_outputs = vision_outputs
        self.alpha_cls = alpha_cls
        self.prompt_dir = prompt_dir
        self.prompt_integration = prompt_integration
        self.ema_alpha = ema_alpha
        self.restoration_p = restoration_p
        self.conf_threshold = conf_threshold
        self.n_augmentations = n_augmentations
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

        print(f"+++ Vision outputs (UAML layers): {self.vision_outputs}")
        assert prompt_integration in ['loss', 'text'], \
            "prompt_integration must be 'loss' or 'text'"

        # ---------- Freeze text encoder, enable LN grads in visual encoder ----------
        self.model.transformer.requires_grad_(False)
        self.model.ln_final.requires_grad_(False)
        self.model.token_embedding.requires_grad_(False)
        self.model.visual = self.set_ln_grads(self.model.visual)
        params, _ = self.collect_ln_params(self.model.visual)

        print_clip_parameters(self.model)

        # ---------- Optimizer ----------
        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        # ---------- Source state (W_0) ----------
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer
        )

        # LN-only source snapshot for fast swap/restore
        self.source_ln_params = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        # ---------- Teacher model (EMA, frozen) ----------
        self.teacher_model = copy.deepcopy(self.model)
        self.teacher_model.requires_grad_(False)

        # ---------- Text embeddings ----------
        # text_x shape: (T+1, C, D)  — T per-template + 1 averaged (index [-1])
        # text_x[:-1] → per-template embeddings used in multi-prompt loss
        # text_x[-1]  → averaged embedding used in evaluate() and teacher pseudo-label
        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(
                self.classes, self.prompt_templates, average=True
            ).squeeze()

        # ---------- Runtime tracking ----------
        if self.runtime:
            self.adapt_times = []
            self.eval_times = []

    # ===========================================================
    # Public API
    # ===========================================================

    def adapt(self, x):
        """
        CTTA adaptation step — no reset, state persists across all calls.

        Args:
            x: Input patch tensor (B*N_patches, C, H, W)

        Returns:
            List[float]: Loss per gradient step (length = self.steps)
        """
        return self.perform_adaptation(x)

    @torch.no_grad()
    def evaluate(self, x):
        """
        Inference using MLMP's uncertainty-aware multi-level weighted fusion.

        Uses the adapted student model (same as MLMP.evaluate).

        Args:
            x: Input patch tensor (B*N_patches, C, H, W)

        Returns:
            torch.Tensor: Per-class logits (B*N_patches, num_classes, H, W)
        """
        t1 = time.time()
        logits, _, _ = self.model(
            x, self.text_x[-1], True,
            vision_outputs=self.vision_outputs,
            interpolate=True,
            vision_out_type="adaptive_weighted_mean",
            save_weights=True
        )
        logits = logits[0]
        if self.runtime:
            self.eval_times.append(time.time() - t1)
        return logits

    def reset(self):
        """
        Full reset to source state (episodic TTA mode only).
        NOT called during CTTA operation.
        """
        self.load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state
        )
        self.teacher_model.load_state_dict(self.model_state, strict=True)
        self.teacher_model.requires_grad_(False)

    # ===========================================================
    # Core CTTA logic
    # ===========================================================

    def perform_adaptation(self, x):
        t1 = time.time()

        # Step 1 — confidence-gated augmentation-averaged teacher pseudo-label
        pseudo_label = self._get_pseudo_label(x)   # (B, C, h, w) soft probs

        # Step 2 — student update via MLMP loss + pseudo-label consistency
        loss_report = []
        for _ in range(self.steps):
            if self.prompt_integration == 'loss':
                # Multi-prompt, multi-level student forward pass
                student_logits, _, _, cls_logits = self.model(
                    x, self.text_x[:-1], True,
                    interpolate=False,
                    vision_outputs=self.vision_outputs,
                    return_vanilla_cls=True,
                    vision_out_type="mean"
                )
                # student_logits: (T, B, C, h, w)
                # cls_logits:     (T, B, C, 1, 1)  — from CLS token

                # Primary loss: CE against teacher pseudo-label, averaged over T prompts
                # pseudo_label is (B, C, h, w) → broadcast to (T, B, C, h, w)
                student_log_prob = student_logits.log_softmax(dim=-3)
                ce_loss = -(pseudo_label.unsqueeze(0) * student_log_prob).sum(dim=-3).mean()

                # Auxiliary loss: image-level CLS entropy (MLMP ILE term)
                cls_entropy = self.softmax_entropy(cls_logits, dim=2).mean()

                loss = ce_loss + self.alpha_cls * cls_entropy

            else:   # 'text' integration
                student_logits, _, _ = self.model(
                    x, self.text_x[-1], True,
                    interpolate=False,
                    vision_outputs=self.vision_outputs
                )
                student_log_prob = student_logits[0].log_softmax(dim=-3)
                loss = -(pseudo_label * student_log_prob).sum(dim=-3).mean()

            loss_report.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Step 3 — EMA teacher update
        with torch.no_grad():
            for t_p, s_p in zip(self.teacher_model.parameters(), self.model.parameters()):
                t_p.data.mul_(self.ema_alpha).add_(s_p.data, alpha=1.0 - self.ema_alpha)

        # Step 4 — stochastic restoration
        self._stochastic_restore()

        if self.runtime:
            self.adapt_times.append(time.time() - t1)

        return loss_report

    @torch.no_grad()
    def _get_pseudo_label(self, x):
        """
        Teacher pseudo-label with confidence-gated augmentation averaging.

        Teacher uses MLMP's evaluate()-style forward:
          - adaptive_weighted_mean layer fusion (β=1, entropy-weighted)
          - averaged prompt embedding (text_x[-1])
        This produces the highest-quality pseudo-label available.

        If source model confidence ≥ conf_threshold: direct teacher prediction.
        Else: N augmented teacher predictions averaged (spatial flips inverted).
        """
        conf = self._source_confidence(x)

        if conf >= self.conf_threshold:
            teacher_logits, _, _ = self.teacher_model(
                x, self.text_x[-1], True,
                vision_outputs=self.vision_outputs,
                interpolate=False,
                vision_out_type="adaptive_weighted_mean"
            )
            pseudo_label = teacher_logits[0].softmax(dim=-3)
        else:
            aug_preds = []
            for _ in range(self.n_augmentations):
                x_aug, flip_h, flip_v = self._random_augment(x)
                aug_logits, _, _ = self.teacher_model(
                    x_aug, self.text_x[-1], True,
                    vision_outputs=self.vision_outputs,
                    interpolate=False,
                    vision_out_type="adaptive_weighted_mean"
                )
                aug_prob = aug_logits[0].softmax(dim=-3)
                if flip_h:
                    aug_prob = torch.flip(aug_prob, dims=[-1])
                if flip_v:
                    aug_prob = torch.flip(aug_prob, dims=[-2])
                aug_preds.append(aug_prob)
            pseudo_label = torch.stack(aug_preds, dim=0).mean(dim=0)

        return pseudo_label

    @torch.no_grad()
    def _source_confidence(self, x):
        """
        Confidence using source (W_0) weights via temporary LN swap.
        Uses MLMP's 'mean' multi-level fusion for fast source confidence estimate.
        """
        current_ln = {
            n: p.data.clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }
        for name, param in self.model.named_parameters():
            if name in self.source_ln_params:
                param.data.copy_(self.source_ln_params[name])

        logits, _, _ = self.model(
            x, self.text_x[-1], True,
            vision_outputs=self.vision_outputs,
            interpolate=False,
            vision_out_type="mean"
        )
        conf = logits[0].softmax(dim=-3).max(dim=-3)[0].mean().item()

        for name, param in self.model.named_parameters():
            if name in current_ln:
                param.data.copy_(current_ln[name])

        return conf

    def _stochastic_restore(self):
        """
        Randomly restore fraction restoration_p of LN weights to source values.
        CoTTA Eq. 7-8: W_{t+1} = M ⊙ W_0 + (1-M) ⊙ W_{t+1}, M ~ Bernoulli(p).
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.source_ln_params:
                    mask = torch.bernoulli(
                        torch.ones_like(param.data) * self.restoration_p
                    ).bool()
                    param.data[mask] = self.source_ln_params[name][mask]

    @staticmethod
    def _random_augment(x):
        """Random H/V flip + small Gaussian noise. Returns (x_aug, flip_h, flip_v)."""
        x_aug = x.clone()
        flip_h = torch.rand(1).item() > 0.5
        flip_v = torch.rand(1).item() > 0.5
        if flip_h:
            x_aug = torch.flip(x_aug, dims=[-1])
        if flip_v:
            x_aug = torch.flip(x_aug, dims=[-2])
        x_aug = x_aug + torch.randn_like(x_aug) * 0.01
        return x_aug, flip_h, flip_v

    # ===========================================================
    # Shared helpers
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

    @staticmethod
    def softmax_entropy(x: torch.Tensor, dim=-3) -> torch.Tensor:
        return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)
