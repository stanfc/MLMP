import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class TENTContinual:
    """
    TENT-Continual: entropy minimization without episodic reset.

    This is the naive continual TTA baseline — standard TENT adapted to the
    continual setting by simply removing the per-sample reset. The model
    accumulates gradient updates across the entire test stream.

    As shown in CoTTA Table 5, TENT-continual degrades over time because:
      1. Accumulated pseudo-label errors compound (error accumulation)
      2. No mechanism prevents the model from drifting away from source knowledge
         (catastrophic forgetting)

    Including this baseline allows direct comparison with CoTTA and MLMP-CoTTA
    to verify that the CoTTA mechanisms are necessary, not just continual updates.

    Architecture: identical to TENT except adapt() does NOT call reset().
    """

    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 prompt_dir=None, runtime_calculation=False, device='cpu'):
        """
        Args:
            ovss_type: OVSS model identifier (e.g., 'naclip')
            ovss_backbone: Backbone name (e.g., 'ViT-L/14')
            lr: Learning rate for the LayerNorm optimizer
            classes: List of class names
            steps: Gradient steps per sample (default 1 for online CTTA)
            prompt_dir: Path to YAML with prompt templates (None → single template)
            runtime_calculation: Record per-sample timing if True
            device: Compute device
        """
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps
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
        """
        CTTA entropy minimization — no reset, model state persists.

        Args:
            x: Input patch tensor (B*N_patches, C, H, W)

        Returns:
            List[float]: Loss per gradient step
        """
        return self.perform_adaptation(x)

    @torch.no_grad()
    def evaluate(self, x):
        """
        Standard TENT evaluation (single-level, single-prompt).

        Args:
            x: Input patch tensor (B*N_patches, C, H, W)

        Returns:
            torch.Tensor: Per-class logits (B*N_patches, num_classes, H, W)
        """
        t1 = time.time()
        logits, _, _ = self.model(x, self.text_x, True, interpolate=True)
        logits = logits[0]
        if self.runtime:
            self.eval_times.append(time.time() - t1)
        return logits

    def reset(self):
        """
        Reset to source state (episodic mode only).
        NOT called during CTTA operation.
        """
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
            logits, _, _ = self.model(x, self.text_x, True, interpolate=False)
            # Pixel-wise entropy minimization (same as TENT)
            loss = self.softmax_entropy(logits).mean()
            loss_report.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

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
    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        return -(x.softmax(-3) * x.log_softmax(-3)).sum(-3)
