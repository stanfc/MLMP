"""
MLMP-MinPrompt-Continual: multi-level multi-prompt loss that picks the
WEAKEST prompt's confidence in the consensus pseudo-label, per pixel.

Differs from `tent_minprompt_continual` by:
  - Uses MLMP's UAML multi-layer fused logits (vision_outputs covers 18
    layers, vision_out_type='mean' during adapt), not a single-layer
    forward.
  - Keeps the optional ILE (alpha_cls * CLS entropy) term -- but defaults
    to alpha_cls=0 because ILE was designed for episodic and may hurt CTTA.

Pixel-level loss:
  probs   = softmax(logits, dim=class)        # (T, B, C, h, w)
  pseudo  = argmax(mean over T)               # (B, h, w)
  probs_y = gather probs at pseudo class       # (T, B, h, w)
  min_p   = probs_y.min(dim=T).values          # (B, h, w)
  L_pix   = -log(min_p).mean()

Total loss:
  L = L_pix + alpha_cls * mean( entropy(cls_logits, dim=class) )
"""

import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class MLMPMinPromptContinual:
    """Multi-level multi-prompt min-margin loss for CTTA."""

    def __init__(self, ovss_type, ovss_backbone, lr, classes,
                 vision_outputs=(-1,), alpha_cls=0.0, steps=1,
                 prompt_dir='prompts.yaml',
                 runtime_calculation=False, device='cpu'):
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps
        self.vision_outputs = vision_outputs
        self.alpha_cls = alpha_cls
        self.runtime = runtime_calculation
        self.device = device

        if classes is None:
            raise ValueError("classes is required in __init__")
        self.classes = classes

        self.model, self.tokenize = load_ovss(ovss_type, ovss_backbone, device=device)

        if prompt_dir:
            self.prompt_templates = load_prompts_from_yaml(prompt_dir)
            print(f"Number of prompt templates: {len(self.prompt_templates)}")
        else:
            self.prompt_templates = [REFERENCE_PROMPT]
            print("WARNING: MLMP-MinPrompt with only 1 template degenerates to log-loss.")

        self.model.transformer.requires_grad_(False)
        self.model.ln_final.requires_grad_(False)
        self.model.token_embedding.requires_grad_(False)
        self.model.visual = self.set_ln_grads(self.model.visual)
        params, _ = self.collect_ln_params(self.model.visual)

        print_clip_parameters(self.model)
        print(f"+++ MLMP-MinPrompt: T={len(self.prompt_templates)} prompts, "
              f"alpha_cls={self.alpha_cls}, vision_outputs={self.vision_outputs}")

        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer
        )

        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(
                self.classes, self.prompt_templates, average=True
            ).squeeze()

        if self.runtime:
            self.adapt_times = []
            self.eval_times = []

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
            save_weights=True
        )
        logits = logits[0]
        if self.runtime:
            self.eval_times.append(time.time() - t1)
        return logits

    def reset(self):
        self.load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state
        )

    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []

        for _ in range(self.steps):
            # Multi-layer fused logits (MLMP UAML), all per-template
            logits, _, _, cls_logits = self.model(
                x, self.text_x[:-1], True,
                interpolate=False,
                vision_outputs=self.vision_outputs,
                return_vanilla_cls=True,
                vision_out_type="mean"
            )
            # logits: (T, B, C, h, w),  cls_logits: (T, B, C, 1, 1)
            probs = logits.softmax(dim=2)                # (T, B, C, h, w)

            with torch.no_grad():
                ens_probs = probs.mean(dim=0)            # (B, C, h, w)
                pseudo = ens_probs.argmax(dim=1)          # (B, h, w)

            T_, B_, C_, H_, W_ = probs.shape
            pseudo_exp = pseudo.unsqueeze(0).unsqueeze(2).expand(T_, B_, 1, H_, W_)
            probs_y = probs.gather(2, pseudo_exp).squeeze(2)  # (T, B, h, w)
            min_p = probs_y.min(dim=0).values                 # (B, h, w)
            loss_pix = -torch.log(min_p.clamp_min(1e-12)).mean()

            if self.alpha_cls > 0:
                entropy_per_cls = self.softmax_entropy(cls_logits, dim=2)
                loss = loss_pix + self.alpha_cls * entropy_per_cls.mean()
            else:
                loss = loss_pix

            loss_report.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

    @staticmethod
    def softmax_entropy(x: torch.Tensor, dim: int = -3) -> torch.Tensor:
        return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)

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
