"""
COMEContinual: port of COME (Test-time Adaption by Conservatively
Minimizing Entropy, ICLR 2025, https://arxiv.org/abs/2410.10894,
github.com/BlueWhaleLab/COME -> vendored at /home/stanfc/TTA-on-OVSS/come/).

COME is a drop-in loss replacement, not a new training loop: the paper's
own README frames it as "replace the loss function of previous TTA
algorithms (Tent, EATA, SAR) from softmax entropy to entropy of opinion".
This port applies it to this repo's own TENT-continual (adapt/tent_continual.py)
-- everything about TENTContinual is kept (LayerNorm-only, single template,
no reset, no filtering) except the loss function itself. This is exactly
the paper's own "Tent_COME" configuration (see come/tent_come.py).

Core mechanism (verbatim formula from come/tent_come.py's `dirichlet_entropy`,
itself Subjective-Logic / evidential-deep-learning style):
  Treat each class's `exp(logit)` as an evidence mass for a Dirichlet
  distribution over a (K+1)-simplex -- K per-class "belief" masses plus one
  extra "uncertainty" mass that shrinks as total evidence (sum of exp(logit))
  grows. Entropy is computed over this augmented (K+1)-dim distribution
  instead of the plain K-way softmax. The uncertainty mass acts as an
  entropy *floor* that only vanishes once genuine evidence dominates it --
  this is the "conservative" refinement: entropy-minimization can no longer
  be satisfied just by making the K-way softmax peaky, it also has to
  accumulate enough absolute evidence to shrink the uncertainty term. This
  is the paper's explicit answer to plain entropy minimization's tendency
  to reward overconfidence -- directly relevant to this repo's own
  documented "entropy minimization collapses to a trivial solution"
  finding (see MLMP/CLAUDE.md's Design Constraints table).

Deviations from the original (documented, not the paper's own choices):
  - The "evidence pseudo-count" constant is `len(classes)` here, not the
    upstream code's hardcoded `1000`. Upstream's own README pseudocode
    names this constant symbolically as `K` (matching the paper, which
    defines it as the number of classes) -- the committed `tent_come.py`
    just never renamed the literal `1000` for ImageNet's own class count.
    We use the actual class count for our ~19-20-class OVSS setting.
  - Dense (B, C, H, W) logits replace (B, C) classification logits: the
    L2-normalize-but-keep-forward-value trick, the per-class evidence sum,
    and the belief/uncertainty concatenation all move from dim=-1/dim=1
    (equivalent for a 2D tensor) to the explicit channel dim=1, and the
    final entropy is a per-pixel (B, H, W) map instead of a per-image
    scalar per sample -- the same generalization every other ported
    entropy-style loss in this repo uses.
  - The L2-normalize-with-detached-magnitude step
    (`x / ||x|| * ||x||.detach()`) is a forward-value no-op (it returns x
    numerically) that only rescales the backward gradient; kept verbatim
    for gradient-behavior fidelity even though it does not affect the
    forward loss value.
"""
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class COMEContinual:
    """TENT-continual with softmax entropy replaced by COME's "entropy of
    opinion" (see module docstring). Everything else -- LayerNorm-only,
    single prompt template, no reset, no reliable/redundant filtering --
    is identical to adapt/tent_continual.py."""

    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 catseg_checkpoint=None, maft_checkpoint=None,
                 prompt_dir=None, runtime_calculation=False, device='cpu'):
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps
        self.runtime = runtime_calculation
        self.device = device

        if classes is None:
            raise ValueError("classes is required in __init__")
        self.classes = classes
        self.num_classes = len(classes)  # COME's evidence pseudo-count K, see docstring

        self.model, self.tokenize = load_ovss(ovss_type, ovss_backbone, device=device,
                                              classes=self.classes, catseg_checkpoint=catseg_checkpoint,
                                              maft_checkpoint=maft_checkpoint)

        if prompt_dir:
            self.prompt_templates = load_prompts_from_yaml(prompt_dir)
        else:
            self.prompt_templates = [REFERENCE_PROMPT]

        self.model.transformer.requires_grad_(False)
        self.model.ln_final.requires_grad_(False)
        self.model.token_embedding.requires_grad_(False)
        self.model.visual = self.set_ln_grads(self.model.visual)
        params, _ = self.collect_ln_params(self.model.visual)

        print_clip_parameters(self.model)

        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer)

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
        return self.perform_adaptation(x)

    def continual_adapt(self, x):
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
            self.model, self.optimizer, self.model_state, self.optimizer_state)

    def state_dict(self):
        """Adapt-state checkpoint for mid-stream resume: every tensor
        currently registered in the optimizer plus optimizer state."""
        opt_params = [p for g in self.optimizer.param_groups for p in g['params']]
        return {
            'params': [p.detach().cpu().clone() for p in opt_params],
            'optimizer': self.optimizer.state_dict(),
        }

    def load_state_dict(self, sd):
        opt_params = [p for g in self.optimizer.param_groups for p in g['params']]
        with torch.no_grad():
            for p, saved in zip(opt_params, sd['params']):
                p.data.copy_(saved.to(device=p.device, dtype=p.dtype))
        self.optimizer.load_state_dict(sd['optimizer'])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

    # ===========================================================
    # Adaptation
    # ===========================================================

    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []
        for _ in range(self.steps):
            logits, _, _ = self.model(x, self.text_x, True, interpolate=False)
            l0 = logits[0]  # (B, C, w, h)
            loss = self.dirichlet_entropy(l0, self.num_classes).mean()
            loss_report.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

    # ===========================================================
    # COME loss (per-pixel, channel dim=1)
    # ===========================================================

    @staticmethod
    def dirichlet_entropy(x: torch.Tensor, K: int) -> torch.Tensor:
        """COME's entropy of opinion, generalized to dim=1 as the class
        channel of a (B, C, ...) tensor (verbatim formula from
        come/tent_come.py's `dirichlet_entropy`, see module docstring for
        the dim and K deviations).

        Computed in fp32: `exp(x)` on a raw CLIP logit (up to ~logit_scale
        ~= 14) already overflows fp16's ~65504 ceiling, and this repo's
        model runs natively in fp16 with no AMP loss scaling -- verified by
        direct repro to collapse every prediction to a single dominant
        class within one adapt step, the exact same NaN-propagation
        signature diagnosed and fixed for adapt/vida_continual.py's Adam
        eps bug."""
        orig_dtype = x.dtype
        x = x.float()
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        x = x / norm * norm.detach()  # forward no-op, rescales the gradient only
        evidence_sum = torch.sum(torch.exp(x), dim=1, keepdim=True)
        brief = torch.exp(x) / (evidence_sum + K)          # (B, C, ...)
        uncertainty = K / (evidence_sum + K)                # (B, 1, ...)
        probability = torch.cat([brief, uncertainty], dim=1) + 1e-7  # (B, C+1, ...)
        entropy = -(probability * torch.log(probability)).sum(1)     # (B, ...)
        return entropy.to(orig_dtype)

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
