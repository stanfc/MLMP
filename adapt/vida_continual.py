"""
ViDAContinual: port of ViDA (Homeostatic Visual Domain Adapter for Continual
Test-Time Adaptation, ICLR 2024, arxiv 2306.04344,
github.com/Yangsenqiao/ViDA -> vendored at /home/stanfc/TTA-on-OVSS/vida/).

Faithful port (user's explicit choice: inject real trainable LoRA-style
adapters, not a LayerNorm-only reinterpretation, even though every other
method in this repo trains only LayerNorm -- ViDA is therefore NOT an
apples-to-apples parameter-budget comparison with our own method, by design).

Core mechanism (see vida/imagenet/vida.py + inject_vida.py for the original):
  1. ViDAInjectedLinear: wraps a frozen nn.Linear with two extra trainable
     low-rank branches -- a low-rank one (r, "domain-shared") and a
     higher-rank one (r2, "domain-specific") -- each gated by a scalar
     (scale1, scale2) set externally per step.
  2. Homeostatic Knowledge Allotment (HKA): run the EMA teacher N times
     through lightweight augmentations, measure prediction variance
     ("uncertainty"). High uncertainty -> lean on the domain-specific
     (high-rank) branch; low uncertainty -> lean on the domain-shared
     (low-rank) branch. This is ViDA's analogue of our own AdaGate's
     trend-based DEEP/SHALLOW switch, just keyed on prediction variance
     instead of gradient-norm trend.
  3. Loss: symmetric cross-entropy between student and EMA-teacher output
     (softmax_entropy below), backprop only touches the injected LoRA
     params (everything else -- backbone weights, LayerNorm, text encoder
     -- stays frozen, matching the original paper).
  4. EMA teacher update: injected-adapter params at `ema_vida` decay,
     everything else at `ema` decay (they're identical/frozen here since
     only the adapters ever change, but kept separate to match upstream).

Deviations from the original (documented, not the paper's own choices):
  - Backbone here is CLIP ViT (nn.MultiheadAttention, a single fused
    in_proj_weight), not the diffusion-style "Attention" modules with
    separate q/k/v nn.Linear the original `inject_trainable_vida` was
    written against. We inject onto the two genuinely separable
    nn.Linear submodules that DO exist per transformer block:
    `attn.out_proj` and `mlp.c_proj`.
  - The uncertainty-estimation augmentations reuse this repo's own
    `adapt/cotta.py::get_tta_transforms` (already validated on our data
    pipeline) instead of ViDA's original ImageNet-classification
    augmentation stack.
  - Loss operates on dense (B, C, H, W) per-pixel logits (softmax over
    the channel/class dim) instead of per-image (B, C) classification
    logits -- everything else about the symmetric CE is unchanged.

Investigated on a real 150-round ACDC run (no early-improvement phase --
mean mIoU declines from Round 1 onward, unlike e.g. TENT-continual's
documented improve-then-collapse arc). First pass concluded this was
purely a property of the published algorithm (see the mechanism argument
below) -- that arguments still holds, but it was *incomplete*: it checked
only `vida/imagenet/vida.py`'s own Python class defaults, not the paper's
*actual tuned* ImageNet-C ViT-benchmark config
(`vida/imagenet/cfgs/vit/vida.yaml`, the one `bash/vida_vit.sh` really
runs). That config reveals a genuine hyperparameter bug in this port:

  - **LR was 50x too large.** The real benchmark splits the optimizer
    into two param groups -- `OPTIM.LR=5e-7` for (in the original) LN/BN
    params, and **`OPTIM.ViDALR=2e-7`** specifically for the injected LoRA
    matrices (see `imagenet/imagenetc.py::setup_optimizer_vida`). Since
    this port trains *only* the LoRA params, `ViDALR=2e-7` is the directly
    applicable rate -- but the bash scripts were using this repo's blanket
    "TENT-style LR=1e-5" convention, **50x larger**. LoRA's low-rank
    bottleneck matrices are far more update-sensitive than a plain
    LayerNorm affine scale/shift, so a 50x-oversized step size plausibly
    explains the *immediate, unstable* decline from Round 1 far better
    than "slow entropy-like drift" alone would. Fixed: `LR=2e-7` in all
    three bash scripts.
  - **`ema_vida` was also checked against the wrong source.** The real
    benchmark's `MT_ViDA=0.8`, not the class file's own untuned default of
    `0.99`. (0.8 is an even *faster*-tracking teacher than 0.99, so this
    correction reinforces the entropy-minimization argument below rather
    than weakening it -- it does not explain the instability by itself,
    the LR did.) Fixed: constructor and CLI default both now `0.8`.

The mechanism argument for why ViDA still won't show a large, sustained
early-gain phase even after these fixes: verified line-by-line against
`vida/imagenet/vida.py`'s `forward_and_adapt`, student and teacher see the
*same, unaugmented* input `x` (the transform is only used inside the HKA
uncertainty estimate, discarded afterward) -- the entire training signal
is `symmetric_ce(student(x), teacher(x).detach())`. With a fast-tracking
teacher (`ema_vida=0.8`, even more so than 0.99), student and teacher stay
close for most of the run, and `symmetric_ce(p, p)` degenerates toward
plain softmax entropy -- precisely the objective this repo's own research
(MLMP/CLAUDE.md's Design Constraints table) already established collapses
under this backbone/no-reset protocol. This part is a property of the
algorithm as published, not something the port introduced. (A separate,
inert-in-this-port `ema` default typo -- 0.999 vs the paper's 0.99 -- was
also found and fixed; `ema` only governs the frozen non-adapter params
here, so it made no numerical difference.)
"""
import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ovss import load_ovss
from adapt.cotta import get_tta_transforms
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class ViDAInjectedLinear(nn.Module):
    """Frozen base linear + two trainable low-rank branches, each scaled by
    an externally-set scalar (scale1=low-rank/domain-shared, scale2=high-
    rank/domain-specific). Verbatim from vida/imagenet/inject_vida.py."""

    def __init__(self, base_linear, r=4, r2=16):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)
        in_f, out_f = base_linear.in_features, base_linear.out_features
        self.vida_down = nn.Linear(in_f, r, bias=False)
        self.vida_up = nn.Linear(r, out_f, bias=False)
        self.vida_down2 = nn.Linear(in_f, r2, bias=False)
        self.vida_up2 = nn.Linear(r2, out_f, bias=False)
        nn.init.normal_(self.vida_down.weight, std=1 / r**2)
        nn.init.zeros_(self.vida_up.weight)
        nn.init.normal_(self.vida_down2.weight, std=1 / r2**2)
        nn.init.zeros_(self.vida_up2.weight)
        self.scale1 = 1.0
        self.scale2 = 1.0

    def forward(self, x):
        return (self.base(x)
                + self.vida_up(self.vida_down(x)) * self.scale1
                + self.vida_up2(self.vida_down2(x)) * self.scale2)


def inject_trainable_vida(visual_model, r=4, r2=16):
    """Replace attn.out_proj and mlp.c_proj (the two separable nn.Linear
    submodules per ResidualAttentionBlock in this repo's CLIP ViT -- see
    module docstring for why these, not the original's q/k/v split) with
    ViDAInjectedLinear. Returns the list of newly-trainable parameters."""
    require_grad_params = []
    for block in visual_model.transformer.resblocks:
        for parent, attr in [(block.attn, 'out_proj'), (block.mlp, 'c_proj')]:
            base = getattr(parent, attr)
            if isinstance(base, ViDAInjectedLinear):
                continue  # already injected (idempotent)
            injected = ViDAInjectedLinear(base, r=r, r2=r2)
            setattr(parent, attr, injected)
            require_grad_params += list(injected.vida_up.parameters())
            require_grad_params += list(injected.vida_down.parameters())
            require_grad_params += list(injected.vida_up2.parameters())
            require_grad_params += list(injected.vida_down2.parameters())
    for p in require_grad_params:
        p.requires_grad_(True)
    return require_grad_params


def set_vida_scale(model, low, high):
    for m in model.modules():
        if isinstance(m, ViDAInjectedLinear):
            m.scale1 = low
            m.scale2 = high


class ViDAContinual:
    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 catseg_checkpoint=None, maft_checkpoint=None,
                 prompt_dir=None, runtime_calculation=False, device='cpu',
                 r=4, r2=16, ema=0.99, ema_vida=0.8, unc_n=10, unc_thr=0.2):
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps
        self.runtime = runtime_calculation
        self.device = device
        self.r, self.r2 = r, r2
        self.alpha_teacher = ema
        self.alpha_vida = ema_vida
        self.unc_n = unc_n
        self.unc_thr = unc_thr

        if classes is None:
            raise ValueError("classes is required in __init__")
        self.classes = classes

        self.model, self.tokenize = load_ovss(ovss_type, ovss_backbone, device=device,
                                              classes=self.classes, catseg_checkpoint=catseg_checkpoint,
                                              maft_checkpoint=maft_checkpoint)

        if prompt_dir:
            self.prompt_templates = load_prompts_from_yaml(prompt_dir)
        else:
            self.prompt_templates = [REFERENCE_PROMPT]

        # freeze everything (text encoder + whole visual backbone) --
        # ViDA trains ONLY the injected adapters, unlike every other method
        # in this repo (which trains LayerNorm).
        self.model.transformer.requires_grad_(False)
        self.model.ln_final.requires_grad_(False)
        self.model.token_embedding.requires_grad_(False)
        self.model.visual.requires_grad_(False)

        vida_params = inject_trainable_vida(self.model.visual, r=r, r2=r2)
        # newly-injected LoRA Linear layers default to CPU/fp32. Move only
        # those (not the whole model!) to (device, model dtype) -- this repo's
        # CLIP.LayerNorm deliberately keeps its weight/bias in fp32 even
        # though the rest of the model is fp16 (see ovss/clip/model.py's
        # `LayerNorm.forward`), so a blanket `self.model.to(dtype=...)` call
        # would silently corrupt every LayerNorm in the model (text encoder
        # included) by casting it to fp16 too.
        for module in self.model.visual.modules():
            if isinstance(module, ViDAInjectedLinear):
                module.to(device=device, dtype=self.model.dtype)
        print_clip_parameters(self.model)

        # eps=1e-4, not Adam's fp32-tuned default 1e-8: the injected adapters
        # (and this optimizer's running moments) live in fp16 the same as
        # the rest of the model, and fp16's smallest representable normal
        # value is ~6e-5 -- 1e-8 underflows to exactly 0, so
        # `denom = sqrt(v_hat) + eps` can be 0 on early steps (when v_hat
        # itself is still tiny) and produce inf/nan updates that permanently
        # poison the adapter weights within the very first optimizer.step().
        # Verified by direct repro: with eps=1e-8, vida_down/up2 weights go
        # NaN after exactly 1 step even though the loss and HKA scale that
        # step were both finite.
        self.optimizer = optim.Adam(vida_params, lr=self.lr, betas=(0.9, 0.999),
                                     eps=1e-4, weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer)
        self.model_ema = copy.deepcopy(self.model)
        for p in self.model_ema.parameters():
            p.requires_grad_(False)

        self.transform = get_tta_transforms(img_size=224)

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
        self.model_ema = copy.deepcopy(self.model)
        for p in self.model_ema.parameters():
            p.requires_grad_(False)

    def state_dict(self):
        """Adapt-state checkpoint for mid-stream resume: every tensor
        registered in the optimizer (LN + LoRA/ViDA params) plus optimizer
        state, plus the EMA teacher's full weights (drifts independently of
        the optimized student, must be captured separately)."""
        opt_params = [p for g in self.optimizer.param_groups for p in g['params']]
        return {
            'params': [p.detach().cpu().clone() for p in opt_params],
            'optimizer': self.optimizer.state_dict(),
            'model_ema': {k: v.detach().cpu().clone() for k, v in self.model_ema.state_dict().items()},
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
        ema_sd = {k: v.to(self.device) for k, v in sd['model_ema'].items()}
        self.model_ema.load_state_dict(ema_sd, strict=True)

    # ===========================================================
    # Adaptation
    # ===========================================================

    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []
        for _ in range(self.steps):
            self.model_ema.eval()
            # ---- Homeostatic Knowledge Allotment: estimate uncertainty ----
            with torch.no_grad():
                outs = []
                for _n in range(self.unc_n):
                    o, _, _ = self.model_ema(self.transform(x), self.text_x, True, interpolate=False)
                    outs.append(o[0])
                variance = torch.var(torch.stack(outs), dim=0)
                uncertainty = variance.mean().item() * 0.1
                # Deviation from the original (vida/imagenet/vida.py: no clamp
                # there either): CLIP's dense per-pixel logits carry a
                # logit_scale (~100) the original's per-image classification
                # logits don't, so this uncertainty statistic can exceed 1.0
                # here in a way the original setting apparently never hit --
                # unclamped, `lambda_low = 1 - uncertainty` goes negative and
                # *flips the sign* of the domain-shared adapter branch inside
                # the forward pass (not just the loss), which is never what
                # "lean toward the other branch" was meant to do. Clamp to
                # [0, 1] so scale1/scale2 stay in [0, 2] and never flip sign.
                uncertainty = min(uncertainty, 1.0)
            if uncertainty >= self.unc_thr:
                lambda_high, lambda_low = 1 + uncertainty, 1 - uncertainty
            else:
                lambda_low, lambda_high = 1 + uncertainty, 1 - uncertainty
            set_vida_scale(self.model.visual, low=lambda_low, high=lambda_high)
            set_vida_scale(self.model_ema.visual, low=lambda_low, high=lambda_high)

            # ---- consistency loss: student vs EMA teacher ----
            with torch.no_grad():
                teacher_logits, _, _ = self.model_ema(x, self.text_x, True, interpolate=False)
                teacher_logits = teacher_logits[0]
            student_logits, _, _ = self.model(x, self.text_x, True, interpolate=False)
            student_logits = student_logits[0]
            loss = self.symmetric_ce(student_logits, teacher_logits.detach()).mean()
            loss_report.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            # ---- EMA teacher update ----
            self._update_ema()
        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

    @torch.no_grad()
    def _update_ema(self):
        for (name_s, p_s), (name_t, p_t) in zip(
                self.model.named_parameters(), self.model_ema.named_parameters()):
            if 'vida_up' in name_s or 'vida_down' in name_s:
                p_t.data.mul_(self.alpha_vida).add_(p_s.data, alpha=1 - self.alpha_vida)
            else:
                p_t.data.mul_(self.alpha_teacher).add_(p_s.data, alpha=1 - self.alpha_teacher)

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
    def copy_model_and_optimizer(model, optimizer):
        return copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())

    @staticmethod
    def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)

    @staticmethod
    def symmetric_ce(x: torch.Tensor, x_ema: torch.Tensor) -> torch.Tensor:
        # dim=1 is the class/channel dim for our (B, C, H, W) dense logits
        # (the original operates on (B, C) classification logits at dim=1
        # too, so this is the same reduction axis, just with extra spatial
        # dims along for the ride).
        return (-0.5 * (x_ema.softmax(1) * x.log_softmax(1)).sum(1)
                - 0.5 * (x.softmax(1) * x_ema.log_softmax(1)).sum(1))
