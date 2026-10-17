"""
SPAContinual: port of SPA (Self-Bootstrapping for Versatile Test-Time
Adaptation, ICML 2025, https://arxiv.org/pdf/2504.08010,
github.com/mr-eggplant/SPA -> vendored at /home/stanfc/TTA-on-OVSS/spa/).

SPA is architecture- and task-agnostic by design (the paper itself reports
a Cityscapes->ACDC *segmentation* result, so this port stays close to the
original mechanism rather than needing a reinterpretation). Core idea: a
BYOL-style asymmetric self-distillation where a small trainable "predictor"
head sits only on the augmented branch, and the augmentation itself is two
geometry-preserving perturbations (a low-frequency amplitude mask in the
Fourier domain, and a learnable adversarial per-patch erasing) chosen
specifically so dense/structured predictions (segmentation, detection)
survive the augmentation without their spatial layout being destroyed --
unlike classification-style crops/color-jitter.

Core mechanism (see spa/tta_library/spa.py + spa/models/{byol_wrapper,
maskers}.py for the original):
  1. Predictor: a single trainable Linear (init'd to identity, matching
     BYOL's predictor head) applied only to the augmented branch's image
     features, right before the cosine-similarity-with-text step.
  2. Two augmentations, applied independently (each contributes its own
     consistency-loss term):
       a. Low-frequency amplitude mask: FFT the image, randomly zero out
          a fraction (`freq_mask_ratio`) of the *low*-frequency amplitude
          components (keeping phase and all high-frequency content
          untouched), IFFT back. No learnable parameters.
       b. Learnable adversarial patch erasing: per-patch scalar `alpha`
          blends each patch toward Gaussian noise; trained with a
          *reversed* gradient (so the adapt step's own backward pass
          pushes alpha to make erasure *harder* to be consistent with,
          not easier), then variance-projected back to `noise_ratio` and
          clipped -- this is a separate tiny SGD optimizer, exactly as
          upstream.
  3. Consistency loss (both augmentations): only pixels/tokens where the
     *clean* branch is more confident than the *augmented* branch
     contribute -- KL(aug_branch.log_softmax, clean_branch.softmax.detach())
     -- i.e. clean predictions (when better) are distilled into the
     harder augmented view, exactly like BYOL/self-distillation.
  4. What's trained: LayerNorm (gamma, beta) of the visual encoder (this
     repo's convention -- upstream trains BatchNorm/GroupNorm/LayerNorm
     affine params depending on backbone, and NA-CLIP ViT only has
     LayerNorm) + the predictor Linear, via one Adam optimizer; the patch
     eraser's `alpha` via its own SGD optimizer (matches upstream exactly
     -- this is an input-augmentation parameter, not a model weight, and
     is deliberately optimized adversarially rather than to minimize the
     task loss).
  5. No entropy-minimization auxiliary term: upstream's SPA+tent/SPA+eata
     variants add an auxiliary entropy loss on the clean branch; those are
     skipped here (vanilla SPA only) since this repo's CLAUDE.md documents
     entropy minimization as collapsing to a trivial solution under our
     continual/no-reset protocol. Vanilla SPA is also the config upstream
     reports as a standalone number (Table 1: "SPA (Ours)"), not just an
     ablation.

Deviations from the original (documented, not the paper's own choices):
  - `patch_size` for the learnable patch eraser is 14 (matching NA-CLIP
    ViT-L/14's actual patch grid -- 16x16=256 patches on our 224x224 input
    tiles), not the original's hardcoded 16 (tuned for ViT-Base/16
    classification backbones). This keeps each erased patch aligned with
    one real visual token, which is the point of the mechanism.
  - The predictor sits on CLIP's L2-normalized per-patch image features
    (there is no separate "pooled feature -> head" split in a dense OVSS
    model the way there is in a classifier); the predictor's output is
    renormalized before the cosine-similarity-with-text step. Since the
    predictor is initialized to the identity, this is a no-op until it
    trains, exactly matching the original's zero-effect-at-init property.
  - Dense (B, C, H, W) logits replace (B, C) classification logits
    throughout (channel dim is dim=1); the confidence-gating and KL loss
    are computed per-pixel instead of per-image, everything else
    (predictor init, both augmentations, the reversed-gradient patch
    eraser, the confidence mask direction) is unchanged.
  - The `imagenet_mask` class-remapping in the original (for evaluating a
    1000-class classifier on a restofimagenet subset) has no equivalent
    here -- our `classes` list is already the exact target vocabulary.
  - Adam `eps=1e-4` instead of PyTorch's fp32-tuned default `1e-8`: the
    predictor + LayerNorm params here run natively in fp16 (no AMP loss
    scaling), and 1e-8 underflows to exactly 0 in fp16 -- this caused a
    real NaN-within-one-step failure when porting ViDA (adapt/vida_continual.py)
    for the identical reason, so it's applied preemptively here too.
"""
import time
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class FrequencyMasker:
    """Low-frequency-only amplitude masking in the Fourier domain. No
    learnable parameters. Verbatim mechanism from spa/tta_library/spa.py's
    `Prompt` class (renamed here since "prompt" would collide with this
    repo's text-prompt-template terminology)."""

    def __init__(self, img_size=224, keep_prob=0.8, device='cpu', dtype=torch.float32):
        promptH = promptW = max(1, int(img_size * 0.2))
        paddingH = (img_size - promptH) // 2
        paddingW = (img_size - promptW) // 2
        init_para = torch.ones((1, 3, promptH, promptW), device=device, dtype=torch.float32)
        self.low_mask = F.pad(
            init_para, [img_size - paddingH - promptH, paddingH, img_size - paddingW - promptW, paddingW],
            mode='constant', value=0.).contiguous()
        self.high_mask = 1 - self.low_mask
        self.keep_prob = keep_prob

    def __call__(self, x):
        orig_dtype = x.dtype
        x = x.float()  # FFT ops need fp32 precision, not fp16
        _, _, imgH, imgW = x.size()
        fft = torch.fft.fft2(x.clone(), dim=(-2, -1))
        amp_src, pha_src = torch.abs(fft), torch.angle(fft)
        amp_src = torch.fft.fftshift(amp_src)
        mask = self.low_mask * torch.empty_like(amp_src).bernoulli_(self.keep_prob)
        mask = mask + self.high_mask
        mask = self._symmetrize(mask)
        amp_src = amp_src * mask
        amp_src = torch.fft.ifftshift(amp_src)
        real = torch.cos(pha_src) * amp_src
        imag = torch.sin(pha_src) * amp_src
        fft_out = torch.complex(real=real, imag=imag)
        out = torch.fft.ifft2(fft_out, dim=(-2, -1), s=[imgH, imgW]).real
        return out.to(orig_dtype)

    @staticmethod
    def _symmetrize(mask):
        startH = 1 - (mask.shape[-2] % 2)
        startW = 1 - (mask.shape[-1] % 2)
        sub_mask = mask[:, :, startH:, startW:]
        maskH, maskW = sub_mask.shape[-2] // 2, sub_mask.shape[-1] // 2
        sub_mask[:, :, -maskH:, :] = 1
        sub_mask[:, :, maskH, :maskW] = 1
        sub_mask = sub_mask * torch.flip(sub_mask, dims=[-1, -2])
        mask[:, :, startH:, startW:] = sub_mask
        return mask


class LearnablePatchErasing(nn.Module):
    """Per-patch learnable blend toward Gaussian noise; alpha is trained
    with a *reversed* gradient by the caller (see SPAContinual.perform_
    adaptation) then variance-projected back to noise_ratio each step.
    Verbatim mechanism from spa/models/maskers.py, generalized to a
    configurable `patch_size` (see module docstring for why 14, not 16)."""

    def __init__(self, patch_size=14, img_size=224, noise_ratio=0.4):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.noise_ratio = noise_ratio
        h = w = img_size // patch_size
        self.alpha = nn.Parameter(torch.ones(h * w) * math.sqrt(noise_ratio))

    def forward(self, img):
        assert img.shape[-1] == img.shape[-2] and img.shape[-1] % self.patch_size == 0
        img_c = img.shape[-3]
        h = w = img.shape[-1] // self.patch_size
        new_img = torch.empty_like(img)
        idx = 0
        for i in range(h):
            for j in range(w):
                y0, x0 = i * self.patch_size, j * self.patch_size
                v = torch.empty([img_c, self.patch_size, self.patch_size],
                                 dtype=img.dtype, device=img.device).normal_()
                a2 = self.alpha[idx] ** 2
                new_img[..., y0:y0 + self.patch_size, x0:x0 + self.patch_size] = (
                    (1 - a2) * img[..., y0:y0 + self.patch_size, x0:x0 + self.patch_size] + a2 * v)
                idx += 1
        new_img.clip_(-3, 3)  # CLIP-normalized images, not [-1,1] raw pixels (see docstring deviation)
        return new_img

    @torch.no_grad()
    def project(self):
        current_mean = (self.alpha ** 2).mean()
        div = current_mean / self.noise_ratio
        self.alpha /= torch.sqrt(div)
        torch.clip_(self.alpha, -1, 1)


class SPAContinual:
    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 catseg_checkpoint=None, maft_checkpoint=None,
                 prompt_dir=None, runtime_calculation=False, device='cpu',
                 noise_ratio=0.4, freq_mask_ratio=0.2, eraser_patch_size=14):
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps
        self.runtime = runtime_calculation
        self.device = device

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

        # ---------- Freeze text encoder, enable LN-only grads on visual ----------
        self.model.transformer.requires_grad_(False)
        self.model.ln_final.requires_grad_(False)
        self.model.token_embedding.requires_grad_(False)
        self.model.visual = self.set_ln_grads(self.model.visual)
        ln_params, _ = self.collect_ln_params(self.model.visual)

        # ---------- Predictor (BYOL-style, init'd to identity) ----------
        embed_dim = self.model.visual.output_dim
        self.predictor = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.eye_(self.predictor.weight)
        self.predictor.to(device=device, dtype=self.model.dtype)  # see vida_continual.py's device/dtype note

        # ---------- Augmentations ----------
        self.freq_mask = FrequencyMasker(img_size=224, keep_prob=1 - freq_mask_ratio,
                                          device=device)
        self.patch_eraser = LearnablePatchErasing(patch_size=eraser_patch_size, img_size=224,
                                                   noise_ratio=noise_ratio)
        self.patch_eraser.to(device=device, dtype=self.model.dtype)

        print_clip_parameters(self.model)

        # ---------- Optimizers ----------
        # eps=1e-4: see module docstring -- fp16 Adam eps=1e-8 underflows to 0.
        self.optimizer = optim.Adam(list(ln_params) + list(self.predictor.parameters()),
                                     lr=self.lr, betas=(0.9, 0.999), eps=1e-4, weight_decay=0.0)
        self.patch_optimizer = optim.SGD([self.patch_eraser.alpha], lr=1.0, momentum=0.9)
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
        with torch.no_grad():
            self.patch_eraser.alpha.data = torch.ones_like(self.patch_eraser.alpha) * \
                math.sqrt(self.patch_eraser.noise_ratio)
        self.patch_optimizer = optim.SGD([self.patch_eraser.alpha], lr=1.0, momentum=0.9)

    def state_dict(self):
        """Adapt-state checkpoint for mid-stream resume: every tensor in the
        main optimizer (LN + predictor) plus its state, plus patch_eraser's
        learnable alpha and its OWN separate optimizer (not part of the main
        one -- see reset())."""
        opt_params = [p for g in self.optimizer.param_groups for p in g['params']]
        return {
            'params': [p.detach().cpu().clone() for p in opt_params],
            'optimizer': self.optimizer.state_dict(),
            'patch_eraser': {k: v.detach().cpu().clone()
                             for k, v in self.patch_eraser.state_dict().items()},
            'patch_optimizer': self.patch_optimizer.state_dict(),
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
        eraser_sd = {k: v.to(self.device) for k, v in sd['patch_eraser'].items()}
        self.patch_eraser.load_state_dict(eraser_sd, strict=True)
        self.patch_optimizer.load_state_dict(sd['patch_optimizer'])
        for state in self.patch_optimizer.state.values():
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
            logits, image_features, text_features = self.model(x, self.text_x, True, interpolate=False)
            logits = logits[0]  # (B, C, w, h), single prompt template
            clean_conf = logits.softmax(dim=1).max(dim=1)[0]  # (B, w, h)

            loss = 0.0
            aug1 = self.freq_mask(x)
            low_logits1 = self._predictor_branch(aug1, text_features, x.shape)
            loss = loss + self._consistency_loss(logits, clean_conf, low_logits1)

            aug2 = self.patch_eraser(x)
            low_logits2 = self._predictor_branch(aug2, text_features, x.shape)
            loss = loss + self._consistency_loss(logits, clean_conf, low_logits2)

            self.optimizer.zero_grad()
            self.patch_optimizer.zero_grad()
            loss.backward()
            # Bug found on a real 150-round V20 run (crashed ~round 70):
            # `_consistency_loss` returns a `clean_logits.sum() * 0.0` stub when
            # no pixel satisfies `clean_conf > aug_conf` -- that stub doesn't
            # depend on aug_logits at all, so when this happens for the
            # patch-eraser branch specifically, the graph never reaches
            # `alpha` and `alpha.grad` stays None after backward(). Upstream
            # (spa/tta_library/spa.py) never hits this because classification
            # batches almost always have *some* sample where the mask fires;
            # our dense, batch_size=1 setting can legitimately have zero
            # qualifying pixels for one augmentation branch on a given step.
            if self.patch_eraser.alpha.grad is not None:
                self.patch_eraser.alpha.grad.neg_()  # adversarial reversal, matches upstream
                self.patch_optimizer.step()
            self.optimizer.step()
            self.patch_eraser.project()
            loss_report.append(loss.item())
        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

    def _predictor_branch(self, x_aug, text_features, image_shape):
        _, image_features_aug, _ = self.model(x_aug, self.text_x, True, interpolate=False)
        projected = self.predictor(image_features_aug)
        projected = projected / projected.norm(dim=-1, keepdim=True)
        return self._logits_from_image_features(projected, text_features, image_shape)[0]

    def _logits_from_image_features(self, image_features, text_features, image_shape):
        """Replicates ovss/clip/model.py CLIP.forward()'s post-encode_image
        logic (cosine-sim-with-text -> patch-grid reshape), so the predictor
        branch can reuse the exact same convention without a second full
        forward through encode_image."""
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * torch.einsum('bsd,tcd->tbsc', image_features, text_features)
        logits = logits[:, :, 1:]  # drop CLS token
        patch_size = self.model.visual.patch_size
        w, h = image_shape[-2] // patch_size, image_shape[-1] // patch_size
        T, B, S, C = logits.shape
        logits = logits.permute(0, 1, 3, 2).reshape(T, B, C, w, h)
        return logits

    @staticmethod
    def _consistency_loss(clean_logits, clean_conf, aug_logits):
        aug_conf = aug_logits.softmax(dim=1).max(dim=1)[0]
        mask = clean_conf > aug_conf  # (B, w, h)
        if mask.sum() == 0:
            return clean_logits.sum() * 0.0
        clean_flat = clean_logits.permute(0, 2, 3, 1)[mask]  # (N, C)
        aug_flat = aug_logits.permute(0, 2, 3, 1)[mask]      # (N, C)
        return F.kl_div(aug_flat.log_softmax(dim=-1),
                         clean_flat.softmax(dim=-1).detach(),
                         reduction='batchmean')

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
