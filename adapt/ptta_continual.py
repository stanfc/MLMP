"""
PTTAContinual: port of PTTA (Purifying Malicious Samples for Test-Time
Model Adaptation, ICML 2025, https://openreview.net/forum?id=SznX4yic20,
github.com/HAIV-Lab/PTTA -> vendored at /home/stanfc/TTA-on-OVSS/ptta/).

Core idea: rather than discarding samples an EATA-style filter judges
"malicious" (unreliable/redundant), PTTA still keeps a memory bank of past
*good* (reliable, non-redundant) samples, and for every good sample it
also Mixup-blends it with the bank entry whose loss-landscape effect is
most *opposite* to the current sample's -- turning would-be-wasted
information into a purification/consistency signal instead.

Core mechanism (see ptta/utils/ptta_eata.py for the original, which is
literally EATA + this purification add-on -- confirmed from
ptta/main_imagenet1k.py: the headline `ptta.PTTA(...)` call never passes
`fishers`, so the reported numbers use the filter + purification only,
no EWC. This port matches that -- no Fisher/EWC term):
  1. EATA-style filtering (verbatim from this repo's own, never-yet-run
     adapt/eata_continual.py, reused here rather than re-derived):
       - Reliable: mean pixel entropy < e_margin.
       - Non-redundant: cosine(mean per-sample *softmax probability*,
         EMA of recent mean probabilities) < 1 - d_margin -- matches
         ptta/utils/ptta_eata.py's own `outputs.softmax(1)` comparison
         exactly (see the deviation note below for why this must be
         probabilities, not raw logits).
  2. Saliency key (closed-form, cheap): d(entropy)/d(logits) at the
     *output* layer -- NOT a backprop through the whole network, just the
     analytic entropy gradient w.r.t. the model's own logits, spatially
     pooled to one (num_classes,)-vector per sample. This is the
     "malicious vs benign direction" descriptor the bank is indexed by.
  3. Memory bank: every reliable+non-redundant sample's (saliency key,
     raw image, per-pixel softmax probs) is pushed into a fixed-size
     ring buffer (`queue_size`, default 1000, matching upstream).
  4. Purification: retrieve the `neighbor` bank entries whose saliency
     key has the *lowest* cosine similarity (most opposite effect) to
     the current sample's key, Mixup-blend the raw images
     (`alpha = 1/(neighbor+1)`) and the per-pixel probs the same way,
     forward the mixed image, and add
     `loss2_weight * KL(mixed_output.log_softmax, mixed_target.detach())`
     to the base (coefficient-reweighted) entropy loss.
  5. What's trained: LayerNorm (gamma, beta) of the visual encoder only
     -- upstream's `collect_params`/`configure_model` already target
     BatchNorm2d/GroupNorm/LayerNorm affine params generically, so no
     BatchNorm-specific incompatibility here.

Deviations from the original (documented, not the paper's own choices):
  - `e_margin=1.198` (this repo's own eata_continual.py's already-tuned
    value for our ~19-20 class OVSS setting) replaces upstream's
    `math.log(1000)*0.4 ~= 2.76` (tuned for 1000-way ImageNet). `d_margin`
    is left at upstream's own default (0.05) since it doesn't depend on
    class count.
  - Dense (B, C, H, W) logits/probs replace (B, C) classification
    logits/probs throughout; the saliency key is the *spatially pooled*
    entropy-gradient (mean over the patch grid) rather than a single
    per-image gradient, and the Mixup target is a *per-pixel* probability
    map (matching the paper's own per-image Mixup, generalized the same
    way standard image Mixup generalizes to dense-prediction Mixup:
    blend inputs and per-pixel targets at the same spatial locations).
    Bank images/probs are stored at the ViT patch-grid resolution (e.g.
    16x16 for ViT-L/14 on a 224 tile), not full pixel resolution, to keep
    the bank small.
  - Assumes `batch_size=1` per adapt step (every continual method's bash
    script in this repo already uses batch_size=1); if a batch ever has
    more than one sample, only the first sample of the batch is used for
    the bank key/update and the Mixup partner (a disclosed simplification,
    not expected to matter given repo convention).
  - Bug found and fixed during a real 150-round ACDC run (mIoU frozen
    bit-identical for 3+ consecutive rounds): the non-redundant filter's
    per-sample descriptor was originally mean-pooled *raw logits*
    (matching this repo's own eata_continual.py, which -- per
    MLMP/CLAUDE.md -- was itself "implemented, never run" and so never
    caught this). Direct repro on real ACDC-fog data: mean cosine
    similarity between consecutive frames' pooled raw-logit vectors was
    ~0.997 (d_margin=0.05 requires < 0.95), so exactly 1 sample was ever
    accepted as non-redundant across 3 full rounds (300 frames) -- PTTA
    silently became a permanent no-op after its first update. Pooling
    *softmax probabilities* instead (matching the original paper's own
    `outputs.softmax(1)` comparison, see mechanism section above) sharpens
    the distribution enough before pooling to restore real discriminative
    power; `d_margin` is otherwise left at the paper's own default.
  - Second bug found on the same real run, after the first fix above
    stopped the freeze but mIoU still declined steadily with no early-gain
    phase: `F.kl_div(..., reduction='batchmean')` on the dense
    (1, C, gh, gw) mixed-sample tensor divides only by dim-0 (=1, our
    batch_size), not by the gh*gw pixels also present -- unlike the
    paper's (B, C) classification tensors, where batchmean's divisor is
    exactly right. This inflated the purification loss (loss2) by a
    factor of gh*gw (256 for a 224-tile ViT-L/14 grid) relative to the
    base entropy loss, so it dominated almost every step once the bank
    held any entries. Fixed by computing the per-pixel KL explicitly
    (sum over channels, mean over pixels) -- see perform_adaptation.
  - No Fisher/EWC term (see above -- matches the paper's own reported
    "PTTA" configuration, not an omission).
  - Adam `eps=1e-4` instead of PyTorch's fp32-tuned default `1e-8`: see
    adapt/vida_continual.py's docstring for why (fp16 params + no AMP loss
    scaling -> 1e-8 underflows to 0 and caused a real NaN-in-one-step bug
    there; applied preemptively here for the same reason).
"""
import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class SaliencyBank:
    """Fixed-size ring buffer of (saliency key, raw image, per-pixel probs)
    triples, retrievable by *most dissimilar* (opposite-effect) key.
    Simplified from ptta/utils/ptta_eata.py's FeatureBank for this repo's
    batch_size=1 convention (no need for the original's chunked/batched
    retrieval)."""

    def __init__(self, queue_size, num_classes, grid_hw, neighbor=1,
                 device='cpu', dtype=torch.float16):
        self.queue_size = queue_size
        self.neighbor = neighbor
        self.keys = torch.zeros(queue_size, num_classes, device=device, dtype=torch.float32)
        self.images = torch.zeros(queue_size, 3, 224, 224, device=device, dtype=dtype)
        gh, gw = grid_hw
        self.probs = torch.zeros(queue_size, num_classes, gh, gw, device=device, dtype=dtype)
        self.ptr = 0
        self.n_stored = 0

    @torch.no_grad()
    def update(self, key, image, probs):
        idx = self.ptr % self.queue_size
        self.keys[idx] = key.detach().float()
        self.images[idx] = image.detach()
        self.probs[idx] = probs.detach()
        self.ptr += 1
        self.n_stored = min(self.n_stored + 1, self.queue_size)

    @torch.no_grad()
    def retrieve_opposite(self, key):
        if self.n_stored == 0:
            return None, None
        bank_keys = self.keys[:self.n_stored]
        sim = F.cosine_similarity(key.float().unsqueeze(0), bank_keys, dim=1)  # (n_stored,)
        k = min(self.neighbor, self.n_stored)
        _, idxs = torch.topk(-sim, k)  # smallest cosine sim = most opposite
        return self.images[idxs].mean(dim=0), self.probs[idxs].mean(dim=0)

    def reset(self):
        self.ptr = 0
        self.n_stored = 0


class PTTAContinual:
    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 catseg_checkpoint=None, maft_checkpoint=None,
                 prompt_dir=None, runtime_calculation=False, device='cpu',
                 e_margin=1.198, d_margin=0.05, loss2_weight=3.0,
                 queue_size=1000, neighbor=1):
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps
        self.runtime = runtime_calculation
        self.device = device

        self.e_margin = float(e_margin)
        self.d_margin = float(d_margin)
        self.loss2_weight = float(loss2_weight)
        self.alpha = 1.0 / (neighbor + 1)

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

        print_clip_parameters(self.model)

        # eps=1e-4: see module docstring.
        self.optimizer = optim.Adam(ln_params, lr=self.lr, betas=(0.9, 0.999),
                                     eps=1e-4, weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer)

        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(
                self.classes, self.prompt_templates, average=False
            ).squeeze()

        patch_size = self.model.visual.patch_size
        grid_hw = (224 // patch_size, 224 // patch_size)
        self.bank = SaliencyBank(queue_size, len(self.classes), grid_hw,
                                  neighbor=neighbor, device=device, dtype=self.model.dtype)
        self.current_logit_ema = None
        self.logit_ema_decay = 0.9

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
        self.bank.reset()
        self.current_logit_ema = None

    def state_dict(self):
        """Adapt-state checkpoint for mid-stream resume: every tensor in the
        optimizer (LN) plus its state, plus the saliency bank's raw tensors
        (keys/images/probs/ptr/n_stored -- a plain Python object, not an
        nn.Module, so nothing here is gradient-tracked or otherwise captured
        elsewhere) and the running logit EMA."""
        opt_params = [p for g in self.optimizer.param_groups for p in g['params']]
        return {
            'params': [p.detach().cpu().clone() for p in opt_params],
            'optimizer': self.optimizer.state_dict(),
            'bank_keys': self.bank.keys.detach().cpu().clone(),
            'bank_images': self.bank.images.detach().cpu().clone(),
            'bank_probs': self.bank.probs.detach().cpu().clone(),
            'bank_ptr': self.bank.ptr,
            'bank_n_stored': self.bank.n_stored,
            'current_logit_ema': (self.current_logit_ema.detach().cpu().clone()
                                  if self.current_logit_ema is not None else None),
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
        self.bank.keys = sd['bank_keys'].to(self.bank.keys.device)
        self.bank.images = sd['bank_images'].to(self.bank.images.device)
        self.bank.probs = sd['bank_probs'].to(self.bank.probs.device)
        self.bank.ptr = sd['bank_ptr']
        self.bank.n_stored = sd['bank_n_stored']
        self.current_logit_ema = (sd['current_logit_ema'].to(self.device)
                                  if sd['current_logit_ema'] is not None else None)

    # ===========================================================
    # Adaptation
    # ===========================================================

    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []
        for _ in range(self.steps):
            logits, _, _ = self.model(x, self.text_x, True, interpolate=False)
            l0 = logits[0]  # (B, C, gh, gw)
            ent_map = self.softmax_entropy(l0)  # (B, gh, gw)
            ent_mean = ent_map.mean()

            if ent_mean.item() > self.e_margin:
                self.optimizer.zero_grad()
                continue

            with torch.no_grad():
                # Softmax probabilities, not raw logits -- matches the
                # original PTTA/EATA's own redundancy check exactly
                # (ptta/utils/ptta_eata.py compares `outputs.softmax(1)`
                # vectors, not logits). Raw-logit cosine similarity was
                # verified by direct repro to collapse to ~0.997 for nearly
                # every ACDC frame within one weather condition (mean-pooling
                # raw logits over the whole patch grid washes out almost all
                # frame-to-frame variation), causing PTTA to accept its first
                # sample and then never again for the rest of a 150-round
                # run. Softmax sharpens the distribution before pooling,
                # restoring the discriminative power the paper relies on.
                mean_prob = l0[0].softmax(dim=0).mean(dim=[1, 2])  # (C,)
                mean_prob_n = mean_prob / mean_prob.norm().clamp(min=1e-8)
                if self.current_logit_ema is None:
                    non_redundant = True
                else:
                    cos_sim = (mean_prob_n * self.current_logit_ema).sum().item()
                    non_redundant = cos_sim < 1.0 - self.d_margin

            if not non_redundant:
                self._update_logit_ema(mean_prob_n)
                self.optimizer.zero_grad()
                continue

            # ----- saliency key: d(entropy)/d(logits), spatially pooled -----
            l0_det = l0.detach().clone().requires_grad_(True)
            ent_for_grad = self.softmax_entropy(l0_det).sum()
            grad = torch.autograd.grad(ent_for_grad, l0_det)[0]  # (B, C, gh, gw)
            key = grad[0].mean(dim=[1, 2])  # (C,)

            # ----- base loss: EATA's confidence-reweighted entropy -----
            coeff = 1.0 / torch.exp(ent_map.detach() - self.e_margin)
            loss = (ent_map * coeff).mean()

            # ----- purification: Mixup with the most-opposite bank entry -----
            if self.bank.n_stored > 0:
                bank_img, bank_probs = self.bank.retrieve_opposite(key)
                mixed_x = x[:1] * self.alpha + bank_img.unsqueeze(0) * (1 - self.alpha)
                clean_probs = l0[:1].softmax(dim=1)
                mixed_target = clean_probs * self.alpha + bank_probs.unsqueeze(0) * (1 - self.alpha)
                logits2, _, _ = self.model(mixed_x, self.text_x, True, interpolate=False)
                l2 = logits2[0]
                # Bug found on a real 150-round run (steady mIoU decline from
                # Round 1, no early-gain phase): F.kl_div's reduction='batchmean'
                # divides only by dim-0 (=1 here, our batch_size), NOT by the
                # gh*gw spatial pixels also present in this (1, C, gh, gw)
                # tensor -- unlike the paper's own (B, C) classification
                # tensors, where batchmean's divisor is exactly right. That
                # silently inflated loss2 by a factor of gh*gw (256 for a
                # 224-tile ViT-L/14 grid), so a single mixed sample's
                # purification term swamped the base entropy loss by ~2-3
                # orders of magnitude on every step once the bank held any
                # entries. Computing the per-pixel KL explicitly (sum over
                # the channel dim, mean over pixels) restores the intended
                # scale relative to the base loss.
                kl_per_pixel = (mixed_target.detach()
                                * (mixed_target.detach().clamp(min=1e-8).log() - l2.log_softmax(dim=1))
                                ).sum(dim=1)  # (1, gh, gw)
                loss2 = kl_per_pixel.mean() * self.loss2_weight
                loss = loss + loss2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            with torch.no_grad():
                self.bank.update(key, x[0], l0[0].softmax(dim=0))
                self._update_logit_ema(mean_prob_n)
            loss_report.append(loss.item())

        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

    @torch.no_grad()
    def _update_logit_ema(self, mean_logit_n):
        if self.current_logit_ema is None:
            self.current_logit_ema = mean_logit_n.detach().clone()
        else:
            self.current_logit_ema.mul_(self.logit_ema_decay).add_(
                mean_logit_n.detach() * (1.0 - self.logit_ema_decay))
            self.current_logit_ema.div_(self.current_logit_ema.norm().clamp(min=1e-8))

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
