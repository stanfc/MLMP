"""KFF (Class-aware Domain Knowledge Fusion and Fission) for OVSS.

Faithful port of the reference implementation at
  https://github.com/zhoujiahuan1991/NeurIPS2025-KFF
  (paper: NeurIPS 2025, arXiv:2510.12150)

Reference files adapted:
  - ours.py   -> this file's `PromptBase` and `KFF` class.
  - vpt.py    -> this file's `cls_prompt_gen`; prompt / cls_prompt injection is
                 implemented as an opt-in path inside `adapt/prompt_vit.py`.

Non-trivial OVSS bridges (vs KFF's ImageNet-C classification setting):

  (a) `output` signature: KFF uses `(B, num_classes)` softmax from the ViT
      classification head. OVSS NA-CLIP produces per-pixel logits of shape
      `(B, num_classes, H, W)`. We aggregate to image-level by spatial mean
      of the softmax, i.e. the per-image class-presence distribution. This is
      the minimal-invasive mapping -- the class signature still lives in
      R^{num_classes} and behaves as "what classes are present in this image".

  (b) Entropy loss: in KFF it is per-sample entropy averaged over the batch.
      We compute per-pixel entropy averaged over (pixels, batch) -- the
      natural generalisation for dense prediction. Coefficient 3 is kept.

  (c) `forward_head`: NA-CLIP has no classification head; we use the standard
      MLMP pipeline `model(images, text_x, ...)` to obtain logits, and take
      `logits[0]` (index 0 out of the `(text_templates+1, ...)` stack corresponds
      to the averaged template -- the one DPCore's `obtain_src_stat` also uses
      via `self.text_x[-1]`; we pass `text_x[-1]` so the ensemble is applied).

All other KFF mechanisms (PromptBase delete-on-overflow via pairwise merge,
EMA updates with softmax weights, cls_prompt_gen fission/fusion clustering,
two separate optimisers for domain prompts vs cls prompts, `lamda` in the
distribution loss, `tau` temperature for softmax weighting, Xavier-uniform
init with VPT's val) are copied verbatim from the reference.
"""

from copy import deepcopy
import math

import torch
import torch.nn as nn
import torch.jit
import torch.optim as optim

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters


REFERENCE_PROMPT = 'a photo of a {}'


# ---------------------------------------------------------------------------
# PromptBase -- verbatim port of ours.py::PromptBase (domain-level prompt
# memory with L2-distance matching, softmax weighting, EMA update, and
# pairwise-merge eviction when capacity exceeded).
# ---------------------------------------------------------------------------

class PromptBase:
    def __init__(self, max_len=20, ema_alpha=0.1, tau=3.0, thr_d=25.):
        self.set = []
        self.num = 0
        self.id = 0
        self.ema_alpha = ema_alpha
        self.tau = tau
        self.max_len = max_len
        self.thr = thr_d

    def __len__(self):
        return self.num

    def delete(self):
        # Find the pair of prompts with the minimum distance and merge them.
        min_distance = float('inf')
        min_pair = None
        for i in range(len(self.set)):
            for j in range(len(self.set)):
                if i != j:
                    distance = torch.norm(self.set[i][0] - self.set[j][0], p=2)
                    if distance < min_distance:
                        min_distance = distance
                        min_pair = (i, j)
        assert min_pair is not None, 'No pair found to delete'
        i, j = min_pair
        max_id = max(self.set[i][3], self.set[j][3])
        self.set[i][0] = (self.set[i][0] + self.set[j][0]) / 2
        self.set[i][1] = (self.set[i][1] + self.set[j][1]) / 2
        for item in self.set[j][2]:
            if item not in self.set[i][2]:
                self.set[i][2].append(item)
        self.set[i][3] = max_id
        self.set.pop(j)
        self.num -= 1

    def add_prompt(self, key, prompt, note):
        self.id += 1
        self.set.append([key, prompt, note, self.id])
        self.num += 1
        while len(self.set) > self.max_len:
            self.delete()

    def get_weighted_prompts(self, key):
        """Return weighted sum of stored prompts and an ID/OOD flag.

        ID iff the nearest stored key is within self.thr (L2) of the query key.
        Weights are softmax(-distance / tau), masked to entries within threshold.
        """
        key_tensor = torch.stack([p[0] for p in self.set])
        is_ID = True
        key_match = torch.norm(key - key_tensor, p=2, dim=1)
        # Debug: print actual L2 distances vs threshold so we can see why
        # ID/OOD decisions go the way they do.
        if getattr(self, 'verbose', False):
            min_d = key_match.min().item()
            print(f"  [PromptBase] thr={self.thr:.3f}  min_dist={min_d:.4f}  "
                  f"all_dists={[f'{d:.3f}' for d in key_match.tolist()]}")
        key_match_masked = torch.where(key_match <= self.thr, key_match,
                                       torch.tensor(float('inf'), dtype=key_match.dtype,
                                                    device=key_match.device))
        if torch.min(key_match_masked).item() > self.thr:
            is_ID = False
        weights = torch.nn.functional.softmax(-key_match_masked / self.tau, dim=0).detach().cpu()
        weighted_prompts = torch.stack([w * p[1] for w, p in zip(weights, self.set)], dim=0).sum(dim=0)
        assert weighted_prompts.shape == self.set[0][1].shape
        return weighted_prompts, weights, is_ID

    def update(self, weights, prompt, key=None):
        for p_idx in range(len(self.set)):
            if key is not None:
                self.set[p_idx][0] += weights[p_idx] * (key - self.set[p_idx][0]) * self.ema_alpha
            self.set[p_idx][1] += weights[p_idx] * (prompt - self.set[p_idx][1])


# ---------------------------------------------------------------------------
# cls_prompt_gen -- verbatim port of vpt.py::cls_prompt_gen. Manages a set of
# (class_distribution, prompt) pairs with cosine-similarity retrieval,
# per-sample confidence-gated update, and capacity fission/fusion clustering.
# ---------------------------------------------------------------------------

class cls_prompt_gen:
    def __init__(self, val, dim=768, num=100, thr_c=0.005, thr_ent=2.0, alpha_c=0.1):
        self.max_num = num
        self.num = 0
        self.set = []
        self.record = []
        self.val = val
        self.dim = dim
        self.id = 0
        self.length = 1
        self.thr = thr_c
        self.thr_e = thr_ent
        self.alpha_c = alpha_c

    def get_cls_prompt(self, output):
        """Return a `(B, length, dim)` tensor: one prompt per sample.

        For each sample, cosine-match its softmax distribution against stored
        class signatures; if max similarity > thr, use weighted sum of matched
        prompts, else produce a fresh random prompt and mark for insertion.
        """
        output = output.softmax(dim=-1)
        res = []
        for i in range(output.shape[0]):
            p = None
            if self.num == 0:
                # Zero-init cls_prompt (departs from KFF's xavier-uniform init).
                # Rationale: MLMP's evaluate-before-adapt protocol attaches this
                # cls_prompt to the model BEFORE adapt() runs. Random init would
                # corrupt evaluate(); zero init keeps evaluate ≈ no-adapt baseline
                # until adapt() actually trains the prompt.
                p = nn.Parameter(torch.zeros(1, self.length, self.dim))
                self.record.append([-1])
            else:
                key_tensor = torch.stack([self.set[j][0].squeeze(0) for j in range(len(self.set))], dim=0)
                key_match = torch.cosine_similarity(key_tensor, output[i], dim=-1)
                key_match = torch.where(key_match > self.thr, key_match,
                                        torch.tensor(float('-inf'), dtype=key_match.dtype,
                                                     device=key_match.device))
                max_val = torch.max(key_match)
                if max_val < self.thr:
                    p = nn.Parameter(torch.zeros(1, self.length, self.dim))
                    nn.init.uniform_(p.data, -self.val, self.val)
                    self.record.append([-1])
                else:
                    weights = torch.softmax(key_match, dim=-1).detach().cpu()
                    p = torch.stack([w * s[1] for w, s in zip(weights, self.set)], dim=0).sum(dim=0)
                    self.record.append(weights)
            res.append(p)
        return nn.Parameter(torch.cat(res, dim=0))

    def update_cls_prompt(self, output, prompts):
        """Confidence-gated update of the class prompt set."""
        ent = _softmax_entropy_1d(output)  # (B,)
        output = output.softmax(dim=-1)
        assert len(self.record) == output.shape[0]
        for i in range(len(self.record)):
            if ent[i] > self.thr_e:
                continue
            if self.record[i][0] == -1:
                self.set.append([output[i].detach().cpu(), prompts[i].unsqueeze(0).detach().cpu(), self.id])
                self.num += 1
                self.id += 1
            else:
                for idx in range(len(self.record[i])):
                    w = self.record[i][idx]
                    self.set[idx][0] = self.set[idx][0] + (output[i].detach().cpu() - self.set[idx][0]) * w * self.alpha_c
                    self.set[idx][1] = self.set[idx][1] + (prompts[i].unsqueeze(0).detach().cpu() - self.set[idx][1]) * w
        self.record = []
        self.resize()

    def resize(self, num=0):
        if num > 0:
            self.max_num = num
        while self.num > self.max_num:
            self._fission_fusion_delete()

    def _fission_fusion_delete(self):
        """Cluster-and-merge deletion from vpt.py::cls_prompt_gen.delete_cls_prompt."""
        tmp = []
        for i in range(len(self.set)):
            for j in range(i + 1, len(self.set)):
                tmp.append([torch.dot(self.set[i][0], self.set[j][0]).item(), i, j])
        tmp = sorted(tmp, key=lambda x: x[0], reverse=True)

        n = self.num
        used = []
        cluster = []
        for entry in tmp:
            if n <= self.max_num:
                break
            a, b = entry[1], entry[2]
            if a not in used and b not in used:
                cluster.append([a, b])
                used.append(a); used.append(b)
                n -= 1
            elif a not in used:
                for j in range(len(cluster)):
                    if b in cluster[j]:
                        cluster[j].append(a); used.append(a); n -= 1
                        break
            elif b not in used:
                for j in range(len(cluster)):
                    if a in cluster[j]:
                        cluster[j].append(b); used.append(b); n -= 1
                        break
            else:
                merged = False
                for j in range(len(cluster)):
                    if a in cluster[j] and b not in cluster[j]:
                        for k in range(j + 1, len(cluster)):
                            if b in cluster[k]:
                                cluster[j].extend(cluster[k])
                                del cluster[k]; n -= 1; merged = True; break
                    elif b in cluster[j] and a not in cluster[j]:
                        for k in range(j + 1, len(cluster)):
                            if a in cluster[k]:
                                cluster[j].extend(cluster[k])
                                del cluster[k]; n -= 1; merged = True; break
                    elif a in cluster[j] and b in cluster[j]:
                        merged = True
                    if merged:
                        break

        todel = []
        for group in cluster:
            tmp1 = self.set[group[0]][0].clone()
            tmp2 = self.set[group[0]][1].clone()
            for j in range(1, len(group)):
                if group[j] not in todel:
                    todel.append(group[j])
                tmp1 = tmp1 + self.set[group[j]][0]
                tmp2 = tmp2 + self.set[group[j]][1]
            self.set[group[0]][0] = tmp1 / len(group)
            self.set[group[0]][1] = tmp2 / len(group)
            self.num -= (len(group) - 1)
        for i in sorted(todel, reverse=True):
            del self.set[i]

    def reset(self):
        self.set = []
        self.record = []
        self.num = 0
        self.id = 0


# ---------------------------------------------------------------------------
# Entropy utils (keep two variants so we can use KFF's classification-style
# batch entropy on the image-level distribution and per-pixel entropy for
# the adaptation loss on dense logits).
# ---------------------------------------------------------------------------

@torch.jit.script
def _softmax_entropy_1d(x: torch.Tensor) -> torch.Tensor:
    """Entropy of (B, C) logits along last dim -> (B,)."""
    x = -(x.softmax(-1) * x.log_softmax(-1)).sum(-1)
    return x


def _per_pixel_entropy_mean(logits: torch.Tensor) -> torch.Tensor:
    """Entropy of (B, C, H, W) logits along class dim -> scalar (mean over B, H, W)."""
    p = logits.softmax(1)
    lp = logits.log_softmax(1)
    return -(p * lp).sum(1).mean()


# ---------------------------------------------------------------------------
# Model / param helpers
# ---------------------------------------------------------------------------

def configure_model(model, prompt_num):
    from .prompt_vit import PromptVisualEncoder
    model.visual = PromptVisualEncoder(model.visual, num_prompts=prompt_num)

    # Compute KFF's xavier-uniform init range for cls_prompt_gen to reuse.
    # (cls_prompt still uses xavier init -- it is rebuilt every batch, so the
    # initial random values never reach evaluate() without first being either
    # overwritten by a coreset match or trained for `steps` iterations.)
    patch_size = model.visual.vit.patch_size
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    from functools import reduce
    from operator import mul
    dim = model.visual.prompt_dim
    val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + dim))
    model.visual.prompt_init_val = val

    # Domain prompts: use zero init (DPCore-style) instead of KFF's xavier init.
    # Rationale: MLMP's protocol calls evaluate() BEFORE adapt() on each batch,
    # so evaluate() sees whatever `self.model.visual.prompts` is. Xavier-random
    # domain prompts destroy the raw NA-CLIP alignment and pre-adapt mIoU; zero
    # init keeps evaluate() ≈ no-adapt baseline until adapt() actually learns
    # something. This departs from KFF's ImageNet-C protocol (which does not
    # have a separate pre-adapt evaluate step).
    with torch.no_grad():
        nn.init.zeros_(model.visual.prompts.data)
    model.visual.prompt_init = model.visual.prompts.clone().detach()

    for p in model.parameters():
        p.requires_grad = False
    model.visual.prompts.requires_grad_(True)
    return model


def copy_model_and_optimizer(model, optimizer):
    return deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


# ---------------------------------------------------------------------------
# KFF main class
# ---------------------------------------------------------------------------

class KFF(nn.Module):
    """Class-aware Domain Knowledge Fusion and Fission for OVSS CTTA.

    Faithful port of KFF's ours.py::Ours, adapted to open-vocabulary
    segmentation. The domain-level PromptBase is 1:1 with the reference;
    the class-level cls_prompt_gen uses image-level softmax as the class
    signature (see module-level docstring, bridge (a)).
    """

    def __init__(self, ovss_type, ovss_backbone, lr, classes,
                 vision_outputs=(-1,), steps=1, episodic=False,
                 prompt_dir='prompts.yaml',
                 # Domain-prompt hyperparameters (mirror KFF OURS.* / OPTIM.*):
                 tau=3.0, ema_alpha=0.1, thr_d=25.0, n_d=20,
                 lr_domain=1e-5, lamda=1.0,
                 # Class-prompt hyperparameters:
                 n_c=100, thr_c=0.005, thr_ent=2.0, alpha_c=0.1,
                 # Shared:
                 prompt_num=8,
                 # OVSS-specific:
                 runtime_calculation=False, verbose_kff=False,
                 micro_batch_size=None, device='cpu'):
        super().__init__()
        self.lr = lr                    # cls_prompt optimiser LR (KFF "OPTIM.LR")
        self.lr_domain = lr_domain      # domain-prompt optimiser LR (KFF "OPTIM.LR_DOMAIN")
        self.lamda = lamda
        self.tau = tau
        self.ema_alpha = ema_alpha
        self.E_ID = 1
        self.E_OOD = steps
        self.steps = steps
        assert steps > 0
        self.episodic = episodic
        self.vision_outputs = vision_outputs
        self.runtime = runtime_calculation
        self.verbose = verbose_kff
        self.micro_batch_size = micro_batch_size
        self.classes = classes
        self.device = device

        # Load NA-CLIP + wrap with prompt encoder.
        base_model, self.tokenize = load_ovss(ovss_type, ovss_backbone, device=device,
                                              classes=classes, catseg_checkpoint=None)
        self.model = configure_model(base_model, prompt_num)

        print_clip_parameters(self.model)
        self.optimizer_domain = optim.AdamW([self.model.visual.prompts], lr=self.lr_domain)

        # Text embeddings (frozen) for NA-CLIP similarity.
        if prompt_dir:
            self.prompt_templates = load_prompts_from_yaml(prompt_dir)
        else:
            self.prompt_templates = [REFERENCE_PROMPT]
        with torch.no_grad():
            self.text_x = self._extract_text_embeddings(classes, self.prompt_templates, average=True).squeeze()
        self.num_classes = len(classes)

        # Snapshot for OOD "reset to source" step.
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer_domain)
        self.prompt_copy = self.model.visual.prompt_init.clone()

        # Prompt bases.
        self.prompt_base = PromptBase(max_len=n_d, ema_alpha=ema_alpha, tau=tau, thr_d=thr_d)
        # Forward verbose flag to PromptBase so it can print L2 distances.
        self.prompt_base.verbose = verbose_kff
        self.cls_prompt_generator = cls_prompt_gen(
            val=self.model.visual.prompt_init_val,
            dim=self.model.visual.prompt_dim,
            num=n_c, thr_c=thr_c, thr_ent=thr_ent, alpha_c=alpha_c,
        )

        self.train_info = None

        print(f"+++ KFF: tau={tau}, thr_d={thr_d}, n_d={n_d}, n_c={n_c}, "
              f"thr_c={thr_c}, thr_ent={thr_ent}, lr={lr}, lr_domain={lr_domain}, "
              f"prompt_num={prompt_num}")

    # --------------------------------------------------------------
    # Public API used by main_continual.py
    # --------------------------------------------------------------

    def adapt(self, x):
        """Episodic TTA: reset, then one adaptation step. Returns [loss]."""
        self.reset()
        _ = self._forward_and_adapt(x)
        return [0.0]

    def continual_adapt(self, x):
        _ = self._forward_and_adapt(x)
        return [0.0]

    @torch.no_grad()
    def evaluate(self, x):
        """Inference with current domain prompt. cls_prompt is only injected
        when the class-prompt memory has at least one entry with a usable
        cosine match for this batch; otherwise we fall back to raw NA-CLIP +
        domain prompt to avoid injecting random tokens during pre-adapt eval.
        """
        # Only build cls_prompt when we have something meaningful to inject.
        self.model.visual.cls_prompt = None
        if len(self.cls_prompt_generator.set) > 0:
            _ = self._prepare_cls_prompt_from_raw(x)
            # If every record is -1 (no match), the generator would have
            # returned random prompts -- discard them.
            if all(r[0] == -1 if isinstance(r, list) else False
                   for r in self.cls_prompt_generator.record[-x.shape[0]:]):
                self.model.visual.cls_prompt = None

        mbs = self.micro_batch_size
        if mbs is not None and mbs < x.shape[0] and self.model.visual.cls_prompt is None:
            chunks = [x[i:i+mbs] for i in range(0, x.shape[0], mbs)]
            outs = []
            for chunk in chunks:
                logits, _, _ = self.model(chunk, self.text_x[-1], text_ensemble=True,
                                          vision_outputs=self.vision_outputs,
                                          interpolate=True,
                                          vision_out_type="adaptive_weighted_mean",
                                          save_weights=True)
                outs.append(logits[0])
            logits = torch.cat(outs, dim=0)
        else:
            logits, _, _ = self.model(x, self.text_x[-1], text_ensemble=True,
                                      vision_outputs=self.vision_outputs,
                                      interpolate=True,
                                      vision_out_type="adaptive_weighted_mean",
                                      save_weights=True)
            logits = logits[0]
        # Clear cls_prompt so subsequent adaptation forward rebuilds it fresh.
        self.model.visual.cls_prompt = None
        return logits

    # --------------------------------------------------------------
    # Core: KFF's forward() ported to OVSS
    # --------------------------------------------------------------

    @torch.enable_grad()
    def _forward_and_adapt(self, x):
        # 1. Raw forward (no prompts) -> CLS features as domain key and
        #    per-pixel logits for class signature.
        self.model.visual.cls_prompt = None  # ensure raw path
        key, image_class_signature = self._raw_key_and_class_signature(x)

        # 2. Domain-prompt base lookup.
        is_ID = False
        weighted_prompts = None
        weights = None
        if len(self.prompt_base) > 0:
            weighted_prompts, weights, is_ID = self.prompt_base.get_weighted_prompts(key)

        # 3. Build per-sample cls_prompt based on class signature.
        #    KFF does this *after* raw forward and before adapt.
        cls_prompt_tensor = self.cls_prompt_generator.get_cls_prompt(image_class_signature)
        # Wrap as a leaf nn.Parameter on device so it's optimisable.
        self.model.visual.cls_prompt = nn.Parameter(cls_prompt_tensor.detach().to(self.device))

        if is_ID:
            # Load matched domain prompt; fine-tune both domain and cls prompts.
            with torch.no_grad():
                self.model.visual.prompts.data.copy_(weighted_prompts.to(self.device))
            self.model.visual.prompts.requires_grad_(True)
            opt_dom = torch.optim.AdamW([self.model.visual.prompts], lr=self.lr_domain)
            opt_cls = torch.optim.AdamW([self.model.visual.cls_prompt], lr=self.lr)

            outputs, _ = self._inner_adapt(x, opt_dom, opt_cls, self.E_ID)

            self.prompt_base.update(weights,
                                    self.model.visual.prompts.clone().detach().cpu(),
                                    key.detach().cpu())
        else:
            # OOD: reset domain prompt to source init, then learn from scratch.
            with torch.no_grad():
                self.model.visual.prompts.data.copy_(self.prompt_copy.to(self.device))
            self.model.visual.prompts.requires_grad_(True)
            opt_dom = torch.optim.AdamW([self.model.visual.prompts], lr=self.lr_domain)
            opt_cls = torch.optim.AdamW([self.model.visual.cls_prompt], lr=self.lr)

            outputs, _ = self._inner_adapt(x, opt_dom, opt_cls, self.E_OOD)

            self.prompt_base.add_prompt(key.detach().cpu(),
                                        self.model.visual.prompts.clone().detach().cpu(),
                                        [])
            if self.verbose:
                print(f"  [KFF OOD] coreset size now {len(self.prompt_base)}")

        # 4. Update cls_prompt memory with final-step outputs.
        self.cls_prompt_generator.update_cls_prompt(
            outputs.mean(dim=(-2, -1)),  # (B, C) image-level logits
            self.model.visual.cls_prompt.detach(),
        )
        # Clear cls_prompt so the next evaluate() starts clean.
        self.model.visual.cls_prompt = None
        return outputs

    # --------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------

    @torch.no_grad()
    def _raw_key_and_class_signature(self, x):
        """Return (domain_key, image_level_logits) under *raw* (no prompt) forward.

        domain_key: (D*2,) concat of batch std and mean of CLS features.
        image_level_logits: (B, num_classes) spatial mean of per-pixel logits.
        """
        self.model.visual.cls_prompt = None  # safety
        cls_feat = self.model.visual.forward_raw_features(x)[:, 0]  # (B, D)
        batch_std, batch_mean = torch.std_mean(cls_feat, dim=0)
        key = torch.cat([batch_mean, batch_std], dim=-1).detach().cpu()

        # logits for class signature
        logits, _, _ = self.model(x, self.text_x[-1], text_ensemble=True,
                                  vision_outputs=self.vision_outputs,
                                  interpolate=True,
                                  vision_out_type="adaptive_weighted_mean",
                                  save_weights=True)
        # KFF feeds softmax logits to cls_prompt_gen. We pre-reduce spatial dims.
        image_level = logits[0].mean(dim=(-2, -1))  # (B, num_classes)
        return key, image_level.detach().cpu()

    @torch.no_grad()
    def _prepare_cls_prompt_from_raw(self, x):
        """For evaluate(): build cls_prompt from raw class signature of x.

        Note: get_cls_prompt() appends to self.cls_prompt_generator.record for
        use in a later update_cls_prompt(). evaluate() does NOT update, so we
        roll back the record length after use to avoid desync with adapt().
        """
        record_len_before = len(self.cls_prompt_generator.record)
        _, image_level = self._raw_key_and_class_signature(x)
        cls_prompt_tensor = self.cls_prompt_generator.get_cls_prompt(image_level)
        # Discard the records we just appended (evaluate must be non-mutating).
        self.cls_prompt_generator.record = self.cls_prompt_generator.record[:record_len_before]
        self.model.visual.cls_prompt = nn.Parameter(cls_prompt_tensor.detach().to(self.device))
        return self.model.visual.cls_prompt

    @torch.enable_grad()
    def _inner_adapt(self, x, opt_dom, opt_cls, iteration):
        """Iterate: forward with domain+cls prompts, compute KFF loss, step."""
        outputs = None
        loss = None
        for i in range(iteration):
            # Prompted forward (features for distribution loss; logits for entropy).
            cls_features = self.model.visual.forward_features(x)[:, 0]
            loss_dist = self._distribution_loss(cls_features)

            logits, _, _ = self.model(x, self.text_x[-1], text_ensemble=True,
                                      vision_outputs=self.vision_outputs,
                                      interpolate=True,
                                      vision_out_type="adaptive_weighted_mean",
                                      save_weights=True)
            outputs = logits[0]  # (B, C, H, W)

            # KFF's 3 * entropy coefficient, but on per-pixel entropy.
            loss_ent = _per_pixel_entropy_mean(outputs)
            loss = loss_dist + 3.0 * loss_ent

            opt_dom.zero_grad()
            opt_cls.zero_grad()
            if i == iteration - 1:
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            opt_dom.step()
            opt_cls.step()

            if self.verbose and (i == 0 or i == iteration - 1):
                print(f"  [KFF step {i:03d}/{iteration}] loss={loss.item():.4f} "
                      f"(dist={loss_dist.item():.4f}, ent={loss_ent.item():.4f})")

        return outputs.detach(), loss.detach() if loss is not None else None

    def _distribution_loss(self, x):
        std, mean = torch.std_mean(x, dim=0)
        ls = torch.norm(std - self.train_info[0].to(x.device), p=2)
        lm = torch.norm(mean - self.train_info[1].to(x.device), p=2)
        return ls + self.lamda * lm

    # --------------------------------------------------------------
    # Source statistics (reuse DPCore's protocol)
    # --------------------------------------------------------------

    @torch.no_grad()
    def obtain_src_stat(self, data_loader, num_samples=5000):
        num = 0
        features = []
        self.model.visual.cls_prompt = None
        for data in data_loader:
            images = data['img_patches'].to(self.device)
            feat = self.model.visual.forward_raw_features(images)  # (B, 1+N, D)
            # Entropy filter using raw OVSS logits (same as DPCore).
            logits, _, _ = self.model(
                images, self.text_x[-1], text_ensemble=True,
                vision_outputs=self.vision_outputs,
                interpolate=False,
                vision_out_type="mean",
            )
            ent = _softmax_entropy_1d(logits[0].permute(0, 2, 3, 1).reshape(-1, logits[0].shape[1]))
            # Flatten to per-sample mean over pixels for the mask threshold.
            per_sample_ent = ent.view(logits[0].shape[0], -1).mean(dim=-1)
            selected_indices = torch.where(per_sample_ent < math.log(self.num_classes) / 2 - 1)[0]
            if len(selected_indices) == 0:
                selected_indices = torch.arange(feat.shape[0], device=feat.device)
            feat = feat[selected_indices]
            features.append(feat[:, 0])
            num += feat.shape[0]
            if num >= num_samples:
                break
        features = torch.cat(features, dim=0)[:num_samples, :]
        print(f'KFF source statistics computed with {features.shape[0]} examples.')
        self.train_info = torch.std_mean(features, dim=0)
        del features

    # --------------------------------------------------------------
    # Reset (episodic only)
    # --------------------------------------------------------------

    def reset(self):
        with torch.no_grad():
            self.model.visual.prompts.data.copy_(self.prompt_copy.to(self.device))
        self.prompt_base = PromptBase(max_len=self.prompt_base.max_len,
                                      ema_alpha=self.ema_alpha,
                                      tau=self.tau, thr_d=self.prompt_base.thr)
        self.cls_prompt_generator.reset()

    # --------------------------------------------------------------
    # Text embeddings (same as DPCore)
    # --------------------------------------------------------------

    def _extract_text_embeddings(self, class_names, prompts, average=True):
        text_features = []
        for class_name in class_names:
            texts = [p.format(class_name) for p in prompts]
            tokens = self.tokenize(texts).to(self.device)
            embeddings = self.model.encode_text(tokens)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            if average:
                avg = embeddings.mean(dim=0)
                avg = avg / avg.norm()
                embeddings = torch.cat([embeddings, avg.unsqueeze(0)], dim=0)
            text_features.append(embeddings)
        return torch.stack(text_features, dim=1).to(self.device)
