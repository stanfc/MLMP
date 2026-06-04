"""
RoTTAContinual — Robust Test-Time Adaptation in Dynamic Scenarios
(Yuan, Xie, Li; CVPR 2023). Ported from https://github.com/BIT-DA/RoTTA
to the MLMP / NA-CLIP / per-pixel-OVSS setting.

RoTTA's four mechanisms (paper §3):
  1. Robust BatchNorm (RBN): replace BN with momentum-BN using a global
     EMA of statistics.  --> NOT APPLICABLE here: NA-CLIP is LayerNorm
     only.  We instead adapt the visual LayerNorm affine params
     (TENT-style).  This is the one mechanism we cannot port; documented
     as a deviation.
  2. CSTU memory bank: Category-balanced Sampling with Timeliness and
     Uncertainty.  Stores (sample, pseudo_label, uncertainty); evicts by
     a heuristic score that balances age (timeliness) and uncertainty,
     keeping the bank class-balanced.  PORTED verbatim (see CSTU class).
  3. Teacher-student with robust training: an EMA teacher produces
     pseudo-labels; the student is trained on STRONG-augmented memory
     samples with a symmetric/soft cross-entropy against the teacher,
     reweighted by sample timeliness.  PORTED.
  4. Timeliness reweighting: exp(-age)/(1+exp(-age)).  PORTED.

Per-pixel adaptation (dense prediction):
  - Each memory "instance" is one 224-px patch (main_continual.py has
    already cropped images into patches by the time we see x).
  - A patch's pseudo_label key for the category-balanced bank is its
    MAJORITY per-pixel class under the teacher (an image-level proxy).
  - A patch's uncertainty is the MEAN per-pixel entropy under the teacher.
  - The student loss is per-pixel soft CE against the teacher, averaged
    over pixels, then timeliness-weighted across memory patches.

CTTA hard-rule compliance:
  - No per-sample reset.  State (memory bank + EMA teacher) persists for
    the whole stream.  reset() exists only for the episodic entry point.

Memory cost: one EMA teacher copy (~1.2 GB for ViT-L/14 visual) plus the
memory bank of `memory_size` patches (~150 KB each -> a few MB).  Much
lighter than CoTTA's aug-averaging or ViDA's multi-copy scheme.

Hyperparameters (paper defaults, retained):
  - memory_size = 64
  - nu          = 0.001   (EMA teacher update rate)
  - lambda_t    = 1.0     (timeliness weight in CSTU score)
  - lambda_u    = 1.0     (uncertainty weight in CSTU score)
  (ALPHA, the RBN momentum, is dropped — no BN to apply it to.)
"""
import time
import copy
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from ovss import load_ovss
from utils.misc import print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


# ---------------------------------------------------------------------------
# CSTU memory bank (ported verbatim from RoTTA/core/utils/memory.py)
# ---------------------------------------------------------------------------
class _MemoryItem:
    def __init__(self, data=None, uncertainty=0, age=0):
        self.data = data
        self.uncertainty = uncertainty
        self.age = age

    def increase_age(self):
        self.age += 1

    def get_data(self):
        return self.data, self.uncertainty, self.age


class CSTU:
    """Category-balanced Sampling with Timeliness and Uncertainty."""
    def __init__(self, capacity, num_class, lambda_t=1.0, lambda_u=1.0):
        self.capacity = capacity
        self.num_class = num_class
        self.per_class = capacity / num_class
        self.lambda_t = lambda_t
        self.lambda_u = lambda_u
        self.data = [[] for _ in range(num_class)]

    def get_occupancy(self):
        return sum(len(c) for c in self.data)

    def per_class_dist(self):
        return [len(c) for c in self.data]

    def add_instance(self, instance):
        x, prediction, uncertainty = instance
        new_item = _MemoryItem(data=x, uncertainty=uncertainty, age=0)
        new_score = self.heuristic_score(0, uncertainty)
        if self.remove_instance(prediction, new_score):
            self.data[prediction].append(new_item)
        self.add_age()

    def remove_instance(self, cls, score):
        class_occupied = len(self.data[cls])
        all_occupancy = self.get_occupancy()
        if class_occupied < self.per_class:
            if all_occupancy < self.capacity:
                return True
            else:
                majority = self.get_majority_classes()
                return self.remove_from_classes(majority, score)
        else:
            return self.remove_from_classes([cls], score)

    def remove_from_classes(self, classes, score_base):
        max_class, max_index, max_score = None, None, None
        for cls in classes:
            for idx, item in enumerate(self.data[cls]):
                score = self.heuristic_score(item.age, item.uncertainty)
                if max_score is None or score >= max_score:
                    max_score, max_index, max_class = score, idx, cls
        if max_class is not None:
            if max_score > score_base:
                self.data[max_class].pop(max_index)
                return True
            return False
        return True

    def get_majority_classes(self):
        dist = self.per_class_dist()
        mx = max(dist)
        return [i for i, o in enumerate(dist) if o == mx]

    def heuristic_score(self, age, uncertainty):
        return (self.lambda_t * 1 / (1 + math.exp(-age / self.capacity))
                + self.lambda_u * uncertainty / math.log(self.num_class))

    def add_age(self):
        for class_list in self.data:
            for item in class_list:
                item.increase_age()

    def get_memory(self):
        tmp_data, tmp_age = [], []
        for class_list in self.data:
            for item in class_list:
                tmp_data.append(item.data)
                tmp_age.append(item.age)
        tmp_age = [a / self.capacity for a in tmp_age]
        return tmp_data, tmp_age


def _timeliness_reweighting(ages, device):
    if isinstance(ages, list):
        ages = torch.tensor(ages, dtype=torch.float, device=device)
    return torch.exp(-ages) / (1 + torch.exp(-ages))


def _get_tta_transforms(img_size=224, gaussian_std=0.005):
    """Strong augmentation for the student. Mirrors MLMP's cotta.py
    transform (valid on CLIP-normalized tensors)."""
    n = img_size
    return transforms.Compose([
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.4, hue=0.05),
        transforms.Pad(padding=int(n / 2), padding_mode='edge'),
        transforms.RandomAffine(degrees=[-15, 15], translate=(1/16, 1/16),
                                scale=(0.9, 1.1)),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.5]),
        transforms.CenterCrop(size=n),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Lambda(lambda x: x + gaussian_std * torch.randn_like(x)),
    ])


class RoTTAContinual:
    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 memory_size=64, nu=0.001, lambda_t=1.0, lambda_u=1.0,
                 update_frequency=None,
                 runtime_calculation=False, device='cpu'):
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps
        self.runtime = runtime_calculation
        self.device = device

        if classes is None:
            raise ValueError("classes is required in __init__")
        self.classes = classes
        self.num_class = len(classes)

        self.memory_size = memory_size
        self.nu = nu
        # update_frequency defaults to memory_size (RoTTA convention)
        self.update_frequency = update_frequency or memory_size
        self.current_instance = 0
        self.total_updates = 0

        self.model, self.tokenize = load_ovss(ovss_type, ovss_backbone, device=device)

        self.prompt_templates = [REFERENCE_PROMPT]

        # Student: adapt visual LayerNorm only (TENT-style; RBN N/A on LN-only).
        self.model.transformer.requires_grad_(False)
        self.model.ln_final.requires_grad_(False)
        self.model.token_embedding.requires_grad_(False)
        self.model.visual = self.set_ln_grads(self.model.visual)
        params, _ = self.collect_ln_params(self.model.visual)

        print_clip_parameters(self.model)
        print(f"+++ RoTTA-Continual: mem_size={self.memory_size}, "
              f"nu={self.nu}, lambda_t={lambda_t}, lambda_u={lambda_u}, "
              f"update_freq={self.update_frequency}, num_class={self.num_class}")

        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        # EMA teacher: full model copy, params detached.
        self.model_ema = copy.deepcopy(self.model)
        for p in self.model_ema.parameters():
            p.detach_()

        # Source snapshot for episodic reset only.
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer)

        self.mem = CSTU(capacity=self.memory_size, num_class=self.num_class,
                        lambda_t=lambda_t, lambda_u=lambda_u)
        self.transform = _get_tta_transforms(img_size=224)

        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(
                self.classes, self.prompt_templates, average=False).squeeze()

        if self.runtime:
            self.adapt_times = []
            self.eval_times = []

    def adapt(self, x):
        return self.perform_adaptation(x)

    def continual_adapt(self, x):
        return self.perform_adaptation(x)

    @torch.no_grad()
    def evaluate(self, x):
        # RoTTA's prediction path is the EMA teacher.
        t1 = time.time()
        self.model_ema.eval()
        logits, _, _ = self.model_ema(x, self.text_x, True, interpolate=True)
        logits = logits[0]
        if self.runtime:
            self.eval_times.append(time.time() - t1)
        return logits

    def reset(self):
        self.load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state)

    # -----------------------------------------------------------
    # Adaptation
    # -----------------------------------------------------------
    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []

        # 1) Teacher pseudo-labels + per-patch (label, uncertainty)
        with torch.no_grad():
            self.model.eval()
            self.model_ema.eval()
            ema_logits, _, _ = self.model_ema(x, self.text_x, True, interpolate=False)
            ema_logits = ema_logits[0]                       # (B, C, h, w)
            prob = ema_logits.softmax(dim=1)
            pseudo = prob.argmax(dim=1)                      # (B, h, w)
            ent = -(prob * torch.log(prob + 1e-6)).sum(dim=1)  # (B, h, w)

        # 2) Add each patch to memory, periodically update student.
        for i in range(x.shape[0]):
            majority_cls = torch.mode(pseudo[i].reshape(-1)).values.item()
            uncertainty = ent[i].mean().item()
            self.mem.add_instance((x[i].detach().cpu(), majority_cls, uncertainty))
            self.current_instance += 1
            if self.current_instance % self.update_frequency == 0:
                lv = self.update_model()
                if lv is not None:
                    loss_report.append(lv)

        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

    def update_model(self):
        sup_data, ages = self.mem.get_memory()
        if len(sup_data) == 0:
            return None

        self.model.train()
        self.model_ema.train()
        batch = torch.stack(sup_data).to(self.device)        # (M, 3, 224, 224)
        strong_aug = self.transform(batch)

        with torch.no_grad():
            ema_logits, _, _ = self.model_ema(batch, self.text_x, True, interpolate=False)
            ema_logits = ema_logits[0]                       # (M, C, h, w)
        stu_logits, _, _ = self.model(strong_aug, self.text_x, True, interpolate=False)
        stu_logits = stu_logits[0]

        # per-pixel soft CE against teacher, mean over pixels -> (M,)
        per_pixel = self.softmax_entropy(stu_logits, ema_logits)  # (M, h, w)
        per_sample = per_pixel.mean(dim=(1, 2))                   # (M,)
        weight = _timeliness_reweighting(ages, self.device)
        loss = (per_sample * weight).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_ema_variables(self.model_ema, self.model, self.nu)
        self.total_updates += 1
        return loss.item()

    @staticmethod
    def update_ema_variables(ema_model, model, nu):
        # in-place, shape-agnostic (NA-CLIP has 0-dim scalar params like
        # logit_scale that break ema_p.data[:] slicing).
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(1 - nu).add_(p.data, alpha=nu)
        return ema_model

    # ===========================================================
    # Helpers
    # ===========================================================
    def extract_text_embeddings(self, class_names, prompts, average=True):
        text_features = []
        for class_name in class_names:
            texts = [p.format(class_name) for p in prompts]
            texts = self.tokenize(texts).to(self.device)
            ce = self.model.encode_text(texts)
            ce = ce / ce.norm(dim=-1, keepdim=True)
            if average:
                avg = ce.mean(dim=0); avg = avg / avg.norm()
                ce = torch.cat([ce, avg.unsqueeze(0)], dim=0)
            text_features.append(ce)
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
    def softmax_entropy(x, x_ema):
        """Soft CE of student logits x against teacher logits x_ema, per
        pixel. Class axis is dim=1 (B, C, h, w) -> (B, h, w)."""
        return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)
