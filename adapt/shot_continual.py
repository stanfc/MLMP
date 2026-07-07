"""
SHOTContinual — SHOT's Information-Maximization (IM) loss (Liang et al.,
ICML 2020, "Do We Really Need to Access the Source Data?") run ONLINE /
CONTINUAL on the NA-CLIP OVSS backbone, as a controlled baseline for
deyo_mlmp_divreg_continual.

SHOT's IM loss (the half we can run online):
    L = L_ent  -  lambda_div * L_div
      L_ent = mean per-pixel prediction entropy  E[H(p)]   (MINIMIZED)
      L_div = H(p_bar), entropy of the marginal class distribution
              p_bar = mean softmax over prompts/batch/space  (MAXIMIZED)
Maximizing H(p_bar) while minimizing E[H(p)] = maximizing mutual information
I(X;Y) = H(p_bar) - E[H(p)]. lambda_div=1.0 is SHOT's default (equal weight).

WHAT IS OMITTED vs the full SHOT paper (and why):
  * SHOT's OTHER half -- self-supervised nearest-centroid pseudo-labeling,
    recomputed per epoch over the whole target set -- is an OFFLINE,
    multi-epoch, transductive procedure. It has no faithful analog in a
    batch=1 single-pass streaming (continual TTA) setting, so it is dropped.
    This is the standard way SHOT appears as a TTA baseline (cf. CoTTA/TENT).
  * SHOT freezes a learned classifier head; here the "head" is the frozen
    CLIP text embedding (open-vocab), which is inherently frozen -- same
    spirit ("freeze hypothesis, adapt features"), different mechanism.

WHY THIS EXACT FORM: it is deyo_mlmp_divreg_continual with the DeYO half
removed. Same backbone, same LN-only params, same MLMP multi-prompt(7) +
multi-layer(18) adapt forward, same UAML evaluation, same evaluate-before-
adapt continual protocol. The ONLY difference from deyo_mlmp_divreg is:
  - SHOT: plain entropy-min over ALL pixels + diversity (no filter).
  - divreg: DeYO reliability filter (ent<margin & PLPD>thr) + confidence
    reweight on the entropy term + diversity.
So a divreg-vs-SHOT comparison isolates the DeYO reliability filtering.
"""
import time
import copy
import math
import re

import torch
import torch.nn as nn
import torch.optim as optim

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class SHOTContinual:
    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 vision_outputs=tuple(range(-1, -19, -1)),
                 prompt_dir='prompts.yaml',
                 lambda_div=1.0,
                 top_block_exclude=6,
                 runtime_calculation=False, device='cpu'):
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps
        self.vision_outputs = vision_outputs
        self.lambda_div = float(lambda_div)
        self.top_block_exclude = top_block_exclude
        self.runtime = runtime_calculation
        self.device = device

        if classes is None:
            raise ValueError("classes is required in __init__")
        self.classes = classes
        self.total_batches = 0

        self.model, self.tokenize = load_ovss(ovss_type, ovss_backbone, device=device)

        if prompt_dir:
            self.prompt_templates = load_prompts_from_yaml(prompt_dir)
            print(f"Number of prompt templates: {len(self.prompt_templates)}")
        else:
            self.prompt_templates = [REFERENCE_PROMPT]

        self.model.transformer.requires_grad_(False)
        self.model.ln_final.requires_grad_(False)
        self.model.token_embedding.requires_grad_(False)
        self.model.visual = self.set_ln_grads(
            self.model.visual, top_block_exclude=self.top_block_exclude)
        params, _ = self.collect_ln_params(
            self.model.visual, top_block_exclude=self.top_block_exclude)

        print_clip_parameters(self.model)
        print(f"+++ SHOT-continual (IM loss, multi-prompt T={len(self.prompt_templates)}): "
              f"lambda_div={self.lambda_div}, UAML layers={len(self.vision_outputs)}, "
              f"top_block_exclude={self.top_block_exclude} -> LN params: {len(params)}")

        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer)

        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(
                self.classes, self.prompt_templates, average=True).squeeze()

        if self.runtime:
            self.adapt_times = []
            self.eval_times = []

        self.collect_diag = False
        self.diag = {}

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
            save_weights=True)
        logits = logits[0]
        if self.runtime:
            self.eval_times.append(time.time() - t1)
        return logits

    def reset(self):
        self.load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state)

    def _adapt_forward(self, x):
        logits, _, _ = self.model(
            x, self.text_x[:-1], True,
            interpolate=False,
            vision_outputs=self.vision_outputs,
            vision_out_type="mean")
        return logits  # (T, B, C, h, w)

    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []
        for _ in range(self.steps):
            self.total_batches += 1

            logits = self._adapt_forward(x)                 # (T, B, C, h, w)
            prob = logits.softmax(dim=-3)                   # differentiable

            # SHOT IM loss:
            #   L_ent = mean per-pixel entropy (minimized)
            ent = -(prob * prob.clamp(min=1e-12).log()).sum(dim=-3)   # (T,B,h,w)
            loss_ent = ent.mean()
            #   L_div = entropy of the marginal class dist (maximized)
            marg = prob.mean(dim=(0, 1, 3, 4))              # (C,)
            marg = marg / marg.sum().clamp(min=1e-8)
            loss_div = -(marg * marg.clamp(min=1e-12).log()).sum()

            loss = loss_ent - self.lambda_div * loss_div

            self.optimizer.zero_grad()
            loss.backward()
            if self.collect_diag:
                gn = 0.0
                for p in self.optimizer.param_groups[0]['params']:
                    if p.grad is not None:
                        gn += float(p.grad.detach().float().pow(2).sum())
                self.diag.update(grad_norm=gn ** 0.5,
                                 h_pixel_mean=float(loss_ent.detach()),
                                 h_margin_loss=float(loss_div.detach()))
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_report.append(loss.item())

        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

    def diagnose(self, x):
        # SHOT baseline: no extra per-layer diagnostic forward.
        return

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
    def _is_excluded(nm, top_block_exclude):
        if 'ln_post' in nm:
            return True
        m = re.search(r'resblocks\.(\d+)\.', nm)
        if m and int(m.group(1)) >= (24 - top_block_exclude):
            return True
        return False

    @classmethod
    def set_ln_grads(cls, model, top_block_exclude=6):
        model.requires_grad_(False)
        for nm, m in model.named_modules():
            if isinstance(m, nn.LayerNorm) and not cls._is_excluded(nm, top_block_exclude):
                m.requires_grad_(True)
        return model

    @classmethod
    def collect_ln_params(cls, model, top_block_exclude=6):
        params, names = [], []
        for nm, m in model.named_modules():
            if isinstance(m, nn.LayerNorm) and not cls._is_excluded(nm, top_block_exclude):
                for np_, p in m.named_parameters():
                    if np_ in ['weight', 'bias']:
                        params.append(p); names.append(f"visual.{nm}.{np_}")
        return params, names

    @staticmethod
    def copy_model_and_optimizer(model, optimizer):
        return copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())

    @staticmethod
    def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)
