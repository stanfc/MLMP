import re
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class TENTEarlyContinual:
    """
    TENT-Early-Continual: pure TENT pixel-wise entropy, but only the
    LayerNorm parameters of the "early" visual-encoder blocks
    (blocks [0, early_cutoff) plus ln_pre) are trained. Every other
    LN parameter is hard-frozen (requires_grad=False, not in optimizer).

    No DivGate, no stochastic restoration. Strict ablation of the
    "drift comes from late layers" hypothesis: if freezing late LNs
    is sufficient to prevent collapse, plain TENT loss should work
    here.

    Default early_cutoff=8 matches the cutoff used by
    cma_layered_continual / tent_divgate_layered_continual so the
    three layered methods share the same partition.
    """

    BLOCK_RE = re.compile(r'resblocks\.(\d+)\.')

    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 early_cutoff=8,
                 prompt_dir=None, runtime_calculation=False, device='cpu'):
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps

        self.early_cutoff = int(early_cutoff)
        if self.early_cutoff < 0:
            raise ValueError(f"early_cutoff must be >= 0, got {self.early_cutoff}")

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

        # All LN params (named) for inspection.
        all_params, all_names = self.collect_ln_params(self.model.visual)
        self.named_ln_params_all = list(zip(all_names, all_params))

        # Hard-freeze every LN param that is NOT in the early group.
        trainable_params = []
        trainable_named = []
        frozen_count = 0
        for name, p in self.named_ln_params_all:
            if self._is_early(name):
                trainable_params.append(p)
                trainable_named.append((name, p))
            else:
                p.requires_grad_(False)
                frozen_count += 1
        self.named_ln_params_trainable = trainable_named

        print_clip_parameters(self.model)
        self._print_layer_groups(frozen_count)

        if not trainable_params:
            raise ValueError(
                f"early_cutoff={self.early_cutoff} produced zero trainable params; "
                f"every LN was frozen. Increase early_cutoff."
            )

        # ---------- Optimizer (only over trainable LN params) ----------
        self.optimizer = optim.Adam(trainable_params, lr=self.lr,
                                    betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        # ---------- Source state snapshot (for optional reset) ----------
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
        return self.perform_adaptation(x)

    def continual_adapt(self, x):
        """Alias for adapt() -- matches main_continual.py protocol."""
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
            loss = self.softmax_entropy(logits).mean()
            loss_report.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

    # ===========================================================
    # Layer grouping (early-only freeze)
    # ===========================================================

    def _is_early(self, name):
        if 'ln_pre' in name:
            return True
        if 'ln_post' in name:
            return False
        m = self.BLOCK_RE.search(name)
        if m is None:
            return False
        idx = int(m.group(1))
        return idx < self.early_cutoff

    def _print_layer_groups(self, frozen_count):
        early = sum(1 for n, _ in self.named_ln_params_all if self._is_early(n))
        late = len(self.named_ln_params_all) - early
        print(f"+++ TENT-Early: early_cutoff={self.early_cutoff}")
        print(f"+++ LN params: early={early} (trainable), "
              f"rest={late} (FROZEN), total={len(self.named_ln_params_all)}")
        if frozen_count != late:
            print(f"!!! sanity: frozen_count={frozen_count} != frozen group size {late}")

    # ===========================================================
    # Loss (pure TENT)
    # ===========================================================

    @staticmethod
    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        return -(x.softmax(-3) * x.log_softmax(-3)).sum(-3)

    # ===========================================================
    # Shared helpers (verbatim from TENTContinual)
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
