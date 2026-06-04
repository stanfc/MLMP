"""
MLMP-SimpleEval-Continual: MLMP adapt loss (multi-prompt × multi-layer entropy
+ optional ILE) BUT evaluate with a TENT-style single-prompt single-layer
forward (NO UAML fusion).

Ablation purpose: isolate whether MLMP's lower peak vs TENT on ACDC is due to
(A) UAML evaluate raising R1 too high (no headroom) or
(B) the ensemble adapt loss being intrinsically weaker than single-prompt TENT.

By forcing the evaluate path to match TENT (single prompt, last layer, no
adaptive weighting), R1 should drop to ~TENT level (~24 on ACDC). If peak then
climbs to ~32 like TENT, cause is (A); if it stays low, cause is (B).

adapt() is byte-identical to mlmp_continual; only evaluate() differs.
"""

import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class MLMPSimpleEvalContinual:
    """MLMP adapt loss + TENT-style simple evaluate."""

    def __init__(self, ovss_type, ovss_backbone, lr, classes,
                 vision_outputs=(-1,), alpha_cls=0.0, steps=1,
                 prompt_dir='prompts.yaml', prompt_integration='loss',
                 runtime_calculation=False, device='cpu'):
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps
        self.vision_outputs = vision_outputs
        self.alpha_cls = alpha_cls
        self.prompt_dir = prompt_dir
        self.prompt_integration = prompt_integration
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

        assert prompt_integration in ['loss', 'text']

        self.model.transformer.requires_grad_(False)
        self.model.ln_final.requires_grad_(False)
        self.model.token_embedding.requires_grad_(False)
        self.model.visual = self.set_ln_grads(self.model.visual)
        params, _ = self.collect_ln_params(self.model.visual)

        print_clip_parameters(self.model)
        print(f"+++ MLMP-SimpleEval: adapt={prompt_integration} (vision_outputs={vision_outputs}, "
              f"alpha_cls={alpha_cls}), EVALUATE=single-prompt single-layer (TENT-style)")

        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer)

        with torch.no_grad():
            # average=True gives (T+1, C, D); we use [:-1] for adapt, [-1] for nothing
            # (simple eval uses a single REFERENCE_PROMPT embedding, see below).
            self.text_x = self.extract_text_embeddings(
                self.classes, self.prompt_templates, average=True).squeeze()
            # Separate single-prompt embedding for TENT-style evaluate
            self.text_x_simple = self.extract_text_embeddings(
                self.classes, [REFERENCE_PROMPT], average=False).squeeze()

        if self.runtime:
            self.adapt_times = []
            self.eval_times = []

    def adapt(self, x):
        return self.perform_adaptation(x)

    def continual_adapt(self, x):
        return self.perform_adaptation(x)

    @torch.no_grad()
    def evaluate(self, x):
        # TENT-style: single prompt, single (last) layer, NO UAML weighting.
        t1 = time.time()
        logits, _, _ = self.model(x, self.text_x_simple, True, interpolate=True)
        logits = logits[0]
        if self.runtime:
            self.eval_times.append(time.time() - t1)
        return logits

    def reset(self):
        self.load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state)

    def perform_adaptation(self, x):
        # Identical to mlmp_continual.perform_adaptation
        t1 = time.time()
        loss_report = []
        for _ in range(self.steps):
            if self.prompt_integration == 'loss':
                logits, _, _, cls_logits = self.model(
                    x, self.text_x[:-1], True,
                    interpolate=False,
                    vision_outputs=self.vision_outputs,
                    return_vanilla_cls=True,
                    vision_out_type="mean")
                entropy_per_pixel = self.softmax_entropy(logits)
                entropy_per_cls = self.softmax_entropy(cls_logits, dim=2)
                loss = entropy_per_pixel.mean() + self.alpha_cls * entropy_per_cls.mean()
            else:
                logits, _, _ = self.model(
                    x, self.text_x[-1], True,
                    interpolate=False, vision_outputs=self.vision_outputs)
                loss = self.softmax_entropy(logits).mean()

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
