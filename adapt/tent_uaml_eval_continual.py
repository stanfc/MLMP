"""
TENT-UAML-Eval-Continual: TENT adapt loss (single-prompt single-layer pixel
entropy) BUT evaluate with MLMP-style UAML (18-layer entropy-weighted fusion).

Mirror of mlmp_simple_eval_continual (which is MLMP-adapt + simple-eval).
Together they complete the 2x2 decoupling of {adapt loss} x {evaluate}:

                    simple eval        UAML eval
  TENT adapt        tent_continual     THIS
  MLMP adapt        mlmp_simple_eval   mlmp_continual

Purpose: test whether UAML's multi-layer fusion (a pure post-hoc inference
trick) stacks on top of TENT's strong single-layer adapt. If TENT+UAML > TENT,
UAML is a plug-and-play boost independent of the adapt loss.

Note: TENT only minimizes the LAST layer's entropy during adapt; the middle
17 layers are never optimized. UAML weights layers by their entropy, so it is
not guaranteed that UAML still helps when the underlying layers were not
adapted toward low entropy. That is exactly the question this run answers.
"""

import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class TENTUAMLEvalContinual:
    """TENT single-layer adapt + UAML multi-layer evaluate."""

    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 vision_outputs=(-1,),
                 prompt_dir=None,
                 runtime_calculation=False, device='cpu'):
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps
        # vision_outputs used ONLY for the UAML evaluate path.
        self.vision_outputs = vision_outputs
        self.runtime = runtime_calculation
        self.device = device

        if classes is None:
            raise ValueError("classes is required in __init__")
        self.classes = classes

        self.model, self.tokenize = load_ovss(ovss_type, ovss_backbone, device=device)

        # adapt uses a single REFERENCE_PROMPT (TENT convention).
        # We still allow a prompt_dir but the adapt loss uses the averaged/first
        # template embedding for the simple single-layer forward.
        self.prompt_templates = [REFERENCE_PROMPT]

        self.model.transformer.requires_grad_(False)
        self.model.ln_final.requires_grad_(False)
        self.model.token_embedding.requires_grad_(False)
        self.model.visual = self.set_ln_grads(self.model.visual)
        params, _ = self.collect_ln_params(self.model.visual)

        print_clip_parameters(self.model)
        print(f"+++ TENT-UAML-Eval: adapt=single-prompt single-layer entropy (TENT); "
              f"EVALUATE=UAML {self.vision_outputs} adaptive_weighted_mean")

        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer)

        with torch.no_grad():
            # Single-prompt embedding for both adapt and UAML evaluate
            # (UAML's multi-layer fusion uses the same single averaged prompt
            #  embedding as MLMP's evaluate text_x[-1]).
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
        # UAML: 18-layer entropy-weighted fusion (MLMP-style).
        t1 = time.time()
        logits, _, _ = self.model(
            x, self.text_x, True,
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

    def perform_adaptation(self, x):
        # TENT: single-prompt single-layer pixel entropy (identical to tent_continual).
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

    @staticmethod
    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        return -(x.softmax(-3) * x.log_softmax(-3)).sum(-3)

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
