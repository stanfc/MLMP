"""
MLMP-TopK-MinPrompt-DivGate-Continual: combines TopK (spatial mask) +
MinPrompt (prompt-view filter) + 3-tier discrete DivGate.

Base loss: -log(min_p over T templates) per pixel, masked by top-K%
ensemble confidence.
Gate: same as tent_divgate / mlmp_smooth_anchor — H_margin from ensemble.
"""

import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'

MODE_AGGRESSIVE = 'aggressive'
MODE_CAUTIOUS = 'cautious'
MODE_BRAKE = 'brake'


class MLMPTopKMinPromptDivGateContinual:

    def __init__(self, ovss_type, ovss_backbone, lr, classes,
                 vision_outputs=(-1,), alpha_cls=0.0, steps=1,
                 top_k_percent=0.2,
                 h_threshold=1.8, h_warning=1.5,
                 monitor_interval=50,
                 cautious_rst=0.005, brake_rst=0.02,
                 prompt_dir='prompts.yaml',
                 runtime_calculation=False, device='cpu'):
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps
        self.vision_outputs = vision_outputs
        self.alpha_cls = alpha_cls
        self.top_k_percent = float(top_k_percent)
        if not (0.0 < self.top_k_percent <= 1.0):
            raise ValueError(f"top_k_percent must be in (0, 1], got {self.top_k_percent}")
        self.h_threshold = float(h_threshold); self.h_warning = float(h_warning)
        if self.h_warning > self.h_threshold: raise ValueError("h_warning > h_threshold")
        self.monitor_interval = int(monitor_interval)
        self.cautious_rst = float(cautious_rst); self.brake_rst = float(brake_rst)
        self.runtime = runtime_calculation
        self.device = device

        if classes is None: raise ValueError("classes is required")
        self.classes = classes

        self.model, self.tokenize = load_ovss(ovss_type, ovss_backbone, device=device)
        if prompt_dir:
            self.prompt_templates = load_prompts_from_yaml(prompt_dir)
        else:
            self.prompt_templates = [REFERENCE_PROMPT]

        self.model.transformer.requires_grad_(False)
        self.model.ln_final.requires_grad_(False)
        self.model.token_embedding.requires_grad_(False)
        self.model.visual = self.set_ln_grads(self.model.visual)
        params, names = self.collect_ln_params(self.model.visual)
        self.named_ln_params = list(zip(names, params))

        print_clip_parameters(self.model)
        print(f"+++ MLMP-TopK-MinPrompt-DivGate: T={len(self.prompt_templates)}, "
              f"top_k={self.top_k_percent}, "
              f"thresholds=(warn {self.h_warning}, agg {self.h_threshold}), "
              f"rsts=(cau {self.cautious_rst}, brake {self.brake_rst}), "
              f"alpha_cls={self.alpha_cls}")

        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer)

        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(
                self.classes, self.prompt_templates, average=True).squeeze()

        self.marginal_buf = []
        self.batch_count = 0
        self.total_batches = 0
        self.current_mode = MODE_AGGRESSIVE
        self.current_rst = 0.0

        if self.runtime:
            self.adapt_times = []; self.eval_times = []

    def adapt(self, x): return self.perform_adaptation(x)
    def continual_adapt(self, x): return self.perform_adaptation(x)

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
        if self.runtime: self.eval_times.append(time.time() - t1)
        return logits

    def reset(self):
        self.load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state)

    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []

        for _ in range(self.steps):
            logits, _, _, cls_logits = self.model(
                x, self.text_x[:-1], True,
                interpolate=False,
                vision_outputs=self.vision_outputs,
                return_vanilla_cls=True,
                vision_out_type="mean")
            probs = logits.softmax(dim=2)            # (T, B, C, h, w)

            with torch.no_grad():
                ens_probs = probs.mean(dim=0)            # (B, C, h, w)
                pseudo = ens_probs.argmax(dim=1)
                # marginal for gate
                self.marginal_buf.append(
                    ens_probs.mean(dim=[0, 2, 3]).detach().float().cpu())
                # TopK mask
                max_p = ens_probs.max(dim=1).values
                flat = max_p.reshape(-1)
                k = max(1, int(self.top_k_percent * flat.numel()))
                threshold = flat.topk(k, sorted=False).values.min()
                mask = (max_p >= threshold).to(probs.dtype)
                denom = mask.sum().clamp_min(1.0)

            T_, B_, C_, H_, W_ = probs.shape
            pseudo_exp = pseudo.unsqueeze(0).unsqueeze(2).expand(T_, B_, 1, H_, W_)
            probs_y = probs.gather(2, pseudo_exp).squeeze(2)
            min_p = probs_y.min(dim=0).values

            loss_per_pixel = -torch.log(min_p.clamp_min(1e-12))
            loss_pix = (loss_per_pixel * mask).sum() / denom

            if self.alpha_cls > 0:
                entropy_per_cls = self.softmax_entropy(cls_logits, dim=2)
                loss = loss_pix + self.alpha_cls * entropy_per_cls.mean()
            else:
                loss = loss_pix

            loss_report.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.current_rst > 0.0:
                self._stochastic_restore_flat(self.current_rst)

        self.batch_count += 1
        self.total_batches += 1
        if self.batch_count >= self.monitor_interval:
            self._update_mode()
        if self.runtime: self.adapt_times.append(time.time() - t1)
        return loss_report

    @staticmethod
    def _pick_mode(h, h_threshold, h_warning):
        if h >= h_threshold: return MODE_AGGRESSIVE
        if h >= h_warning:   return MODE_CAUTIOUS
        return MODE_BRAKE

    def _mode_to_rst(self, mode):
        if mode == MODE_AGGRESSIVE: return 0.0
        if mode == MODE_CAUTIOUS:   return self.cautious_rst
        if mode == MODE_BRAKE:      return self.brake_rst
        raise ValueError(mode)

    @torch.no_grad()
    def _update_mode(self):
        if len(self.marginal_buf) == 0:
            self.batch_count = 0; return
        agg = torch.stack(self.marginal_buf, dim=0).mean(dim=0)
        agg = agg / agg.sum().clamp(min=1e-8)
        h = -(agg * agg.clamp(min=1e-12).log()).sum().item()
        new_mode = self._pick_mode(h, self.h_threshold, self.h_warning)
        if new_mode != self.current_mode:
            print(f"[MLMP-TM-DivGate] B{self.total_batches}: H={h:.3f}  "
                  f"{self.current_mode}->{new_mode}")
        self.current_mode = new_mode
        self.current_rst = self._mode_to_rst(new_mode)
        self.marginal_buf.clear()
        self.batch_count = 0

    @torch.no_grad()
    def _stochastic_restore_flat(self, rst):
        for name, p in self.named_ln_params:
            mask = (torch.rand(p.shape, device=p.device) < rst).to(p.dtype)
            src = self.model_state[name].to(p.device, dtype=p.dtype)
            p.data.mul_(1.0 - mask).add_(src * mask)

    @staticmethod
    def softmax_entropy(x, dim=-3):
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
