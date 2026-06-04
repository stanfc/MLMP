"""
MLMP-TopK-MinPrompt-SmoothAnchor-Continual: TopK + MinPrompt base loss
combined with smooth-anchor (continuous H_margin -> lag) gate.
"""

import time
import copy
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class MLMPTopKMinPromptSmoothAnchorContinual:

    def __init__(self, ovss_type, ovss_backbone, lr, classes,
                 vision_outputs=(-1,), alpha_cls=0.0, steps=1,
                 top_k_percent=0.2,
                 h_ceil=1.8, h_floor=1.5,
                 lag_scale=90.0,
                 max_lag=3000,
                 rst=0.005,
                 monitor_interval=50,
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
        self.h_ceil = float(h_ceil); self.h_floor = float(h_floor)
        if self.h_floor >= self.h_ceil: raise ValueError("h_floor >= h_ceil")
        self.lag_scale = float(lag_scale)
        self.max_lag = int(max_lag)
        self.rst = float(rst)
        self.monitor_interval = int(monitor_interval)
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
        print(f"+++ MLMP-TopK-MinPrompt-SmoothAnchor: T={len(self.prompt_templates)}, "
              f"top_k={self.top_k_percent}, "
              f"h_ceil={self.h_ceil}, h_floor={self.h_floor}, "
              f"lag_scale={self.lag_scale}, max_lag={self.max_lag}, rst={self.rst}, "
              f"alpha_cls={self.alpha_cls}")

        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer)
        self._source_ln_snapshot = {name: self.model_state[name].detach().clone()
                                     for name, _ in self.named_ln_params}
        self._anchor_buf = deque(maxlen=self.max_lag + 1)
        self._anchor_buf.append(self._snapshot_ln_weights())

        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(
                self.classes, self.prompt_templates, average=True).squeeze()

        self.marginal_buf = []
        self.batch_count = 0
        self.total_batches = 0
        self.current_lag = 0
        self.current_rst = 0.0

        if self.runtime:
            self.adapt_times = []; self.eval_times = []

    def _lag_from_h(self, h):
        if h >= self.h_ceil: return 0
        if h <= self.h_floor: return None
        raw = self.lag_scale / (h - self.h_floor)
        if raw > self.max_lag: return None
        return max(1, int(round(raw)))

    @torch.no_grad()
    def _snapshot_ln_weights(self):
        return {name: p.detach().cpu().to(torch.float16).clone()
                for name, p in self.named_ln_params}

    @torch.no_grad()
    def _push_current_to_anchor_buf(self):
        self._anchor_buf.append(self._snapshot_ln_weights())

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
        self._anchor_buf.clear()
        self._anchor_buf.append(self._snapshot_ln_weights())

    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []
        self._push_current_to_anchor_buf()

        for _ in range(self.steps):
            logits, _, _, cls_logits = self.model(
                x, self.text_x[:-1], True,
                interpolate=False,
                vision_outputs=self.vision_outputs,
                return_vanilla_cls=True,
                vision_out_type="mean")
            probs = logits.softmax(dim=2)

            with torch.no_grad():
                ens_probs = probs.mean(dim=0)
                pseudo = ens_probs.argmax(dim=1)
                self.marginal_buf.append(
                    ens_probs.mean(dim=[0, 2, 3]).detach().float().cpu())
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
                self._stochastic_restore(self.current_rst, self.current_lag)

        self.batch_count += 1
        self.total_batches += 1
        if self.batch_count >= self.monitor_interval:
            self._update_lag()
        if self.runtime: self.adapt_times.append(time.time() - t1)
        return loss_report

    @torch.no_grad()
    def _update_lag(self):
        if len(self.marginal_buf) == 0:
            self.batch_count = 0; return
        agg = torch.stack(self.marginal_buf, dim=0).mean(dim=0)
        agg = agg / agg.sum().clamp(min=1e-8)
        h = -(agg * agg.clamp(min=1e-12).log()).sum().item()
        new_lag = self._lag_from_h(h)
        new_rst = 0.0 if new_lag == 0 else self.rst

        def _label(lg):
            if lg is None: return "source"
            if lg == 0: return "off"
            return f"lag={lg}"
        old = self.current_lag
        log_change = (old == 0) != (new_lag == 0) or (old is None) != (new_lag is None)
        if isinstance(old, int) and old > 0 and isinstance(new_lag, int) and new_lag > 0:
            if abs(new_lag - old) / max(old, 1) > 0.25: log_change = True
        if log_change:
            print(f"[MLMP-TM-SA] B{self.total_batches}: H={h:.3f}  "
                  f"{_label(old)} -> {_label(new_lag)}")
        self.current_lag = new_lag
        self.current_rst = new_rst
        self.marginal_buf.clear()
        self.batch_count = 0

    @torch.no_grad()
    def _stochastic_restore(self, rst, lag):
        if lag is None:
            anchor = self._source_ln_snapshot
        else:
            idx = -1 - lag
            if -idx > len(self._anchor_buf):
                anchor = self._anchor_buf[0]
            else:
                anchor = self._anchor_buf[idx]
        for name, p in self.named_ln_params:
            mask = (torch.rand(p.shape, device=p.device) < rst).to(p.dtype)
            src = anchor[name].to(p.device, dtype=p.dtype)
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
