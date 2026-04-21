from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import torch.optim as optim

from torch.autograd import Variable
from .prompt_vit import PromptVisualEncoder
import numpy as np
import math

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class DPCore(nn.Module):
    def __init__(self, ovss_type, ovss_backbone, lr, classes,
                 vision_outputs=(-1,), steps=1, episodic=False,
                 prompt_dir='prompts.yaml',
                 temp_tau=3.0, ema_alpha=0.999, thr_rho=0.8,
                 prompt_num=1, runtime_calculation=False, verbose_dpcore=False,
                 micro_batch_size=None, catseg_checkpoint=None, device='cpu'):
        super().__init__()
        self.lamda = 1.0
        self.lr = lr
        self.verbose = verbose_dpcore
        self.temp_tau = temp_tau
        self.ema_alpha = ema_alpha
        self.thr_rho = thr_rho
        self.E_ID = 1
        self.E_OOD = steps

        self.vision_outputs = vision_outputs
        self.steps = steps
        assert steps > 0, "dpcore requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.runtime = runtime_calculation
        self.micro_batch_size = micro_batch_size
        self.catseg_checkpoint = catseg_checkpoint
        self.is_catseg = (ovss_type == 'catseg')
        self.classes = classes
        self.device = device

        print(f"+++ DPCore: vision_outputs={vision_outputs}, temp_tau={temp_tau}, thr_rho={thr_rho}, ema_alpha={ema_alpha}")

        # Load NA-CLIP and wrap visual encoder with prompt injection
        base_model, self.tokenize = load_ovss(ovss_type, ovss_backbone, device=device,
                                              classes=self.classes, catseg_checkpoint=self.catseg_checkpoint)
        self.model = configure_model(base_model, prompt_num)
        prompt_params = collect_params(self.model)
        print_clip_parameters(self.model)
        self.optimizer = optim.AdamW(prompt_params, lr=lr)
        print_optimizer_parameters(self.optimizer, self.model)

        # Prompt templates
        if prompt_dir:
            self.prompt_templates = load_prompts_from_yaml(prompt_dir)
            print(f"Number of prompt templates: {len(self.prompt_templates)}")
        else:
            self.prompt_templates = [REFERENCE_PROMPT]

        # Pre-compute text embeddings (frozen)
        if not self.is_catseg:
            with torch.no_grad():
                self.text_x = self._extract_text_embeddings(
                    classes, self.prompt_templates, average=True
                ).squeeze()  # (num_templates+1, num_classes, embed_dim)
        else:
            self.text_x = None

        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

        self.coreset = []
        self.train_info = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, x):
        if self.episodic:
            self.reset()
        outputs, loss_raw, loss_new, loss = self.forward_and_adapt(x, self.model, self.optimizer)
        return outputs, loss_raw, loss_new, loss

    def adapt(self, x):
        """Standard TTA: reset then adapt. Returns per-iteration loss list."""
        self.episodic = True
        _, loss_raw, loss_new, loss = self.forward(x)
        return [loss.item()]

    def continual_adapt(self, x):
        """Continual TTA: adapt without resetting (state carries across batches)."""
        self.episodic = False
        _, loss_raw, loss_new, loss = self.forward(x)
        return [loss.item()]

    @torch.no_grad()
    def evaluate(self, x):
        """Inference with current student model."""
        mbs = self.micro_batch_size
        if self.is_catseg:
            if mbs is not None and mbs < x.shape[0]:
                chunks = [x[i:i+mbs] for i in range(0, x.shape[0], mbs)]
                logits = torch.cat([
                    self.model(chunk, interpolate=True)[0][0]
                    for chunk in chunks
                ], dim=0)
            else:
                logits, _, _ = self.model(x, interpolate=True)
                logits = logits[0]
        else:
            if mbs is not None and mbs < x.shape[0]:
                chunks = [x[i:i+mbs] for i in range(0, x.shape[0], mbs)]
                logits = torch.cat([
                    self.model(chunk, self.text_x[-1], text_ensemble=True,
                               vision_outputs=self.vision_outputs,
                               interpolate=True,
                               vision_out_type="adaptive_weighted_mean",
                               save_weights=True)[0][0]
                    for chunk in chunks
                ], dim=0)
            else:
                logits, _, _ = self.model(
                    x, self.text_x[-1], text_ensemble=True,
                    vision_outputs=self.vision_outputs,
                    interpolate=True,
                    vision_out_type="adaptive_weighted_mean",
                    save_weights=True,
                )
                logits = logits[0]
        return logits  # (batch, num_classes, H, W)

    def reset(self):
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.coreset = []

    # ------------------------------------------------------------------
    # Coreset operations
    # ------------------------------------------------------------------

    def _update_coreset(self, weights, batch_mean, batch_std):
        """Update overall test statistics"""
        updated_prompts = self.model.visual.prompts.clone().detach().cpu()
        for p_idx in range(len(self.coreset)):
            self.coreset[p_idx][0] += self.ema_alpha * weights[p_idx] * (batch_mean - self.coreset[p_idx][0])
            self.coreset[p_idx][1] += self.ema_alpha * weights[p_idx] * torch.clamp(batch_std - self.coreset[p_idx][1], min=0.0)
            self.coreset[p_idx][2] += self.ema_alpha * weights[p_idx] * (updated_prompts - self.coreset[p_idx][2])

    @torch.no_grad()
    def _eval_coreset(self, x):
        """Evaluate the coreset on a batch of samples."""
        if self.train_info is None:
            raise RuntimeError("train_info is not set. Call obtain_src_stat() before adaptation.")
        loss, batch_mean, batch_std = forward_and_get_loss(x, self.model, self.lamda, self.train_info,
                                                           with_prompt=False, micro_batch_size=self.micro_batch_size)
        is_ID = False
        weights = None
        weighted_prompts = None
        if self.coreset:
            weights = calculate_weights(self.coreset, batch_mean, batch_std, self.lamda, self.temp_tau)
            weighted_prompts = torch.stack([w * p[2] for w, p in zip(weights, self.coreset)], dim=0).sum(dim=0)
            assert weighted_prompts.shape == self.model.visual.prompts.shape, f'{weighted_prompts.shape} != {self.model.visual.prompts.shape}'
            self.model.visual.prompts = torch.nn.Parameter(weighted_prompts.to(self.device))
            self.model.visual.prompts.requires_grad_(False)

            loss_new, _, _ = forward_and_get_loss(x, self.model, self.lamda, self.train_info,
                                                  with_prompt=True, micro_batch_size=self.micro_batch_size)
            if self.verbose:
                print(f"[coreset eval] loss_raw={loss:.4f}, loss_new={loss_new:.4f}, ratio={loss_new/loss:.4f}, thr={self.thr_rho}, is_ID={loss_new < loss * self.thr_rho}")
            if loss_new < loss * self.thr_rho:
                self.model.visual.prompts.requires_grad_(True)
                self.optimizer = torch.optim.AdamW([self.model.visual.prompts], lr=self.lr)
                is_ID = True
        else:
            loss_new = loss

        return is_ID, batch_mean, batch_std, weighted_prompts, weights, loss, loss_new

    # ------------------------------------------------------------------
    # Adaptation loop
    # ------------------------------------------------------------------

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        is_ID, batch_mean, batch_std, weighted_prompts, weights, loss_raw, loss_new = self._eval_coreset(x)
        if is_ID:
            for step in range(self.E_ID):
                self.model.visual.prompts = torch.nn.Parameter(weighted_prompts.to(self.device))
                optimizer = torch.optim.AdamW([self.model.visual.prompts], lr=self.lr)
                outputs, loss, batch_mean, batch_std = forward_and_adapt(
                    x, self.model, optimizer, self.lamda, self.train_info,
                    self.text_x, self.vision_outputs,
                    return_output=(step == self.E_ID - 1),
                    micro_batch_size=self.micro_batch_size,
                    is_catseg=self.is_catseg)
            self._update_coreset(weights, batch_mean, batch_std)

        else:
            load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
            self.model.visual.prompts.requires_grad_(True)
            self.optimizer = torch.optim.AdamW([self.model.visual.prompts], lr=self.lr)

            for step in range(self.E_OOD):
                outputs, loss, _, _ = forward_and_adapt(
                    x, self.model, self.optimizer, self.lamda, self.train_info,
                    self.text_x, self.vision_outputs,
                    return_output=(step == self.E_OOD - 1),
                    micro_batch_size=self.micro_batch_size,
                    is_catseg=self.is_catseg)
                if self.verbose:
                    print(f"  [OOD step {step:03d}] loss={loss.item():.4f}")

            self.coreset.append([batch_mean, batch_std, self.model.visual.prompts.clone().detach().cpu()])

        return outputs, loss_raw, loss_new, loss

    # ------------------------------------------------------------------
    # Source statistics
    # ------------------------------------------------------------------

    def obtain_src_stat(self, data_loader, num_samples=5000):
        num = 0
        features = []
        mbs = self.micro_batch_size
        with torch.no_grad():
            for data in data_loader:
                images = data['img_patches'].to(self.device)
                if self.is_catseg:
                    if mbs is not None and mbs < images.shape[0]:
                        feature = torch.cat([forward_raw_features(self.model, images[i:i+mbs])
                                             for i in range(0, images.shape[0], mbs)], dim=0)
                        logits_list = [self.model(images[i:i+mbs], interpolate=False)[0][0]
                                       for i in range(0, images.shape[0], mbs)]
                        logits_0 = torch.cat(logits_list, dim=0)
                    else:
                        feature = forward_raw_features(self.model, images)
                        logits, _, _ = self.model(images, interpolate=False)
                        logits_0 = logits[0]
                else:
                    if mbs is not None and mbs < images.shape[0]:
                        feature = torch.cat([forward_raw_features(self.model, images[i:i+mbs])
                                             for i in range(0, images.shape[0], mbs)], dim=0)
                        logits_list = [self.model(images[i:i+mbs], self.text_x[-1], text_ensemble=True,
                                                 vision_outputs=self.vision_outputs,
                                                 interpolate=False, vision_out_type="mean")[0][0]
                                       for i in range(0, images.shape[0], mbs)]
                        logits_0 = torch.cat(logits_list, dim=0)
                    else:
                        feature = forward_raw_features(self.model, images)
                        logits, _, _ = self.model(
                            images, self.text_x[-1], text_ensemble=True,
                            vision_outputs=self.vision_outputs,
                            interpolate=False,
                            vision_out_type="mean",
                        )
                        logits_0 = logits[0]

                ent = softmax_entropy(logits_0)
                num_classes = logits_0.shape[1]
                selected_indices = torch.where(ent.mean(dim=(-2, -1)) < math.log(num_classes) / 2 - 1)[0]
                feature = feature[selected_indices]

                features.append(feature[:, 0])
                num += feature.shape[0]
                if num >= num_samples:
                    break

            features = torch.cat(features, dim=0)
            features = features[:num_samples, :]
            print(f'Source Statistics computed with {features.shape[0]} examples.')
            self.train_info = torch.std_mean(features, dim=0)
        del features

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temperature = 1
    x = x / temperature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x


def collect_params(model):
    return [model.visual.prompts]


def forward_raw_features(model, images):
    """Extract CLS features without prompts (raw visual encoder features)."""
    return model.visual.forward_raw_features(images)


def forward_and_get_loss(images, model, lamda, train_info, with_prompt=False, micro_batch_size=None):
    fwd = model.visual.forward_features if with_prompt else model.visual.forward_raw_features
    mbs = micro_batch_size
    if mbs is not None and mbs < images.shape[0]:
        cls_features = torch.cat([fwd(images[i:i+mbs])[:, 0] for i in range(0, images.shape[0], mbs)], dim=0)
    else:
        cls_features = fwd(images)[:, 0]

    """discrepancy loss"""
    batch_std, batch_mean = torch.std_mean(cls_features, dim=0)
    std_loss = torch.norm(batch_std - train_info[0].to(cls_features.device), p=2)
    mean_loss = torch.norm(batch_mean - train_info[1].to(cls_features.device), p=2)

    loss = lamda * std_loss + mean_loss

    return loss, batch_mean, batch_std


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def calculate_weights(coreset, batch_mean, batch_std, lamda, temp_tau):
    mean_tensor = torch.stack([p[0] for p in coreset])
    std_tensor = torch.stack([p[1] for p in coreset])

    mean_match = torch.norm(batch_mean - mean_tensor, p=2, dim=1)
    std_match = torch.norm(batch_std - std_tensor, p=2, dim=1)

    match_loss = mean_match + lamda * std_match
    weights = torch.nn.functional.softmax(-match_loss / temp_tau, dim=0)
    return weights.detach().cpu()


@torch.enable_grad()
def forward_and_adapt(x, model, optimizer, lamda, train_info, text_x, vision_outputs,
                      return_output=False, micro_batch_size=None, is_catseg=False):
    """Forward and adapt model on batch of data.
    Measure discrepancy loss, take gradients, and update params.

    When micro_batch_size is set, each chunk independently computes its own
    mean/std loss against the source statistics and accumulates gradients.
    Only one chunk's computation graph lives in memory at a time.

    Only computes segmentation output when return_output=True (last step only).
    """
    mbs = micro_batch_size
    if mbs is not None and mbs < x.shape[0]:
        n_chunks = math.ceil(x.shape[0] / mbs)
        # Accumulate detached stats for reporting
        all_cls = []
        for i in range(0, x.shape[0], mbs):
            cls_feat = model.visual.forward_features(x[i:i+mbs])[:, 0]
            all_cls.append(cls_feat.detach())
            chunk_std, chunk_mean = torch.std_mean(cls_feat, dim=0)
            std_loss = torch.norm(chunk_std - train_info[0].to(cls_feat.device), p=2)
            mean_loss = torch.norm(chunk_mean - train_info[1].to(cls_feat.device), p=2)
            ((lamda * std_loss + mean_loss) / n_chunks).backward()

        optimizer.step()
        optimizer.zero_grad()

        # Report full-batch statistics
        with torch.no_grad():
            cls_all = torch.cat(all_cls, dim=0)
            batch_std, batch_mean = torch.std_mean(cls_all, dim=0)
            loss = lamda * torch.norm(batch_std - train_info[0].to(batch_std.device), p=2) \
                 + torch.norm(batch_mean - train_info[1].to(batch_mean.device), p=2)
    else:
        features = model.visual.forward_features(x)
        cls_features = features[:, 0]
        batch_std, batch_mean = torch.std_mean(cls_features, dim=0)

        std_loss = torch.norm(batch_std - train_info[0].to(cls_features.device), p=2)
        mean_loss = torch.norm(batch_mean - train_info[1].to(cls_features.device), p=2)
        loss = lamda * std_loss + mean_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    output = None
    if return_output:
        with torch.no_grad():
            if is_catseg:
                if mbs is not None and mbs < x.shape[0]:
                    output = torch.cat([
                        model(x[i:i+mbs], interpolate=False)[0][0]
                        for i in range(0, x.shape[0], mbs)
                    ], dim=0)
                else:
                    out, _, _ = model(x, interpolate=False)
                    output = out[0]
            else:
                if mbs is not None and mbs < x.shape[0]:
                    output = torch.cat([
                        model(x[i:i+mbs], text_x[-1], text_ensemble=True,
                              vision_outputs=vision_outputs,
                              interpolate=False, vision_out_type="mean")[0][0]
                        for i in range(0, x.shape[0], mbs)
                    ], dim=0)
                else:
                    out, _, _ = model(
                        x, text_x[-1], text_ensemble=True,
                        vision_outputs=vision_outputs,
                        interpolate=False,
                        vision_out_type="mean",
                    )
                    output = out[0]

    return output, loss, batch_mean, batch_std


def configure_model(model, prompt_num=1):
    """Configure model for use with dpcore."""
    # train mode, because dpcore optimizes the model to minimize discrepancy
    model.train()
    # disable grad for everything
    model.requires_grad_(False)
    # freeze text encoder entirely
    model.transformer.requires_grad_(False)
    model.ln_final.requires_grad_(False)
    model.token_embedding.requires_grad_(False)
    # wrap visual encoder with prompt injection; only prompts are trainable
    model.visual = PromptVisualEncoder(model.visual, prompt_num)
    return model


def check_model(model):
    """Check model for compatibility with dpcore."""
    is_training = model.training
    assert is_training, "dpcore needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "dpcore needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "dpcore should not update all params: " \
                               "check which require grad"
