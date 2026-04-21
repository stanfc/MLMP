from copy import deepcopy
import time

import torch
import torch.nn as nn
import torch.jit
import torch.optim as optim

import torchvision.transforms as transforms

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


def get_tta_transforms(img_size=224, gaussian_std: float = 0.005, soft=False):
    n_pixels = img_size

    clip_min, clip_max = 0.0, 1.0

    p_hflip = 0.5

    tta_transforms = transforms.Compose([
        transforms.Lambda(lambda x: x.clamp(clip_min, clip_max)),
        transforms.ColorJitter(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        transforms.Lambda(lambda x: x + gaussian_std * torch.randn_like(x)),
        transforms.Lambda(lambda x: x.clamp(clip_min, clip_max)),
    ])
    return tta_transforms


def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha_teacher).add_(param.data, alpha=1 - alpha_teacher)
    return ema_model


class CoTTA(nn.Module):
    """CoTTA adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, ovss_type, ovss_backbone, lr, classes,
                 vision_outputs=(-1,), alpha_cls=0.0, steps=1, episodic=False,
                 prompt_dir='prompts.yaml', prompt_integration='text',
                 mt=0.999, rst=0.01, ap=0.92,
                 aug_n=32, runtime_calculation=False, device='cpu',
                 finetune_mode='ln',
                 catseg_checkpoint=None):
        super().__init__()
        assert finetune_mode in ('ln', 'full'), f"finetune_mode must be 'ln' or 'full', got {finetune_mode!r}"

        self.vision_outputs = vision_outputs
        self.alpha_cls = alpha_cls
        self.steps = steps
        assert steps > 0, "cotta requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.prompt_integration = prompt_integration
        self.mt = mt
        self.rst = rst
        self.ap = ap
        self.aug_n = aug_n
        self.runtime = runtime_calculation
        self.device = device
        self.finetune_mode = finetune_mode
        self.classes = classes
        self.catseg_checkpoint = catseg_checkpoint
        self.is_catseg = (ovss_type == 'catseg')

        print(f"+++ CoTTA: vision_outputs={self.vision_outputs}, mt={mt}, rst={rst}, ap={ap}, finetune_mode={finetune_mode}")

        # Load NA-CLIP
        self.model, self.tokenize = load_ovss(
            ovss_type, ovss_backbone, device=device,
            classes=self.classes, catseg_checkpoint=self.catseg_checkpoint,
        )
        self.model = configure_model(self.model, finetune_mode)
        params, _ = collect_params(self.model, finetune_mode)

        if self.is_catseg:
            self.model.aggregator = set_ln_grads(self.model.aggregator)
            self.model.sem_seg_head = set_ln_grads(self.model.sem_seg_head)
            agg_params, _ = collect_ln_params(self.model.aggregator)
            head_params, _ = collect_ln_params(self.model.sem_seg_head)
            existing_ids = {id(p) for p in params}
            for p in agg_params + head_params:
                if id(p) not in existing_ids:
                    params.append(p)
                    existing_ids.add(id(p))
            print(f"+++ CAT-Seg CoTTA: total LN params for TTA: {len(params)}")

        print_clip_parameters(self.model)
        self.optimizer = optim.Adam(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        # Prompt templates
        if prompt_dir:
            self.prompt_templates = load_prompts_from_yaml(prompt_dir)
            print(f"Number of prompt templates: {len(self.prompt_templates)}")
        else:
            self.prompt_templates = [REFERENCE_PROMPT]

        assert prompt_integration in ['loss', 'text']

        # Pre-compute text embeddings (frozen)
        if not self.is_catseg:
            with torch.no_grad():
                self.text_x = self._extract_text_embeddings(
                    classes, self.prompt_templates, average=True
                ).squeeze()  # (num_templates+1, num_classes, embed_dim)
        else:
            self.text_x = None

        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.transform = get_tta_transforms(img_size=224)

        if self.runtime:
            self.adapt_times = []
            self.eval_times = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, x):
        if self.episodic:
            self.reset()
        loss_report = []
        for _ in range(self.steps):
            outputs, loss_val = self.forward_and_adapt(x, self.model, self.optimizer)
            loss_report.append(loss_val)
        return outputs, loss_report

    def adapt(self, x):
        """Standard TTA: reset then adapt. Returns per-iteration loss list."""
        self.episodic = True
        _, loss_report = self.forward(x)
        return loss_report

    def continual_adapt(self, x):
        """Continual TTA: adapt without resetting (state carries across batches)."""
        self.episodic = False
        _, loss_report = self.forward(x)
        return loss_report

    @torch.no_grad()
    def evaluate(self, x):
        """Inference with current student model."""
        t1 = time.time()
        if self.is_catseg:
            logits, _, _ = self.model(x, interpolate=True)
        else:
            logits, _, _ = self.model(
                x, self.text_x[-1], text_ensemble=True,
                vision_outputs=self.vision_outputs,
                interpolate=True,
                vision_out_type="adaptive_weighted_mean",
                save_weights=True,
            )
        logits = logits[0]  # (batch, num_classes, H, W)
        if self.runtime:
            self.eval_times.append(time.time() - t1)
        return logits

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)

    # ------------------------------------------------------------------
    # Adaptation loop
    # ------------------------------------------------------------------

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
        self.model_ema.train()

        # Student forward
        if self.is_catseg:
            outputs, _, _ = self.model(x, interpolate=False)
            cls_logits = None
        elif self.prompt_integration == 'loss':
            outputs, _, _, cls_logits = self.model(
                x, self.text_x[:-1], text_ensemble=True,
                vision_outputs=self.vision_outputs,
                interpolate=False,
                return_vanilla_cls=True,
                vision_out_type="mean",
            )
            # outputs:     (T, B, C, H, W)
            # cls_logits:  (T, B, C)
        else:
            outputs, _, _ = self.model(
                x, self.text_x[-1], text_ensemble=True,
                vision_outputs=self.vision_outputs,
                interpolate=False,
                vision_out_type="mean",
            )
            cls_logits = None

        # Teacher Prediction
        if self.is_catseg:
            anchor_prob = self.model_anchor(
                x, interpolate=False,
            )[0][0].softmax(dim=1).max(dim=1)[0]  # (B, H, W)
            standard_ema = self.model_ema(
                x, interpolate=False,
            )[0][0]  # (B, C, H, W)
        else:
            anchor_prob = self.model_anchor(
                x, self.text_x[-1], text_ensemble=True,
                vision_outputs=self.vision_outputs,
                interpolate=False,
                vision_out_type="mean",
            )[0][0].softmax(dim=1).max(dim=1)[0]  # (B, H, W)
            standard_ema = self.model_ema(
                x, self.text_x[-1], text_ensemble=True,
                vision_outputs=self.vision_outputs,
                interpolate=False,
                vision_out_type="mean",
            )[0][0]  # (B, C, H, W)

        # Augmentation-averaged Prediction
        outputs_emas = []
        to_aug = anchor_prob.mean() < self.ap
        if to_aug:
            for i in range(self.aug_n):
                if self.is_catseg:
                    outputs_ = self.model_ema(
                        self.transform(x), interpolate=False,
                    )[0][0].detach()
                else:
                    outputs_ = self.model_ema(
                        self.transform(x), self.text_x[-1], text_ensemble=True,
                        vision_outputs=self.vision_outputs,
                        interpolate=False,
                        vision_out_type="mean",
                    )[0][0].detach()
                outputs_emas.append(outputs_)

        # Threshold choice discussed in supplementary
        if to_aug:
            outputs_ema = torch.stack(outputs_emas).mean(0)
        else:
            outputs_ema = standard_ema

        # Student update
        loss = softmax_entropy(outputs.mean(0), outputs_ema.detach()).mean()
        if cls_logits is not None and self.alpha_cls > 0:
            loss = loss + self.alpha_cls * softmax_entropy(cls_logits.mean(0), dim=2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Teacher update
        self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=self.mt)

        # Stochastic restore
        for nm, m in self.model.named_modules():
            for npp, p in m.named_parameters(recurse=False):
                if npp in ['weight', 'bias'] and p.requires_grad:
                    mask = (torch.rand(p.shape) < self.rst).float().to(p.device)
                    with torch.no_grad():
                        p.data = self.model_state[f"{nm}.{npp}"] * mask + p * (1. - mask)

        return outputs_ema, loss.item()

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
def softmax_entropy(x, x_ema):  # -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -0.5*(x_ema.softmax(1) * x.log_softmax(1)).sum(1)-0.5*(x.softmax(1) * x_ema.log_softmax(1)).sum(1)


def collect_params(model, finetune_mode='ln'):
    """Collect trainable parameters from the visual encoder.

    Walk the model's modules and collect parameters.
    finetune_mode='ln'   : LayerNorm weight & bias only (same spirit as BN-only in original)
    finetune_mode='full' : all visual encoder parameters
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.visual.named_modules():
        if finetune_mode == 'full' or isinstance(m, nn.LayerNorm):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"visual.{nm}.{np}")
                    print(nm, np)
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model, finetune_mode='ln'):
    """Configure model for use with cotta."""
    # train mode, because cotta optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what we update
    model.requires_grad_(False)
    # freeze text encoder entirely
    model.transformer.requires_grad_(False)
    model.ln_final.requires_grad_(False)
    model.token_embedding.requires_grad_(False)
    # enable visual encoder: LayerNorm only, or all params
    for m in model.visual.modules():
        if finetune_mode == 'full' or isinstance(m, nn.LayerNorm):
            m.requires_grad_(True)
    return model


def set_ln_grads(module):
    """Disable grads globally on *module*, then re-enable LayerNorm params."""
    module.requires_grad_(False)
    for m in module.modules():
        if isinstance(m, nn.LayerNorm):
            m.requires_grad_(True)
    return module


def collect_ln_params(module):
    """Collect LayerNorm weight/bias from *module*."""
    params, names = [], []
    for nm, m in module.named_modules():
        if isinstance(m, nn.LayerNorm):
            for np_, p in m.named_parameters():
                if np_ in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np_}")
    return params, names


def check_model(model):
    """Check model for compatibility with cotta."""
    is_training = model.training
    assert is_training, "cotta needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "cotta needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "cotta should not update all params: " \
                               "check which require grad"
    has_ln = any([isinstance(m, nn.LayerNorm) for m in model.modules()])
    assert has_ln, "cotta needs normalization for its optimization"
