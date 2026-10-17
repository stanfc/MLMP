import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class DAFMLMPContinual:
    """
    Port of DAF (Diversify, Anchor, Filter) on top of MLMP-continual, from the
    official repo https://github.com/chandlerbing65nm/DAF (itself a fork of
    dosowiechi/MLMP -- our own upstream). Paper: "Towards Continual Test-Time
    Adaptation of Vision-Language Models in Open-Vocabulary Semantic
    Segmentation" (Doloriel et al. 2026, arXiv:2608.29923).

    Adds three mechanisms on top of MLMP's entropy+ILE loss, ported verbatim
    from adapt/mlmp.py in the DAF repo:
      - Diversify (loss_div): maximize entropy of the marginal class
        distribution over pixels, to resist collapse to one class.
      - Anchor (loss_cmac): cross-modal anchor consistency -- penalizes the
        adapted model's similarity to the source-assigned class dropping
        below the frozen source model's similarity ("loss_away"), and
        penalizes similarity to OTHER classes rising above the source's
        ("loss_toward"). Requires a frozen deep-copied source model.
      - Filter (module_safs): per-sample "salience-aware feature skip" --
        compute each sample's average patch-level feature shift vs the
        source model, then keep only samples whose shift exceeds
        (batch mean - alpha_safs * batch std), skipping the backward pass
        for the rest. This is a BATCH statistic: at batch_size=1 the
        keep-mask degenerates to always-empty (shift never exceeds its own
        mean), so the Filter component is a no-op unless batch_size > 1.
        The DAF paper itself uses batch_size=8 -- we follow that here
        rather than our usual batch_size=1, matching the precedent already
        set for DPCore (batch_size=16) in this codebase.

    Only visual encoder LayerNorm (gamma, beta) are updated; text encoder
    and the frozen source_model copy are never trained. No reset() call in
    continual_adapt() -- state persists across the whole stream.
    """

    def __init__(self, ovss_type, ovss_backbone, lr, classes,
                 vision_outputs=(-1,), alpha_cls=0.0, steps=1,
                 prompt_dir='prompts.yaml', prompt_integration='loss',
                 lamb_ent=1.0, lamb_div=2.0, lamb_cmac=0.5, alpha_safs=0.5,
                 optimizer='sgd',
                 runtime_calculation=False, device='cpu'):
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.optimizer_name = optimizer
        self.steps = steps
        self.vision_outputs = vision_outputs
        self.alpha_cls = alpha_cls
        self.prompt_dir = prompt_dir
        self.prompt_integration = prompt_integration
        self.lamb_ent = lamb_ent
        self.lamb_div = lamb_div
        self.lamb_cmac = lamb_cmac
        self.alpha_safs = alpha_safs
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

        print(f"+++ Vision outputs (UAML layers): {self.vision_outputs}")
        assert prompt_integration in ['loss', 'text'], \
            "prompt_integration must be 'loss' or 'text'"

        # ---------- Freeze text encoder, enable LN grads in visual encoder ----------
        self.model.transformer.requires_grad_(False)
        self.model.ln_final.requires_grad_(False)
        self.model.token_embedding.requires_grad_(False)
        self.model.visual = self.set_ln_grads(self.model.visual)
        params, _ = self.collect_ln_params(self.model.visual)

        print_clip_parameters(self.model)

        # ---------- Frozen source model (for Anchor + Filter) ----------
        # Snapshot BEFORE any adaptation happens -- this is the "zero-shot"
        # reference both cmac_loss and the SAFS shift measure drift against.
        self.source_model = copy.deepcopy(self.model)
        self.source_model.requires_grad_(False)
        self.source_model.eval()

        # ---------- Optimizer ----------
        if self.optimizer_name == 'sgd':
            # DAF's own README specifies --optimizer sgd -- Adam's per-parameter
            # adaptive scaling makes the same nominal LR far more aggressive
            # than SGD, which plausibly explains why the Adam version collapsed
            # much faster than DAF's own paper reports.
            self.optimizer = optim.SGD(params, lr=self.lr, momentum=0.9)
        else:
            self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        # ---------- Source state (kept only for reset(), not used in CTTA) ----------
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer
        )

        # ---------- Text embeddings ----------
        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(
                self.classes, self.prompt_templates, average=True
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
        """Alias for adapt() -- compatible with main_continual.py protocol."""
        return self.perform_adaptation(x)

    @torch.no_grad()
    def evaluate(self, x):
        t1 = time.time()
        logits, _, _ = self.model(
            x, self.text_x[-1], True,
            vision_outputs=self.vision_outputs,
            interpolate=True,
            vision_out_type="adaptive_weighted_mean",
            save_weights=True
        )
        logits = logits[0]
        if self.runtime:
            self.eval_times.append(time.time() - t1)
        return logits

    def reset(self):
        """Full reset to source state. NOT called during normal CTTA operation."""
        self.load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state
        )

    def state_dict(self):
        """Adapt-state checkpoint for mid-stream resume: adapted LayerNorm
        params + optimizer state. source_model needs no entry -- it is a
        frozen deepcopy taken at __init__ time and never updated, so a fresh
        process reconstructs an identical copy from the same pretrained
        weights regardless of checkpoint."""
        names, params = self._named_ln_params()
        return {
            'ln_params': {n: p.detach().cpu().clone() for n, p in zip(names, params)},
            'optimizer': self.optimizer.state_dict(),
        }

    def load_state_dict(self, sd):
        names, params = self._named_ln_params()
        with torch.no_grad():
            for n, p in zip(names, params):
                p.data.copy_(sd['ln_params'][n].to(device=p.device, dtype=p.dtype))
        self.optimizer.load_state_dict(sd['optimizer'])
        # optimizer.state_dict() was captured from GPU params but the checkpoint
        # is loaded on CPU (load_checkpoint uses map_location='cpu'); moving
        # state back to the param device is required or the next step() fails.
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

    def _named_ln_params(self):
        params, names = self.collect_ln_params(self.model.visual)
        return names, params

    # ===========================================================
    # Core adaptation logic: MLMP entropy+ILE, plus Diversify/Anchor/Filter
    # ===========================================================

    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []

        for _ in range(self.steps):
            logits, image_features, _, cls_logits = self.model(
                x, self.text_x[:-1], True,
                interpolate=False,
                vision_outputs=self.vision_outputs,
                return_vanilla_cls=True,
                vision_out_type="mean"
            )
            # logits:          (T, B, C, h, w)
            # image_features:  (B, tokens, D), CLS at index 0
            # cls_logits:      (T, B, C, 1, 1)

            with torch.no_grad():
                logits_src, image_features_src, _, _ = self.source_model(
                    x, self.text_x[:-1], True,
                    interpolate=False,
                    vision_outputs=self.vision_outputs,
                    return_vanilla_cls=True,
                    vision_out_type="mean"
                )

            # ---------------- Filter (SAFS): per-sample skip mask ----------------
            keep_mask = None
            f_adapt_patches = image_features[:, 1:]
            f_src_patches = image_features_src[:, 1:]
            cos_sim = (f_adapt_patches * f_src_patches).sum(dim=-1)
            shift = (1.0 - cos_sim).mean(dim=-1)
            threshold = shift.mean() - self.alpha_safs * shift.std(unbiased=False)
            keep_mask = shift > threshold
            if not keep_mask.any():
                keep_mask = None
            elif not keep_mask.all().item():
                logits = logits[:, keep_mask]
                image_features = image_features[keep_mask]
                image_features_src = image_features_src[keep_mask]
                logits_src = logits_src[:, keep_mask]
                if cls_logits is not None:
                    cls_logits = cls_logits[:, keep_mask]

            loss = torch.tensor(0.0, device=x.device)

            # ---------------- entropy + ILE (same as MLMP-continual) ----------------
            entropy_per_pixel = self.softmax_entropy(logits)
            if cls_logits is not None:
                entropy_per_cls = self.softmax_entropy(cls_logits, dim=2)
                loss = loss + self.lamb_ent * entropy_per_pixel.mean() + self.alpha_cls * entropy_per_cls.mean()
            else:
                loss = loss + self.lamb_ent * entropy_per_pixel.mean()

            # ---------------- Diversify: marginal-class entropy ----------------
            div_loss = self.diversity_loss(logits)
            loss = loss + self.lamb_div * div_loss

            # ---------------- Anchor: cross-modal anchor consistency ----------------
            cmac_loss = self.cmac_loss(image_features, image_features_src, self.text_x[:-1])
            loss = loss + cmac_loss

            loss_report.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.runtime:
            self.adapt_times.append(time.time() - t1)

        return loss_report

    # ===========================================================
    # DAF-specific loss terms (ported from adapt/mlmp.py in the DAF repo)
    # ===========================================================

    @staticmethod
    def diversity_loss(logits: torch.Tensor) -> torch.Tensor:
        """Maximize entropy of the marginal class distribution over pixels,
        preventing all pixels from collapsing to a single class.
        logits: [#templates, batch, #classes, W, H]
        """
        p = logits.softmax(dim=-3)
        marginal = p.mean(dim=(1, 3, 4))  # [#templates, #classes]
        entropy = -(marginal * torch.log(marginal + 1e-8)).sum(dim=-1)  # [#templates]
        return -entropy.mean()

    def cmac_loss(self, image_features_adapt: torch.Tensor,
                  image_features_src: torch.Tensor,
                  text_features: torch.Tensor) -> torch.Tensor:
        """Cross-Modal Anchor Consistency: feature-level text-anchored drift.
        image_features_*: [batch, tokens, dim] (normalized), CLS at index 0.
        text_features: [#templates, #classes, dim] or [#classes, dim] (normalized).
        """
        T = text_features.mean(dim=0) if text_features.dim() == 3 else text_features
        num_classes = T.shape[0]

        f_adapt = image_features_adapt[:, 1:]
        f_src = image_features_src[:, 1:]

        logit_scale = self.model.logit_scale.exp()
        sim_adapt = logit_scale * (f_adapt @ T.t())
        sim_src = logit_scale * (f_src @ T.t())

        assigned = sim_src.argmax(dim=-1)

        sim_adapt_assigned = sim_adapt.gather(-1, assigned.unsqueeze(-1)).squeeze(-1)
        sim_src_assigned = sim_src.gather(-1, assigned.unsqueeze(-1)).squeeze(-1)

        loss_away = F.relu(sim_src_assigned - sim_adapt_assigned).mean()

        loss_toward_all = F.relu(sim_adapt - sim_src)
        mask = F.one_hot(assigned, num_classes=num_classes).bool()
        loss_toward_all = loss_toward_all.masked_fill(mask, 0)
        loss_toward = (loss_toward_all.sum(dim=-1) / (num_classes - 1)).mean()

        return self.lamb_cmac * (loss_away + loss_toward)

    # ===========================================================
    # Shared helpers (identical to mlmp_continual.py)
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

    @staticmethod
    def softmax_entropy(x: torch.Tensor, dim=-3) -> torch.Tensor:
        return -(x.softmax(dim) * x.log_softmax(dim)).sum(dim)
