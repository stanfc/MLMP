import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class LCoTTAMLMPContinual:
    """
    LCoTTA + MLMP — MLMP adaptation with LCoTTA's subspace-projected gradient update.

    Base loss / evaluation: identical to mlmp_continual (7-prompt, 18-layer
    mean-fusion adapt + UAML evaluate, LayerNorm-only). MLMP is kept because it
    significantly boosts the base performance.

    LCoTTA mechanism (from "Lifelong Test-Time Adaptation via Online Learning in
    Tracked Low-Dimensional Subspace", NeurIPS 2025, repo ThunderDavid/LCoTTA,
    method `Subspace_plus`): harmful "entropy-deceptive" samples (confident but
    wrong) produce gradients OUTSIDE the dominant correlated subspace of recent
    gradients. Every `batch_step` batches we record the current LN-gradient
    vector into a moving window of size `w_num`; once the window holds
    >= `n_components` vectors we PCA it to get the principal gradient subspace P,
    and PROJECT the current gradient onto P before stepping (P_SGD). Updates are
    confined to the tracked low-dimensional subspace, suppressing the off-subspace
    (deceptive) component that drives long-horizon drift / collapse.

    OUR combination: LCoTTA's update rule on MLMP's loss (the paper used plain
    filtered entropy). No reset — inherently continual.

    Only visual encoder LayerNorm (γ, β) are updated.
    """

    def __init__(self, ovss_type, ovss_backbone, lr, classes,
                 vision_outputs=(-1,), alpha_cls=0.0, steps=1,
                 prompt_dir='prompts.yaml', prompt_integration='loss',
                 w_num=100, n_components=20, batch_step=50,
                 runtime_calculation=False, device='cpu'):
        """
        Args:
            ovss_type: OVSS model identifier (e.g. 'naclip')
            ovss_backbone: Backbone name (e.g. 'ViT-L/14')
            lr: Learning rate for Adam optimizer
            classes: List of class names for text embeddings
            vision_outputs: Tuple of ViT layer indices for UAML (e.g. tuple(range(-1,-19,-1)))
            alpha_cls: Weight for CLS entropy term (ILE). Default 0.0 (disabled)
            steps: Gradient steps per sample — use 1 for online CTTA
            prompt_dir: Path to YAML with prompt templates
            prompt_integration: 'loss' (per-template losses averaged) or 'text' (averaged embedding)
            runtime_calculation: Track adapt/eval wall-clock times
            device: Torch device string
        """
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
        self.ln_params = params   # the only trainable params (LN γ,β); grad vector basis

        print_clip_parameters(self.model)

        # ---------- Optimizer ----------
        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        # ---------- LCoTTA subspace state ----------
        self.w_num = int(w_num)                 # moving window size (gradient vectors kept)
        self.n_components = int(n_components)    # PCA components = subspace dimension
        self.batch_step = int(batch_step)        # record a gradient every this many batches
        self.moving_W = []                       # list of past LN-gradient vectors (np float32)
        self.batch_num = 0
        print(f"+++ LCoTTA subspace: w_num={self.w_num}, n_components={self.n_components}, "
              f"batch_step={self.batch_step}")

        # ---------- Source state (kept only for reset(), not used in CTTA) ----------
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer
        )

        # ---------- Text embeddings ----------
        # text_x shape: (T+1, C, D) — T per-template + 1 averaged (index [-1])
        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(
                self.classes, self.prompt_templates, average=True
            ).squeeze()

        # ---------- Runtime tracking ----------
        if self.runtime:
            self.adapt_times = []
            self.eval_times = []

    # ===========================================================
    # Public API
    # ===========================================================

    def adapt(self, x):
        """
        CTTA adaptation step — NO reset. State persists across all calls.

        Loss: MLMP pixel entropy + ILE (same as episodic MLMP), applied
        continuously without resetting to source weights.

        Args:
            x: Input patch tensor (B*N_patches, C, H, W)
        Returns:
            List[float]: Loss per gradient step
        """
        return self.perform_adaptation(x)

    def continual_adapt(self, x):
        """Alias for adapt() — compatible with main_continual.py protocol."""
        return self.perform_adaptation(x)

    @torch.no_grad()
    def evaluate(self, x):
        """
        Inference using MLMP's uncertainty-aware multi-level weighted fusion.

        Args:
            x: Input patch tensor (B*N_patches, C, H, W)
        Returns:
            torch.Tensor: Per-class logits (B*N_patches, num_classes, H, W)
        """
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
        """
        Full reset to source state (for episodic TTA mode or debugging).
        NOT called during normal CTTA operation.
        """
        self.load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state
        )
        self.moving_W = []
        self.batch_num = 0

    # ===========================================================
    # Core adaptation logic (same as MLMP, minus the reset call)
    # ===========================================================

    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []

        for _ in range(self.steps):
            if self.prompt_integration == 'loss':
                logits, _, _, cls_logits = self.model(
                    x, self.text_x[:-1], True,
                    interpolate=False,
                    vision_outputs=self.vision_outputs,
                    return_vanilla_cls=True,
                    vision_out_type="mean"
                )
                # logits:     (T, B, C, h, w)
                # cls_logits: (T, B, C, 1, 1)

                entropy_per_pixel = self.softmax_entropy(logits)          # (T, B, h, w)
                entropy_per_cls = self.softmax_entropy(cls_logits, dim=2) # (T, B, 1, 1)
                loss = entropy_per_pixel.mean() + self.alpha_cls * entropy_per_cls.mean()

            elif self.prompt_integration == 'text':
                logits, _, _ = self.model(
                    x, self.text_x[-1], True,
                    interpolate=False,
                    vision_outputs=self.vision_outputs
                )
                loss = self.softmax_entropy(logits).mean()

            else:
                raise ValueError("prompt_integration must be 'loss' or 'text'")

            loss_report.append(loss.item())
            loss.backward()
            self._subspace_step()
            self.optimizer.zero_grad()

        if self.runtime:
            self.adapt_times.append(time.time() - t1)

        return loss_report

    # ===========================================================
    # LCoTTA subspace-projected update
    # ===========================================================

    def _get_grad_vec(self):
        """Flatten LN grads into one float32 vector (on the model's device)."""
        return torch.cat([p.grad.detach().reshape(-1).float() for p in self.ln_params], dim=0)

    def _set_grad_vec(self, vec):
        """Write a flat vector back into the LN .grad tensors (cast to their dtype)."""
        idx = 0
        for p in self.ln_params:
            n = p.grad.numel()
            p.grad.data.copy_(vec[idx:idx + n].reshape(p.grad.shape).to(p.grad.dtype))
            idx += n

    @torch.no_grad()
    def _subspace_step(self):
        """Record the gradient every batch_step batches; once enough history exists,
        project the current gradient onto the principal subspace of the window
        before stepping (LCoTTA's P_SGD). Otherwise a plain Adam step.

        PCA principal axes = top right singular vectors of the centred window, so
        we use a torch SVD (no sklearn): for W (m x D), m << D, the rows of Vh are
        the principal components."""
        if self.batch_num != 0 and self.batch_num % self.batch_step == 0:
            self.moving_W.append(self._get_grad_vec().cpu())
            if len(self.moving_W) > self.w_num:
                self.moving_W.pop(0)
        if len(self.moving_W) >= self.n_components:
            g = self._get_grad_vec()                                  # (D,) on device
            W = torch.stack(self.moving_W, dim=0).to(g.device)        # (m, D)
            W = W - W.mean(dim=0, keepdim=True)
            _, _, Vh = torch.linalg.svd(W, full_matrices=False)       # Vh: (m, D)
            P = Vh[:self.n_components]                                 # (k, D)
            self._set_grad_vec(P.t() @ (P @ g))                       # project onto subspace
        self.optimizer.step()
        self.batch_num += 1

    # ===========================================================
    # Shared helpers (identical to MLMP)
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
