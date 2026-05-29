"""Source NA-CLIP model loader + text feature builder.

Mirrors adapt/tent_continual.py setup exactly so that the gradient signal
measured by exp1 matches the gradient that TENT-DivGate would actually see.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml

# Mirror REFERENCE_PROMPT from adapt/tent.py (single-prompt convention)
REFERENCE_PROMPT = "a photo of a {}"


def load_source_model(
    device: str = "cuda",
    ovss_type: str = "naclip",
    ovss_backbone: str = "ViT-L/14",
) -> tuple[nn.Module, callable, list[nn.Parameter], list[str]]:
    """Load NA-CLIP and prepare it for gradient computation on visual LN.

    - All params frozen except visual encoder's LN γ,β (matches TENT/TENT-DivGate).
    - Model is in eval() mode (no dropout/BN drift), but LN params have grad enabled.

    Returns:
        model      — NA-CLIP nn.Module on `device`
        tokenize   — text tokenizer callable
        ln_params  — list of nn.Parameter (visual LN γ,β); fixed iteration order
        ln_names   — parallel list of state_dict-style names (for debugging)
    """
    model, tokenize = load_ovss(ovss_type, ovss_backbone, device=device)
    model.eval()

    # Freeze everything
    model.requires_grad_(False)
    # Enable LN on visual encoder only
    for m in model.visual.modules():
        if isinstance(m, nn.LayerNorm):
            m.requires_grad_(True)

    # Collect LN params in deterministic order (matches
    # adapt.tent_continual.TENTContinual.collect_ln_params)
    ln_params: list[nn.Parameter] = []
    ln_names: list[str] = []
    for nm, m in model.visual.named_modules():
        if isinstance(m, nn.LayerNorm):
            for pname, p in m.named_parameters():
                if pname in ("weight", "bias"):
                    ln_params.append(p)
                    ln_names.append(f"visual.{nm}.{pname}")
    return model, tokenize, ln_params, ln_names


def compute_text_features(
    model: nn.Module,
    tokenize: callable,
    class_names: list[str],
    prompt_template_idx: int = 0,
    prompts_yaml: str | None = None,
    device: str = "cuda",
) -> torch.Tensor:
    """Build text features matching TENT-DivGate's runtime layout.

    - prompts_yaml=None: single prompt = REFERENCE_PROMPT (matches the bash
      scripts that don't pass --prompt_dir, i.e. tent_divgate_continual.sh).
    - prompts_yaml='prompts.yaml': uses prompts.yaml[prompt_template_idx] as a
      single template (still single-prompt convention).

    Returns:
        text_features — shape (num_prompts=1, num_classes, D), L2-normalized.
                        Shape matches what `self.text_x` is when fed to
                        model.forward(...) in adapt/tent_continual.py.
    """
    if prompts_yaml is None:
        templates = [REFERENCE_PROMPT]
    else:
        all_templates = load_prompts_from_yaml(prompts_yaml)
        templates = [all_templates[prompt_template_idx]]

    text_features = []
    for class_name in class_names:
        texts = [t.format(class_name) for t in templates]
        tok = tokenize(texts).to(device)
        with torch.no_grad():
            emb = model.encode_text(tok)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        text_features.append(emb)
    # Stack along class dim → (num_prompts=1, num_classes, D)
    return torch.stack(text_features, dim=1).to(device)
