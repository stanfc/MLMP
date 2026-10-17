import torch
import ovss.clip as clip
from ovss.clip import tokenize as clip_tokenize


def load_ovss(ovss_type, ovss_backbone, device='cpu', classes=None, catseg_checkpoint=None,
              maft_checkpoint=None):
    """
    Load the OVSS model based on the specified type and backbone.

    Args:
        ovss_type: Type of the OVSS model.
        ovss_backbone: Backbone architecture of the OVSS model.
        device: Device to load the model on (e.g., 'cpu' or 'cuda').
        classes: List of class names (required for 'catseg' type).
        catseg_checkpoint: Path to a CAT-Seg pretrained checkpoint (optional).

    Returns:
        ovss_model: Loaded OVSS model.
        tokenize: Tokenizer function.
    """
    if ovss_type == 'clip':
        arch = "vanilla"
        attn_strategy = "vanilla"
        gaussian_std = 5.0
        ovss_model, _ = clip.load(ovss_backbone, device)
        ovss_model.visual.set_params(arch, attn_strategy, gaussian_std)
        tokenize = clip_tokenize

    elif ovss_type == 'sclip':
        arch = "vanilla"
        attn_strategy = "csa"
        gaussian_std = 5.0
        ovss_model, _ = clip.load(ovss_backbone, device)
        ovss_model.visual.set_params(arch, attn_strategy, gaussian_std)
        tokenize = clip_tokenize

    elif ovss_type == 'naclip':
        arch = "reduced"
        attn_strategy = "naclip"
        gaussian_std = 5.0
        ovss_model, _ = clip.load(ovss_backbone, device)
        ovss_model.visual.set_params(arch, attn_strategy, gaussian_std)
        tokenize = clip_tokenize

    elif ovss_type == 'clearclip_kk':
        # ClearCLIP (ECCV'24): last layer drops MLP+residuals (arch=reduced),
        # k-k self-similarity attention (same formula naclip's 'kk' already had).
        arch = "reduced"; attn_strategy = "kk"; gaussian_std = 0.0
        ovss_model, _ = clip.load(ovss_backbone, device)
        ovss_model.visual.set_params(arch, attn_strategy, gaussian_std)
        tokenize = clip_tokenize

    elif ovss_type == 'clearclip_qq':
        # ClearCLIP (ECCV'24): last layer drops MLP+residuals, q-q self-similarity.
        arch = "reduced"; attn_strategy = "qq"; gaussian_std = 0.0
        ovss_model, _ = clip.load(ovss_backbone, device)
        ovss_model.visual.set_params(arch, attn_strategy, gaussian_std)
        tokenize = clip_tokenize

    elif ovss_type == 'vv_attention':
        # value-value self-similarity -- unpublished third member of the qq/kk/vv
        # attention-surgery family (ClearCLIP=qq, existing naclip 'kk' path=kk).
        arch = "reduced"; attn_strategy = "vv"; gaussian_std = 0.0
        ovss_model, _ = clip.load(ovss_backbone, device)
        ovss_model.visual.set_params(arch, attn_strategy, gaussian_std)
        tokenize = clip_tokenize

    elif ovss_type == 'maskclip':
        # MaskCLIP (ECCV'22, arxiv 2112.01071): identity attention (no q/k),
        # value projection + out_proj act as a per-token 1x1 conv.
        arch = "reduced"; attn_strategy = "maskclip"; gaussian_std = 0.0
        ovss_model, _ = clip.load(ovss_backbone, device)
        ovss_model.visual.set_params(arch, attn_strategy, gaussian_std)
        tokenize = clip_tokenize

    elif ovss_type == 'sclip_reduced':
        # SCLIP's csa formula, but with arch=reduced output (bare attention
        # output, no MLP/residual) instead of SCLIP's original arch=vanilla.
        arch = "reduced"; attn_strategy = "csa"; gaussian_std = 0.0
        ovss_model, _ = clip.load(ovss_backbone, device)
        ovss_model.visual.set_params(arch, attn_strategy, gaussian_std)
        tokenize = clip_tokenize

    elif ovss_type == 'naclip_gaussonly':
        # naclip's existing 'nonly' attn_strategy: pure gaussian-neighbourhood
        # addition, no k-k similarity term.
        arch = "reduced"; attn_strategy = "nonly"; gaussian_std = 5.0
        ovss_model, _ = clip.load(ovss_backbone, device)
        ovss_model.visual.set_params(arch, attn_strategy, gaussian_std)
        tokenize = clip_tokenize

    elif ovss_type == 'maft':
        # MAFT+ (arxiv 2408.00744): Mask2Former-style, ConvNeXt-CLIP backbone,
        # open-vocab via mask-proposal + text classification. Not a plain-CLIP
        # ViT -- see ovss/maft/wrapper.py module docstring for the 3 scoping
        # decisions this integration makes (no UAML/multi-layer weighting,
        # ConvNeXt LayerNorm-only adaptation, MAFT+'s own internal detach()
        # patched out so that adaptation's gradient can actually reach it).
        from ovss.maft.wrapper import MAFTWrapper
        if classes is None:
            raise ValueError("classes must be provided for MAFT+ model")
        if maft_checkpoint is None:
            raise ValueError("maft_checkpoint must be provided for MAFT+ model")
        ovss_model = MAFTWrapper(classes, maft_checkpoint, device=device)
        tokenize = lambda texts: torch.zeros(len(texts), 1)

    elif ovss_type == 'itaclip':
        # ported core (attention) mechanism of ITACLIP (arxiv 2411.12044):
        # SCLIP-style attention at the last layer, blended 50/50 with the
        # average of that same attention captured at a few intermediate
        # layers. Does not include ITACLIP's other two components (LLM
        # class-synonym prompts, inference-time image-augmentation ensemble).
        arch = "reduced"
        attn_strategy = "itaclip"
        gaussian_std = 0.0
        ovss_model, _ = clip.load(ovss_backbone, device)
        ovss_model.visual.set_params(arch, attn_strategy, gaussian_std)
        tokenize = clip_tokenize

    elif ovss_type == 'catseg':
        from ovss.catseg.wrapper import CATSegWrapper
        from ovss.catseg.third_party.clip import tokenize as catseg_tokenize

        if classes is None:
            raise ValueError("classes must be provided for CAT-Seg model")

        if ovss_backbone == 'ViT-B/16':
            agg_cfg = dict(
                text_guidance_dim=512, text_guidance_proj_dim=128,
                appearance_guidance_dim=512, appearance_guidance_proj_dim=128,
                decoder_dims=[64, 32], decoder_guidance_dims=[256, 128],
                decoder_guidance_proj_dims=[32, 16],
                num_heads=4, num_layers=2, hidden_dims=128,
                pooling_sizes=(2, 2), feature_resolution=(24, 24),
                window_sizes=12,
            )
        elif ovss_backbone == 'ViT-L/14':
            agg_cfg = dict(
                text_guidance_dim=768, text_guidance_proj_dim=128,
                appearance_guidance_dim=768, appearance_guidance_proj_dim=128,
                decoder_dims=[64, 32], decoder_guidance_dims=[256, 128],
                decoder_guidance_proj_dims=[32, 16],
                num_heads=4, num_layers=2, hidden_dims=128,
                pooling_sizes=(2, 2), feature_resolution=(24, 24),
                window_sizes=12,
            )
        else:
            raise ValueError(f"CAT-Seg only supports ViT-B/16 and ViT-L/14, got: {ovss_backbone}")

        ovss_model = CATSegWrapper(
            clip_pretrained=ovss_backbone,
            class_names=classes,
            prompt_ensemble_type="single",
            prompt_depth=0,
            prompt_length=0,
            attention_type="linear",
            device=device,
            **agg_cfg,
        )

        if catseg_checkpoint is not None:
            print(f"Loading CAT-Seg checkpoint from: {catseg_checkpoint}")
            state_dict = torch.load(catseg_checkpoint, map_location=device)
            if 'model' in state_dict:
                state_dict = state_dict['model']

            pos_key = 'sem_seg_head.predictor.clip_model.visual.positional_embedding'
            if pos_key in state_dict:
                ckpt_pos = state_dict[pos_key]
                model_pos = ovss_model.state_dict()[pos_key]
                if ckpt_pos.shape != model_pos.shape:
                    import math
                    print(f"Resizing visual positional_embedding: {model_pos.shape} -> {ckpt_pos.shape}")
                    ckpt_side = int(math.sqrt(ckpt_pos.shape[0] - 1))
                    with torch.no_grad():
                        ovss_model.clip_model.visual.positional_embedding = torch.nn.Parameter(ckpt_pos)
                    ovss_model.feature_resolution = (ckpt_side, ckpt_side)
                    ovss_model.sem_seg_head.feature_resolution = [ckpt_side, ckpt_side]

            ovss_model.load_state_dict(state_dict, strict=False)

        tokenize = catseg_tokenize

    else:
        raise ValueError(f"Unsupported OVSS type: {ovss_type}")

    return ovss_model, tokenize
