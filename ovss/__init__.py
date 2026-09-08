import ovss.clip as clip
from ovss.clip import tokenize as clip_tokenize


def load_ovss(ovss_type, ovss_backbone, device='cpu', classes=None, catseg_checkpoint=None):
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

    elif ovss_type == 'clearclip':
        # ClearCLIP (ECCV'24): last-block self-self attention with the residual
        # and the FFN removed from the dense output.  In this codebase that is
        # exactly arch='reduced' (out_features take `reduced`, i.e. the attention
        # branch alone) + attn_strategy='kk' (k-k self-self).  It is NA-CLIP
        # minus the Gaussian neighbourhood prior, which makes it the tightest
        # possible control for the NA-CLIP arm.  gaussian_std is unused here.
        arch = "reduced"
        attn_strategy = "kk"
        gaussian_std = 5.0
        ovss_model, _ = clip.load(ovss_backbone, device)
        ovss_model.visual.set_params(arch, attn_strategy, gaussian_std)
        tokenize = clip_tokenize

    elif ovss_type in ('maskclip', 'clearclip_qq', 'vvclip', 'naclip_nonly', 'sclip_reduced'):
        # Four further training-free OVSS formulations, all sharing the same CLIP
        # weights and all using arch='reduced' so the dense features stay in the
        # text-aligned attention-branch space that MLMP's multi-level UAML average
        # requires (arch='vanilla' does not -- see the SCLIP note in
        # bash/backbone_ablation.sh).
        #   maskclip      MaskCLIP (ECCV'22)  -- no spatial mixing; the canonical
        #                 training-free OVSS baseline
        #   clearclip_qq  ClearCLIP (ECCV'24) as published -- q-q self-self
        #                 (ovss_type 'clearclip' is the k-k variant)
        #   vvclip        v-v self-self, the CLIP-Surgery / GEM family
        #   naclip_nonly  NA-CLIP's Gaussian prior WITHOUT the k-k term -- an
        #                 ablation of our own default rather than a new method
        arch = "reduced"
        #   sclip_reduced SCLIP's csa attention emitted through the reduced branch,
        #                 i.e. SCLIP made compatible with MLMP's UAML stack
        attn_strategy = {'maskclip': 'identity', 'clearclip_qq': 'qq',
                         'vvclip': 'vv', 'naclip_nonly': 'nonly',
                         'sclip_reduced': 'csa'}[ovss_type]
        gaussian_std = 5.0
        ovss_model, _ = clip.load(ovss_backbone, device)
        ovss_model.visual.set_params(arch, attn_strategy, gaussian_std)
        tokenize = clip_tokenize

    elif ovss_type == 'catseg':
        from ovss.catseg.wrapper import CATSegWrapper
        from ovss.catseg.third_party.clip import tokenize as catseg_tokenize
        import torch

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
