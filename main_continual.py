# Standard
import os
import time
import argparse

# Third-party
import torch
# Use 'file_system' sharing strategy to avoid 'Connection reset by peer' errors
# when many dataloader workers run simultaneously across multiple experiments.
# Recommended by PyTorch docs for high-concurrency dataloader scenarios.
torch.multiprocessing.set_sharing_strategy('file_system')
# Limit PyTorch intra/inter-op threads. On a 256-core box PyTorch defaults to
# spawning ~256 threads per process; with many concurrent experiments this
# saturates the user thread limit and causes OpenCV/MKL spawn failures.
# Honor OMP_NUM_THREADS / MKL_NUM_THREADS env vars when present, else fall back to 4.
_n_intra = int(os.environ.get('OMP_NUM_THREADS', '4'))
torch.set_num_threads(_n_intra)
torch.set_num_interop_threads(min(2, _n_intra))
import numpy as np
from tqdm import tqdm

# Local
from adapt import get_method
from utils import segmentation_datasets
from utils.metrics import intersect_and_union, process_metrics
from utils.misc import set_global_seeds, save_configuration, aggregate_pred_patches


"""
Continual Test-Time Adaptation (CTTA) entry point.

Key differences from main.py (episodic TTA):

  1. The adaptation method is instantiated ONCE before all rounds. It is
     NEVER deleted or re-created between conditions or rounds.

  2. adapt_method.reset() is NEVER called between samples. Model state
     accumulates continuously across the entire test stream.

  3. The outer loop is `continual_rounds` (default 10 for ACDC).
     Each round processes all conditions in order: fog → night → rain → snow.

  4. Results are saved per-round per-condition, matching CoTTA Table 5.

  5. shuffle=False is enforced so the stream order is deterministic.

Supported datasets: ACDCDataset (primary), CityscapesDataset (for ablation).
Supported methods:  cotta, mlmp_cotta, tent_continual (plus any episodic
                    method run in continual mode for comparison).
"""


def argparser():
    parser = argparse.ArgumentParser(
        description="Continual Test-Time Adaptation for Open-Vocabulary Semantic Segmentation"
    )

    # ----------------------------------------
    # I/O
    # ----------------------------------------
    parser.add_argument('--save_dir', type=str, default='save/')
    parser.add_argument('--data_dir', type=str, default='.data/')
    parser.add_argument('--prompt_dir', type=str, default='')

    # ----------------------------------------
    # Dataset
    # ----------------------------------------
    parser.add_argument(
        '--dataset', type=str, default='ACDCDataset',
        choices=('ACDCDataset', 'ACDCMerged3Dataset', 'ACDCMerged6Dataset',
                 'ACDCMerged10Dataset',
                 'CityscapesDataset',
                 'COCOStuffDataset', 'COCOObjectDataset',
                 'PascalVOC20Dataset', 'PascalVOC21Dataset',
                 'PascalContext59Dataset', 'PascalContext60Dataset',
                 'DarkZurichDataset', 'NighttimeDrivingDataset',
                 'DZ_ND_Combined'),
    )
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument(
        '--init_resize', nargs='+', type=int, default=None,
        help='Resize images before patch extraction (H W). Must be multiples of patch_stride.'
    )
    parser.add_argument('--patch_size', nargs='+', type=int, default=None)
    parser.add_argument('--patch_stride', type=int, default=None)
    parser.add_argument(
        '--corruptions_list', nargs='+', type=str, default=None,
        help=(
            'For ACDCDataset: ordered list of conditions (fog night rain snow). '
            'For other datasets: standard corruption names. '
            'The full sequence is repeated continual_rounds times.'
        )
    )
    parser.add_argument('--class_extensions', action='store_true')
    parser.add_argument(
        '--split', type=str, default='val',
        help=(
            'Dataset split (ACDC only). Default "val" preserves existing behaviour. '
            'Use "train+val" to concatenate ACDC train and val splits per condition.'
        )
    )
    parser.add_argument(
        '--subset_size', type=int, default=None,
        help=(
            'Optional: randomly subsample N images per corruption (deterministic '
            'by --subset_seed). Used to align per-round update count across '
            'datasets of different val sizes (e.g. 100 to match ACDC).'
        )
    )
    parser.add_argument('--subset_seed', type=int, default=0,
                        help='Seed for --subset_size sampling')
    parser.add_argument(
        '--acdc_overlay_corruptions', nargs='+', type=str, default=None,
        help=(
            'ACDC only: paired list (same length as --corruptions_list). '
            'For each ACDC condition, overlay this synthetic corruption on top '
            'of the weather images. Example: --acdc_overlay_corruptions '
            'gaussian_noise shot_noise defocus_blur jpeg_compression '
            '(paired with fog night rain snow).'
        )
    )
    parser.add_argument(
        '--corruption_severity', type=int, default=5,
        help='Synthetic corruption severity (1-5). Default 5 matches existing Cityscapes/V20 runs.'
    )

    # ----------------------------------------
    # Model
    # ----------------------------------------
    parser.add_argument('--ovss_type', type=str, default='naclip')
    parser.add_argument('--ovss_backbone', type=str, default='ViT-L/14')
    parser.add_argument('--catseg_checkpoint', type=str, default=None,
                        help='Path to a CAT-Seg pretrained checkpoint '
                             '(only used when --ovss_type catseg)')

    # ----------------------------------------
    # Adaptation
    # ----------------------------------------
    parser.add_argument('--adapt', action='store_true',
                        help='Enable test-time adaptation (omit for source-only baseline)')
    parser.add_argument('--method', type=str, default='cotta',
                        help='Adaptation method name')
    parser.add_argument('--batch_size', '--batch-size', type=int, default=1, dest='batch_size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate — use lower values (1e-4 to 1e-5) for CTTA')
    parser.add_argument('--steps', type=int, default=1,
                        help='Gradient steps per batch — 1 recommended for online CTTA')

    # ----------------------------------------
    # CTTA-specific
    # ----------------------------------------
    parser.add_argument(
        '--continual_rounds', type=int, default=10,
        help='Number of times to repeat the full conditions sequence (10 for ACDC)'
    )

    # ----------------------------------------
    # DPCore source statistics
    # ----------------------------------------
    parser.add_argument(
        '--src_dataset', type=str, default=None,
        help=(
            'Dataset for DPCore source statistics. Defaults to --dataset. '
            'For ACDCDataset, set to CityscapesDataset to use clean images as source '
            '(fog-as-source causes loss_raw≈0, making ID detection impossible).'
        )
    )
    parser.add_argument(
        '--src_data_dir', type=str, default=None,
        help='Data root for src_dataset. Defaults to --data_dir.'
    )
    parser.add_argument(
        '--src_corruption', type=str, default=None,
        help=(
            'Corruption / condition key to use as source proxy for prototype '
            'initialization (e.g., "fog" for ACDC, "original" for Cityscapes). '
            'Used by cma_proto_continual; defaults to conditions_list[0] if unset. '
            'DPCore has its own hardcoded default of "original".'
        )
    )

    # ----------------------------------------
    # Misc
    # ----------------------------------------
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--runtime_calculation', action='store_true')
    parser.add_argument('--debug', action='store_true',
                        help='Process only 5 batches per condition for quick testing')
    parser.add_argument('--ann_file', type=str, default=None,
                        help='Override dataset ann_file path (e.g., a VOC subset split). '
                             'Only affects PascalVOC20Dataset / PascalVOC21Dataset; '
                             'ignored for ACDC / Cityscapes / COCO. Applied to stream '
                             'loaders only -- source-stat loaders still use full split.')

    return parser


def add_method_specific_args(parser, method):
    """Add per-method CLI arguments. Mirrors the structure in main.py."""

    # --- MLMP (episodic, included for ablation in continual mode) ---
    if method == 'mlmp':
        parser.add_argument('--vision_outputs', nargs='+', type=int, default=(-1,))
        parser.add_argument('--prompt_integration', type=str, default='loss')
        parser.add_argument('--alpha_cls', type=float, default=1.0)

    # --- WATT ---
    elif method == 'watt':
        parser.add_argument('--watt_l', default=2, type=int)
        parser.add_argument('--watt_m', default=5, type=int)

    # --- CLIPArTT ---
    elif method == 'clipartt':
        parser.add_argument('--clipartt_k', default=3, type=int)

    # --- TPT ---
    elif method == 'tpt':
        parser.add_argument('--n_ctx', default=4, type=int)

    # --- CoTTA on NA-CLIP (OVSS) ---
    # Uses constructor defaults for mt/rst/ap/aug_n unless overridden here.
    elif method == 'cotta':
        parser.add_argument('--mt', type=float, default=0.999,
                            help='EMA smoothing factor for teacher (CoTTA mt)')
        parser.add_argument('--rst', type=float, default=0.01,
                            help='Stochastic restoration probability (CoTTA rst)')
        parser.add_argument('--ap', type=float, default=0.92,
                            help='Anchor confidence threshold for augmentation gating (CoTTA ap)')
        parser.add_argument('--aug_n', type=int, default=32,
                            help='Number of augmented teacher views (CoTTA aug_n)')
        parser.add_argument('--vision_outputs', nargs='+', type=int, default=(-1,))
        parser.add_argument('--prompt_integration', type=str, default='text')

    # --- MLMP-Continual (naive continual: MLMP without reset) ---
    elif method == 'mlmp_continual':
        parser.add_argument('--vision_outputs', nargs='+', type=int, default=(-1,))
        parser.add_argument('--prompt_integration', type=str, default='loss')
        parser.add_argument('--alpha_cls', type=float, default=1.0)

    elif method == 'mlmp_topk_continual':
        parser.add_argument('--vision_outputs', nargs='+', type=int, default=(-1,))
        parser.add_argument('--prompt_integration', type=str, default='loss')
        parser.add_argument('--alpha_cls', type=float, default=0.0,
                            help='ILE weight; 0 = no ILE (recommended for CTTA)')
        parser.add_argument('--top_k_percent', type=float, default=0.2,
                            help='Fraction of highest-confidence pixels to keep in loss')

    elif method == 'mlmp_minprompt_continual':
        parser.add_argument('--vision_outputs', nargs='+', type=int, default=(-1,))
        parser.add_argument('--alpha_cls', type=float, default=0.0,
                            help='ILE weight; 0 = no ILE (recommended for CTTA)')

    elif method == 'mlmp_topk_minprompt_continual':
        parser.add_argument('--vision_outputs', nargs='+', type=int, default=(-1,))
        parser.add_argument('--alpha_cls', type=float, default=0.0,
                            help='ILE weight; 0 = no ILE (recommended for CTTA)')
        parser.add_argument('--top_k_percent', type=float, default=0.2)

    elif method == 'mlmp_topk_divgate_continual':
        parser.add_argument('--vision_outputs', nargs='+', type=int, default=(-1,))
        parser.add_argument('--alpha_cls', type=float, default=0.0)
        parser.add_argument('--top_k_percent', type=float, default=0.2)
        parser.add_argument('--h_threshold', type=float, default=1.8)
        parser.add_argument('--h_warning', type=float, default=1.5)
        parser.add_argument('--monitor_interval', type=int, default=50)
        parser.add_argument('--cautious_rst', type=float, default=0.005)
        parser.add_argument('--brake_rst', type=float, default=0.02)

    elif method == 'mlmp_topk_smooth_anchor_continual':
        parser.add_argument('--vision_outputs', nargs='+', type=int, default=(-1,))
        parser.add_argument('--alpha_cls', type=float, default=0.0)
        parser.add_argument('--top_k_percent', type=float, default=0.2)
        parser.add_argument('--h_ceil', type=float, default=1.8)
        parser.add_argument('--h_floor', type=float, default=1.5)
        parser.add_argument('--lag_scale', type=float, default=90.0)
        parser.add_argument('--max_lag', type=int, default=3000)
        parser.add_argument('--rst', type=float, default=0.005)
        parser.add_argument('--monitor_interval', type=int, default=50)

    elif method == 'mlmp_minprompt_divgate_continual':
        parser.add_argument('--vision_outputs', nargs='+', type=int, default=(-1,))
        parser.add_argument('--alpha_cls', type=float, default=0.0)
        parser.add_argument('--h_threshold', type=float, default=1.8)
        parser.add_argument('--h_warning', type=float, default=1.5)
        parser.add_argument('--monitor_interval', type=int, default=50)
        parser.add_argument('--cautious_rst', type=float, default=0.005)
        parser.add_argument('--brake_rst', type=float, default=0.02)

    elif method == 'mlmp_minprompt_smooth_anchor_continual':
        parser.add_argument('--vision_outputs', nargs='+', type=int, default=(-1,))
        parser.add_argument('--alpha_cls', type=float, default=0.0)
        parser.add_argument('--h_ceil', type=float, default=1.8)
        parser.add_argument('--h_floor', type=float, default=1.5)
        parser.add_argument('--lag_scale', type=float, default=90.0)
        parser.add_argument('--max_lag', type=int, default=3000)
        parser.add_argument('--rst', type=float, default=0.005)
        parser.add_argument('--monitor_interval', type=int, default=50)

    elif method == 'mlmp_topk_minprompt_divgate_continual':
        parser.add_argument('--vision_outputs', nargs='+', type=int, default=(-1,))
        parser.add_argument('--alpha_cls', type=float, default=0.0)
        parser.add_argument('--top_k_percent', type=float, default=0.2)
        parser.add_argument('--h_threshold', type=float, default=1.8)
        parser.add_argument('--h_warning', type=float, default=1.5)
        parser.add_argument('--monitor_interval', type=int, default=50)
        parser.add_argument('--cautious_rst', type=float, default=0.005)
        parser.add_argument('--brake_rst', type=float, default=0.02)

    elif method == 'mlmp_topk_minprompt_smooth_anchor_continual':
        parser.add_argument('--vision_outputs', nargs='+', type=int, default=(-1,))
        parser.add_argument('--alpha_cls', type=float, default=0.0)
        parser.add_argument('--top_k_percent', type=float, default=0.2)
        parser.add_argument('--h_ceil', type=float, default=1.8)
        parser.add_argument('--h_floor', type=float, default=1.5)
        parser.add_argument('--lag_scale', type=float, default=90.0)
        parser.add_argument('--max_lag', type=int, default=3000)
        parser.add_argument('--rst', type=float, default=0.005)
        parser.add_argument('--monitor_interval', type=int, default=50)

    elif method == 'mlmp_simple_eval_continual':
        parser.add_argument('--vision_outputs', nargs='+', type=int, default=(-1,))
        parser.add_argument('--prompt_integration', type=str, default='loss')
        parser.add_argument('--alpha_cls', type=float, default=0.0)

    elif method == 'tent_uaml_eval_continual':
        parser.add_argument('--vision_outputs', nargs='+', type=int, default=(-1,))

    elif method == 'mlmp_smooth_anchor_continual':
        parser.add_argument('--vision_outputs', nargs='+', type=int, default=(-1,))
        parser.add_argument('--prompt_integration', type=str, default='loss')
        parser.add_argument('--alpha_cls', type=float, default=1.0)
        parser.add_argument('--h_ceil', type=float, default=1.8,
                            help='H_margin >= h_ceil -> no restore.')
        parser.add_argument('--h_floor', type=float, default=1.5,
                            help='H_margin <= h_floor -> restore toward frozen source.')
        parser.add_argument('--lag_scale', type=float, default=90.0,
                            help='lag(H) = lag_scale / (H - h_floor). Default 90 gives lag=300 at H=1.8.')
        parser.add_argument('--max_lag', type=int, default=3000,
                            help='Cap on lag; lag > max_lag falls back to source snapshot.')
        parser.add_argument('--rst', type=float, default=0.005,
                            help='Fixed restoration rate when restore is active.')
        parser.add_argument('--monitor_interval', type=int, default=50)

    # --- TENT-Continual ---
    elif method == 'tent_continual':
        pass   # no extra args beyond base parser

    # --- SAR (Sharpness-Aware Reliable TTA, ICLR 2023) ---
    elif method == 'sar_continual':
        parser.add_argument('--e_margin', type=float, default=1.198,
                            help='Per-sample mean entropy threshold; samples > this are skipped. '
                                 'Default 1.198 = 0.4 * ln(20) for VOC20.')
        parser.add_argument('--sam_rho', type=float, default=0.05,
                            help='SAM perturbation radius (default 0.05 from SAR paper)')
        parser.add_argument('--e_0', type=float, default=0.2,
                            help='If loss EMA exceeds this, reset model to source')
        parser.add_argument('--ema_factor', type=float, default=0.9,
                            help='EMA decay for the loss moving average')
        parser.add_argument('--recovery_warmup', type=int, default=50,
                            help='Number of batches before recovery can fire')

    # --- DeYO (Entropy is not Enough, ICLR 2024 spotlight) ---
    elif method == 'deyo_continual':
        parser.add_argument('--deyo_margin_factor', type=float, default=0.5,
                            help='DeYO entropy filter = factor*log(C) (default 0.5)')
        parser.add_argument('--deyo_margin_e0_factor', type=float, default=0.4,
                            help='Reweighting margin = factor*log(C) (default 0.4)')
        parser.add_argument('--plpd_threshold', type=float, default=0.2,
                            help='PLPD filter threshold (default 0.2, wild setting)')
        parser.add_argument('--aug_type', type=str, default='patch',
                            choices=['patch', 'pixel', 'occ'],
                            help='Object-destructive transform (default patch shuffle)')
        parser.add_argument('--patch_len', type=int, default=4,
                            help='patch_len x patch_len grid for patch shuffle (default 4)')
        parser.add_argument('--reweight_ent', type=int, default=1,
                            help='Entropy reweighting on/off (default 1)')
        parser.add_argument('--reweight_plpd', type=int, default=1,
                            help='PLPD reweighting on/off (default 1)')
        parser.add_argument('--top_block_exclude', type=int, default=6,
                            help='Number of trailing ViT blocks excluded (default 6)')

    # --- DeYO + multi-layer adapt + UAML eval (single-prompt / full-MLMP) ---
    elif method in ('deyo_uaml_continual', 'deyo_mlmp_continual'):
        parser.add_argument('--vision_outputs', nargs='+', type=int,
                            default=tuple(range(-1, -19, -1)))
        parser.add_argument('--deyo_margin_factor', type=float, default=0.5)
        parser.add_argument('--deyo_margin_e0_factor', type=float, default=0.4)
        parser.add_argument('--plpd_threshold', type=float, default=0.2)
        parser.add_argument('--aug_type', type=str, default='patch',
                            choices=['patch', 'pixel', 'occ'])
        parser.add_argument('--patch_len', type=int, default=4)
        parser.add_argument('--reweight_ent', type=int, default=1)
        parser.add_argument('--reweight_plpd', type=int, default=1)
        parser.add_argument('--top_block_exclude', type=int, default=6)

    # --- DeYO + MLMP + DivGate ---
    elif method == 'deyo_mlmp_divgate_continual':
        parser.add_argument('--vision_outputs', nargs='+', type=int,
                            default=tuple(range(-1, -19, -1)))
        parser.add_argument('--deyo_margin_factor', type=float, default=0.5)
        parser.add_argument('--deyo_margin_e0_factor', type=float, default=0.4)
        parser.add_argument('--plpd_threshold', type=float, default=0.2)
        parser.add_argument('--aug_type', type=str, default='patch',
                            choices=['patch', 'pixel', 'occ'])
        parser.add_argument('--patch_len', type=int, default=4)
        parser.add_argument('--reweight_ent', type=int, default=1)
        parser.add_argument('--reweight_plpd', type=int, default=1)
        parser.add_argument('--top_block_exclude', type=int, default=6)
        parser.add_argument('--h_threshold', type=float, default=2.0)
        parser.add_argument('--h_warning', type=float, default=1.7)
        parser.add_argument('--monitor_interval', type=int, default=50)
        parser.add_argument('--cautious_rst', type=float, default=0.005)
        parser.add_argument('--brake_rst', type=float, default=0.02)

    # --- DeYO + MLMP + SmoothAnchor ---
    elif method == 'deyo_mlmp_smooth_anchor_continual':
        parser.add_argument('--vision_outputs', nargs='+', type=int,
                            default=tuple(range(-1, -19, -1)))
        parser.add_argument('--deyo_margin_factor', type=float, default=0.5)
        parser.add_argument('--deyo_margin_e0_factor', type=float, default=0.4)
        parser.add_argument('--plpd_threshold', type=float, default=0.2)
        parser.add_argument('--aug_type', type=str, default='patch',
                            choices=['patch', 'pixel', 'occ'])
        parser.add_argument('--patch_len', type=int, default=4)
        parser.add_argument('--reweight_ent', type=int, default=1)
        parser.add_argument('--reweight_plpd', type=int, default=1)
        parser.add_argument('--top_block_exclude', type=int, default=6)
        parser.add_argument('--h_ceil', type=float, default=2.9)
        parser.add_argument('--h_floor', type=float, default=2.2)
        parser.add_argument('--lag_scale', type=float, default=150.0)
        parser.add_argument('--max_lag', type=int, default=3000)
        parser.add_argument('--rst', type=float, default=0.005)
        parser.add_argument('--monitor_interval', type=int, default=50)

    # --- DAT (Distribution-Aware Tuning, CVPR 2024) ---
    elif method == 'dat_continual':
        parser.add_argument('--k_percent', type=float, default=0.001,
                            help='Fraction of visual layers selected per backward (default 0.001)')
        parser.add_argument('--select_until', type=int, default=100,
                            help='PAU: accumulate selected layers for first N batches (default 100)')
        parser.add_argument('--trp_thr', type=float, default=0.85,
                            help='Low-uncertainty cutoff -> TRP pixels (default 0.85)')
        parser.add_argument('--dsp_thr', type=float, default=0.99,
                            help='High-uncertainty cutoff -> DSP pixels (default 0.99)')
        parser.add_argument('--conf_thr', type=float, default=0.69,
                            help='Anchor-confidence cutoff for pseudo-label (default 0.69)')
        parser.add_argument('--aug_n', type=int, default=8,
                            help='Multi-aug forwards for uncertainty (default 8)')
        parser.add_argument('--mt', type=float, default=0.999,
                            help='EMA teacher rate (default 0.999)')

    # --- EATA (Efficient Anti-forgetting TTA, ICML 2022) ---
    elif method == 'eata_continual':
        parser.add_argument('--e_margin', type=float, default=1.198,
                            help='Per-sample mean entropy threshold for reliable filter')
        parser.add_argument('--d_margin', type=float, default=0.05,
                            help='Cosine-similarity gap for non-redundant filter; '
                                 'a sample is kept if cos(sample, ema) < 1 - d_margin')
        parser.add_argument('--fisher_alpha', type=float, default=2000.0,
                            help='Weight on EWC penalty in total loss')
        parser.add_argument('--fisher_size', type=int, default=2000,
                            help='Number of clean source samples used to estimate Fisher')

    # --- RoTTA (Robust TTA in Dynamic Scenarios, CVPR 2023) ---
    elif method == 'rotta_continual':
        parser.add_argument('--memory_size', type=int, default=64,
                            help='CSTU memory bank capacity (default 64)')
        parser.add_argument('--nu', type=float, default=0.001,
                            help='EMA teacher update rate (default 0.001)')
        parser.add_argument('--lambda_t', type=float, default=1.0,
                            help='Timeliness weight in CSTU score (default 1.0)')
        parser.add_argument('--lambda_u', type=float, default=1.0,
                            help='Uncertainty weight in CSTU score (default 1.0)')
        parser.add_argument('--update_frequency', type=int, default=None,
                            help='Update every N instances (default = memory_size)')

    # --- CMA-Continual (Cross-Modal Alignment, proposed) ---
    elif method == 'cma_continual':
        parser.add_argument('--top_k_percent', type=float, default=0.2,
                            help='Fraction of highest-confidence pixels per batch '
                                 'that contribute to the CMA loss (default 0.2)')

    # --- CMA-Proto-Continual (CMA + source/target prototype bank, proposed) ---
    elif method == 'cma_proto_continual':
        parser.add_argument('--top_k_percent', type=float, default=0.2,
                            help='Top-K%% confidence mask for the loss (default 0.2)')
        parser.add_argument('--lambda_cma', type=float, default=1.0,
                            help='Weight of text-alignment (CMA) term (default 1.0)')
        parser.add_argument('--lambda_src', type=float, default=1.0,
                            help='Weight of fixed source-prototype term (default 1.0)')
        parser.add_argument('--lambda_tgt', type=float, default=0.5,
                            help='Weight of EMA target-prototype term; set to 0 '
                                 'to degrade to pure source-anchor variant (default 0.5)')
        parser.add_argument('--ema_alpha', type=float, default=0.999,
                            help='EMA momentum for target prototypes (default 0.999)')
        parser.add_argument('--src_conf_threshold', type=float, default=0.5,
                            help='Confidence threshold for source prototype init (default 0.5)')
        parser.add_argument('--src_max_samples', type=int, default=5000,
                            help='Max source images used for prototype init (default 5000)')

    # --- CMA-Layered-Continual (CMA + layer-stratified restoration, Direction A) ---
    elif method == 'cma_layered_continual':
        parser.add_argument('--top_k_percent', type=float, default=0.2,
                            help='Top-K%% confidence mask for the CMA loss (default 0.2)')
        parser.add_argument('--early_rst', type=float, default=0.001,
                            help='Stochastic restore probability for early blocks '
                                 '([0, early_cutoff) and ln_pre) (default 0.001)')
        parser.add_argument('--mid_rst', type=float, default=0.01,
                            help='Stochastic restore probability for mid blocks '
                                 '([early_cutoff, late_cutoff)) (default 0.01)')
        parser.add_argument('--late_rst', type=float, default=0.05,
                            help='Stochastic restore probability for late blocks '
                                 '([late_cutoff, num_blocks) and ln_post) (default 0.05)')
        parser.add_argument('--early_cutoff', type=int, default=8,
                            help='Block index boundary: blocks [0, early_cutoff) are early')
        parser.add_argument('--late_cutoff', type=int, default=16,
                            help='Block index boundary: blocks [early_cutoff, late_cutoff) are mid; '
                                 '[late_cutoff, num_blocks) are late')

    # --- CMA-DivGate-Continual (CMA + diversity gate, Direction B) ---
    elif method == 'cma_divgate_continual':
        parser.add_argument('--top_k_percent', type=float, default=0.2,
                            help='Top-K%% confidence mask for the CMA loss (default 0.2)')
        parser.add_argument('--h_threshold', type=float, default=1.8,
                            help='H_margin >= this -> aggressive mode (rst=0)')
        parser.add_argument('--h_warning', type=float, default=1.2,
                            help='h_warning <= H_margin < h_threshold -> cautious; '
                                 '< h_warning -> brake')
        parser.add_argument('--monitor_interval', type=int, default=50,
                            help='Batches between H_margin re-evaluations (default 50)')
        parser.add_argument('--cautious_rst', type=float, default=0.005,
                            help='Stochastic restore probability in cautious mode')
        parser.add_argument('--brake_rst', type=float, default=0.05,
                            help='Stochastic restore probability in brake mode')

    # --- TENT-DivGate-Continual (TENT + diversity gate, Direction B on TENT base) ---
    elif method == 'tent_divgate_continual':
        parser.add_argument('--h_threshold', type=float, default=1.8,
                            help='H_margin >= this -> aggressive mode (rst=0)')
        parser.add_argument('--h_warning', type=float, default=1.2,
                            help='h_warning <= H_margin < h_threshold -> cautious; '
                                 '< h_warning -> brake')
        parser.add_argument('--monitor_interval', type=int, default=50,
                            help='Batches between H_margin re-evaluations (default 50)')
        parser.add_argument('--cautious_rst', type=float, default=0.005,
                            help='Stochastic restore probability in cautious mode')
        parser.add_argument('--brake_rst', type=float, default=0.05,
                            help='Stochastic restore probability in brake mode')

    elif method == 'mlmp_divgate_continual':
        parser.add_argument('--vision_outputs', nargs='+', type=int, default=(-1,))
        parser.add_argument('--prompt_integration', type=str, default='loss')
        parser.add_argument('--alpha_cls', type=float, default=1.0)
        parser.add_argument('--h_threshold', type=float, default=1.8)
        parser.add_argument('--h_warning', type=float, default=1.5)
        parser.add_argument('--monitor_interval', type=int, default=50)
        parser.add_argument('--cautious_rst', type=float, default=0.005)
        parser.add_argument('--brake_rst', type=float, default=0.02)

    elif method == 'tent_divgate_recent_anchor':
        parser.add_argument('--h_threshold', type=float, default=1.8)
        parser.add_argument('--h_warning', type=float, default=1.5)
        parser.add_argument('--monitor_interval', type=int, default=50)
        parser.add_argument('--cautious_rst', type=float, default=0.005)
        parser.add_argument('--brake_rst', type=float, default=0.02)
        parser.add_argument('--anchor_lag', type=int, default=300,
                            help='Restoration target = LN snapshot from N batches ago')

    elif method == 'tent_divgate_decoupled':
        parser.add_argument('--h_threshold', type=float, default=1.8)
        parser.add_argument('--h_warning', type=float, default=1.5)
        parser.add_argument('--marginal_buf_size', type=int, default=50,
                            help='Rolling window size for H_margin averaging')
        parser.add_argument('--monitor_interval', type=int, default=50,
                            help='Batches between mode re-evaluations')
        parser.add_argument('--cautious_rst', type=float, default=0.005)
        parser.add_argument('--brake_rst', type=float, default=0.02)

    elif method == 'tent_divgate_hybrid_anchor':
        parser.add_argument('--h_threshold', type=float, default=1.8)
        parser.add_argument('--h_warning', type=float, default=1.5)
        parser.add_argument('--monitor_interval', type=int, default=50)
        parser.add_argument('--cautious_rst', type=float, default=0.005)
        parser.add_argument('--brake_rst', type=float, default=0.02)
        parser.add_argument('--anchor_lag', type=int, default=300,
                            help='Cautious restore target = LN snapshot from N batches ago; '
                                 'brake restore always targets the frozen source.')

    elif method == 'tent_divgate_smooth_anchor':
        parser.add_argument('--h_ceil', type=float, default=1.8,
                            help='H_margin >= h_ceil -> no restore.')
        parser.add_argument('--h_floor', type=float, default=1.5,
                            help='H_margin <= h_floor -> restore toward frozen source.')
        parser.add_argument('--lag_scale', type=float, default=90.0,
                            help='lag(H) = lag_scale / (H - h_floor). '
                                 'Default 90 gives lag=300 at H=1.8.')
        parser.add_argument('--max_lag', type=int, default=3000,
                            help='Cap on lag; lag > max_lag falls back to source snapshot.')
        parser.add_argument('--rst', type=float, default=0.005,
                            help='Fixed restoration rate when restore is active.')
        parser.add_argument('--monitor_interval', type=int, default=50)

    elif method == 'tent_contgate_continual':
        parser.add_argument('--h_high', type=float, default=1.8,
                            help='H_margin >= this -> rst = 0 (healthy, no brake)')
        parser.add_argument('--h_low', type=float, default=0.8,
                            help='H_margin <= this -> rst = max_rst (full brake)')
        parser.add_argument('--max_rst', type=float, default=0.01,
                            help='Upper bound on stochastic restore probability')
        parser.add_argument('--monitor_interval', type=int, default=50,
                            help='Batches between H_margin re-evaluations (default 50)')

    elif method == 'tent_divgate_continual_catseg':
        parser.add_argument('--h_threshold', type=float, default=1.8,
                            help='H_margin >= this -> aggressive (rst=0)')
        parser.add_argument('--h_warning', type=float, default=1.5,
                            help='h_warning <= H < h_threshold -> cautious; '
                                 '< h_warning -> brake')
        parser.add_argument('--monitor_interval', type=int, default=50,
                            help='Batches between H_margin re-evaluations')
        parser.add_argument('--cautious_rst', type=float, default=0.005,
                            help='Restore probability in cautious mode')
        parser.add_argument('--brake_rst', type=float, default=0.02,
                            help='Restore probability in brake mode')

    elif method == 'tent_siggate_continual':
        parser.add_argument('--h_high', type=float, default=1.8,
                            help='Upper anchor: rst saturates near 0 here')
        parser.add_argument('--h_low', type=float, default=0.8,
                            help='Lower anchor: rst saturates near max_rst here')
        parser.add_argument('--max_rst', type=float, default=0.01,
                            help='Upper bound on stochastic restore probability')
        parser.add_argument('--sigmoid_k', type=float, default=-1.0,
                            help='Sigmoid steepness; <=0 -> auto = 6/(h_high-h_low)')
        parser.add_argument('--monitor_interval', type=int, default=50,
                            help='Batches between H_margin re-evaluations (default 50)')
        parser.add_argument('--gate_log_path', type=str, default=None,
                            help='Optional path to per-batch gate log csv (h_margin, rst)')

    elif method == 'clipartt_continual':
        parser.add_argument('--clipartt_k', type=int, default=3,
                            help='Top-K classes per pixel for CLIPArTT prompt construction')

    elif method == 'clipartt_siggate_continual':
        parser.add_argument('--clipartt_k', type=int, default=3,
                            help='Top-K classes per pixel for CLIPArTT prompt construction')

    elif method == 'clipartt_divgate_continual':
        parser.add_argument('--clipartt_k', type=int, default=3,
                            help='Top-K classes per pixel for CLIPArTT prompt construction')
        parser.add_argument('--h_threshold', type=float, default=1.8)
        parser.add_argument('--h_warning', type=float, default=1.5)
        parser.add_argument('--monitor_interval', type=int, default=50)
        parser.add_argument('--cautious_rst', type=float, default=0.005)
        parser.add_argument('--brake_rst', type=float, default=0.02)
        parser.add_argument('--gate_log_path', type=str, default=None,
                            help='Optional path to per-batch gate log csv (h_margin, mode, rst)')

    elif method == 'tent_topk_continual':
        parser.add_argument('--top_k_percent', type=float, default=0.2,
                            help='Fraction of highest-confidence pixels to keep in TENT loss')

    elif method == 'tent_minprompt_continual':
        pass  # only takes prompt_dir / steps / lr (already global)

    elif method == 'tent_topk_divgate_continual':
        parser.add_argument('--top_k_percent', type=float, default=0.2,
                            help='Fraction of highest-confidence pixels to keep in TENT loss')
        parser.add_argument('--h_threshold', type=float, default=1.8)
        parser.add_argument('--h_warning', type=float, default=1.5)
        parser.add_argument('--monitor_interval', type=int, default=50)
        parser.add_argument('--cautious_rst', type=float, default=0.005)
        parser.add_argument('--brake_rst', type=float, default=0.02)

    elif method == 'tent_minprompt_divgate_continual':
        parser.add_argument('--h_threshold', type=float, default=1.8)
        parser.add_argument('--h_warning', type=float, default=1.5)
        parser.add_argument('--monitor_interval', type=int, default=50)
        parser.add_argument('--cautious_rst', type=float, default=0.005)
        parser.add_argument('--brake_rst', type=float, default=0.02)

    elif method == 'tent_early_continual':
        parser.add_argument('--early_cutoff', type=int, default=8,
                            help='Block index < this -> trainable; '
                                 'all other LN params (incl. ln_post) FROZEN')

    elif method == 'tent_divgate_layered_continual':
        parser.add_argument('--h_threshold', type=float, default=1.8,
                            help='H_margin >= this -> aggressive mode (rst=0)')
        parser.add_argument('--h_warning', type=float, default=1.2,
                            help='h_warning <= H_margin < h_threshold -> cautious; '
                                 '< h_warning -> brake')
        parser.add_argument('--monitor_interval', type=int, default=50,
                            help='Batches between H_margin re-evaluations (default 50)')
        parser.add_argument('--cautious_rst', type=float, default=0.005,
                            help='Stochastic restore probability in cautious mode')
        parser.add_argument('--brake_rst', type=float, default=0.05,
                            help='Stochastic restore probability in brake mode')
        parser.add_argument('--early_cutoff', type=int, default=8,
                            help='Block index < this -> "early" group (trainable)')
        parser.add_argument('--late_cutoff', type=int, default=16,
                            help='Block index >= this -> "late" group (FROZEN, '
                                 'no gradients, no restoration)')

    # --- SAR-DivGate-Continual (SAR base + DivGate replaces hard recovery) ---
    elif method == 'sar_divgate_continual':
        # SAR-side
        parser.add_argument('--e_margin', type=float, default=1.8,
                            help='Per-sample mean entropy threshold; samples > this are skipped')
        parser.add_argument('--sam_rho', type=float, default=0.05,
                            help='SAM perturbation radius (default 0.05 from SAR paper)')
        # DivGate-side
        parser.add_argument('--h_threshold', type=float, default=1.6,
                            help='H_margin >= this -> aggressive mode (rst=0)')
        parser.add_argument('--h_warning', type=float, default=1.4,
                            help='h_warning <= H_margin < h_threshold -> cautious; '
                                 '< h_warning -> brake')
        parser.add_argument('--monitor_interval', type=int, default=50,
                            help='Batches between H_margin re-evaluations (default 50)')
        parser.add_argument('--cautious_rst', type=float, default=0.01,
                            help='Stochastic restore probability in cautious mode')
        parser.add_argument('--brake_rst', type=float, default=0.05,
                            help='Stochastic restore probability in brake mode')

    # --- SAR + full MLMP (multi-prompt/layer + UAML) + SmoothAnchor ---
    elif method == 'sar_mlmp_smooth_anchor_continual':
        # MLMP / UAML
        parser.add_argument('--vision_outputs', nargs='+', type=int,
                            default=tuple(range(-1, -19, -1)),
                            help='ViT layer indices for UAML multi-layer fusion')
        parser.add_argument('--alpha_cls', type=float, default=0.0,
                            help='ILE (CLS-entropy) weight; >0 enables the term')
        parser.add_argument('--uaml_in_adapt', type=int, default=1,
                            help='1: multi-layer fusion in the adapt loss; '
                                 '0: single-layer adapt (eval still multi-layer)')
        # SAR
        parser.add_argument('--e_margin', type=float, default=1.8,
                            help='Per-sample mean entropy threshold; samples > this are skipped')
        parser.add_argument('--sam_rho', type=float, default=0.05,
                            help='SAM perturbation radius (default 0.05 from SAR paper)')
        # SmoothAnchor
        parser.add_argument('--h_ceil', type=float, default=2.9,
                            help='H_margin >= this -> no restore (~3.0 healthy for adapt-time ensemble)')
        parser.add_argument('--h_floor', type=float, default=2.2,
                            help='H_margin <= this -> restore toward frozen source')
        parser.add_argument('--lag_scale', type=float, default=150.0,
                            help='lag(H) = lag_scale / (H - h_floor)')
        parser.add_argument('--max_lag', type=int, default=3000,
                            help='Cap on lag; lag > max_lag falls back to source snapshot')
        parser.add_argument('--rst', type=float, default=0.005,
                            help='Fixed restoration rate when restore is active')
        parser.add_argument('--monitor_interval', type=int, default=50)

    # --- DELTA-Continual (DOT-only, ICLR 2023) ---
    elif method == 'delta_continual':
        parser.add_argument('--dot_momentum', type=float, default=0.9,
                            help='EMA decay for class-frequency tracker. '
                                 'Higher = slower frequency updates.')
        parser.add_argument('--dot_alpha', type=float, default=1.0,
                            help='Exponent on inverse class frequency for the per-pixel '
                                 'weight. 0 = plain TENT, 1 = full inverse-frequency.')

    # --- DPCore (Dynamic Prompt Coreset) ---
    elif method == 'dpcore':
        parser.add_argument('--vision_outputs', nargs='+', type=int, default=(-1,))
        parser.add_argument('--temp_tau', type=float, default=3.0,
                            help='Temperature for coreset weight softmax')
        parser.add_argument('--ema_alpha', type=float, default=0.999,
                            help='EMA momentum for coreset statistics update')
        parser.add_argument('--thr_rho', type=float, default=0.9,
                            help='Loss threshold ratio for ID/OOD decision')
        parser.add_argument('--prompt_num', type=int, default=8,
                            help='Number of visual prompt tokens')
        parser.add_argument('--verbose_dpcore', action='store_true',
                            help='Print per-step loss and coreset eval info')

    # --- KFF (Class-aware domain knowledge Fusion & Fission, NeurIPS 2025) ---
    elif method == 'kff':
        parser.add_argument('--vision_outputs', nargs='+', type=int, default=(-1,))
        # Domain-prompt params (mirror KFF OURS.*/OPTIM.*):
        parser.add_argument('--tau', type=float, default=3.0,
                            help='Temperature for domain-prompt softmax weighting')
        parser.add_argument('--ema_alpha', type=float, default=0.1,
                            help='EMA momentum for domain-prompt key update (KFF: 0.1)')
        parser.add_argument('--thr_d', type=float, default=25.0,
                            help='L2 threshold for domain-prompt ID/OOD decision')
        parser.add_argument('--n_d', type=int, default=20,
                            help='Max domain-prompt base capacity')
        parser.add_argument('--lr_domain', type=float, default=1e-5,
                            help='LR for domain-prompt optimiser')
        parser.add_argument('--lamda', type=float, default=1.0,
                            help='Weight on mean term in distribution loss')
        # Class-prompt params:
        parser.add_argument('--n_c', type=int, default=100,
                            help='Max class-prompt base capacity')
        parser.add_argument('--thr_c', type=float, default=0.005,
                            help='Cosine threshold for class-prompt match')
        parser.add_argument('--thr_ent', type=float, default=2.0,
                            help='Entropy threshold gating class-prompt update')
        parser.add_argument('--alpha_c', type=float, default=0.1,
                            help='EMA momentum for class-prompt key update')
        # Shared:
        parser.add_argument('--prompt_num', type=int, default=8,
                            help='Number of domain visual prompt tokens')
        parser.add_argument('--verbose_kff', action='store_true',
                            help='Print per-step loss and coreset eval info')

    return parser


def save_round_results(all_round_results, save_dir, conditions):
    """
    Write a summary table matching CoTTA Table 5 format.

    Output: save_dir/results_all_rounds.txt
    Columns: Round | fog | night | rain | snow | Mean
    """
    summary_path = os.path.join(save_dir, "results_all_rounds.txt")

    header = "Round, " + ", ".join(conditions) + ", Mean_mIoU"
    lines = [header]

    for rnd, cond_results in sorted(all_round_results.items()):
        miou_vals = []
        row = f"Round {rnd:02d}"
        for cond in conditions:
            if cond in cond_results:
                v = cond_results[cond]['mIoU']
                row += f", {v:.2f}"
                miou_vals.append(v)
            else:
                row += ", N/A"
        if miou_vals:
            row += f", {np.mean(miou_vals):.2f}"
        lines.append(row)

    with open(summary_path, 'w') as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n[Summary saved → {summary_path}]")


def main(args):
    save_configuration(args)
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    conditions = args.corruptions_list
    assert conditions, "--corruptions_list must be provided (e.g., fog night rain snow)"

    # ----------------------------------------------------------------
    # Resolve class list from first condition's dataset
    # (classes are condition-independent for ACDC/Cityscapes)
    # ----------------------------------------------------------------
    first_loader, org_classes = segmentation_datasets.prepare_data(
        args.dataset, args.data_dir, args.init_resize,
        args.patch_size, args.patch_stride,
        corruption=conditions[0],
        batch_size=args.batch_size, num_workers=args.workers,
        shuffle=False,
        ann_file=args.ann_file,
        corruption_severity=args.corruption_severity,
        split=args.split,
    )

    if args.class_extensions and first_loader.dataset.class_extensions is not None:
        args.classes = first_loader.dataset.class_extensions
        print(f"\n+++ Using class extensions: {len(org_classes)} → {len(args.classes)} classes")
    else:
        args.classes = org_classes
        print(f"\n+++ Classes: {len(org_classes)}")

    num_org_classes = len(org_classes)
    ignore_index = first_loader.dataset.ignore_index
    del first_loader  # free memory before loading full model

    # ----------------------------------------------------------------
    # Create adaptation method ONCE — this is the KEY CTTA invariant.
    # The model is NEVER deleted or re-created for the rest of the experiment.
    # ----------------------------------------------------------------
    adapt_method = get_method(args, device)

    # DPCore requires source statistics before adaptation.
    # Using fog-as-source causes loss_raw≈0 when testing on fog, making the ID
    # condition (loss_new < loss_raw * thr_rho) impossible → every batch is OOD.
    # Solution: use a clean source domain (CityscapesDataset) for ACDC experiments.
    if args.method in ('dpcore', 'kff'):
        src_dataset  = args.src_dataset  or args.dataset
        src_data_dir = args.src_data_dir or args.data_dir
        src_corruption = 'original'

        print(f"\n[{args.method}] Computing source statistics from '{src_dataset}' ({src_data_dir}) ...")
        src_loader, _ = segmentation_datasets.prepare_data(
            src_dataset, src_data_dir, args.init_resize,
            args.patch_size, args.patch_stride,
            corruption=src_corruption,
            batch_size=args.batch_size, num_workers=args.workers,
            shuffle=False,
            corruption_severity=args.corruption_severity,
        )
        adapt_method.obtain_src_stat(src_loader)
        del src_loader

    # EATA-Continual requires a one-time forward pass on clean source data to
    # compute the per-LN-param Fisher diagonal (used as EWC weights). Mirrors
    # DPCore's obtain_src_stat pattern; uses corruption="original" so no
    # CorruptTransform is inserted into the pipeline.
    if args.method == 'eata_continual':
        src_dataset  = args.src_dataset    or args.dataset
        src_data_dir = args.src_data_dir   or args.data_dir
        # Datasets with a clean val split (VOC, Cityscapes) use "original".
        # ACDC has no clean split — pass --src_corruption fog (or another
        # condition) to use it as a source proxy, mirroring cma_proto_continual.
        src_corruption = args.src_corruption or 'original'

        print(f"\n[EATA] Loading source: dataset='{src_dataset}', "
              f"data_dir='{src_data_dir}', corruption='{src_corruption}' ...")
        src_loader, _ = segmentation_datasets.prepare_data(
            src_dataset, src_data_dir, args.init_resize,
            args.patch_size, args.patch_stride,
            corruption=src_corruption,
            batch_size=args.batch_size, num_workers=args.workers,
            shuffle=False,
            corruption_severity=args.corruption_severity,
        )
        adapt_method.obtain_src_fisher(src_loader)
        del src_loader

    # CMA-Proto-continual requires per-class source prototypes before adaptation.
    # Unlike DPCore (unsupervised feature stats), the prototype bank groups visual
    # features by CLIP pseudo-label. ACDC has no clean source, so by default we
    # use the first condition (fog) as a source proxy — filters in
    # obtain_src_prototypes (confidence + cross-prompt agreement) control noise.
    if args.method == 'cma_proto_continual':
        src_dataset    = args.src_dataset    or args.dataset
        src_data_dir   = args.src_data_dir   or args.data_dir
        src_corruption = args.src_corruption or conditions[0]

        print(f"\n[CMA-Proto] Loading source proxy: "
              f"dataset='{src_dataset}', corruption='{src_corruption}'")
        src_loader, _ = segmentation_datasets.prepare_data(
            src_dataset, src_data_dir, args.init_resize,
            args.patch_size, args.patch_stride,
            corruption=src_corruption,
            batch_size=args.batch_size, num_workers=args.workers,
            shuffle=False,
            corruption_severity=args.corruption_severity,
        )
        adapt_method.obtain_src_prototypes(src_loader)
        del src_loader

    all_round_results = {}   # round_num → {condition → {mIoU, mDice, mAcc}}
    headers = "mIoU, mDice, mAcc"

    # ----------------------------------------------------------------
    # Universal entropy log: written for EVERY method, every batch.
    # Captures both per-pixel and marginal-class entropy so any method's
    # entropy dynamics can be analyzed post-hoc without re-running.
    # Columns:
    #   total_batch : monotonic batch counter across the whole stream
    #   round       : 1..continual_rounds
    #   condition   : current corruption / condition name
    #   batch_idx   : index within (round, condition)
    #   h_margin    : entropy of the batch's marginal class distribution
    #                 (i.e. -sum(p log p) where p = mean softmax over B,H,W)
    #   h_pixel_mean: mean per-pixel softmax entropy (TENT-style signal)
    #   max_logit   : mean of max softmax probability (per pixel) — confidence
    # ----------------------------------------------------------------
    os.makedirs(args.save_dir, exist_ok=True)
    entropy_log_path = os.path.join(args.save_dir, "entropy_log.csv")
    with open(entropy_log_path, 'w') as f:
        f.write("total_batch,round,condition,batch_idx,"
                "h_margin,h_pixel_mean,max_logit\n")
    _entropy_total_batches = 0  # monotonic across the whole stream

    print(f"\n{'='*65}")
    print(f"  Starting CTTA: {args.continual_rounds} rounds × {len(conditions)} conditions")
    print(f"  Conditions: {conditions}")
    print(f"  Adapt: {args.adapt}  |  Method: {args.method}  |  LR: {args.lr}")
    print(f"  Entropy log: {entropy_log_path}")
    print(f"{'='*65}")

    for round_idx in range(args.continual_rounds):
        round_num = round_idx + 1
        round_results = {}

        print(f"\n{'─'*65}")
        print(f"  Round {round_num:2d} / {args.continual_rounds}")
        print(f"{'─'*65}")

        for cond_idx, condition in enumerate(conditions):
            # ----------------------------------------------------------
            # Load data for this condition.
            # shuffle=False guarantees a deterministic stream order,
            # which is required for a fair CTTA evaluation.
            # ----------------------------------------------------------
            # ACDC overlay corruption: pick the i-th overlay corruption from
            # --acdc_overlay_corruptions for the i-th condition (paired by index).
            _overlay = None
            if args.acdc_overlay_corruptions:
                if len(args.acdc_overlay_corruptions) != len(conditions):
                    raise ValueError(
                        f"--acdc_overlay_corruptions length ({len(args.acdc_overlay_corruptions)}) "
                        f"must match --corruptions_list length ({len(conditions)})")
                _overlay = args.acdc_overlay_corruptions[cond_idx]

            data_loader, _ = segmentation_datasets.prepare_data(
                args.dataset, args.data_dir, args.init_resize,
                args.patch_size, args.patch_stride,
                corruption=condition,
                batch_size=args.batch_size, num_workers=args.workers,
                shuffle=False,
                ann_file=args.ann_file,
                split=args.split,
                subset_size=args.subset_size,
                subset_seed=args.subset_seed,
                corruption_severity=args.corruption_severity,
                acdc_overlay_corruption=_overlay,
            )

            results = []

            pbar = tqdm(
                enumerate(data_loader),
                total=len(data_loader),
                desc=f"  Rd {round_num:02d} | {condition:5s}"
            )
            for batch_idx, data in pbar:
                if args.debug and batch_idx >= 5:
                    break

                inputs = data['img_patches']
                original_gts = data['gt']
                patch_grid_shape = data['meta']['patch_grid_shape']
                image_shapes = data['meta']['img_shape']

                inputs = inputs.to(device, non_blocking=True)

                # *** CTTA: NO reset() — model state accumulates ***
                # Evaluate BEFORE adapt (pre-update prediction), matching main.py protocol
                with torch.no_grad():
                    patch_preds = adapt_method.evaluate(inputs)

                # ----- Universal entropy logging -----
                # patch_preds is per-pixel logits; shape (B*Npatch, C, H, W).
                # Compute marginal-class entropy (DivGate signal),
                # mean per-pixel entropy (TENT signal), and confidence.
                with torch.no_grad():
                    _probs = patch_preds.softmax(dim=1)              # (N, C, H, W)
                    # Marginal class distribution (over batch and spatial dims)
                    _marginal = _probs.mean(dim=(0, 2, 3))            # (C,)
                    _marginal = _marginal / _marginal.sum().clamp(min=1e-8)
                    _h_margin = float(-(
                        _marginal * _marginal.clamp(min=1e-12).log()
                    ).sum().item())
                    # Per-pixel softmax entropy averaged over (N, H, W)
                    _h_pixel = float(-(
                        _probs * _probs.clamp(min=1e-12).log()
                    ).sum(dim=1).mean().item())
                    # Mean of max softmax probability (confidence)
                    _max_logit = float(_probs.max(dim=1).values.mean().item())
                _entropy_total_batches += 1
                with open(entropy_log_path, 'a') as _f:
                    _f.write(f"{_entropy_total_batches},{round_num},{condition},"
                             f"{batch_idx},{_h_margin:.6f},"
                             f"{_h_pixel:.6f},{_max_logit:.6f}\n")
                # ----- end entropy logging -----

                if args.adapt:
                    adapt_method.continual_adapt(inputs)

                # Show coreset size for DPCore
                if hasattr(adapt_method, 'coreset'):
                    pbar.set_postfix(coreset=len(adapt_method.coreset))

                # Reconstruct full-resolution segmentation maps
                if args.init_resize:
                    reconstructed_preds = aggregate_pred_patches(
                        patch_preds, patch_grid_shape, image_shapes,
                        args.patch_size, args.patch_stride
                    )
                else:
                    reconstructed_preds = patch_preds

                # Per-image metric computation (same as main.py)
                for pd, gt in zip(reconstructed_preds, original_gts):
                    pd = pd.softmax(dim=0)

                    # Map extended class indices back to original classes if needed
                    if args.class_extensions and data_loader.dataset.class_extensions is not None:
                        ext_to_real = torch.Tensor(
                            data_loader.dataset.extentions_to_real_class_idx
                        ).to(torch.int64).to(device)
                        num_cls = max(ext_to_real) + 1
                        num_queries = len(ext_to_real)
                        ext_oh = torch.nn.functional.one_hot(ext_to_real).T.view(
                            num_cls, num_queries, 1, 1
                        )
                        pd = (pd.unsqueeze(0) * ext_oh).max(1)[0]

                    pd = pd.argmax(dim=0).to(gt.device)
                    gt = gt[0]
                    results.append(intersect_and_union(pd, gt, num_org_classes, ignore_index))

            # ----------------------------------------------------------
            # Compute and record metrics for this (round, condition)
            # ----------------------------------------------------------
            metrics = process_metrics(results, org_classes)
            round_results[condition] = metrics

            print(
                f"  Rd {round_num:02d} | {condition:5s}: "
                f"mIoU={metrics['mIoU']:.2f}  "
                f"mDice={metrics['mDice']:.2f}  "
                f"mAcc={metrics['mAcc']:.2f}"
            )

            # Save per-condition per-round results (matches other datasets' format)
            c_dir = os.path.join(args.save_dir, f"round_{round_num:02d}", condition)
            os.makedirs(c_dir, exist_ok=True)
            with open(os.path.join(c_dir, "results.txt"), 'w') as f:
                f.write(headers + "\n")
                f.write(
                    f"{metrics['mIoU']:.4f} +/- 0.0000, "
                    f"{metrics['mDice']:.4f} +/- 0.0000, "
                    f"{metrics['mAcc']:.4f} +/- 0.0000\n"
                )

        # Round summary
        mean_miou = np.mean([v['mIoU'] for v in round_results.values()])
        print(f"\n  → Round {round_num} Mean mIoU: {mean_miou:.2f}")

        all_round_results[round_num] = round_results

        # Incrementally update the summary file after each round
        save_round_results(all_round_results, args.save_dir, conditions)

    # ----------------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------------
    total_duration = time.time() - start_time
    gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    final_results_path = os.path.join(args.save_dir, "results.txt")
    with open(final_results_path, 'w') as f:
        f.write(headers + "\n")
        for rnd, cond_results in sorted(all_round_results.items()):
            for cond, metrics in cond_results.items():
                f.write(
                    f"Round {rnd:02d}/{cond}, "
                    f"{metrics['mIoU']:.4f}, "
                    f"{metrics['mDice']:.4f}, "
                    f"{metrics['mAcc']:.4f}\n"
                )
        f.write(f"\nGPU: {gpu_info}\n")
        f.write(f"Total Duration (s): {total_duration:.2f}\n")

    print(f"\nTotal duration: {total_duration:.1f}s  |  GPU: {gpu_info}")
    print(f"All results saved to: {args.save_dir}")


if __name__ == '__main__':
    # Two-pass parsing: first get method name, then add method-specific args
    initial_parser = argparser()
    initial_args, _ = initial_parser.parse_known_args()

    parser = argparser()
    parser = add_method_specific_args(parser, initial_args.method)
    args = parser.parse_args()

    set_global_seeds(args.seed)
    main(args)
