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
# OpenCV defaults to one thread per core (~256). Forked DataLoader workers each
# try to spawn that pool; under many concurrent experiments this exhausts the
# per-user thread limit (ulimit -u) and kills workers with "Can't spawn new
# thread (res=11)". Cap it (must be set before DataLoader workers fork).
try:
    import cv2
    cv2.setNumThreads(int(os.environ.get('OPENCV_NUM_THREADS', '2')))
except Exception:
    pass
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
    # --- analysis experiments (opt-in; default behaviour unchanged) ---
    parser.add_argument(
        '--ood_corruptions', nargs='+', type=str, default=None,
        help=('Analysis: per round, after adapting on --corruptions_list, also run a '
              'FROZEN-weight evaluation on these held-out corruptions (no adapt). '
              'Produces results_ood.txt — an OOD generalization curve. If it trends '
              'like the in-distribution curve, adaptation is improving the model '
              'overall rather than fitting the training corruptions.'))
    parser.add_argument(
        '--resample_subset', action='store_true',
        help=('Analysis: re-draw a fresh random --subset_size subset every (round, '
              'corruption) instead of a fixed one. Tests whether the method overfits '
              'to the specific 100 images. Effective seed = subset_seed + 1000*round + cond_idx.'))
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
    parser.add_argument('--log_signals', action='store_true',
                        help='Also log the full collapse/degradation signal panel to '
                             'signals_log.csv (utils/collapse_signals.py). Opt-in; off by default.')
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

    # --- LCoTTA + MLMP: MLMP loss + subspace-projected gradient update ---
    elif method == 'lcotta_mlmp_continual':
        parser.add_argument('--vision_outputs', nargs='+', type=int, default=(-1,))
        parser.add_argument('--prompt_integration', type=str, default='loss')
        parser.add_argument('--alpha_cls', type=float, default=1.0)
        parser.add_argument('--w_num', type=int, default=100,
                            help='LCoTTA moving window size (gradient vectors kept)')
        parser.add_argument('--n_components', type=int, default=20,
                            help='LCoTTA PCA components = subspace dimension')
        parser.add_argument('--batch_step', type=int, default=50,
                            help='record a gradient every this many batches')

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
    elif method in ('deyo_uaml_continual', 'deyo_mlmp_continual', 'deyo_mlmp_divloss_continual'):
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
        if method == 'deyo_mlmp_divloss_continual':
            parser.add_argument('--lambda_div', type=float, default=0.5,
                                help='weight of the marginal-diversity loss term (-lambda*H_margin)')

    # --- DeYO + MLMP + Diversity Regularizer (loss-side) ---
    elif method == 'deyo_mlmp_divreg_continual':
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
        parser.add_argument('--lambda_div', type=float, default=0.0,
                            help='Weight of the -H_margin diversity term added '
                                 'to the DeYO loss. 0 = bit-identical to '
                                 'deyo_mlmp_continual.')

    # --- SHOT (Information-Maximization loss) continual baseline ---
    elif method == 'shot_continual':
        parser.add_argument('--vision_outputs', nargs='+', type=int,
                            default=tuple(range(-1, -19, -1)))
        parser.add_argument('--top_block_exclude', type=int, default=6)
        parser.add_argument('--lambda_div', type=float, default=1.0,
                            help="Weight of SHOT's diversity term L_div in "
                                 "L = L_ent - lambda_div*L_div. SHOT default 1.0.")

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

    elif method == 'deyo_mlmp_gradpen_divgate_continual':
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
        parser.add_argument('--grad_pen_lambda', type=float, default=0.0,
                            help='strength of the ||grad|| penalty; 0 = divgate baseline')
        parser.add_argument('--grad_pen_form', type=str, default='linear',
                            choices=['sq', 'linear'])
        parser.add_argument('--grad_pen_mode', type=str, default='raw',
                            choices=['raw', 'excess', 'ema'],
                            help='raw=always-on; excess/ema=penalise rise above baseline')
        parser.add_argument('--grad_pen_ema_decay', type=float, default=0.99)
        parser.add_argument('--grad_clip', type=float, default=0.0,
                            help='clip_grad_norm_ on LN params (0=off); only when lambda>0')

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

    # --- DeYO+MLMP COMPOSITE gate (mean_conf trigger + grad_norm depth) ---
    elif method == 'deyo_mlmp_composite_gate_continual':
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
        parser.add_argument('--conf_ceil', type=float, default=0.80,
                            help='mean_conf trigger: gate restores once windowed '
                                 'confidence rises past this (past-peak)')
        parser.add_argument('--base_rst', type=float, default=0.005,
                            help='restore prob at trigger (grad ratio = 1)')
        parser.add_argument('--grad_mult_max', type=float, default=4.0,
                            help='max multiplier on base_rst as grad_norm rises '
                                 'above its trigger baseline')
        parser.add_argument('--monitor_interval', type=int, default=50)

    # --- DeYO+MLMP DivReg loss + COMPOSITE gate (combined method) ---
    elif method == 'deyo_mlmp_divreg_composite_continual':
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
        parser.add_argument('--lambda_div', type=float, default=0.3,
                            help='diversity term weight in L_DeYO - lambda_div*H_margin')
        parser.add_argument('--conf_ceil', type=float, default=0.80,
                            help='mean_conf trigger (RE-CALIBRATE for divreg: '
                                 'diversity lowers mean_conf)')
        parser.add_argument('--base_rst', type=float, default=0.005)
        parser.add_argument('--grad_mult_max', type=float, default=4.0)
        parser.add_argument('--monitor_interval', type=int, default=50)

    # --- DeYO+MLMP grad_norm-SLOPE gate (rising grad -> lagged restore) ---
    elif method == 'deyo_mlmp_gradslope_continual':
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
        parser.add_argument('--slope_window', type=int, default=10,
                            help='# monitor windows for the grad_norm slope (~500 batches)')
        parser.add_argument('--slope_deadzone', type=float, default=0.002,
                            help='slope <= this -> no restore (grad flat/falling). '
                                 'Observed slopes: ~0.002 (VOC20) .. ~0.008 (ACDC collapse)/window')
        parser.add_argument('--lag_gain', type=float, default=100000.0,
                            help='lag = lag_gain * slope (batches to restore back); '
                                 'slope 0.005 -> lag 500')
        parser.add_argument('--max_lag', type=int, default=3000)
        parser.add_argument('--base_rst', type=float, default=0.01,
                            help='restore prob when gate is active')
        parser.add_argument('--monitor_interval', type=int, default=50)

    # --- DeYO+MLMP grad-ANCHOR gate (restore toward best-state, rst ~ slope) ---
    elif method == 'deyo_mlmp_gradanchor_continual':
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
        parser.add_argument('--slope_window', type=int, default=10)
        parser.add_argument('--slope_deadzone', type=float, default=0.002)
        parser.add_argument('--rst_gain', type=float, default=2.0,
                            help='rst = clamp(rst_gain * slope, 0, max_rst)')
        parser.add_argument('--max_rst', type=float, default=0.1,
                            help='cap on restore prob toward the best-state anchor')
        parser.add_argument('--monitor_interval', type=int, default=50)

    # --- DeYO+MLMP ADAPTIVE-lag gate (method1: lag capped at grad-min distance) ---
    elif method == 'deyo_mlmp_gradlagadapt_continual':
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
        parser.add_argument('--slope_window', type=int, default=10)
        parser.add_argument('--slope_deadzone', type=float, default=0.002)
        parser.add_argument('--lag_gain', type=float, default=1500.0,
                            help='lag(windows) = lag_gain * slope, capped at grad-min distance')
        parser.add_argument('--base_rst', type=float, default=0.01)
        parser.add_argument('--max_windows', type=int, default=2000)
        parser.add_argument('--slope_unlock', type=float, default=0.0,
                            help='slope < this -> shallow cap (maxlag_shallow); '
                                 '>= this -> adaptive deep cap. 0 = always deep (pure method1)')
        parser.add_argument('--maxlag_shallow', type=int, default=6,
                            help='shallow lag cap in windows (6w=300batch) for the gentle regime')
        parser.add_argument('--monitor_interval', type=int, default=50)

    # --- DeYO+MLMP grad-RATIO gate (proposed: EMA-grad/min ratio trigger) ---
    elif method == 'deyo_mlmp_gradratio_continual':
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
        parser.add_argument('--grad_ema_alpha', type=float, default=0.99,
                            help='EMA decay for grad_norm smoothing (noise immunity)')
        parser.add_argument('--trigger_ratio', type=float, default=1.1,
                            help='restore when g_ema/g_min exceeds this (relative rise)')
        parser.add_argument('--rst_gain', type=float, default=0.15,
                            help='rst = clamp(rst_gain*(r - trigger_ratio), 0, max_rst)')
        parser.add_argument('--max_rst', type=float, default=0.1)
        parser.add_argument('--monitor_interval', type=int, default=50)

    # --- DeYO+MLMP H-MARGIN-regime gate (method1, deep cap gated by H_margin drop) ---
    elif method == 'deyo_mlmp_hmgate_continual':
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
        parser.add_argument('--slope_window', type=int, default=10)
        parser.add_argument('--slope_deadzone', type=float, default=0.002)
        parser.add_argument('--lag_gain', type=float, default=1500.0)
        parser.add_argument('--base_rst', type=float, default=0.01)
        parser.add_argument('--max_windows', type=int, default=2000)
        parser.add_argument('--h_drop_ratio', type=float, default=0.9,
                            help='collapse regime when windowed H_margin < this * running-max H '
                                 '-> deep cap; else shallow cap (preserve climb)')
        parser.add_argument('--maxlag_shallow', type=int, default=6)
        parser.add_argument('--monitor_interval', type=int, default=50)

    # --- DeYO+MLMP HMGate2: deep restore -> PERMANENT best anchor (never evicted) ---
    elif method in ('deyo_mlmp_hmgate2_continual', 'deyo_mlmp_promptw_hmgate2_continual',
                    'deyo_mlmp_textres_hmgate2_continual', 'deyo_mlmp_adagate_continual'):
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
        parser.add_argument('--slope_window', type=int, default=10)
        parser.add_argument('--slope_deadzone', type=float, default=0.002)
        parser.add_argument('--lag_gain', type=float, default=1500.0)
        parser.add_argument('--base_rst', type=float, default=0.01)
        parser.add_argument('--max_windows', type=int, default=2000)
        parser.add_argument('--h_drop_ratio', type=float, default=0.9,
                            help='collapse regime when windowed H_margin < this * running-max H '
                                 '-> deep restore toward PERMANENT best anchor')
        parser.add_argument('--maxlag_shallow', type=int, default=6)
        parser.add_argument('--monitor_interval', type=int, default=50)
        if method == 'deyo_mlmp_adagate_continual':
            # self-calibrating trigger (A) and lag (B); see the module docstring for
            # why the absolute slope_deadzone / lag_gain are inert in hmgate2.
            parser.add_argument('--trend_stat', type=str, default='mad',
                                choices=['abs', 'rel', 'mad', 'tstat'],
                                help="normalisation of the grad-norm OLS slope before the "
                                     "deadzone test. 'abs' = hmgate2 (raw slope); 'mad' = "
                                     "robust z vs recent slope spread; 'tstat' = slope/SE; "
                                     "'rel' = slope/grad_norm.")
            parser.add_argument('--trend_thr', type=float, default=0.5,
                                help='unitless deadzone on the normalised trend z. Ignored '
                                     'for trend_stat=abs, which uses --slope_deadzone.')
            parser.add_argument('--trend_hist', type=int, default=50,
                                help='window count for the MAD scale and the ECDF rank.')
            parser.add_argument('--lag_mode', type=str, default='ecdf',
                                choices=['gain', 'sat', 'ecdf'],
                                help="shallow-restore depth. 'gain' = hmgate2 "
                                     "(round(lag_gain*slope), saturates in practice); "
                                     "'sat' = ceil(cap*clamp(z/lag_sat)); 'ecdf' = "
                                     "ceil(cap*rank(z)), tunable-free and scale-invariant.")
            parser.add_argument('--lag_sat', type=float, default=1.5,
                                help='z at which lag_mode=sat reaches the full budget.')
            parser.add_argument('--shallow_cap_mode', type=str, default='fixed',
                                choices=['fixed', 'growing', 'growing_scaled',
                                        'growing_hmargin', 'growing_hmargin_scaled'],
                                help="(C) how far back SHALLOW restore may reach. 'fixed' = "
                                     "min(maxlag_shallow, windows_since_min), AdaGate A/B "
                                     "behaviour. 'growing' = min(len(win_buf)-1, "
                                     "windows_since_min): deletes maxlag_shallow as a cap, so "
                                     "reach grows with time-since-best -- targets post-peak "
                                     "tail decline on datasets whose H_margin never triggers "
                                     "collapse_regime (e.g. VOC20). 'growing_scaled' = same but "
                                     "scaled by today's ecdf severity rank u, so a still-"
                                     "improving run isn't over-restored just because grad_norm "
                                     "bottomed out early. 'growing_hmargin' = distance measured "
                                     "since H_margin's own peak instead of grad_norm's minimum. "
                                     "'growing_hmargin_scaled' combines both.")

        if method == 'deyo_mlmp_promptw_hmgate2_continual':
            # entropy-weighted prompt aggregation (prompt/text-template axis)
            parser.add_argument('--prompt_weight_beta', type=float, default=0.0,
                                help='adapt-side: w^t = softmax(-beta*h^t) over templates. '
                                     '0 = bit-identical GDG-PA (uniform).')
            parser.add_argument('--eval_prompt_mode', type=str, default='avg_embed',
                                choices=['avg_embed', 'ent_weight'],
                                help="avg_embed = GDG-PA original eval; ent_weight = "
                                     "entropy-weighted logit ensemble over templates.")
            parser.add_argument('--eval_prompt_beta', type=float, default=1.0,
                                help='eval-side beta for ent_weight mode.')
        if method == 'deyo_mlmp_textres_hmgate2_continual':
            # learnable text-embedding residual + orthogonality regularizer
            parser.add_argument('--text_res_lr', type=float, default=0.0,
                                help='LR for the (C,D) text residual. '
                                     '0 = no residual created = bit-identical GDG-PA.')
            parser.add_argument('--lambda_orth', type=float, default=0.0,
                                help='weight of the off-diagonal cosine penalty on the '
                                     'class text vectors. No effect unless text_res_lr>0.')
            parser.add_argument('--text_res_max_norm', type=float, default=0.5,
                                help='per-class L2 cap on the residual (0 = uncapped).')

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
        parser.add_argument('--marginal_buf_size', type=int, default=None,
                            help='Decouple H_margin averaging window from lag-update '
                                 'cadence. None -> coupled (H over a fresh '
                                 'monitor_interval window). If set, H is a rolling mean '
                                 'over the last marginal_buf_size batches while lag still '
                                 'refreshes every monitor_interval batches.')

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


def save_round_results(all_round_results, save_dir, conditions, filename="results_all_rounds.txt"):
    """
    Write a summary table matching CoTTA Table 5 format.

    Output: save_dir/<filename>
    Columns: Round | fog | night | rain | snow | Mean
    """
    summary_path = os.path.join(save_dir, filename)

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


def _ckpt_path(save_dir):
    return os.path.join(save_dir, "ckpt.pt")


def save_checkpoint(save_dir, adapt_method, round_idx, cond_idx, round_results,
                    all_round_results, entropy_total_batches):
    """Atomically persist adapt-method state (LN params, optimizer, gate
    bookkeeping) plus stream position, so a killed/rebooted run can resume from
    the last completed (round, condition) instead of restarting from scratch.
    No-op if the method doesn't implement state_dict() (older methods)."""
    if not hasattr(adapt_method, "state_dict"):
        return
    ckpt = {
        'adapt_state': adapt_method.state_dict(),
        'round_idx': round_idx,
        'cond_idx': cond_idx,
        'round_results': round_results,
        'all_round_results': all_round_results,
        'entropy_total_batches': entropy_total_batches,
    }
    path = _ckpt_path(save_dir)
    tmp_path = path + ".tmp"
    torch.save(ckpt, tmp_path)
    os.replace(tmp_path, path)  # atomic on POSIX -- never leaves a half-written ckpt.pt


def load_checkpoint(save_dir, device):
    path = _ckpt_path(save_dir)
    if not os.path.exists(path):
        return None
    # Keep everything on CPU as it was saved -- marginal_buf/_win_buf/best_snapshot
    # are deliberately CPU-resident throughout the class (GPU memory). Only
    # ln_params get moved to `device`, done explicitly inside load_state_dict().
    # map_location=device here would force EVERYTHING to GPU, creating a mixed
    # CPU/CUDA marginal_buf once new (still-CPU) entries get appended post-resume.
    # weights_only=False: torch>=2.6 defaults to True, which refuses to unpickle
    # the numpy float64 mIoU/mDice/mAcc values inside round_results/all_round_results.
    # Safe here -- this checkpoint is one we wrote ourselves, not a third-party file.
    return torch.load(path, map_location='cpu', weights_only=False)


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

    # ----------------------------------------------------------------
    # Resume from a previous run's checkpoint, if present (survives kills /
    # machine maintenance). Restores LN params, optimizer, and gate state, and
    # picks up right after the last fully-completed (round, condition).
    # ----------------------------------------------------------------
    resume_round_idx, resume_cond_idx = 0, -1
    round_results = {}
    ckpt = load_checkpoint(args.save_dir, device)
    if ckpt is not None and hasattr(adapt_method, "load_state_dict"):
        adapt_method.load_state_dict(ckpt['adapt_state'])
        resume_round_idx = ckpt['round_idx']
        resume_cond_idx = ckpt['cond_idx']
        round_results = ckpt['round_results']
        print(f"\n+++ Resumed from checkpoint: round {resume_round_idx + 1}, "
              f"after condition idx {resume_cond_idx} "
              f"({args.corruptions_list[resume_cond_idx] if resume_cond_idx >= 0 else 'none'})")
    elif ckpt is not None:
        print(f"\n+++ WARNING: checkpoint found at {_ckpt_path(args.save_dir)} but "
              f"method '{args.method}' has no load_state_dict() -- ignoring checkpoint, "
              f"starting fresh (results may be overwritten).")

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

    all_round_results = (ckpt['all_round_results'] if ckpt is not None
                        and hasattr(adapt_method, "load_state_dict") else {})
    all_ood_results = {}     # round_num → {ood_corruption → metrics} (--ood_corruptions)
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
    _resuming = ckpt is not None and hasattr(adapt_method, "load_state_dict")
    entropy_log_path = os.path.join(args.save_dir, "entropy_log.csv")
    if not (_resuming and os.path.exists(entropy_log_path)):
        with open(entropy_log_path, 'w') as f:
            f.write("total_batch,round,condition,batch_idx,"
                    "h_margin,h_pixel_mean,max_logit\n")
    _entropy_total_batches = ckpt['entropy_total_batches'] if _resuming else 0

    # ----- Optional full signal panel (opt-in via --log_signals) -----
    _signal_monitor = None
    _signals_log_path = None
    if args.log_signals:
        from utils.collapse_signals import SignalMonitor, SIGNAL_NAMES
        _signal_monitor = SignalMonitor(adapt_method)
        # Turn on the method's side-effect-free diagnostic stashing, if it supports it.
        if hasattr(adapt_method, "collect_diag"):
            adapt_method.collect_diag = True
        _signals_log_path = os.path.join(args.save_dir, "signals_log.csv")
        if not (_resuming and os.path.exists(_signals_log_path)):
            with open(_signals_log_path, 'w') as f:
                f.write("total_batch,round,condition,batch_idx," + ",".join(SIGNAL_NAMES) + "\n")
        print(f"  Signal panel log: {_signals_log_path} ({len(SIGNAL_NAMES)} signals)")

    print(f"\n{'='*65}")
    print(f"  Starting CTTA: {args.continual_rounds} rounds × {len(conditions)} conditions")
    print(f"  Conditions: {conditions}")
    print(f"  Adapt: {args.adapt}  |  Method: {args.method}  |  LR: {args.lr}")
    print(f"  Entropy log: {entropy_log_path}")
    print(f"{'='*65}")

    for round_idx in range(resume_round_idx, args.continual_rounds):
        round_num = round_idx + 1
        if round_idx != resume_round_idx:
            round_results = {}   # fresh round; the resume round keeps its loaded partial results

        print(f"\n{'─'*65}")
        print(f"  Round {round_num:2d} / {args.continual_rounds}")
        print(f"{'─'*65}")

        _cond_start_idx = (resume_cond_idx + 1) if round_idx == resume_round_idx else 0
        for cond_idx, condition in enumerate(conditions):
            if cond_idx < _cond_start_idx:
                continue  # already completed before the checkpoint was taken
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

            # Analysis (--resample_subset): fresh random subset each (round, cond).
            _subset_seed = args.subset_seed
            if args.resample_subset:
                _subset_seed = args.subset_seed + 1000 * round_num + cond_idx

            data_loader, _ = segmentation_datasets.prepare_data(
                args.dataset, args.data_dir, args.init_resize,
                args.patch_size, args.patch_stride,
                corruption=condition,
                batch_size=args.batch_size, num_workers=args.workers,
                shuffle=False,
                ann_file=args.ann_file,
                split=args.split,
                subset_size=args.subset_size,
                subset_seed=_subset_seed,
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

                # ----- Optional diagnostic forward (per-layer/feature signals) -----
                # RNG-free (model has dropout_p=0), so it does NOT perturb the
                # adaptation trajectory. Run before adapt to capture pre-update state.
                if _signal_monitor is not None and hasattr(adapt_method, "diagnose"):
                    adapt_method.diagnose(inputs)

                if args.adapt:
                    adapt_method.continual_adapt(inputs)

                # ----- Optional full signal panel (written AFTER adapt so the -----
                # ----- method's adapt-internal diag reflects THIS batch) -----
                if _signal_monitor is not None:
                    _sig = _signal_monitor.update(
                        patch_preds, round_num, condition, batch_idx)
                    with open(_signals_log_path, 'a') as _f:
                        _f.write(f"{_entropy_total_batches},{round_num},{condition},"
                                 f"{batch_idx}," +
                                 ",".join(f"{v:.6f}" for v in _sig.values()) + "\n")
                # ----- end signal panel -----

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

            # Checkpoint after every (round, condition): survives a kill/reboot
            # with at most one condition's worth of adaptation re-done on resume.
            save_checkpoint(args.save_dir, adapt_method, round_idx, cond_idx,
                            round_results, all_round_results, _entropy_total_batches)

        # ----------------------------------------------------------
        # Analysis (--ood_corruptions): frozen-weight eval on held-out
        # corruptions, AFTER this round's adaptation. No adapt, no state change.
        # ----------------------------------------------------------
        if args.ood_corruptions:
            ood_results = {}
            for ood_cond in args.ood_corruptions:
                ood_loader, _ = segmentation_datasets.prepare_data(
                    args.dataset, args.data_dir, args.init_resize,
                    args.patch_size, args.patch_stride,
                    corruption=ood_cond,
                    batch_size=args.batch_size, num_workers=args.workers,
                    shuffle=False, split=args.split,
                    subset_size=args.subset_size, subset_seed=args.subset_seed,
                    corruption_severity=args.corruption_severity,
                )
                ood_res = []
                for data in ood_loader:
                    inputs = data['img_patches'].to(device, non_blocking=True)
                    with torch.no_grad():
                        patch_preds = adapt_method.evaluate(inputs)   # FROZEN
                    if args.init_resize:
                        rec = aggregate_pred_patches(
                            patch_preds, data['meta']['patch_grid_shape'],
                            data['meta']['img_shape'], args.patch_size, args.patch_stride)
                    else:
                        rec = patch_preds
                    for pd, gt in zip(rec, data['gt']):
                        pd = pd.softmax(dim=0)
                        if args.class_extensions and ood_loader.dataset.class_extensions is not None:
                            ext_to_real = torch.Tensor(
                                ood_loader.dataset.extentions_to_real_class_idx
                            ).to(torch.int64).to(device)
                            num_cls = max(ext_to_real) + 1
                            num_queries = len(ext_to_real)
                            ext_oh = torch.nn.functional.one_hot(ext_to_real).T.view(
                                num_cls, num_queries, 1, 1)
                            pd = (pd.unsqueeze(0) * ext_oh).max(1)[0]
                        pd = pd.argmax(dim=0).to(gt.device)
                        ood_res.append(intersect_and_union(pd, gt[0], num_org_classes, ignore_index))
                ood_metrics = process_metrics(ood_res, org_classes)
                ood_results[ood_cond] = ood_metrics
                print(f"  Rd {round_num:02d} | OOD {ood_cond:5s}: mIoU={ood_metrics['mIoU']:.2f}")
            all_ood_results[round_num] = ood_results
            save_round_results(all_ood_results, args.save_dir, args.ood_corruptions,
                               filename="results_ood.txt")

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
