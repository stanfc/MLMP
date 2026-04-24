# Standard
import os
import time
import argparse

# Third-party
import torch
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
        choices=('ACDCDataset', 'CityscapesDataset',
                 'COCOStuffDataset', 'COCOObjectDataset',
                 'PascalVOC20Dataset', 'PascalVOC21Dataset',
                 'PascalContext59Dataset', 'PascalContext60Dataset'),
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

    # ----------------------------------------
    # Model
    # ----------------------------------------
    parser.add_argument('--ovss_type', type=str, default='naclip')
    parser.add_argument('--ovss_backbone', type=str, default='ViT-L/14')

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

    # --- TENT-Continual ---
    elif method == 'tent_continual':
        pass   # no extra args beyond base parser

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
        shuffle=False
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
    if args.method == 'dpcore':
        src_dataset  = args.src_dataset  or args.dataset
        src_data_dir = args.src_data_dir or args.data_dir
        src_corruption = 'original'

        print(f"\n[DPCore] Computing source statistics from '{src_dataset}' ({src_data_dir}) ...")
        src_loader, _ = segmentation_datasets.prepare_data(
            src_dataset, src_data_dir, args.init_resize,
            args.patch_size, args.patch_stride,
            corruption=src_corruption,
            batch_size=args.batch_size, num_workers=args.workers,
            shuffle=False
        )
        adapt_method.obtain_src_stat(src_loader)
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
            shuffle=False
        )
        adapt_method.obtain_src_prototypes(src_loader)
        del src_loader

    all_round_results = {}   # round_num → {condition → {mIoU, mDice, mAcc}}
    headers = "mIoU, mDice, mAcc"

    print(f"\n{'='*65}")
    print(f"  Starting CTTA: {args.continual_rounds} rounds × {len(conditions)} conditions")
    print(f"  Conditions: {conditions}")
    print(f"  Adapt: {args.adapt}  |  Method: {args.method}  |  LR: {args.lr}")
    print(f"{'='*65}")

    for round_idx in range(args.continual_rounds):
        round_num = round_idx + 1
        round_results = {}

        print(f"\n{'─'*65}")
        print(f"  Round {round_num:2d} / {args.continual_rounds}")
        print(f"{'─'*65}")

        for condition in conditions:
            # ----------------------------------------------------------
            # Load data for this condition.
            # shuffle=False guarantees a deterministic stream order,
            # which is required for a fair CTTA evaluation.
            # ----------------------------------------------------------
            data_loader, _ = segmentation_datasets.prepare_data(
                args.dataset, args.data_dir, args.init_resize,
                args.patch_size, args.patch_stride,
                corruption=condition,
                batch_size=args.batch_size, num_workers=args.workers,
                shuffle=False
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
