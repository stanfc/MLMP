"""Exp 1 — Per-image cosine between TENT entropy gradient and supervised CE
gradient over visual LayerNorm parameters.

Usage:
    python -m experiments.h1_validation.exp1_gradient_cosine.run \
        --dataset ACDC --condition fog --n 100 \
        [--out experiments/h1_validation/results/exp1/all.csv] \
        [--prompt_idx 0] [--seed 0] [--device cuda] [--skip_done]
"""
from __future__ import annotations
import argparse
import time

import torch
import torch.nn.functional as F

from experiments.h1_validation.common import (
    DATASET_REGISTRY, build_loader, iterate_limited, get_class_names,
    load_source_model, compute_text_features,
    flatten_grads, cosine,
    append_row, already_done,
)


def parse_n(n: str):
    if n == "all":
        return "all"
    return int(n)


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Pixel-wise entropy along class dim (-3); identical to
    adapt.tent_continual.TENTContinual.softmax_entropy."""
    return -(x.softmax(-3) * x.log_softmax(-3)).sum(-3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASET_REGISTRY))
    ap.add_argument("--condition", required=True)
    ap.add_argument("--n", default="100", help="int or 'all'")
    ap.add_argument("--out", default="experiments/h1_validation/results/exp1/all.csv")
    ap.add_argument("--prompt_idx", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip_done", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    entry = DATASET_REGISTRY[args.dataset]
    kind = entry["kind"]
    classes = get_class_names(args.dataset)

    print(f"[exp1] Loading source NA-CLIP on {args.device}...", flush=True)
    model, tokenize, ln_params, _ = load_source_model(device=args.device)
    text_features = compute_text_features(
        model, tokenize, classes, prompt_template_idx=args.prompt_idx, device=args.device
    )
    print(f"[exp1] #LN params={len(ln_params)}, text_features={tuple(text_features.shape)}",
          flush=True)

    n_samples = parse_n(args.n)
    print(f"[exp1] Building loader {args.dataset}/{args.condition} n={n_samples}", flush=True)
    loader = build_loader(args.dataset, args.condition, severity=5, seed=args.seed)

    t0 = time.time()
    n_done = 0
    for idx, batch in iterate_limited(loader, n_samples):
        # Resume support
        if args.skip_done and already_done(
            args.out, {"dataset": args.dataset, "condition": args.condition, "idx": idx}
        ):
            print(f"  [skip done] idx={idx}", flush=True)
            n_done += 1
            continue

        img_patches, gt_patches = _unpack_batch(batch, args.device)

        # NA-CLIP weights are fp16; cast logits to fp32 before loss to avoid
        # softmax/CE underflow. logit_scale = exp(2.66) ≈ 100 also saturates
        # softmax in CE — divide it out (uniform monotonic transform that does
        # not change the supervised direction we measure).
        logit_scale = model.logit_scale.exp().detach().float()

        # Both losses run at the patch-token resolution (interpolate=False).
        # Apples-to-apples: same model output, same forward path, same backward
        # graph — only the loss differs. The interpolate=True bilinear upsample
        # was a route to GT-resolution CE, but its backward pass saturates
        # fp16 ops and produces a zero gradient artefact.

        # ----- g_tent (entropy at patch-token resolution) -----
        for p in ln_params:
            if p.grad is not None:
                p.grad.zero_()
        logits, _, _ = model(img_patches, text_features, True, interpolate=False)
        if logits.dim() == 5:   # (num_prompts=1, N, C, h, w)
            logits = logits[0]
        L_tent = softmax_entropy(logits.float()).mean()
        L_tent.backward()
        g_tent = flatten_grads(ln_params).cpu()

        # ----- g_sup (CE at patch-token resolution; GT nearest-downsampled) -----
        for p in ln_params:
            if p.grad is not None:
                p.grad.zero_()
        logits_sup, _, _ = model(img_patches, text_features, True, interpolate=False)
        if logits_sup.dim() == 5:
            logits_sup = logits_sup[0]    # (N, C, h, w)
        # nearest-mode downsample preserves label classes (no class blending)
        gt_small = F.interpolate(
            gt_patches.unsqueeze(1).float(),
            size=logits_sup.shape[-2:],
            mode="nearest",
        ).squeeze(1).long()
        L_sup = F.cross_entropy(logits_sup.float() / logit_scale, gt_small, ignore_index=255)
        L_sup.backward()
        g_sup = flatten_grads(ln_params).cpu()

        cos, n_tent, n_sup = cosine(g_tent, g_sup)
        n_pixels = int((gt_patches != 255).sum().item())
        append_row(args.out, {
            "dataset": args.dataset,
            "kind": kind,
            "condition": args.condition,
            "idx": idx,
            "cos": cos,
            "norm_tent": n_tent,
            "norm_sup": n_sup,
            "n_pixels": n_pixels,
        })
        n_done += 1
        if n_done % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [progress] idx={idx} cos={cos:+.4f} norm_sup={n_sup:.2e} "
                  f"({elapsed/n_done:.2f}s/img)", flush=True)

    print(f"[exp1] DONE  {args.dataset}/{args.condition}  total={time.time()-t0:.1f}s",
          flush=True)


def _unpack_batch(batch: dict, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Pull img_patches and gt_patches out of a prepare_data batch.

    Real schema (from utils/segmentation_datasets.py custom collate):
      batch['img_patches'] — tensor (N_patches, 3, H, W) float
      batch['gt_patches']  — tensor (N_patches, 1, H, W) long
    """
    if "img_patches" not in batch or "gt_patches" not in batch:
        raise RuntimeError(
            f"unexpected batch schema: keys={list(batch)}. "
            "Expected 'img_patches' and 'gt_patches' from prepare_data collate."
        )
    img_patches = batch["img_patches"].to(device, non_blocking=True)
    gt_patches = batch["gt_patches"].to(device, non_blocking=True).long()
    if gt_patches.dim() == 4 and gt_patches.shape[1] == 1:
        gt_patches = gt_patches.squeeze(1)
    return img_patches, gt_patches


if __name__ == "__main__":
    main()
