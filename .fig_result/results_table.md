# CAGA benchmark results: R150 / Peak mIoU per dataset x method

Generated 2026-09-22 from `results_all_rounds.txt` of each run (script: `.fig_result/make_results_table.py`, full CSV: `.fig_result/results_table.csv`).
Every cell is **R150 mIoU / Peak mIoU (round of the peak)**. All runs are single-seed (seed 0), 150 rounds, all complete at generation time.

## Definitions
- **R150**: Mean mIoU (average over the domains of that round) at the end of round 150.
- **Peak**: highest per-round Mean mIoU over the whole 150-round trajectory; the number in brackets is the round where it occurs.
- **No-Adapt**: source model, one number per dataset (no adaptation).
- **MLMP-episodic**: episodic MLMP (reset every sample), mean over conditions; it has no trajectory, so there is no peak. Run with steps=1, lr 1e-3 (the MLMP repo default is steps=10).
- **CAGA** = `deyo_mlmp_adagate_hgate_continual` (locked config, anchor_h_min_ratio 0.95).
- Protocol: NA-CLIP ViT-L/14, LayerNorm-only, batch 1, evaluate-before-adapt, no reset. ACDC = full val split (406 imgs/round, fog -> night -> rain -> snow). Cityscapes / VOC20 / V21 / P59 / P60 / COCO-Stuff / COCO-Object = 100-image subset x 15 ImageNet-C corruptions (severity 5), 1500 imgs/round.

## Main datasets

| Method | ACDC | Cityscapes | V20 |
|---|---|---|---|
| **No-Adapt** | 23.34 | 22.67 | 68.51 |
| MLMP-episodic | 29.84 | 23.40 | 74.09 |
| **CAGA** | 30.95 / 31.55 (43) | 24.22 / 24.31 (74) | 73.48 / 73.79 (13) |
| DAF | 16.55 / 29.73 (4) | 1.97 / 22.94 (1) | 40.56 / 70.80 (2) |
| TENT | 7.90 / 32.90 (19) | 1.22 / 21.94 (1) | 0.25 / 67.53 (1) |
| SAR | 25.67 / 27.32 (62) | 22.69 / 22.69 (106) | 62.39 / 68.45 (59) |
| SPA | 19.20 / 24.91 (13) | 10.11 / 22.12 (3) | 79.15 / 79.15 (149) |
| TRIBE | 23.29 / 24.05 (50) | 21.37 / 23.84 (28) | 71.72 / 73.99 (34) |
| DeYO | 9.52 / 32.11 (30) | 2.21 / 25.56 (7) | 68.85 / 72.12 (12) |
| ViDA | 4.03 / 27.98 (6) | 0.01 / 3.73 (1) | 0.25 / 36.61 (1) |
| PTTA | 22.27 / 23.38 (1) | 20.22 / 21.45 (1) | 10.35 / 67.17 (1) |
| COME | 1.51 / 30.58 (28) | 1.18 / 21.62 (1) | 0.26 / 67.66 (1) |
| MGP | 6.12 / 34.04 (49) | 2.00 / 25.14 (8) | 66.62 / 76.35 (27) |

## Appendix datasets

| Method | V21 | P59 | P60 | COCO-Stuff | COCO-Object |
|---|---|---|---|---|---|
| **No-Adapt** | 42.17 | 24.14 | 22.44 | 12.50 | 18.71 |
| MLMP-episodic | 45.44 | 24.78 | 23.02 | 12.70 | 21.00 |
| **CAGA** | 50.01 / 50.04 (77) | 26.92 / 26.97 (147) | 24.57 / 24.64 (110) | 13.30 / 13.54 (23) | 21.73 / 21.75 (120) |
| DAF | 33.54 / 46.25 (7) | 16.64 / 25.80 (9) | 15.16 / 23.70 (11) | 7.89 / 13.11 (3) | 10.18 / 20.40 (1) |
| TENT | 0.10 / 41.30 (5) | 0.01 / 23.41 (4) | 0.01 / 21.93 (5) | 0.04 / 12.96 (7) | 1.35 / 22.45 (18) |
| SAR | 40.79 / 42.22 (99) | 25.11 / 26.07 (12) | 22.44 / 24.18 (77) | 12.61 / 12.87 (88) | 20.02 / 23.05 (131) |
| SPA | 41.57 / 41.58 (148) | 25.45 / 25.45 (150) | 23.04 / 23.04 (150) | 13.24 / 13.26 (148) | 19.86 / 19.86 (150) |
| TRIBE | 40.29 / 40.29 (150) | 22.70 / 22.70 (149) | 20.62 / 20.62 (149) | 11.33 / 11.33 (150) | 17.95 / 17.95 (150) |
| DeYO | 53.08 / 53.09 (139) | 29.11 / 29.12 (149) | 26.07 / 26.07 (150) | 10.22 / 13.03 (14) | 22.13 / 22.15 (148) |
| ViDA | 0.34 / 39.67 (1) | 0.05 / 12.68 (1) | 0.01 / 12.15 (1) | 0.00 / 7.18 (1) | 0.07 / 11.13 (1) |
| PTTA | 19.98 / 40.49 (1) | 2.98 / 22.10 (1) | 1.75 / 20.61 (1) | 9.20 / 11.81 (1) | 15.55 / 17.55 (1) |
| COME | 37.60 / 42.25 (5) | 0.01 / 24.01 (6) | 0.01 / 22.48 (7) | 0.00 / 12.83 (6) | 0.02 / 20.15 (9) |
| MGP | 52.91 / 52.91 (150) | 22.37 / 27.44 (46) | 20.93 / 24.93 (42) | 0.36 / 13.48 (26) | 20.03 / 21.67 (61) |

## Ablation variants of CAGA (not drawn in the main figure)

CAGA-gradgate = no H-margin check on anchor update; CAGA-scale6 = fixed shallow cap (maxlag_shallow=6).

| Method | ACDC | Cityscapes | V20 | V21 | P59 | P60 | COCO-Stuff | COCO-Object |
|---|---|---|---|---|---|---|---|---|
| CAGA | 30.95 / 31.55 (43) | 24.22 / 24.31 (74) | 73.48 / 73.79 (13) | 50.01 / 50.04 (77) | 26.92 / 26.97 (147) | 24.57 / 24.64 (110) | 13.30 / 13.54 (23) | 21.73 / 21.75 (120) |
| CAGA-gradgate | 31.92 / 32.00 (123) | 21.29 / 24.46 (7) | 72.33 / 73.79 (14) | 50.69 / 50.72 (124) | 27.01 / 27.05 (144) | 24.91 / 24.93 (133) | 13.36 / 13.54 (24) | 22.01 / 22.01 (139) |
| CAGA-scale6 | 31.80 / 31.97 (84) | 24.38 / 24.39 (8) | 71.36 / 73.78 (10) | 50.57 / 50.60 (126) | 27.49 / 27.49 (109) | 24.94 / 24.97 (65) | 13.25 / 13.59 (16) | 21.95 / 22.02 (49) |

## Reading notes (sanity checks)
1. Peak at round 1 means the method already degraded or collapsed inside the first round (TENT, ViDA, PTTA, COME, DAF on Cityscapes); its peak is then near or below No-Adapt.
2. Peak equal to R150 (SPA, TRIBE, MGP on V21, ...) means the curve is flat or still slowly rising; it is not a "rise then fall" pattern.
3. SAR is a periodic sawtooth on Cityscapes and V20 (reset cycle), so its R150 depends on where round 150 falls inside a cycle (Cityscapes: R150 = peak = 22.69, V20: R150 = 62.39, mid-cycle).
4. CoTTA is excluded (stopped by decision). Baseline hyperparameters are not identical across methods (see the separate hyperparameter notes).
