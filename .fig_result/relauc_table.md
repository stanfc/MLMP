# RelAUC and AvgRelAUC (150-round whole-stream, relative to No-Adapt)

Generated 2026-09-22 from `results_all_rounds.txt` of each run
(script: `.fig_result/make_relauc.py`, source CSV: `.fig_result/relauc_table.csv`).

```
RelAUC(method, dataset) = (1/150) * sum_{r=1..150} (mIoU_r - NoAdapt_mIoU) / NoAdapt_mIoU * 100
AvgRelAUC(method)        = (1/8)   * sum_d RelAUC(method, d)
```

MLMP-episodic is excluded (no continual trajectory). Sanity check: all 88 runs (11 methods x 8 datasets) have exactly rounds 1-150.

## RelAUC(method, dataset), %

| Method | ACDC | Cityscapes | V20 | V21 | P59 | P60 | COCO-Stuff | COCO-Object |
|---|---|---|---|---|---|---|---|---|
| CAGA | 32.75 | 6.86 | 7.20 | 16.96 | 10.77 | 8.90 | 7.07 | 15.32 |
| DAF | -4.99 | -72.39 | -21.18 | -6.94 | -13.87 | -15.03 | -23.01 | -29.02 |
| TENT | -16.58 | -90.78 | -94.75 | -82.12 | -93.04 | -92.72 | -88.01 | -69.37 |
| SAR | 6.37 | -22.09 | -7.99 | -2.28 | 1.77 | 1.54 | -0.08 | 13.66 |
| SPA | -3.06 | -34.05 | 12.05 | -2.05 | 1.78 | 0.02 | 2.67 | 2.65 |
| TRIBE | 0.53 | -1.09 | 5.97 | -7.57 | -8.75 | -10.75 | -11.75 | -7.86 |
| DeYO | -8.78 | -66.85 | 1.65 | 20.00 | 16.96 | 13.50 | -11.78 | 13.34 |
| ViDA | -70.81 | -96.95 | -97.90 | -82.12 | -98.20 | -98.03 | -98.49 | -93.08 |
| PTTA | -3.89 | -10.37 | -53.54 | -21.87 | -50.16 | -50.57 | -13.25 | -11.00 |
| COME | -23.33 | -92.44 | -94.49 | -8.06 | -92.50 | -92.09 | -83.69 | -87.09 |
| MGP | 1.96 | -68.86 | 2.11 | 18.65 | 3.87 | 2.78 | -48.54 | 12.42 |

## AvgRelAUC(method), ranked

| Rank | Method | AvgRelAUC |
|---|---|---|
| 1 | **CAGA** | 13.23 |
| 2 | SAR | -1.14 |
| 3 | SPA | -2.50 |
| 4 | DeYO | -2.74 |
| 5 | TRIBE | -5.16 |
| 6 | MGP | -9.45 |
| 7 | DAF | -23.30 |
| 8 | PTTA | -26.83 |
| 9 | COME | -71.71 |
| 10 | TENT | -78.42 |
| 11 | ViDA | -91.95 |
