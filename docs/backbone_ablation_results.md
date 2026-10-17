# Backbone ablation: CAGA on ACDC across 14 backbones (naclip_gaussonly_L14 and sclip_uaml1_B16 dropped; includes the default naclip_L14 row, from `/home/stanfc/TTA-on-OVSS/MLMP/save/ACDCDataset/adagate_hgate`)

Generated 2026-09-24 from `/home/stanfc/TTA-on-OVSS/MLMP/save/ACDCDataset/adagate_hgate_backbone_<name>/results_all_rounds.txt` (CAGA, locked config, only `--ovss_type`/`--ovss_backbone` changed; 150 rounds, seed 0, single run) and `/home/stanfc/TTA-on-OVSS/MLMP/save/ACDCDataset/mlmp_episodic_backbone_ablation/<name>/results.txt` (MLMP-episodic on the same backbone, lr 1e-3, steps 1). Rows sorted by CAGA R150. All numbers are mean mIoU over fog/night/rain/snow. Default backbone = `naclip_L14` (main paper).

Columns: **R1** = round-1 mean of the CAGA run; **R150** / **Peak (round)** / **Mean** = end / best / average over 150 rounds; **Epi.** = MLMP-episodic mean; **Δ** = CAGA R150 - Epi.

| Backbone | ovss_type | arch | UAML layers | R1 | R150 | Peak (round) | Mean | Epi. | Δ (R150-Epi.) |
|---|---|---|---|---|---|---|---|---|---|
| itaclip_L14 | itaclip | ViT-L/14 | 18 | 31.80 | **32.85** | 33.34 (118) | 32.96 | 31.88 | +0.97 |
| sclip_reduced_L14 | sclip_reduced | ViT-L/14 | 18 | 31.54 | **32.66** | 33.10 (144) | 32.73 | 31.64 | +1.02 |
| naclip_B16 | naclip | ViT-B/16 | 18 | 29.02 | **31.25** | 31.54 (32) | 31.22 | 29.38 | +1.87 |
| clearclip_kk_B16 | clearclip_kk | ViT-B/16 | 18 | 29.14 | **31.23** | 31.69 (71) | 31.30 | 29.50 | +1.73 |
| sclip_reduced_B16 | sclip_reduced | ViT-B/16 | 18 | 30.21 | **31.22** | 31.54 (96) | 31.31 | 30.61 | +0.61 |
| clearclip_qq_L14 | clearclip_qq | ViT-L/14 | 18 | 30.17 | **31.21** | 31.37 (105) | 31.20 | 30.21 | +1.00 |
| clearclip_kk_L14 | clearclip_kk | ViT-L/14 | 18 | 29.55 | **30.95** | 31.36 (90) | 30.89 | 29.56 | +1.39 |
| naclip_L14 (default) | naclip | ViT-L/14 | 18 | 29.84 | **30.95** | 31.55 (43) | 30.98 | 29.84 | +1.11 |
| vv_attention_L14 | vv_attention | ViT-L/14 | 18 | 29.18 | **30.58** | 31.02 (31) | 30.60 | 29.23 | +1.35 |
| clearclip_qq_B16 | clearclip_qq | ViT-B/16 | 18 | 28.55 | **30.06** | 30.32 (96) | 30.13 | 28.94 | +1.12 |
| naclip_B32 | naclip | ViT-B/32 | 18 | 26.35 | **27.85** | 28.19 (73) | 27.81 | 26.61 | +1.24 |
| maskclip_L14 | maskclip | ViT-L/14 | 18 | 23.89 | **25.32** | 25.54 (32) | 25.20 | 23.95 | +1.37 |
| maskclip_B16 | maskclip | ViT-B/16 | 18 | 21.98 | **25.06** | 25.59 (138) | 24.96 | 22.19 | +2.87 |
| naclip_L14_336 | naclip | ViT-L/14@336px | 18 | 23.04 | **25.04** | 25.80 (57) | 25.27 | 22.94 | +2.10 |

## Per-condition R150 (CAGA)

| Backbone | fog | night | rain | snow |
|---|---|---|---|---|
| itaclip_L14 | 35.31 | 27.56 | 34.07 | 34.46 |
| sclip_reduced_L14 | 35.67 | 26.72 | 34.40 | 33.84 |
| naclip_B16 | 33.32 | 24.09 | 32.48 | 35.13 |
| clearclip_kk_B16 | 33.48 | 23.60 | 32.83 | 35.03 |
| sclip_reduced_B16 | 33.92 | 23.77 | 32.26 | 34.94 |
| clearclip_qq_L14 | 34.84 | 24.91 | 33.87 | 31.21 |
| clearclip_kk_L14 | 34.31 | 24.90 | 33.29 | 31.31 |
| naclip_L14 (default) | 34.28 | 25.18 | 32.91 | 31.44 |
| vv_attention_L14 | 33.06 | 26.06 | 30.98 | 32.22 |
| clearclip_qq_B16 | 32.29 | 21.66 | 32.21 | 34.10 |
| naclip_B32 | 29.08 | 20.58 | 30.81 | 30.92 |
| maskclip_L14 | 27.72 | 20.50 | 25.67 | 27.41 |
| maskclip_B16 | 26.91 | 19.62 | 25.82 | 27.90 |
| naclip_L14_336 | 28.25 | 18.99 | 26.78 | 26.13 |

All 15 CAGA runs completed 150 rounds.

## Notes / caveats

- **Source (No-Adapt) floor per backbone is NOT available** for these 15 rows (only NA-CLIP L14 `/home/stanfc/TTA-on-OVSS/MLMP/save/ACDCDataset/naclip_no_adapt` = 23.34 and ITA-CLIP L14 `/home/stanfc/TTA-on-OVSS/MLMP/save/ACDCDataset/itaclip_no_adapt`). The table therefore shows CAGA's absolute level and its gap to MLMP-episodic, not its gain over the source model. A no-gate (plain continual) column is also missing.
- Δ vs MLMP-episodic is the only per-backbone reference available; MLMP-episodic uses steps=1 (official MLMP uses 10).
- Single seed; differences below about 0.3 mIoU are not reliable.
- `naclip_L14_336` uses ViT-L/14@336px; whether its low score comes from the window/stride setting (still 224 window, stride 112) was not checked.
- Figure: `/home/stanfc/TTA-on-OVSS/MLMP/.figs_additional_experiments/fig9_backbone_ablation.png` (script `plot_new_results.py`).
