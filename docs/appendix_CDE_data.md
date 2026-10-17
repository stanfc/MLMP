# Appendix C / D / E: data, figures, open issues

Generated 2026-09-24. Everything is single seed (0), 150 rounds, NA-CLIP ViT-L/14, LayerNorm-only, batch 1 unless stated. CAGA = `deyo_mlmp_adagate_hgate_continual` (locked config).

---

## C. Experimental and baseline configurations

### C.1 Shared protocol
| Item | Value |
|---|---|
| Backbone | NA-CLIP ViT-L/14 (ovss_type naclip), text encoder frozen, only visual LayerNorm gamma/beta updated |
| Protocol | evaluate-before-adapt, no reset, model built once, 150 rounds, seed 0 |
| Input | sliding window 224x224, stride 112; init_resize 1120x560 (ACDC, Cityscapes), 224x224 (VOC) |
| ACDC | full val split, 406 imgs/round, order fog, night, rain, snow (real shift, no synthetic corruption) |
| Cityscapes-C, V20-C, V21-C, P59-C, P60-C, COCO-Stuff-C, COCO-Object-C | 100-image subset (subset_seed 0) x 15 ImageNet-C corruptions, severity 5, 1500 imgs/round |
| Analysis runs (5corr) | 100-image subset x {snow, frost, fog, brightness, contrast} |
| Prompts | 7 templates (`prompts.yaml`); TENT/SAR/... single-prompt convention where the port follows the official method |

### C.2 Per-method hyperparameters (from each run's `cmd.sh`; identical across all 8 datasets unless noted)
| Method | Optimizer / lr | Method-specific settings | Per-dataset differences |
|---|---|---|---|
| **CAGA** | Adam, lr 5e-6, steps 1 | trend_stat mad, trend_thr 0.5, lag_mode ecdf, shallow_cap_mode growing_scaled, base_rst 0.01, anchor_h_min_ratio 0.95, h_drop_ratio 0.9, h_max_mode lifetime, monitor_interval 50, slope_window 10, trend_hist 50, max_windows 2000, DeYO filter (margin 0.5, e0 0.4, plpd 0.2, patch aug len 4, reweight ent+plpd), top_block_exclude 6, vision_outputs -1..-18 (18-layer UAML) | none |
| TENT | Adam, lr 1e-5, steps 1 | none | none |
| SAR | Adam wrapped in SAM, lr 1e-5 | sam_rho 0.05; other datasets: e_margin 1.8, e_0 0.1, ema 0.9, recovery_warmup 50 | **ACDC uses an older arg set**: margin_e0_factor 0.4, reset_constant_em 0.2, top_block_exclude 6 (no e_margin/e_0) |
| SPA | Adam, lr 1e-5 | eraser_patch_size 14, freq_mask_ratio 0.2, noise_ratio 0.4 | none |
| TRIBE | Adam, lr 1e-5 | eta 0.005, gamma 0.0, h0 0.05, lam 0.5 | none |
| DeYO | Adam, lr 1e-5 | margin 0.5, e0 0.4, plpd 0.2, patch aug len 4, reweight ent+plpd, top_block_exclude 6 | none |
| ViDA | Adam, lr 1e-5 | ema 0.99, ema_vida 0.8, r 4, r2 16, unc_n 10, unc_thr 0.2 | **ACDC lr 2e-7** |
| PTTA | Adam, lr 1e-5 | queue_size 1000, neighbor 1, e_margin 1.198, d_margin 0.05, loss2_weight 3.0 | none |
| COME | Adam, lr 1e-5 | none | none |
| MGP | Adam, lr 5e-6 | DeYO filter as above + buffer 32, collect_freq 40, distill_freq 100, max_rank 32, residual_thr 0.75, vision_outputs -1..-18 | none |
| DAF | **SGD**, prompt_integration loss | alpha_cls 1.0, alpha_safs 0.5, lamb_ent 1.0, lamb_div 2.0, lamb_cmac 0.5, vision_outputs -1..-18 | **ACDC / Cityscapes / V20: batch 1, lr 1.25e-4; V21 / P59 / P60 / COCO-Stuff / COCO-Object: batch 8, lr 1e-3** |
| MLMP-episodic | Adam, lr 1e-3, **steps 1** | reset every sample, trials 1, alpha_cls 1.0, 18-layer UAML (official MLMP uses steps 10) | none |
| No-Adapt | no update | run through tent_continual without `--adapt`; lr value in cmd.sh (1e-4 ACDC, 5e-6 others) is unused | none |

Notes for the config table caption: CoTTA was dropped (stopped by decision). Baseline hyperparameters were not tuned per dataset and are not uniform (see problems 3-5 below). SegFormer (closed-set) uses a separate repo (`/home/stanfc/TTA-on-OVSS/DeYO`, ports of the same methods).

Raw parsed configs (104 dataset x method): `/tmp/appx/cfg.json` (temporary; regenerate with `/tmp/appx/extract.py`). Method docs: `/home/stanfc/TTA-on-OVSS/MLMP/docs/CAGA_method_and_hyperparameters.md`.

---

## D. Additional long-horizon results (5 benchmarks)

Figure (five panels + legend cell, all methods, CAGA thick solid, No-Adapt dashed, MLMP-episodic dotted):
- `/home/stanfc/TTA-on-OVSS/MLMP/.figs_additional_experiments/appendixD_additional_benchmarks_5panel.png` (+ `.pdf`)
- script: `/home/stanfc/TTA-on-OVSS/MLMP/.figs_additional_experiments/plot_appendixD_5panel.py`
- CoTTA excluded; the older per-dataset figures `/home/stanfc/TTA-on-OVSS/MLMP/.fig_result/<dataset>_all_methods.{png,svg}` are also available.

Raw table: R150 / Peak (round of peak). Full CSV incl. ACDC/Cityscapes/V20: `/home/stanfc/TTA-on-OVSS/MLMP/.fig_result/results_table.csv`, markdown: `/home/stanfc/TTA-on-OVSS/MLMP/.fig_result/results_table.md`.

| Method | V21 | P59 | P60 | COCO-Stuff | COCO-Object |
|---|---|---|---|---|---|
| No-Adapt | 42.17 | 24.14 | 22.44 | 12.50 | 18.71 |
| MLMP-episodic | 45.44 | 24.78 | 23.02 | 12.70 | 21.00 |
| **CAGA** | 50.01 / 50.04 (77) | 26.92 / 26.97 (147) | 24.57 / 24.64 (110) | 13.30 / 13.54 (23) | 21.73 / 21.75 (120) |
| DAF | 33.54 / 46.25 (7) | 16.64 / 25.80 (9) | 15.16 / 23.70 (11) | 7.89 / 13.11 (3) | 10.18 / 20.40 (1) |
| TENT | 0.10 / 41.30 (5) | 0.01 / 23.41 (4) | 0.01 / 21.93 (5) | 0.04 / 12.96 (7) | 1.35 / 22.45 (18) |
| SAR | 40.79 / 42.22 (99) | 25.11 / 26.07 (12) | 22.44 / 24.18 (77) | 12.61 / 12.87 (88) | 20.02 / 23.05 (131) |
| SPA | 41.57 / 41.58 (148) | 25.45 / 25.45 (150) | 23.04 / 23.04 (150) | 13.24 / 13.26 (148) | 19.86 / 19.86 (150) |
| TRIBE | 40.29 / 40.29 (150) | 22.70 / 22.70 (149) | 20.62 / 20.62 (149) | 11.33 / 11.33 (150) | 17.95 / 17.95 (150) |
| DeYO | **53.08** / 53.09 (139) | **29.11** / 29.12 (149) | **26.07** / 26.07 (150) | 10.22 / 13.03 (14) | 22.13 / 22.15 (148) |
| ViDA | 0.34 / 39.67 (1) | 0.05 / 12.68 (1) | 0.01 / 12.15 (1) | 0.00 / 7.18 (1) | 0.07 / 11.13 (1) |
| PTTA | 19.98 / 40.49 (1) | 2.98 / 22.10 (1) | 1.75 / 20.61 (1) | 9.20 / 11.81 (1) | 15.55 / 17.55 (1) |
| COME | 37.60 / 42.25 (5) | 0.01 / 24.01 (6) | 0.01 / 22.48 (7) | 0.00 / 12.83 (6) | 0.02 / 20.15 (9) |
| MGP | 52.91 / 52.91 (150) | 22.37 / 27.44 (46) | 20.93 / 24.93 (42) | 0.36 / 13.48 (26) | 20.03 / 21.67 (61) |

**Honest reading of D:** CAGA is the most consistent (never collapses, above No-Adapt on all five) but is **not** the best R150 everywhere: DeYO beats CAGA on V21 (+3.1), P59 (+2.2), P60 (+1.5) and COCO-Object (+0.4), and MGP beats it on V21 (+2.9). CAGA wins where DeYO/MGP eventually collapse (COCO-Stuff: DeYO 10.22, MGP 0.36; also MGP on P59/P60). The safe claim is "CAGA never collapses and is competitive/best among stable methods", not "CAGA is best on every benchmark". DeYO/MGP on V21/P59/P60 are still rising or flat at R150; whether they collapse beyond 150 rounds is untested.

---

## E. Additional robustness and generalization analyses

All figures in `/home/stanfc/TTA-on-OVSS/MLMP/.figs_additional_experiments/` (each has `.png` + `.pdf`, script `plot_*.py` next to it).

| # | Analysis | Figure (absolute path, .png) | Key numbers (R150 mIoU unless noted) | Status |
|---|---|---|---|---|
| E1 | Shuffled domain order | `/home/stanfc/TTA-on-OVSS/MLMP/.figs_additional_experiments/fig7a_domain_order_delta.png` | ACDC 30.95 fixed -> 31.07 shuffled; V20-5corr 75.86 -> 75.78; Cityscapes-5corr 23.92 -> 23.80 | drawn (as delta bars) |
| E2 | Abrupt domain switch | `/home/stanfc/TTA-on-OVSS/MLMP/.figs_additional_experiments/fig7b_abrupt_domain_shift.png` | V20-5corr phase-switch: R150 68.44, peak 76.18, mean 72.03 (fixed-order run: 75.86 / 76.18 / 75.72) | drawn |
| E3 | Resample robustness | `/home/stanfc/TTA-on-OVSS/MLMP/.figs_additional_experiments/appendix_resample_robustness.png` | V20-5corr resample: R150 71.77, peak 77.29, mean 74.09 | drawn |
| E4 | OOD generalization (held-out corruptions probe) | `/home/stanfc/TTA-on-OVSS/MLMP/.figs_additional_experiments/appendix_ood_generalization.png` | curve file `/home/stanfc/TTA-on-OVSS/MLMP/save/PascalVOC20Dataset/adagate_hgate_5corr_sub100_oodprobe/results_ood.txt` | drawn |
| E5 | Backbones (15 ACDC runs) | `/home/stanfc/TTA-on-OVSS/MLMP/.figs_additional_experiments/fig9_backbone_ablation.png` | naclip_L14 (default) 30.95; ITA-CLIP L14 32.85; SCLIP-reduced L14 32.66; naclip_B16 31.25; naclip_B32 27.85; naclip_L14_336 25.04; MaskCLIP L14 25.32 / B16 25.06 | drawn; **no No-Adapt/floor column** (see problems) |
| E6 | Closed-set (SegFormer, ACDC) | `/home/stanfc/TTA-on-OVSS/MLMP/.figs_additional_experiments/fig10_segformer_all_methods.png` | No-Adapt 50.62; all stable methods within +-0.15 of it | drawn |
| E7 | Monitoring-signal ablation (all 4 runs finished, fig8 regenerated) | `/home/stanfc/TTA-on-OVSS/MLMP/.figs_additional_experiments/fig8_monitoring_signal_ablation.png` | see table below | redrawn with final curves |
| E8 | Stochastic restoration / hyperparameter sensitivity | `/home/stanfc/TTA-on-OVSS/MLMP/.figs_additional_experiments/fig6a_stochastic_restoration.png`, `/home/stanfc/TTA-on-OVSS/MLMP/.figs_additional_experiments/fig6b_hparam_sensitivity_bars.png` | base_rst {0, 0.005, 0.01, 0.02}, trend_thr {0.25, 0.5, 1.0}, anchor ratio, monitor interval, lr | drawn |

Compact panel choice for 4-6 slots: E1+E2 (domain order/switch), E3 (resample), E4 (OOD), E5 (backbones), E6 (closed-set) = 6 panels; E7/E8 belong in the main-text ablation section.

### E7 numbers (ACDC, 150 rounds, CAGA variants)
| Variant | R150 | Peak (round) | Mean |
|---|---|---|---|
| CAGA (full) | 30.95 | 31.55 (43) | 30.98 |
| always-restore (no trigger) | 30.77 | 30.81 (122) | 30.49 |
| random-restore (p = 0.296) | 30.96 | 31.22 (38) | 30.99 |
| grad_norm-only | 30.65 | 32.15 (17) | 31.02 |
| h_margin-only | 30.61 | 30.71 (38) | 30.61 |

V20 sub100 15corr random-restore (p = 0.2927 = CAGA's measured firing rate there): R150 73.48, peak 73.78 (round 17), mean 73.47 vs CAGA 73.48 / 73.79 / 73.44. Trajectories differ (sum abs diff 7.2 mIoU over 150 rounds), so the equal R150 is a coincidence, not a duplicated run. Actual firing rate of the random run 0.282.

**Honest reading:** the ablation does **not** show that the grad_norm trigger is necessary. Random-restore at the same rate matches CAGA on ACDC (30.96 vs 30.95) and on V20 (73.47 vs 73.44 mean). Only always-restore and h_margin-only are clearly worse (-0.2 to -0.5 mean). grad_norm-only has the highest mean (31.02) but peaks at R17 then decays. Random-restore keeps the informed anchor (best_snapshot chosen by grad_norm + H health), so it isolates the trigger only. Any paper claim must be "restoring at a rate ~0.3 toward a well-chosen anchor is what matters; grad-norm timing is not shown to help" unless a random/source-anchor variant is added.

---

## Problems and questions for you

1. **Random-restore matches CAGA (ACDC and V20).** The trigger's contribution is not supported by our data. Do you want a random-*anchor* / source-anchor variant to test whether the anchor is the real contributor, or do we reframe the paper claim?
2. **CAGA is not best everywhere (D).** DeYO/MGP win on V21/P59/P60 at R150. Is the intended claim "no collapse, best among stable methods"? The appendix text must match that.
3. **Baseline configs are not uniform:** DAF uses batch 1/lr 1.25e-4 on ACDC/Cityscapes/V20 but batch 8/lr 1e-3 on the other five; ViDA ACDC lr 2e-7 vs 1e-5 elsewhere; SAR ACDC uses an older arg set (no e_margin/e_0). Need to state this in the config table or rerun for consistency (ask before any run).
4. **MLMP-episodic uses steps=1** (official 10) in our runs. Either footnote or rerun.
5. **SegFormer:** no measurable CAGA gain (all stable methods within +-0.15 of No-Adapt); ports of TRIBE/PTTA/ViDA/COME are reinterpretations. Should E6 stay as a "limitation/closed-set" panel?
6. **Backbone figure (E5)** lacks the No-Adapt floor and the no-gate column per backbone, so it shows CAGA's absolute values but not its gain. Need those runs (ask first) or drop the gain claim.
7. **Protocol mismatch:** E1-E4 analysis numbers use 5corr sub100 (V20 CAGA 75.86) while the main table uses 15corr sub100 (V20 73.48). The paper must say so; the only 15corr analysis run is the V20 random-restore one.
8. **Single seed everywhere;** differences under ~0.3 mIoU (E7, E1) are within plausible seed noise.
9. **CoTTA excluded** (stopped by decision); reviewers may ask.
10. **Cleanup:** superseded files still present: `ablation_study.{png,svg}`, `fig6_stochastic_restoration_and_sensitivity.*`, `fig7_domain_order_and_abrupt_shift.*`. Delete? Also `/tmp/appx/cfg.json` is temporary.
11. Run-script quirks: `main_segmentation.py` appears in the V20 random-restore `cmd.sh` (the run finished correctly with method `deyo_mlmp_adagate_hgate_randomrestore_continual`; noted only for traceability).
