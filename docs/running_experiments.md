# Running Experiments Tracker

Full-benchmark runs for the paper table. **Claude updates the Status column when
asked for progress.** Last updated: 2026-06-30.

## Settings (all rows unless noted)
- **Full val set** (no subset), **50 continual rounds**, evaluate-before-adapt, no-reset.
- NA-CLIP ViT-L/14, LayerNorm-only, seed 0.
- Adapt methods: LR 5e-6, steps 1. GDG-PA = `deyo_mlmp_hmgate2_continual` with
  h_drop_ratio 0.9, slope_window 10, deadzone 0.002, lag_gain 1500, base_rst 0.01,
  maxlag_shallow 6, monitor_interval 50.
- no-gate = `deyo_mlmp_continual` (--adapt, no gate). no-adapt = `tent_continual` (no --adapt, 1 round).

## The 4 methods per (dataset × corruption-set)
- **no-adapt** and **MLMP-episodic** are corruption-set-independent (frozen / per-sample reset)
  → ONE full run per dataset covers both 5corr and 15corr (average the relevant rows).
- **no-gate** and **GDG-PA** accumulate over the specific corruption sequence → one run per config.

## MLMP-episodic (REUSE existing — no new run, lr 0.001 / steps 10 / full val)
| Dataset | dir | status |
|---|---|---|
| VOC20 | `.save/PascalVOC20Dataset/mlmp` | done (ALL15) |
| VOC21 | `.save/PascalVOC21Dataset/mlmp` | done (ALL15) |
| PContext59 | `.save/PascalContext59Dataset/mlmp` | done (ALL15) |
| PContext60 | `.save/PascalContext60Dataset/mlmp` | done (ALL15) |
| COCO-Object | `.save/COCOObjectDataset/mlmp` | done (ALL15) |
| COCO-Stuff | `.save/COCOStuffDataset/mlmp` + `mlmp_gaussian_noise` patch | **patch QUEUED** (orig missing gaussian_noise) |
| Cityscapes | `.save/CityscapesDataset/mlmp_lr1e-3` | done (ALL15) |

## no-adapt (full, 15corr → covers 5/15; per dataset)
| Dataset | save_dir | GPU | status |
|---|---|---|---|
| VOC20 | `.save/PascalVOC20Dataset/No_Adaptation` | — | done (exists) |
| Cityscapes | `.save/CityscapesDataset/No_Adaptation` | — | done (exists) |
| VOC21 | `save/PascalVOC21Dataset/No_Adaptation_full` | 0 | queued |
| PContext59 | `save/PascalContext59Dataset/No_Adaptation_full` | 0 | queued |
| PContext60 | `save/PascalContext60Dataset/No_Adaptation_full` | 1 | queued |
| COCO-Object | `save/COCOObjectDataset/No_Adaptation_full` | 2 | queued |
| COCO-Stuff | `save/COCOStuffDataset/No_Adaptation_full` | 3 | queued |

## no-gate + GDG-PA (full, per config, 50R)
| Dataset | corr | method | GPU | save_dir | status |
|---|---|---|---|---|---|
| VOC20 | 5 | no-gate | 2 | `deyo_mlmp_continual_full_5corr` | queued |
| VOC20 | 5 | GDG-PA | 3 | `deyo_mlmp_hmgate2_continual_full_5corr` | queued |
| Cityscapes | 5 | no-gate | 2 | `deyo_mlmp_continual_full_5corr` | queued |
| Cityscapes | 5 | GDG-PA | 3 | `deyo_mlmp_hmgate2_continual_full_5corr` | queued |
| VOC21 | 15 | no-gate | 0 | `deyo_mlmp_continual_full_15corr` | queued |
| VOC21 | 15 | GDG-PA | 1 | `deyo_mlmp_hmgate2_continual_full_15corr` | queued |
| VOC21 | 5 | no-gate | 2 | `deyo_mlmp_continual_full_5corr` | queued |
| VOC21 | 5 | GDG-PA | 3 | `deyo_mlmp_hmgate2_continual_full_5corr` | queued |
| PContext59 | 15 | no-gate | 0 | `deyo_mlmp_continual_full_15corr` | queued |
| PContext59 | 15 | GDG-PA | 1 | `deyo_mlmp_hmgate2_continual_full_15corr` | queued |
| PContext59 | 5 | no-gate | 0 | `deyo_mlmp_continual_full_5corr` | queued |
| PContext59 | 5 | GDG-PA | 1 | `deyo_mlmp_hmgate2_continual_full_5corr` | queued |
| PContext60 | 15 | no-gate | 2 | `deyo_mlmp_continual_full_15corr` | queued |
| PContext60 | 15 | GDG-PA | 3 | `deyo_mlmp_hmgate2_continual_full_15corr` | queued |
| PContext60 | 5 | no-gate | 2 | `deyo_mlmp_continual_full_5corr` | queued |
| PContext60 | 5 | GDG-PA | 3 | `deyo_mlmp_hmgate2_continual_full_5corr` | queued |
| COCO-Object | 15 | no-gate | 0 | `deyo_mlmp_continual_full_15corr` | queued |
| COCO-Object | 15 | GDG-PA | 1 | `deyo_mlmp_hmgate2_continual_full_15corr` | queued |
| COCO-Object | 5 | no-gate | 0 | `deyo_mlmp_continual_full_5corr` | queued |
| COCO-Object | 5 | GDG-PA | 1 | `deyo_mlmp_hmgate2_continual_full_5corr` | queued |
| COCO-Stuff | 15 | no-gate | 2 | `deyo_mlmp_continual_full_15corr` | queued |
| COCO-Stuff | 15 | GDG-PA | 3 | `deyo_mlmp_hmgate2_continual_full_15corr` | queued |
| COCO-Stuff | 5 | no-gate | 2 | `deyo_mlmp_continual_full_5corr` | queued |
| COCO-Stuff | 5 | GDG-PA | 3 | `deyo_mlmp_hmgate2_continual_full_5corr` | queued |

## Already-complete full+15corr (from earlier, reuse)
| Dataset | method | save_dir | status |
|---|---|---|---|
| VOC20 | GDG-PA | `deyo_mlmp_hmgate2_continual_full_15corr` | done R150 |
| VOC20 | no-gate | `deyo_mlmp_continual_full_15corr` | done R50 |
| VOC20 | GDG 0.9/0.95 | `deyo_mlmp_hmgate_continual_full_15corr[_hdr095]` | done |
| Cityscapes | GDG-PA | `deyo_mlmp_hmgate2_continual_full_15corr` | running (~R27) |
| Cityscapes | no-gate | `deyo_mlmp_continual_full_15corr` | done R50 |
| Cityscapes | GDG 0.95 | `deyo_mlmp_hmgate_continual_full_15corr_hdr095` | done R50 |

## Note
COCO/PContext full val are large (10k / 5k imgs) → full×15corr×50R is multi-day per run.

---

## Launch commands (paste one block per GPU — runs sequentially, fast→slow)

**GPU 0**
```bash
for s in v21/no_adapt_full coco_stuff/mlmp_gaussian_noise p59/no_adapt_full v21/deyo_mlmp_continual_full_15corr p59/deyo_mlmp_continual_full_5corr coco_obj/deyo_mlmp_continual_full_5corr p59/deyo_mlmp_continual_full_15corr coco_obj/deyo_mlmp_continual_full_15corr; do bash "bash/$s.sh"; done
```

**GPU 1**
```bash
for s in p60/no_adapt_full v21/deyo_mlmp_hmgate2_continual_full_15corr p59/deyo_mlmp_hmgate2_continual_full_5corr coco_obj/deyo_mlmp_hmgate2_continual_full_5corr p59/deyo_mlmp_hmgate2_continual_full_15corr coco_obj/deyo_mlmp_hmgate2_continual_full_15corr; do bash "bash/$s.sh"; done
```

**GPU 2** (launch after PA Cityscapes-full on GPU2 finishes)
```bash
for s in cityscapes/deyo_mlmp_continual_full_5corr v20/deyo_mlmp_continual_full_5corr v21/deyo_mlmp_continual_full_5corr coco_obj/no_adapt_full p60/deyo_mlmp_continual_full_5corr coco_stuff/deyo_mlmp_continual_full_5corr p60/deyo_mlmp_continual_full_15corr coco_stuff/deyo_mlmp_continual_full_15corr; do bash "bash/$s.sh"; done
```

**GPU 3**
```bash
for s in cityscapes/deyo_mlmp_hmgate2_continual_full_5corr v20/deyo_mlmp_hmgate2_continual_full_5corr v21/deyo_mlmp_hmgate2_continual_full_5corr coco_stuff/no_adapt_full p60/deyo_mlmp_hmgate2_continual_full_5corr coco_stuff/deyo_mlmp_hmgate2_continual_full_5corr p60/deyo_mlmp_hmgate2_continual_full_15corr coco_stuff/deyo_mlmp_hmgate2_continual_full_15corr; do bash "bash/$s.sh"; done
```
