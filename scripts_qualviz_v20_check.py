"""Quick check: does the naclip VOC20 no-gate ckpt (150R complete, mean
73.25->28.98) show the same single-class over-confidence collapse as the
ACDC/MaFT one? Reuses the same load-checkpoint-and-infer pattern."""
import numpy as np
import torch
from PIL import Image

from adapt.deyo_mlmp_adagate_continual import DeYOMLMPAdaGateContinual
from utils import segmentation_datasets
from utils.misc import aggregate_pred_patches, custom_collate
from qualitative import apply_palette

device = 'cuda'
CORRUPTION = 'frost'  # lowest final-round mIoU among the 5

loader, org_classes = segmentation_datasets.prepare_data(
    'PascalVOC20Dataset', '.data/VOC2012/', [224, 224], [224, 224], 112,
    corruption=CORRUPTION, batch_size=1, num_workers=0, shuffle=False,
    corruption_severity=5, split='val', ann_file='ImageSets/Segmentation/trainval.txt')

classes = loader.dataset.class_extensions or org_classes
palette = loader.dataset.METAINFO['palette']

m = DeYOMLMPAdaGateContinual(
    ovss_type='naclip', ovss_backbone='ViT-L/14', lr=5e-6, classes=classes,
    steps=1, vision_outputs=list(range(-1, -19, -1)),
    deyo_margin_factor=0.5, deyo_margin_e0_factor=0.4, plpd_threshold=0.2,
    aug_type='patch', patch_len=4, reweight_ent=True, reweight_plpd=True,
    top_block_exclude=6, slope_window=10, slope_deadzone=0.002, lag_gain=1500.0,
    base_rst=0.0, max_windows=2000, h_drop_ratio=0.9, maxlag_shallow=6,
    monitor_interval=5, trend_stat='mad', trend_thr=0.5, trend_hist=50,
    lag_mode='ecdf', lag_sat=1.5, shallow_cap_mode='fixed', device=device)

ckpt = torch.load(
    'save/PascalVOC20Dataset/v20_batch_ablation/adagate_nogate_batch8_fulldata/ckpt.pt',
    map_location='cpu', weights_only=False)
print('round_idx', ckpt['round_idx'], 'cond_idx', ckpt['cond_idx'])
# This checkpoint predates a code change that now always trains all 100
# LN affine tensors (comment at _is_excluded(): "top_block_exclude/ln_post
# exclusion removed"). At save time, top_block_exclude=6 meant only ln_pre +
# blocks 0-17 (74 tensors) were ever trained; blocks 18-23 + ln_post stayed
# frozen at source values for the entire real 150-round run. Loading only
# the keys present therefore reconstructs that run faithfully -- the
# untouched tensors are correctly left at fresh source init.
saved_ln = ckpt['adapt_state']['ln_params']
missing = [n for n, _ in m.named_ln_params if n not in saved_ln]
print(f"{len(saved_ln)}/{len(m.named_ln_params)} LN tensors in checkpoint; "
      f"{len(missing)} left at source init (correct: those were excluded "
      f"from training in this historical run)")
with torch.no_grad():
    for n, p in m.named_ln_params:
        if n in saved_ln:
            p.data.copy_(saved_ln[n].to(device=p.device, dtype=p.dtype))
m.model.to(device)

for img_idx in [0, 1, 2]:
    sample = loader.dataset[img_idx]
    batch = custom_collate([sample])
    inputs = batch['img_patches'].to(device)
    with torch.no_grad():
        patch_preds = m.evaluate(inputs)
    reconstructed = aggregate_pred_patches(
        patch_preds, batch['meta']['patch_grid_shape'], batch['meta']['img_shape'], [224, 224], 112)
    pd = reconstructed[0].softmax(dim=0)
    if loader.dataset.class_extensions is not None:
        ext_to_real = torch.tensor(loader.dataset.extentions_to_real_class_idx, dtype=torch.int64, device=device)
        num_cls = int(ext_to_real.max()) + 1
        one_hot = torch.nn.functional.one_hot(ext_to_real).T.view(num_cls, len(ext_to_real), 1, 1)
        pd = (pd.unsqueeze(0) * one_hot).max(1)[0]
    pd_label = pd.argmax(dim=0).cpu().numpy()
    vals, counts = np.unique(pd_label, return_counts=True)
    frac = counts / counts.sum()
    order = np.argsort(-frac)
    print(f"--- img_idx={img_idx} ({CORRUPTION}) ---")
    for i in order[:5]:
        cls_name = classes[vals[i]] if vals[i] < len(classes) else f"idx{vals[i]}"
        print(f"  {cls_name}: {frac[i]*100:.1f}%")
    print(f"  mean top-1 conf: {pd.max(0)[0].mean().item()*100:.1f}%")

    CLIP_MEAN = torch.tensor([122.7709, 116.7460, 104.0937]).view(3, 1, 1)
    CLIP_STD = torch.tensor([68.5005, 66.6322, 70.3232]).view(3, 1, 1)
    img_tensor = batch['img'][0].cpu().float()
    raw_img = (img_tensor * CLIP_STD + CLIP_MEAN).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    pred_colour = apply_palette(pd_label, palette)
    gt_colour = apply_palette(batch['gt'][0][0].numpy(), palette)
    Image.fromarray(raw_img).save(f'.qualviz_v20_check/img{img_idx}_image.png')
    Image.fromarray(pred_colour).save(f'.qualviz_v20_check/img{img_idx}_pred.png')
    Image.fromarray(gt_colour).save(f'.qualviz_v20_check/img{img_idx}_gt.png')
