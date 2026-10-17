"""Render one fixed ACDC sample (same image as .qualviz_tent_collapse_snow/image.png) with every method at R0 (source) and R150
(from each run's final ckpt.pt), into one folder. Usage (from MLMP/):  CUDA_VISIBLE_DEVICES=0 python scripts_qualviz_render_all.py"""
import os, sys, json, shlex, gc
import numpy as np, torch
from PIL import Image
import main_continual as mc
from adapt import get_method
from utils import segmentation_datasets
from utils.misc import aggregate_pred_patches, custom_collate
from qualitative import apply_palette, save_legend

OUT = '.qualviz_acdc_snow_all_methods/'
REF = '.qualviz_tent_collapse_snow/image.png'
COND = 'snow'
RUNS = [('TENT', 'save/ACDCDataset/qual_tent'), ('SAR', 'save/ACDCDataset/qual_sar'), ('SPA', 'save/ACDCDataset/qual_spa'),
        ('TRIBE', 'save/ACDCDataset/qual_tribe'), ('DeYO', 'save/ACDCDataset/qual_deyo'), ('ViDA', 'save/ACDCDataset/qual_vida'),
        ('PTTA', 'save/ACDCDataset/qual_ptta'), ('COME', 'save/ACDCDataset/qual_come'), ('MGP', 'save/ACDCDataset/qual_mgp'),
        ('DAF', 'save/ACDCDataset/qual_daf'), ('CAGA', 'save/ACDCDataset/adagate_hgate')]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs(OUT, exist_ok=True)

loader, org_classes = segmentation_datasets.prepare_data('ACDCDataset', '.data/ACDC/', [1120, 560], [224, 224], 112, corruption=COND,
        batch_size=1, num_workers=0, shuffle=False, corruption_severity=5, split='val')
classes = loader.dataset.class_extensions or org_classes
palette = loader.dataset.METAINFO['palette']
CM = torch.tensor([122.7709, 116.7460, 104.0937]).view(3, 1, 1); CS = torch.tensor([68.5005, 66.6322, 70.3232]).view(3, 1, 1)
def raw_of(sample):
    return (sample['img'].float() * CS + CM).clamp(0, 255).byte().permute(1, 2, 0).numpy()
ref = np.array(Image.open(REF).convert('RGB')).astype(np.float32)
idx = None
for i in range(len(loader.dataset)):
    a = raw_of(loader.dataset[i]).astype(np.float32)
    if a.shape == ref.shape and np.abs(a - ref).mean() < 1.0: idx = i; break
assert idx is not None, "reference image not found in ACDC snow val"
print('matched img_idx =', idx)
sample = loader.dataset[idx]; batch = custom_collate([sample])
inputs = batch['img_patches'].to(device); grid = batch['meta']['patch_grid_shape']; shp = batch['meta']['img_shape']
gt = batch['gt'][0][0].numpy()
Image.fromarray(raw_of(sample)).save(OUT + 'image.png'); Image.fromarray(apply_palette(gt, palette)).save(OUT + 'gt_mask.png')
save_legend(list(org_classes), palette, OUT + 'legend.png'); np.save(OUT + 'gt.npy', gt)

def make(run_dir):
    toks = shlex.split(open(os.path.join(run_dir, 'cmd.sh')).read().strip())
    toks = toks[toks.index('main_segmentation.py') + 1:]
    fixed = []
    skip = 0
    for i, t in enumerate(toks):
        if skip: skip -= 1; continue
        if t == '--r': skip = 1; continue                                                     # ambiguous abbreviation now; default is 4 anyway
        if t == '--prompt_dir' and (i + 1 >= len(toks) or toks[i + 1].startswith('--')): continue   # empty flag in cmd.sh
        fixed.append(t)
    sys.argv = ['x'] + fixed
    p0 = mc.argparser(); a0, _ = p0.parse_known_args()
    p = mc.argparser(); p = mc.add_method_specific_args(p, a0.method); args = p.parse_args()
    args.classes = classes
    return get_method(args, device), args

@torch.no_grad()
def predict(m):
    pp = m.evaluate(inputs)
    rec = aggregate_pred_patches(pp, grid, shp, [224, 224], 112)
    pd = rec[0].softmax(dim=0)
    if loader.dataset.class_extensions is not None:
        e2r = torch.tensor(loader.dataset.extentions_to_real_class_idx, dtype=torch.int64, device=device)
        n = int(e2r.max()) + 1
        oh = torch.nn.functional.one_hot(e2r).T.view(n, len(e2r), 1, 1)
        pd = (pd.unsqueeze(0) * oh).max(1)[0]
    conf = float(pd.max(0)[0].mean()); lab = pd.argmax(0).cpu().numpy()
    valid = gt != 255
    return lab, conf, float((lab[valid] == gt[valid]).mean())

stats = {}
def save(name, lab, conf, acc):
    np.save(OUT + f'pd_{name}.npy', lab); Image.fromarray(apply_palette(lab, palette)).save(OUT + f'pred_{name}.png')
    u, c = np.unique(lab, return_counts=True)
    stats[name] = dict(n_active=int(len(u)), top1=float(c.max() / c.sum()), conf=conf, pixel_acc=acc)
    print(f"{name:10s} classes={len(u):2d} top1={c.max()/c.sum()*100:4.0f}% conf={conf*100:3.0f}% pixacc={acc*100:4.1f}%")

# R0 = source model (no adaptation): TENT class, nothing loaded
m, args = make(RUNS[0][1]); save('source_R0', *predict(m)); del m; gc.collect(); torch.cuda.empty_cache()
for name, d in RUNS:
    m, args = make(d)
    ck = torch.load(os.path.join(d, 'ckpt.pt'), map_location='cpu', weights_only=False)
    assert ck['round_idx'] == 149 and ck['cond_idx'] == 3, (name, ck['round_idx'], ck['cond_idx'])
    m.load_state_dict(ck['adapt_state']); 
    if hasattr(m, 'model'): m.model.to(device)
    save(f'{name}_R150', *predict(m)); del m, ck; gc.collect(); torch.cuda.empty_cache()
json.dump(dict(img_idx=idx, condition=COND, stats=stats), open(OUT + 'stats.json', 'w'), indent=1)
print('saved to', OUT)
