# Gradient-Norm Penalty (DeYO+MLMP+DivGate) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit gradient-norm penalty `L' = L + λ‖∇_θ L‖²` (double-backward) to the DeYO+MLMP+DivGate adapt loss, and run a λ sweep on ACDC / VOC20 / Cityscapes, keeping the H_margin DivGate intact.

**Architecture:** New method `DeYOMLMPGradPenDivGateContinual` = byte-copy of `deyo_mlmp_divgate_continual.py` with only the loss/backward block changed. λ=0 branches to the exact original path (bit-identical control). New `gradpen_log.csv` records the raw `grad_norm` so we can verify the penalty engages. Three dataset runners + a λ-sweep launcher + an analysis script.

**Tech Stack:** PyTorch (double-backward via `torch.autograd.grad(..., create_graph=True)`), NA-CLIP ViT-L/14, existing `main_continual.py` CTTA harness. No pytest in this repo — correctness is verified by smoke runs and by comparing `results_all_rounds.txt` against the divgate baseline.

**Spec:** [docs/superpowers/specs/2026-06-21-gradnorm-penalty-design.md](../specs/2026-06-21-gradnorm-penalty-design.md)

---

## Conventions for verification in this repo

There is no unit-test harness. Each implementation task is verified by one of:
- **Static check:** `python -c "import ast; ast.parse(open('<file>').read())"` (syntax) or an import.
- **Factory check:** confirm `adapt.get_method` knows the method name.
- **Smoke run:** a tiny real run (`--subset_size 20 --continual_rounds 1`) on VOC20 (fastest: 224×224) confirming no crash + expected log files.
- **Bit-identical invariant:** λ=0 new method must match `deyo_mlmp_divgate_continual` on the same tiny config.

GPU: use `GPU_ID` / `CUDA_VISIBLE_DEVICES` for an idle GPU (default 3). Conda env is `MLMP` (uppercase).

---

## Task 1: Create the new method file (copy + penalty edits)

**Files:**
- Create: `adapt/deyo_mlmp_gradpen_divgate_continual.py` (copied from `adapt/deyo_mlmp_divgate_continual.py`, then edited)

- [ ] **Step 1: Byte-copy the divgate base**

```bash
cd /home/tekai324/MLMP
cp adapt/deyo_mlmp_divgate_continual.py adapt/deyo_mlmp_gradpen_divgate_continual.py
```

- [ ] **Step 2: Rename the class**

In `adapt/deyo_mlmp_gradpen_divgate_continual.py`, change the class declaration:

```python
# from:
class DeYOMLMPDivGateContinual:
# to:
class DeYOMLMPGradPenDivGateContinual:
```

- [ ] **Step 3: Update the top-of-file module docstring header**

Replace the first docstring line (currently `DeYOMLMPDivGateContinual — DeYO (ICLR 2024) adapt loss in the full MLMP`) with:

```python
"""
DeYOMLMPGradPenDivGateContinual — DeYO+MLMP+DivGate with an explicit
gradient-norm penalty L' = L + lambda * ||grad_theta L||^2 (double backward).

Identical to deyo_mlmp_divgate_continual except the loss/backward block: when
grad_pen_lambda > 0 we add a penalty on the L2 norm of the LN-param gradient
(the signal 學長 found highly anti-correlated with mIoU, docs/2026-06-18-contribution.md
§7) so the optimiser actively avoids high-grad_norm states. lambda == 0 falls
back to the exact divgate path (bit-identical control). H_margin DivGate is
unchanged. New log: gradpen_log.csv (total_batches, grad_norm, penalty, lambda).
"""
```

(Leave the rest of the original docstring body below it untouched.)

- [ ] **Step 4: Add the two new constructor args**

In the `__init__` signature, insert the two args immediately before `save_dir=None,`:

```python
                 cautious_rst=0.005, brake_rst=0.02,
                 # --- gradient-norm penalty ---
                 grad_pen_lambda=0.0, grad_pen_form='sq',
                 save_dir=None,
                 runtime_calculation=False, device='cpu'):
```

- [ ] **Step 5: Store the new args + init the gradpen log file**

Find the gate-log block in `__init__`:

```python
        import os
        self.gate_log_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self.gate_log_path = os.path.join(save_dir, "gate_log.csv")
            with open(self.gate_log_path, 'w') as _f:
                _f.write("total_batches,h_margin,mode,rst\n")
```

Replace it with (adds `grad_pen_*` storage + `gradpen_log.csv`):

```python
        import os
        self.grad_pen_lambda = float(grad_pen_lambda)
        self.grad_pen_form = str(grad_pen_form)
        self.gate_log_path = None
        self.gradpen_log_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self.gate_log_path = os.path.join(save_dir, "gate_log.csv")
            with open(self.gate_log_path, 'w') as _f:
                _f.write("total_batches,h_margin,mode,rst\n")
            self.gradpen_log_path = os.path.join(save_dir, "gradpen_log.csv")
            with open(self.gradpen_log_path, 'w') as _f:
                _f.write("total_batches,grad_norm,penalty,lambda\n")
```

- [ ] **Step 6: Replace the loss/backward block with the penalty version**

Find this block inside `adapt()` (under `if final_mask.sum() > 0:`):

```python
                    loss_report.append(loss.item())
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # gate stochastic restore after the step
                    if self.current_rst > 0.0:
                        self._stochastic_restore_flat(self.current_rst)
```

Replace it with:

```python
                    loss_report.append(loss.item())
                    ln_params = [p for _, p in self.named_ln_params]
                    if self.grad_pen_lambda > 0.0:
                        # explicit grad-norm penalty (double backward, fp32)
                        g = torch.autograd.grad(loss, ln_params, create_graph=True)
                        grad_sq = sum((gi.float() ** 2).sum() for gi in g)
                        if self.grad_pen_form == 'linear':
                            penalty = grad_sq.clamp(min=1e-12).sqrt()
                        else:
                            penalty = grad_sq
                        total = loss + self.grad_pen_lambda * penalty
                        total.backward()
                        grad_norm_raw = float(grad_sq.detach().sqrt())
                        penalty_val = float(penalty.detach())
                    else:
                        # lambda == 0: exact divgate path (bit-identical control)
                        loss.backward()
                        with torch.no_grad():
                            gsq = sum((p.grad.float() ** 2).sum()
                                      for _, p in self.named_ln_params
                                      if p.grad is not None)
                        grad_norm_raw = float(gsq.sqrt())
                        penalty_val = 0.0
                    self._log_gradpen(grad_norm_raw, penalty_val)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # gate stochastic restore after the step
                    if self.current_rst > 0.0:
                        self._stochastic_restore_flat(self.current_rst)
```

- [ ] **Step 7: Add the `_log_gradpen` helper method**

Immediately after the `adapt()` method returns (i.e. just before the
`# ===================== Diagnostics` comment, or before the first `@staticmethod`
helper), add:

```python
    def _log_gradpen(self, grad_norm, penalty):
        if self.gradpen_log_path is None:
            return
        with open(self.gradpen_log_path, 'a') as _f:
            _f.write(f"{self.total_batches},{grad_norm:.6f},"
                     f"{penalty:.6f},{self.grad_pen_lambda:.6g}\n")
```

- [ ] **Step 8: Syntax check**

Run: `python -c "import ast; ast.parse(open('adapt/deyo_mlmp_gradpen_divgate_continual.py').read()); print('syntax OK')"`
Expected: `syntax OK`

- [ ] **Step 9: Commit**

```bash
git add adapt/deyo_mlmp_gradpen_divgate_continual.py
git commit -m "feat(adapt): deyo_mlmp_gradpen_divgate_continual — grad-norm penalty method

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Register the method (factory + argparse)

**Files:**
- Modify: `adapt/__init__.py` (import near line 58, registry near line 163)
- Modify: `main_continual.py` (new args block after the divgate block, ~line 425)

- [ ] **Step 1: Add the import in `adapt/__init__.py`**

After the line `from .deyo_mlmp_divgate_continual import DeYOMLMPDivGateContinual`, add:

```python
from .deyo_mlmp_gradpen_divgate_continual import DeYOMLMPGradPenDivGateContinual
```

- [ ] **Step 2: Add the registry entry in `adapt/__init__.py`**

In the method-name dict, after the line `'deyo_mlmp_divgate_continual': DeYOMLMPDivGateContinual,`, add:

```python
    'deyo_mlmp_gradpen_divgate_continual': DeYOMLMPGradPenDivGateContinual,
```

- [ ] **Step 3: Add the argparse block in `main_continual.py`**

Find the end of the `elif method == 'deyo_mlmp_divgate_continual':` block (the line
`parser.add_argument('--brake_rst', type=float, default=0.02)` around line 425).
Immediately after it (before `# --- DeYO + MLMP + SmoothAnchor ---`), insert:

```python

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
                            help='strength of the ||grad||^2 penalty; 0 = divgate baseline')
        parser.add_argument('--grad_pen_form', type=str, default='sq',
                            choices=['sq', 'linear'])
```

- [ ] **Step 4: Factory check**

Run:
```bash
python -c "import adapt; m=adapt.METHODS if hasattr(adapt,'METHODS') else None; import adapt as a; print('deyo_mlmp_gradpen_divgate_continual' in [k for k in a.__dict__.get('method_classes', {}) ] or 'check via get_method')"
python -c "from adapt import get_method; print('factory import OK')"
```
Expected: `factory import OK` (the import succeeding proves the registry line is valid Python and the new class imports cleanly).

- [ ] **Step 5: Argparse check**

Run:
```bash
python main_continual.py --method deyo_mlmp_gradpen_divgate_continual --help 2>&1 | grep -E "grad_pen_lambda|grad_pen_form"
```
Expected: both `--grad_pen_lambda` and `--grad_pen_form` appear in the help.

- [ ] **Step 6: Commit**

```bash
git add adapt/__init__.py main_continual.py
git commit -m "feat: register deyo_mlmp_gradpen_divgate_continual (factory + argparse)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Correctness invariant — λ=0 is bit-identical to divgate

**Files:** none (verification only). Uses a tiny VOC20 config for speed.

- [ ] **Step 1: Run the existing divgate baseline on a tiny config**

```bash
cd /home/tekai324/MLMP
conda run -n MLMP env OPENCV_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=3 \
  python main_continual.py --adapt --method deyo_mlmp_divgate_continual \
  --ovss_type naclip --ovss_backbone ViT-L/14 --prompt_dir prompts.yaml \
  --dataset PascalVOC20Dataset --data_dir data/VOC/VOC2012/ \
  --init_resize 224 224 --patch_size 224 224 --patch_stride 112 \
  --corruptions_list snow frost fog brightness contrast \
  --subset_size 20 --subset_seed 0 --workers 1 \
  --lr 0.000005 --steps 1 --batch_size 1 --continual_rounds 1 --seed 0 \
  --vision_outputs -1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18 \
  --h_threshold 3.0 --h_warning 2.7 --monitor_interval 50 \
  --cautious_rst 0.005 --brake_rst 0.02 \
  --save_dir save/_smoke/divgate_ref/ --class_extensions
```
Expected: completes 1 round, writes `save/_smoke/divgate_ref/results_all_rounds.txt`.

- [ ] **Step 2: Run the new method with λ=0 on the identical config**

```bash
conda run -n MLMP env OPENCV_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=3 \
  python main_continual.py --adapt --method deyo_mlmp_gradpen_divgate_continual \
  --ovss_type naclip --ovss_backbone ViT-L/14 --prompt_dir prompts.yaml \
  --dataset PascalVOC20Dataset --data_dir data/VOC/VOC2012/ \
  --init_resize 224 224 --patch_size 224 224 --patch_stride 112 \
  --corruptions_list snow frost fog brightness contrast \
  --subset_size 20 --subset_seed 0 --workers 1 \
  --lr 0.000005 --steps 1 --batch_size 1 --continual_rounds 1 --seed 0 \
  --vision_outputs -1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18 \
  --h_threshold 3.0 --h_warning 2.7 --monitor_interval 50 \
  --cautious_rst 0.005 --brake_rst 0.02 \
  --grad_pen_lambda 0.0 --grad_pen_form sq \
  --save_dir save/_smoke/gradpen_l0/ --class_extensions
```
Expected: completes 1 round, writes `save/_smoke/gradpen_l0/results_all_rounds.txt`.

- [ ] **Step 3: Diff the two results files (the invariant)**

Run:
```bash
diff save/_smoke/divgate_ref/results_all_rounds.txt save/_smoke/gradpen_l0/results_all_rounds.txt && echo "BIT-IDENTICAL ✅"
```
Expected: `BIT-IDENTICAL ✅` (no diff). If they differ, the λ=0 branch is not the
exact original path — re-check Task 1 Step 6 before proceeding.

- [ ] **Step 4: Confirm gradpen_log.csv exists and is well-formed (λ=0 → penalty 0)**

Run:
```bash
head -3 save/_smoke/gradpen_l0/gradpen_log.csv
```
Expected: header `total_batches,grad_norm,penalty,lambda` then rows with a non-zero
`grad_norm`, `penalty=0.000000`, `lambda=0`.

- [ ] **Step 5: Commit (note: smoke outputs are under save/, normally git-ignored)**

```bash
git add -A && git status --short
# If save/ is gitignored (expected), nothing to commit here — this task is verification only.
echo "Task 3 verification done"
```

---

## Task 4: Mechanism check — λ>0 engages and lowers grad_norm

**Files:** none (verification only).

- [ ] **Step 1: Run the new method with λ=0.03 on the same tiny config**

```bash
cd /home/tekai324/MLMP
conda run -n MLMP env OPENCV_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=3 \
  python main_continual.py --adapt --method deyo_mlmp_gradpen_divgate_continual \
  --ovss_type naclip --ovss_backbone ViT-L/14 --prompt_dir prompts.yaml \
  --dataset PascalVOC20Dataset --data_dir data/VOC/VOC2012/ \
  --init_resize 224 224 --patch_size 224 224 --patch_stride 112 \
  --corruptions_list snow frost fog brightness contrast \
  --subset_size 20 --subset_seed 0 --workers 1 \
  --lr 0.000005 --steps 1 --batch_size 1 --continual_rounds 2 --seed 0 \
  --vision_outputs -1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18 \
  --h_threshold 3.0 --h_warning 2.7 --monitor_interval 50 \
  --cautious_rst 0.005 --brake_rst 0.02 \
  --grad_pen_lambda 0.03 --grad_pen_form sq \
  --save_dir save/_smoke/gradpen_l003/ --class_extensions
```
Expected: completes 2 rounds without NaN/crash.

- [ ] **Step 2: Confirm the penalty is non-zero and grad_norm is recorded**

Run:
```bash
python - <<'PY'
import csv
rows = list(csv.DictReader(open('save/_smoke/gradpen_l003/gradpen_log.csv')))
gn = [float(r['grad_norm']) for r in rows]
pen = [float(r['penalty']) for r in rows]
print(f"steps={len(rows)}  grad_norm[min..max]={min(gn):.3f}..{max(gn):.3f}  penalty>0 in {sum(p>0 for p in pen)}/{len(pen)} rows")
assert any(p > 0 for p in pen), "penalty never fired — double-backward path not taken"
print("MECHANISM OK ✅")
PY
```
Expected: `MECHANISM OK ✅`, penalty > 0 in most rows. (A side-by-side grad_norm
*reduction* vs λ=0 only becomes meaningful over many rounds — that is what the
full sweep + Task 7 plots assess; here we only confirm the penalty path executes.)

- [ ] **Step 3: Clean up smoke dirs**

```bash
rm -rf save/_smoke/
echo "Task 4 done"
```

---

## Task 5: Dataset runner scripts (ACDC / VOC20 / Cityscapes)

**Files:**
- Create: `bash/ACDC_10_round/deyo_mlmp_gradpen_divgate_continual.sh`
- Create: `bash/v20/deyo_mlmp_gradpen_divgate_continual_5corr_sub100.sh`
- Create: `bash/cityscapes/deyo_mlmp_gradpen_divgate_continual_5corr_sub100.sh`

- [ ] **Step 1: Write the ACDC runner**

Create `bash/ACDC_10_round/deyo_mlmp_gradpen_divgate_continual.sh`:

```bash
#!/bin/bash
# deyo_mlmp_gradpen_divgate on ACDCDataset:
#   DeYO loss + MLMP + DivGate + lambda*||grad||^2 gradient-norm penalty.
# Gate tuned for this dataset (h_margin healthy ~ 2.0+). GRAD_PEN_LAMBDA env-overridable.

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2

GPU_ID=${GPU_ID:-3}

DATASET=ACDCDataset
DATA_DIR="data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=1

METHOD="deyo_mlmp_gradpen_divgate_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"
OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"
PROMPT_DIR="prompts.yaml"

BATCH_SIZE=1
LR=0.000005
STEPS=1
CONTINUAL_ROUNDS=${CONTINUAL_ROUNDS:-150}

H_THRESHOLD=${H_THRESHOLD:-2.0}
H_WARNING=${H_WARNING:-1.7}
MONITOR_INTERVAL=50
CAUTIOUS_RST=${CAUTIOUS_RST:-0.005}
BRAKE_RST=${BRAKE_RST:-0.02}

GRAD_PEN_LAMBDA=${GRAD_PEN_LAMBDA:-0.01}
GRAD_PEN_FORM=${GRAD_PEN_FORM:-sq}

SAVE_DIR=${SAVE_DIR:-"save/${DATASET}/${METHOD}_l${GRAD_PEN_LAMBDA}/"}

CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --adapt \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone $OVSS_BACKBONE \
                        --prompt_dir $PROMPT_DIR \
                        \
                        --dataset $DATASET \
                        --data_dir $DATA_DIR \
                        --init_resize $INIT_RESIZE \
                        --patch_size 224 224 \
                        --patch_stride 112 \
                        --corruptions_list $CONDITIONS \
                        --workers $WORKERS \
                        \
                        --lr $LR \
                        --steps $STEPS \
                        --batch_size $BATCH_SIZE \
                        --continual_rounds $CONTINUAL_ROUNDS \
                        --seed 0 \
                        \
                        --vision_outputs $OUT_VISION \
                        --h_threshold $H_THRESHOLD \
                        --h_warning $H_WARNING \
                        --monitor_interval $MONITOR_INTERVAL \
                        --cautious_rst $CAUTIOUS_RST \
                        --brake_rst $BRAKE_RST \
                        --grad_pen_lambda $GRAD_PEN_LAMBDA \
                        --grad_pen_form $GRAD_PEN_FORM \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
```

- [ ] **Step 2: Write the VOC20 runner**

Create `bash/v20/deyo_mlmp_gradpen_divgate_continual_5corr_sub100.sh` — identical to
Step 1 except the dataset block and gate defaults:

```bash
#!/bin/bash
# deyo_mlmp_gradpen_divgate on PascalVOC20Dataset (5corr, subset 100).
# NOTE: DivGate never fires here (H_margin ~3.0) -> the lambda penalty is the ONLY
# active defence. This is the cleanest test of the grad_norm hypothesis.

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENCV_NUM_THREADS=2

GPU_ID=${GPU_ID:-3}

DATASET=PascalVOC20Dataset
DATA_DIR="data/VOC/VOC2012/"
INIT_RESIZE="224 224"
CONDITIONS="snow frost fog brightness contrast"
WORKERS=1

METHOD="deyo_mlmp_gradpen_divgate_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"
OUT_VISION="-1 -2 -3 -4 -5 -6 -7 -8 -9 -10 -11 -12 -13 -14 -15 -16 -17 -18"
PROMPT_DIR="prompts.yaml"

BATCH_SIZE=1
LR=0.000005
STEPS=1
CONTINUAL_ROUNDS=${CONTINUAL_ROUNDS:-150}

H_THRESHOLD=${H_THRESHOLD:-3.0}
H_WARNING=${H_WARNING:-2.7}
MONITOR_INTERVAL=50
CAUTIOUS_RST=${CAUTIOUS_RST:-0.005}
BRAKE_RST=${BRAKE_RST:-0.02}

GRAD_PEN_LAMBDA=${GRAD_PEN_LAMBDA:-0.01}
GRAD_PEN_FORM=${GRAD_PEN_FORM:-sq}

SUBSET_SIZE=${SUBSET_SIZE:-100}

SAVE_DIR=${SAVE_DIR:-"save/${DATASET}/v20_acdc_matched/${METHOD}_l${GRAD_PEN_LAMBDA}/"}

CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --adapt \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone $OVSS_BACKBONE \
                        --prompt_dir $PROMPT_DIR \
                        \
                        --dataset $DATASET \
                        --data_dir $DATA_DIR \
                        --init_resize $INIT_RESIZE \
                        --patch_size 224 224 \
                        --patch_stride 112 \
                        --corruptions_list $CONDITIONS \
                        --subset_size $SUBSET_SIZE \
                        --subset_seed 0 \
                        --workers $WORKERS \
                        \
                        --lr $LR \
                        --steps $STEPS \
                        --batch_size $BATCH_SIZE \
                        --continual_rounds $CONTINUAL_ROUNDS \
                        --seed 0 \
                        \
                        --vision_outputs $OUT_VISION \
                        --h_threshold $H_THRESHOLD \
                        --h_warning $H_WARNING \
                        --monitor_interval $MONITOR_INTERVAL \
                        --cautious_rst $CAUTIOUS_RST \
                        --brake_rst $BRAKE_RST \
                        --grad_pen_lambda $GRAD_PEN_LAMBDA \
                        --grad_pen_form $GRAD_PEN_FORM \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
```

- [ ] **Step 3: Write the Cityscapes runner**

Create `bash/cityscapes/deyo_mlmp_gradpen_divgate_continual_5corr_sub100.sh` —
identical to Step 2 except:

```bash
# header comment:
# deyo_mlmp_gradpen_divgate on CityscapesDataset (5corr, subset 100).
DATASET=CityscapesDataset
DATA_DIR="data/Cityscape/"
INIT_RESIZE="1120 560"
# gate defaults:
H_THRESHOLD=${H_THRESHOLD:-2.1}
H_WARNING=${H_WARNING:-1.8}
# SAVE_DIR (no v20_acdc_matched subdir):
SAVE_DIR=${SAVE_DIR:-"save/${DATASET}/${METHOD}_l${GRAD_PEN_LAMBDA}/"}
```

All other lines (CONDITIONS `snow frost fog brightness contrast`, `--subset_size`,
the full python invocation) are the same as the VOC20 runner in Step 2.

- [ ] **Step 4: chmod + syntax-check all three scripts**

```bash
chmod +x bash/ACDC_10_round/deyo_mlmp_gradpen_divgate_continual.sh \
         bash/v20/deyo_mlmp_gradpen_divgate_continual_5corr_sub100.sh \
         bash/cityscapes/deyo_mlmp_gradpen_divgate_continual_5corr_sub100.sh
for f in bash/ACDC_10_round/deyo_mlmp_gradpen_divgate_continual.sh \
         bash/v20/deyo_mlmp_gradpen_divgate_continual_5corr_sub100.sh \
         bash/cityscapes/deyo_mlmp_gradpen_divgate_continual_5corr_sub100.sh; do
  bash -n "$f" && echo "OK: $f"
done
```
Expected: `OK: ...` for all three (`bash -n` = syntax check, no execution).

- [ ] **Step 5: Commit**

```bash
git add bash/ACDC_10_round/deyo_mlmp_gradpen_divgate_continual.sh \
        bash/v20/deyo_mlmp_gradpen_divgate_continual_5corr_sub100.sh \
        bash/cityscapes/deyo_mlmp_gradpen_divgate_continual_5corr_sub100.sh
git commit -m "feat(bash): gradpen_divgate runners for ACDC/VOC20/Cityscapes

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: λ-sweep launcher (Stage 1)

**Files:**
- Create: `bash/sweep_gradpen_divgate.sh`

- [ ] **Step 1: Write the sweep launcher**

Create `bash/sweep_gradpen_divgate.sh`:

```bash
#!/bin/bash
# Stage-1 lambda sweep for deyo_mlmp_gradpen_divgate_continual.
# Sweeps GRAD_PEN_LAMBDA over the grid on one dataset, packing runs onto the GPUs
# in GPUS, staggered so model loads do not collide. Gate stays at each runner's
# proven default. Usage:
#   bash bash/sweep_gradpen_divgate.sh acdc
#   bash bash/sweep_gradpen_divgate.sh v20
#   bash bash/sweep_gradpen_divgate.sh cityscapes
set -u

DATASET_KEY="${1:-acdc}"
LAMBDAS=(${LAMBDAS:-0.003 0.01 0.03 0.1})   # 0 == divgate baseline; run separately if needed
GPUS=(${GPUS:-0 1 2 3})
STAGGER="${STAGGER:-90}"                     # seconds between launches

case "$DATASET_KEY" in
  acdc)       RUNNER="bash/ACDC_10_round/deyo_mlmp_gradpen_divgate_continual.sh" ;;
  v20)        RUNNER="bash/v20/deyo_mlmp_gradpen_divgate_continual_5corr_sub100.sh" ;;
  cityscapes) RUNNER="bash/cityscapes/deyo_mlmp_gradpen_divgate_continual_5corr_sub100.sh" ;;
  *) echo "unknown dataset key: $DATASET_KEY (use acdc|v20|cityscapes)"; exit 1 ;;
esac

mkdir -p save/_sweep_logs
i=0
for lam in "${LAMBDAS[@]}"; do
  gpu="${GPUS[$(( i % ${#GPUS[@]} ))]}"
  log="save/_sweep_logs/gradpen_${DATASET_KEY}_l${lam}_gpu${gpu}.log"
  echo "[launch] $DATASET_KEY lambda=$lam on GPU $gpu -> $log"
  GPU_ID="$gpu" GRAD_PEN_LAMBDA="$lam" nohup bash "$RUNNER" > "$log" 2>&1 &
  i=$(( i + 1 ))
  sleep "$STAGGER"
done
echo "launched ${#LAMBDAS[@]} runs for $DATASET_KEY; tail save/_sweep_logs/ to monitor."
wait
echo "sweep ($DATASET_KEY) done."
```

- [ ] **Step 2: Syntax-check**

```bash
chmod +x bash/sweep_gradpen_divgate.sh
bash -n bash/sweep_gradpen_divgate.sh && echo "OK"
```
Expected: `OK`

- [ ] **Step 3: Dry-run the launch logic without starting real training**

Run:
```bash
LAMBDAS="0.01 0.03" GPUS="0 1" STAGGER=0 bash -c '
set -u; DATASET_KEY=acdc; LAMBDAS=(0.01 0.03); GPUS=(0 1); i=0
for lam in "${LAMBDAS[@]}"; do gpu="${GPUS[$(( i % ${#GPUS[@]} ))]}";
  echo "would launch acdc lambda=$lam on GPU $gpu"; i=$((i+1)); done'
```
Expected: two `would launch ...` lines mapping λ to GPUs 0 and 1 (confirms the
round-robin GPU assignment; no training started).

- [ ] **Step 4: Commit**

```bash
git add bash/sweep_gradpen_divgate.sh
git commit -m "feat(bash): Stage-1 lambda sweep launcher for gradpen_divgate

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: Analysis + plotting script

**Files:**
- Create: `scripts/plot_gradpen_sweep.py`

- [ ] **Step 1: Write the analysis script**

Create `scripts/plot_gradpen_sweep.py`:

```python
#!/usr/bin/env python
"""Collect + plot the deyo_mlmp_gradpen_divgate lambda sweep for one dataset.

Reads, for each lambda, <save_dir>/results_all_rounds.txt (per-round mean mIoU,
last column) and <save_dir>/gradpen_log.csv (per-step grad_norm). Prints a
mean/peak/last table and writes two figures:
  * <out>_miou.png       — per-round mean mIoU, one line per lambda
  * <out>_gradnorm.png   — per-round mean grad_norm, one line per lambda
                           (the mechanism panel: does higher lambda lower grad_norm?)

Usage:
  python scripts/plot_gradpen_sweep.py \
      --base save/ACDCDataset --prefix deyo_mlmp_gradpen_divgate_continual \
      --lambdas 0 0.003 0.01 0.03 0.1 --out save/_compare/gradpen_acdc
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_round_means(path):
    """results_all_rounds.txt: header then 'Round, c1, ..., Mean_mIoU'. Returns
    list of per-round mean (last numeric column)."""
    means = []
    if not os.path.exists(path):
        return means
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("round"):
                continue
            parts = [p for p in line.replace(",", " ").split() if p]
            try:
                means.append(float(parts[-1]))
            except (ValueError, IndexError):
                continue
    return means


def load_gradnorm_per_round(path, n_rounds):
    """gradpen_log.csv: total_batches,grad_norm,penalty,lambda. Split the
    grad_norm series into n_rounds equal chunks and average each (a per-round
    proxy; batch counts per round are equal in this protocol)."""
    if not os.path.exists(path) or n_rounds == 0:
        return []
    gn = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                gn.append(float(r["grad_norm"]))
            except (ValueError, KeyError):
                continue
    if not gn:
        return []
    per = max(1, len(gn) // n_rounds)
    out = []
    for k in range(n_rounds):
        chunk = gn[k * per:(k + 1) * per]
        if chunk:
            out.append(sum(chunk) / len(chunk))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="e.g. save/ACDCDataset")
    ap.add_argument("--prefix", default="deyo_mlmp_gradpen_divgate_continual")
    ap.add_argument("--lambdas", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="output path stem")
    ap.add_argument("--subdir", default="", help="optional subdir under base, e.g. v20_acdc_matched")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig_m, ax_m = plt.subplots(figsize=(8, 5))
    fig_g, ax_g = plt.subplots(figsize=(8, 5))

    print(f"{'lambda':>8} {'meanAll':>8} {'peak':>8} {'last':>8} {'rounds':>7}")
    for lam in args.lambdas:
        run_dir = os.path.join(args.base, args.subdir, f"{args.prefix}_l{lam}")
        means = load_round_means(os.path.join(run_dir, "results_all_rounds.txt"))
        if not means:
            print(f"{lam:>8} {'(no results_all_rounds.txt at '+run_dir+')':>8}")
            continue
        rounds = list(range(1, len(means) + 1))
        mean_all = sum(means) / len(means)
        print(f"{lam:>8} {mean_all:>8.2f} {max(means):>8.2f} {means[-1]:>8.2f} {len(means):>7}")
        ax_m.plot(rounds, means, marker="", label=f"λ={lam}")
        gn = load_gradnorm_per_round(os.path.join(run_dir, "gradpen_log.csv"), len(means))
        if gn:
            ax_g.plot(range(1, len(gn) + 1), gn, label=f"λ={lam}")

    for ax, ttl, yl, fig, suf in [
        (ax_m, "mean mIoU vs round", "mean mIoU", fig_m, "miou"),
        (ax_g, "grad_norm vs round", "mean grad_norm", fig_g, "gradnorm"),
    ]:
        ax.set_xlabel("round"); ax.set_ylabel(yl); ax.set_title(ttl)
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(f"{args.out}_{suf}.png", dpi=130)
        print(f"wrote {args.out}_{suf}.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax check**

Run: `python -c "import ast; ast.parse(open('scripts/plot_gradpen_sweep.py').read()); print('syntax OK')"`
Expected: `syntax OK`

- [ ] **Step 3: Smoke-test the parser on a fabricated minimal input**

Run:
```bash
mkdir -p save/_t/deyo_mlmp_gradpen_divgate_continual_l0.01
printf 'Round, fog, night, rain, snow, Mean_mIoU\n1, 30, 28, 29, 27, 28.5\n2, 31, 29, 30, 28, 29.5\n' \
  > save/_t/deyo_mlmp_gradpen_divgate_continual_l0.01/results_all_rounds.txt
printf 'total_batches,grad_norm,penalty,lambda\n1,5.0,25.0,0.01\n2,4.5,20.2,0.01\n' \
  > save/_t/deyo_mlmp_gradpen_divgate_continual_l0.01/gradpen_log.csv
python scripts/plot_gradpen_sweep.py --base save/_t --lambdas 0.01 --out save/_t/out
ls save/_t/out_miou.png save/_t/out_gradnorm.png && echo "PLOT OK"
rm -rf save/_t
```
Expected: a one-row table (λ=0.01, meanAll≈29.00, peak 29.50, last 29.50), then
`PLOT OK`.

- [ ] **Step 4: Commit**

```bash
git add scripts/plot_gradpen_sweep.py
git commit -m "feat(scripts): plot_gradpen_sweep — lambda mean/peak/last + grad_norm panels

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: Document the experiment (EXPERIMENT_STATUS + memory)

**Files:**
- Modify: `docs/EXPERIMENT_STATUS.md` (append a Phase O section)
- Modify: `/home/tekai324/.claude/projects/-home-tekai324-MLMP/memory/MEMORY.md` + a new memory file

- [ ] **Step 1: Append Phase O to EXPERIMENT_STATUS.md**

Add this section at the end of `docs/EXPERIMENT_STATUS.md`, before the closing
`*End of research-arc document...*` line:

```markdown
## 18. Phase O: grad_norm penalty in DeYO+MLMP+DivGate (2026-06-21)

**Idea:** 學長's 0618 hunt (§ docs/2026-06-18-contribution.md §7) found grad_norm is
the only signal anti-correlated with mIoU on ALL datasets incl. VOC20's uniform
degradation (where H_margin is blind). Turn it from a passive gate signal into an
active loss term: `L' = L + λ‖∇_θ L‖²`, explicit double-backward, on top of the
proven DeYO+MLMP+DivGate (gate kept as safety net).

- Method: `adapt/deyo_mlmp_gradpen_divgate_continual.py` (divgate base; λ=0 →
  bit-identical control). New log `gradpen_log.csv` (grad_norm, penalty).
- Spec: `docs/superpowers/specs/2026-06-21-gradnorm-penalty-design.md`.
- Plan: `docs/superpowers/plans/2026-06-21-gradnorm-penalty.md`.
- Stage-1 grid: λ ∈ {0, 0.003, 0.01, 0.03, 0.1} × {ACDC h2.0/1.7, Cityscapes
  h2.1/1.8, VOC20 h3.0/2.7}, 150R. Launcher `bash/sweep_gradpen_divgate.sh`,
  analysis `scripts/plot_gradpen_sweep.py`.
- VOC20 is the cleanest test (gate never fires → λ penalty is the only defence).
- Risk tracked: penalising grad_norm could seek flat minima (good) OR the
  zero-gradient degenerate state (bad) — gate is the backstop; mechanism panel
  (grad_norm vs round per λ) checks which way it goes.
- **Results: TODO after the sweep finishes.**
```

- [ ] **Step 2: Add a memory file**

Create `/home/tekai324/.claude/projects/-home-tekai324-MLMP/memory/gradnorm_penalty_phase_o.md`:

```markdown
---
name: gradnorm-penalty-phase-o
description: Phase O (2026-06-21) — grad_norm penalty added to DeYO+MLMP+DivGate loss (L+λ‖∇L‖², double-backward); method/spec/plan/sweep locations + design decisions.
metadata:
  type: project
---

Phase O turns 學長's grad_norm↔mIoU finding into an active loss term:
`L' = L + λ‖∇_θ L‖²` (explicit double-backward, create_graph=True, fp32 norm) on
top of DeYO+MLMP+DivGate. H_margin gate kept as safety net (penalising grad_norm
could seek flat minima OR the 0-gradient collapse state).

- Method: `adapt/deyo_mlmp_gradpen_divgate_continual.py` (copy of divgate; only
  loss/backward changed; λ=0 → exact original path = bit-identical control).
  New CLI `--grad_pen_lambda` (default 0), `--grad_pen_form sq|linear`. New log
  `gradpen_log.csv` (total_batches,grad_norm,penalty,lambda).
- Why explicit penalty, not SAM: SAM is the implicit version (SAR already uses it);
  explicit λ ties directly to the measured signal. See [[merge_2026-06-04_senpai]].
- Decisions: squared form primary (smooth grad near 0; linear diverges); always-on
  (fallbacks if peak suppressed: gate-gated / warmup — documented in spec §3.3).
- Grid: λ ∈ {0,0.003,0.01,0.03,0.1} × {ACDC h2.0/1.7, Citys h2.1/1.8, VOC20 h3.0/2.7},
  150R. Launcher `bash/sweep_gradpen_divgate.sh <acdc|v20|cityscapes>`; analysis
  `scripts/plot_gradpen_sweep.py`. VOC20 = cleanest test (DivGate never fires there).
- Spec `docs/superpowers/specs/2026-06-21-gradnorm-penalty-design.md`; plan
  `docs/superpowers/plans/2026-06-21-gradnorm-penalty.md`; status EXPERIMENT_STATUS §18.
- Targets: beat divgate λ=0 baseline, then MLMP-episodic (ACDC 29.84/30.6, VOC20
  76.21, Cityscapes 20.0). Results: TODO.
```

- [ ] **Step 3: Add the index pointer in MEMORY.md**

Append to `/home/tekai324/.claude/projects/-home-tekai324-MLMP/memory/MEMORY.md`:

```markdown
- [gradnorm_penalty_phase_o.md](gradnorm_penalty_phase_o.md) — Phase O (2026-06-21): grad_norm added to DeYO+MLMP+DivGate loss as L+λ‖∇L‖² (double-backward); λ=0 bit-identical control; sweep λ×3 datasets. Method/spec/plan/launcher locations + why-explicit-not-SAM. Results TODO.
```

- [ ] **Step 4: Commit**

```bash
git add docs/EXPERIMENT_STATUS.md
git commit -m "docs(EXPERIMENT_STATUS): Phase O — grad_norm penalty experiment

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
(The memory files under ~/.claude are outside the repo and are not committed.)

---

## Task 9: Launch Stage-1 sweeps (execution, not code)

**Files:** none. This task runs the experiment; do it only when GPUs are free.

- [ ] **Step 1: Launch VOC20 first (fastest + cleanest hypothesis test)**

```bash
cd /home/tekai324/MLMP
GPUS="0 1 2 3" bash bash/sweep_gradpen_divgate.sh v20
```
Also run the λ=0 baseline separately for the control line:
```bash
GPU_ID=0 GRAD_PEN_LAMBDA=0 bash bash/v20/deyo_mlmp_gradpen_divgate_continual_5corr_sub100.sh
```

- [ ] **Step 2: Monitor**

```bash
tail -n 2 save/PascalVOC20Dataset/v20_acdc_matched/deyo_mlmp_gradpen_divgate_continual_l*/results_all_rounds.txt
```
Expected: each run's latest round line advancing toward Round 150.

- [ ] **Step 3: When VOC20 done, analyse**

```bash
python scripts/plot_gradpen_sweep.py \
  --base save/PascalVOC20Dataset --subdir v20_acdc_matched \
  --prefix deyo_mlmp_gradpen_divgate_continual \
  --lambdas 0 0.003 0.01 0.03 0.1 --out save/_compare/gradpen_v20
```
Inspect: (a) does any λ>0 beat λ=0 mean? (b) does grad_norm panel show higher λ →
lower grad_norm? Decide whether to repeat for ACDC / Cityscapes and whether to run
Stage 2 (gate sweep at the best λ).

- [ ] **Step 4: Repeat for ACDC and Cityscapes**

```bash
GPUS="0 1 2 3" bash bash/sweep_gradpen_divgate.sh acdc
GPUS="0 1 2 3" bash bash/sweep_gradpen_divgate.sh cityscapes
# + the matching λ=0 baseline run for each, then plot_gradpen_sweep.py per dataset.
```

- [ ] **Step 5: Fill in EXPERIMENT_STATUS §18 results + update the memory file**

Replace the `Results: TODO` lines in `docs/EXPERIMENT_STATUS.md` §18 and the memory
file with the per-dataset mean/peak/last vs baseline + episodic, and the grad_norm
mechanism finding. Commit the EXPERIMENT_STATUS change.

---

## Self-review notes (coverage vs spec)

- Spec §3.1–3.2 (method + penalty) → Task 1. §3.3 decisions (squared, always-on,
  fp32, λ=0 branch) → Task 1 Steps 4/6. §3.4 CLI → Task 2. §3.5 log → Task 1 Steps 5/7.
- Spec §4 grid + per-dataset gate table → Tasks 5/6/9.
- Spec §5 file table → Tasks 1/2/5/6/7/8 (every file covered).
- Spec §6 success criteria: correctness (1) → Task 3; mechanism (2) → Task 4 + Task 7
  grad_norm panel; main result (3) + milestone (4) + story (5) → Task 9 analysis.
- Spec §7 out-of-scope fallbacks → documented in spec §3.3 and the Phase O note;
  not implemented (correct).
- `linear` variant: arg wired (Task 2), branch implemented (Task 1 Step 6),
  default `sq` — covered.
```
