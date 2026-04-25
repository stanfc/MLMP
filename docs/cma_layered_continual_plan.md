# CMA-Layered-Continual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `CMALayeredContinual` — CMA loss + layer-stratified stochastic restoration — and ship a runnable ACDC bash script per [docs/cma_layered_continual_spec.md](cma_layered_continual_spec.md).

**Architecture:** A new self-contained adapter class in `adapt/cma_layered_continual.py`, registered in the existing `get_method()` factory, exposed via `main_continual.py` CLI args, and runnable via a new bash script under `bash/ACDC_10_round/`. The class reuses verbatim the CMA loss tensor ops from `adapt/cma_continual.py` and adds a single new mechanism — `_stochastic_layered_restore()` — that runs after every `optimizer.step()`.

**Tech Stack:** PyTorch, NA-CLIP ViT-L/14 (via `ovss.load_ovss`), Adam optimizer over LayerNorm γ/β only, ACDC dataset (existing loader), no new dependencies.

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `adapt/cma_layered_continual.py` | NEW | `class CMALayeredContinual` — full method implementation |
| `adapt/__init__.py` | MODIFY | Add import and `METHOD_CLASSES` entry |
| `main_continual.py` | MODIFY | Add `add_method_specific_args` branch for new method |
| `bash/ACDC_10_round/cma_layered_continual.sh` | NEW | Runnable experiment script |
| `parse_acdc_results.py` | MODIFY | Append new method to `METHODS` table |

No tests directory exists in this project — the established pattern is manual sanity checks. The plan includes (a) a pure-logic check for layer classification (no GPU needed) and (b) a debug-mode end-to-end smoke run before committing.

---

### Task 1: Create the adapter class

**Files:**
- Create: `adapt/cma_layered_continual.py`

- [ ] **Step 1: Write the full file**

```python
import re
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'


class CMALayeredContinual:
    """
    CMA-Layered-Continual: Cross-Modal Alignment loss + layer-stratified
    stochastic restoration toward source weights (CTTA, no reset).

    Same forward and CMA loss as CMAContinual. After each gradient step
    every visual-encoder LayerNorm parameter is stochastically restored
    toward its source value with a probability that depends on which
    transformer block the parameter belongs to:

        early ([0, early_cutoff) + ln_pre)         -> early_rst (default 0.001)
        mid   ([early_cutoff, late_cutoff))         -> mid_rst   (default 0.01)
        late  ([late_cutoff, num_blocks) + ln_post) -> late_rst  (default 0.05)

    Setting any rate to 0 disables restoration for that group; setting it
    to 1 freezes those parameters. With all three rates at 0 the method
    reduces exactly to CMAContinual.

    See docs/cma_layered_continual_spec.md for the full design.
    """

    BLOCK_RE = re.compile(r'resblocks\.(\d+)\.')

    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 top_k_percent=0.2,
                 early_rst=0.001, mid_rst=0.01, late_rst=0.05,
                 early_cutoff=8, late_cutoff=16,
                 prompt_dir=None,
                 runtime_calculation=False, device='cpu'):
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps
        self.top_k_percent = float(top_k_percent)
        if not (0.0 < self.top_k_percent <= 1.0):
            raise ValueError(
                f"top_k_percent must be in (0, 1], got {self.top_k_percent}"
            )

        self.early_rst = float(early_rst)
        self.mid_rst = float(mid_rst)
        self.late_rst = float(late_rst)
        for n, v in [('early_rst', self.early_rst),
                     ('mid_rst', self.mid_rst),
                     ('late_rst', self.late_rst)]:
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{n} must be in [0, 1], got {v}")

        self.early_cutoff = int(early_cutoff)
        self.late_cutoff = int(late_cutoff)
        if self.early_cutoff < 0 or self.late_cutoff < self.early_cutoff:
            raise ValueError(
                f"need 0 <= early_cutoff <= late_cutoff, "
                f"got early={self.early_cutoff}, late={self.late_cutoff}"
            )

        self.runtime = runtime_calculation
        self.device = device

        if classes is None:
            raise ValueError("classes is required in __init__")
        self.classes = classes

        # ---------- OVSS model ----------
        self.model, self.tokenize = load_ovss(ovss_type, ovss_backbone, device=device)

        # ---------- Prompts ----------
        if prompt_dir:
            self.prompt_templates = load_prompts_from_yaml(prompt_dir)
            print(f"Number of prompt templates: {len(self.prompt_templates)}")
        else:
            self.prompt_templates = [REFERENCE_PROMPT]

        # ---------- Freeze text encoder, enable LN grads ----------
        self.model.transformer.requires_grad_(False)
        self.model.ln_final.requires_grad_(False)
        self.model.token_embedding.requires_grad_(False)
        self.model.visual = self.set_ln_grads(self.model.visual)
        params, names = self.collect_ln_params(self.model.visual)
        self.named_ln_params = list(zip(names, params))

        print_clip_parameters(self.model)
        self._print_layer_groups()

        # ---------- Optimizer ----------
        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        # ---------- Source state snapshot (used by layered restoration) ----------
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer
        )

        # ---------- Text features ----------
        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(
                self.classes, self.prompt_templates, average=False
            ).squeeze()

        if self.runtime:
            self.adapt_times = []
            self.eval_times = []

    # ===========================================================
    # Public API
    # ===========================================================

    def adapt(self, x):
        return self.perform_adaptation(x)

    def continual_adapt(self, x):
        """Alias for adapt() -- matches main_continual.py protocol."""
        return self.perform_adaptation(x)

    @torch.no_grad()
    def evaluate(self, x):
        t1 = time.time()
        logits, _, _ = self.model(x, self.text_x, True, interpolate=True)
        logits = logits[0]
        if self.runtime:
            self.eval_times.append(time.time() - t1)
        return logits

    def reset(self):
        self.load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state
        )

    # ===========================================================
    # Adaptation
    # ===========================================================

    def perform_adaptation(self, x):
        t1 = time.time()
        loss_report = []
        for _ in range(self.steps):
            logits, image_features, text_features = self.model(
                x, self.text_x, True, interpolate=False
            )
            loss = self.cma_loss(logits, image_features, text_features)
            loss_report.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self._stochastic_layered_restore()
        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

    def cma_loss(self, logits, image_features, text_features):
        """Identical to CMAContinual.cma_loss -- see docs/cma_continual_spec.md §2."""
        # (1) Prompt-averaged logits and text vector (re-normalized)
        avg_logits = logits.mean(dim=0)
        avg_text = text_features.mean(dim=0)
        avg_text = avg_text / avg_text.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        # (2) Pseudo-label and confidence (no_grad)
        with torch.no_grad():
            probs = avg_logits.softmax(dim=1)
            confidence, pred_cls = probs.max(dim=1)
            n_total = confidence.numel()
            k = max(1, int(round(n_total * self.top_k_percent)))
            threshold = torch.topk(confidence.flatten(), k, sorted=True).values[-1]
            mask = confidence >= threshold

        # (3) Drop CLS, reshape patch features
        B, _, w, h = avg_logits.shape
        patch_feats = image_features[:, 1:, :]
        D = patch_feats.shape[-1]
        vis_feat = patch_feats.reshape(B, w, h, D)

        # (4) Per-pixel target text vector and cosine similarity
        text_target = avg_text[pred_cls]
        cos_sim = (vis_feat * text_target).sum(dim=-1)

        # (5) Mean over confident pixels, negated to minimize
        return -cos_sim[mask].mean()

    # ===========================================================
    # Layer-stratified restoration
    # ===========================================================

    def _get_restoration_rate(self, name):
        """Map a state-dict parameter name to its restoration rate."""
        if 'ln_pre' in name:
            return self.early_rst
        if 'ln_post' in name:
            return self.late_rst
        m = self.BLOCK_RE.search(name)
        if m is None:
            return 0.0
        idx = int(m.group(1))
        if idx < self.early_cutoff:
            return self.early_rst
        if idx < self.late_cutoff:
            return self.mid_rst
        return self.late_rst

    @torch.no_grad()
    def _stochastic_layered_restore(self):
        """For each LN parameter, stochastically restore elements toward source."""
        for name, p in self.named_ln_params:
            rst = self._get_restoration_rate(name)
            if rst <= 0.0:
                continue
            mask = (torch.rand(p.shape, device=p.device) < rst).to(p.dtype)
            src = self.model_state[name].to(p.device, dtype=p.dtype)
            p.data.mul_(1.0 - mask).add_(src * mask)

    def _print_layer_groups(self):
        """Diagnostic: report how many LN params fall into each group."""
        groups = {'early': 0, 'mid': 0, 'late': 0}
        for name, _ in self.named_ln_params:
            if 'ln_pre' in name:
                groups['early'] += 1
                continue
            if 'ln_post' in name:
                groups['late'] += 1
                continue
            m = self.BLOCK_RE.search(name)
            if m is None:
                continue
            idx = int(m.group(1))
            if idx < self.early_cutoff:
                groups['early'] += 1
            elif idx < self.late_cutoff:
                groups['mid'] += 1
            else:
                groups['late'] += 1
        total = sum(groups.values())
        print(f"+++ CMA-Layered: cutoffs=({self.early_cutoff}, {self.late_cutoff}), "
              f"rsts=({self.early_rst}, {self.mid_rst}, {self.late_rst})")
        print(f"+++ LN params/group: early={groups['early']}, "
              f"mid={groups['mid']}, late={groups['late']}, total={total}")

    # ===========================================================
    # Shared helpers (verbatim from CMAContinual / TENTContinual)
    # ===========================================================

    def extract_text_embeddings(self, class_names, prompts, average=True):
        text_features = []
        for class_name in class_names:
            texts = [p.format(class_name) for p in prompts]
            texts = self.tokenize(texts).to(self.device)
            class_embeddings = self.model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            if average:
                avg = class_embeddings.mean(dim=0)
                avg = avg / avg.norm()
                class_embeddings = torch.cat([class_embeddings, avg.unsqueeze(0)], dim=0)
            text_features.append(class_embeddings)
        return torch.stack(text_features, dim=1).to(self.device)

    @staticmethod
    def set_ln_grads(model):
        model.requires_grad_(False)
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)
        return model

    @staticmethod
    def collect_ln_params(model):
        params, names = [], []
        for nm, m in model.named_modules():
            if isinstance(m, nn.LayerNorm):
                for np_, p in m.named_parameters():
                    if np_ in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"visual.{nm}.{np_}")
        return params, names

    @staticmethod
    def copy_model_and_optimizer(model, optimizer):
        return copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())

    @staticmethod
    def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)
```

- [ ] **Step 2: Verify the file is syntactically valid Python**

Run: `python -c "import ast; ast.parse(open('adapt/cma_layered_continual.py').read()); print('OK')"`
Expected: `OK`

---

### Task 2: Register the method in `adapt/__init__.py`

**Files:**
- Modify: `adapt/__init__.py`

- [ ] **Step 1: Add the import**

Find the line `from .cma_proto_continual import CMAProtoContinual` and add a new import directly below it.

```python
from .cma_proto_continual import CMAProtoContinual
from .cma_layered_continual import CMALayeredContinual
```

- [ ] **Step 2: Add the entry to `METHOD_CLASSES`**

Find `'cma_proto_continual': CMAProtoContinual,` in `METHOD_CLASSES` and append the new entry right after it. The relevant section should read:

```python
    # Continual TTA with CMA + source/target prototype memory bank (proposed D1+D2)
    'cma_proto_continual': CMAProtoContinual,
    # Continual TTA with CMA + layer-stratified stochastic restoration (Direction A)
    'cma_layered_continual': CMALayeredContinual,
}
```

- [ ] **Step 3: Verify the import resolves**

Run: `python -c "from adapt import METHOD_CLASSES; assert 'cma_layered_continual' in METHOD_CLASSES; print(METHOD_CLASSES['cma_layered_continual'].__name__)"`
Expected: `CMALayeredContinual`

---

### Task 3: Add CLI arguments in `main_continual.py`

**Files:**
- Modify: `main_continual.py` — `add_method_specific_args()` function

- [ ] **Step 1: Insert the new branch**

Find the existing `cma_proto_continual` branch in `add_method_specific_args()` (currently around lines 197–212) and add the new branch directly below it, before the `dpcore` branch:

```python
    # --- CMA-Layered-Continual (CMA + layer-stratified restoration, Direction A) ---
    elif method == 'cma_layered_continual':
        parser.add_argument('--top_k_percent', type=float, default=0.2,
                            help='Top-K%% confidence mask for the CMA loss (default 0.2)')
        parser.add_argument('--early_rst', type=float, default=0.001,
                            help='Stochastic restore probability for early blocks '
                                 '([0, early_cutoff) and ln_pre) (default 0.001)')
        parser.add_argument('--mid_rst', type=float, default=0.01,
                            help='Stochastic restore probability for mid blocks '
                                 '([early_cutoff, late_cutoff)) (default 0.01)')
        parser.add_argument('--late_rst', type=float, default=0.05,
                            help='Stochastic restore probability for late blocks '
                                 '([late_cutoff, num_blocks) and ln_post) (default 0.05)')
        parser.add_argument('--early_cutoff', type=int, default=8,
                            help='Block index boundary: blocks [0, early_cutoff) are early')
        parser.add_argument('--late_cutoff', type=int, default=16,
                            help='Block index boundary: blocks [early_cutoff, late_cutoff) are mid; '
                                 '[late_cutoff, num_blocks) are late')
```

- [ ] **Step 2: Verify the parser builds**

Run:
```bash
python -c "
import sys; sys.argv = ['main_continual.py', '--method', 'cma_layered_continual']
from main_continual import argparser, add_method_specific_args
p = argparser()
p = add_method_specific_args(p, 'cma_layered_continual')
args, _ = p.parse_known_args(['--method', 'cma_layered_continual', '--corruptions_list', 'fog'])
print(args.early_rst, args.mid_rst, args.late_rst, args.early_cutoff, args.late_cutoff)
"
```
Expected: `0.001 0.01 0.05 8 16`

---

### Task 4: Pure-logic sanity check for layer classification (no GPU)

**Files:** none modified — inline check.

- [ ] **Step 1: Run the classification check**

This validates the regex + cutoff logic against hand-constructed parameter names that match what `collect_ln_params` will emit.

```bash
python << 'EOF'
import re
BLOCK_RE = re.compile(r'resblocks\.(\d+)\.')

def classify(name, e_rst=0.001, m_rst=0.01, l_rst=0.05, e_cut=8, l_cut=16):
    if 'ln_pre' in name:  return e_rst
    if 'ln_post' in name: return l_rst
    m = BLOCK_RE.search(name)
    if m is None: return 0.0
    idx = int(m.group(1))
    if idx < e_cut: return e_rst
    if idx < l_cut: return m_rst
    return l_rst

cases = [
    ('visual.ln_pre.weight',                          0.001, 'early'),
    ('visual.ln_pre.bias',                            0.001, 'early'),
    ('visual.transformer.resblocks.0.ln_1.weight',    0.001, 'early'),
    ('visual.transformer.resblocks.7.ln_2.bias',      0.001, 'early (last early block)'),
    ('visual.transformer.resblocks.8.ln_1.weight',    0.01,  'mid (first mid block)'),
    ('visual.transformer.resblocks.15.ln_2.bias',     0.01,  'mid (last mid block)'),
    ('visual.transformer.resblocks.16.ln_1.weight',   0.05,  'late (first late block)'),
    ('visual.transformer.resblocks.23.ln_2.bias',     0.05,  'late (last late block)'),
    ('visual.ln_post.weight',                         0.05,  'late'),
    ('visual.ln_post.bias',                           0.05,  'late'),
]
for name, expected, label in cases:
    got = classify(name)
    assert got == expected, f"{name}: got {got}, expected {expected} ({label})"
    print(f"OK  {label:30s}  {name}  ->  {got}")
print("\nAll layer classification cases passed.")
EOF
```
Expected: 10 lines of `OK ...` followed by `All layer classification cases passed.`

---

### Task 5: Create the bash run script

**Files:**
- Create: `bash/ACDC_10_round/cma_layered_continual.sh`

- [ ] **Step 1: Write the script**

```bash
#!/bin/bash
# CMA-Layered-Continual on ACDC (CTTA, 150 rounds).
# Cross-Modal Alignment loss + layer-stratified stochastic restoration:
# early ViT blocks adapt freely, late blocks are strongly anchored to source.
# See docs/cma_layered_continual_spec.md for the full design.

# GPU Configuration
GPU_ID=3

# Dataset Configuration
DATASET=ACDCDataset
DATA_DIR="data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=4

# Method Configuration
METHOD="cma_layered_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# Hyperparameters (shared with CMA baseline for clean comparison)
BATCH_SIZE=1
LR=0.00001
STEPS=1
TOP_K_PERCENT=0.2

# Layer-stratified restoration (proposal_after_cma.md §1.4 defaults)
# - all 0   -> equivalent to cma_continual (no restoration)
# - all 1   -> LN params fully frozen
# - any 0   -> that group is unrestored
EARLY_RST=0.001
MID_RST=0.01
LATE_RST=0.05

# Group cutoffs for ViT-L/14 (24 blocks)
# blocks [0, EARLY_CUTOFF)            + ln_pre  -> early
# blocks [EARLY_CUTOFF, LATE_CUTOFF)            -> mid
# blocks [LATE_CUTOFF, 24)            + ln_post -> late
EARLY_CUTOFF=8
LATE_CUTOFF=16

# Experiment (match CMA baseline run length for fair comparison)
CONTINUAL_ROUNDS=150

# Output
SAVE_DIR="save/${DATASET}/${METHOD}_step_${STEPS}/"

# Run
CUDA_VISIBLE_DEVICES=$GPU_ID python main_continual.py \
                        --adapt \
                        --method $METHOD \
                        --ovss_type $OVSS_TYPE \
                        --ovss_backbone $OVSS_BACKBONE \
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
                        --top_k_percent $TOP_K_PERCENT \
                        --early_rst $EARLY_RST \
                        --mid_rst $MID_RST \
                        --late_rst $LATE_RST \
                        --early_cutoff $EARLY_CUTOFF \
                        --late_cutoff $LATE_CUTOFF \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
```

- [ ] **Step 2: Make it executable and verify it parses**

```bash
chmod +x bash/ACDC_10_round/cma_layered_continual.sh
bash -n bash/ACDC_10_round/cma_layered_continual.sh && echo "Bash syntax OK"
```
Expected: `Bash syntax OK`

---

### Task 6: Register the method in `parse_acdc_results.py`

**Files:**
- Modify: `parse_acdc_results.py` — `METHODS` list

- [ ] **Step 1: Append the new entry**

Find the `METHODS = [...]` list and add a new tuple after the `CMA-Proto-continual` line. The relevant section becomes:

```python
    ("CMA-continual",   "cma_continual_step_1",  False),
    ("CMA-Proto-continual", "cma_proto_continual_step_1", False),
    ("CMA-Layered-continual", "cma_layered_continual_step_1", False),
    ("MLMP (episodic)", "mlmp_batch_1",           True),
```

The save-dir string `cma_layered_continual_step_1` must match the `SAVE_DIR` produced by the bash script (which formats as `${METHOD}_step_${STEPS}` with `STEPS=1`).

- [ ] **Step 2: Verify the file still imports**

Run: `python -c "import ast; ast.parse(open('parse_acdc_results.py').read()); print('OK')"`
Expected: `OK`

---

### Task 7: Smoke test — debug-mode end-to-end run

**Goal:** verify model construction, layer-group reporting, forward, loss, restoration, and metric writing all work together on real GPU before launching the 150-round experiment.

- [ ] **Step 1: Run the debug variant (5 batches per condition, 1 round)**

This uses the existing `--debug` flag and overrides `--continual_rounds 1` so it finishes in a few minutes.

```bash
CUDA_VISIBLE_DEVICES=3 python main_continual.py \
    --adapt \
    --method cma_layered_continual \
    --ovss_type naclip --ovss_backbone ViT-L/14 \
    --dataset ACDCDataset --data_dir data/ACDC/ \
    --init_resize 1120 560 --patch_size 224 224 --patch_stride 112 \
    --corruptions_list fog night rain snow --workers 4 \
    --lr 0.00001 --steps 1 --batch_size 1 \
    --continual_rounds 1 --seed 0 \
    --top_k_percent 0.2 \
    --early_rst 0.001 --mid_rst 0.01 --late_rst 0.05 \
    --early_cutoff 8 --late_cutoff 16 \
    --save_dir save/_debug_cma_layered/ \
    --class_extensions \
    --debug
```

Expected output — verify each of these:
- `+++ CMA-Layered: cutoffs=(8, 16), rsts=(0.001, 0.01, 0.05)` is printed once.
- `+++ LN params/group: early=..., mid=..., late=..., total=100` for ViT-L/14 — totals 100 (24 blocks × 2 LN × 2 params + ln_pre × 2 + ln_post × 2 = 100). early should equal `2 + 8*2*2 = 34`, mid = `8*2*2 = 32`, late = `8*2*2 + 2 = 34`.
- 4 condition lines `Rd 01 | fog: mIoU=...` etc. with non-NaN mIoU values.
- File `save/_debug_cma_layered/results_all_rounds.txt` exists and has one row.

- [ ] **Step 2: Inspect the debug output**

Run: `cat save/_debug_cma_layered/results_all_rounds.txt`
Expected: a header row plus a single `Round 01, ..., ..., ..., ..., ...` row with finite numeric mIoU values.

- [ ] **Step 3: Clean up the debug save directory**

Run: `rm -rf save/_debug_cma_layered/`
Expected: no error.

---

### Task 8: Commit the implementation

- [ ] **Step 1: Verify git status reflects only the intended files**

Run: `git status --short`
Expected — exactly these lines:
```
 M adapt/__init__.py
 M main_continual.py
 M parse_acdc_results.py
?? adapt/cma_layered_continual.py
?? bash/ACDC_10_round/cma_layered_continual.sh
?? docs/cma_layered_continual_plan.md
```

(`docs/cma_layered_continual_plan.md` is this plan file. `proposal_after_cma.md` may also be untracked from a prior session — leave it alone unless asked.)

- [ ] **Step 2: Stage the implementation files**

```bash
git add adapt/cma_layered_continual.py \
        adapt/__init__.py \
        main_continual.py \
        parse_acdc_results.py \
        bash/ACDC_10_round/cma_layered_continual.sh \
        docs/cma_layered_continual_plan.md
```

- [ ] **Step 3: Create the commit**

```bash
git commit -m "$(cat <<'EOF'
Implement CMA-Layered-Continual (Direction A minimal)

CMA loss + layer-stratified stochastic restoration on top of
cma_continual. Early ViT blocks adapt freely (rst=0.001), late
blocks strongly anchored (rst=0.05). All rates and cutoffs are
exposed as CLI/bash hyperparameters for ablation.

Adds adapt/cma_layered_continual.py, registers the method in
adapt/__init__.py and main_continual.py, ships
bash/ACDC_10_round/cma_layered_continual.sh and updates
parse_acdc_results.py to include the new method in the table.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: Verify the commit landed**

Run: `git log --oneline -1`
Expected: a line beginning with the new commit's short hash and `Implement CMA-Layered-Continual (Direction A minimal)`.

---

## Self-Review Checklist (run before declaring complete)

- **Spec §3 (algorithm)** → Task 1 implements `_get_restoration_rate`, `_stochastic_layered_restore`, and the `perform_adaptation` ordering (forward → loss → backward → step → zero_grad → restore).
- **Spec §4 (file layout)** → all five files in the table are touched by Tasks 1–6.
- **Spec §5 (public API)** → constructor signature, `adapt`/`continual_adapt`/`evaluate`/`reset` all present in Task 1.
- **Spec §6 (hyperparameters)** → defaults match Task 3 CLI defaults and Task 5 bash defaults (`top_k=0.2`, `early_rst=0.001`, `mid_rst=0.01`, `late_rst=0.05`, `early_cutoff=8`, `late_cutoff=16`, `continual_rounds=150`).
- **Spec §6.1 (degradation paths)** → all-rates-0 path verified inherently by `if rst <= 0.0: continue` short-circuit; layer-classification edge cases tested in Task 4.
- **Spec §7 (CLI)** → Task 3 adds exactly the listed arguments with matching defaults and help strings.
- **Spec §8 (bash script)** → Task 5 produces the listed script.
- **Spec §9 (sanity checks)** → check #5 (layer classification) is Task 4; checks #1, #2, #6 are covered by Task 7's debug run output validation; checks #3 and #4 (`all rates 0` and `all rates 1`) are degradation paths and are not load-bearing for the experiment — defer unless smoke fails.
- **Spec §10 (success criteria)** → not validated in this plan; that is the experiment's job after the 150-round run.

If any item is unchecked, fix the plan before proceeding to execution.
