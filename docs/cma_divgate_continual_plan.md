# CMA-DivGate-Continual Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `CMADivGateContinual` (Direction B) per [docs/cma_divgate_continual_spec.md](cma_divgate_continual_spec.md): CMA loss + Buffer-N marginal-entropy gate that switches stochastic restoration between aggressive / cautious / brake modes. Ship a runnable ACDC bash script.

**Architecture:** Standalone adapter class in `adapt/cma_divgate_continual.py`. CMA forward and loss tensor ops are copied verbatim from `adapt/cma_continual.py` with one extra line in the `no_grad` block to buffer per-batch marginal class probabilities. After every `optimizer.step()`, a single CoTTA-style flat stochastic restore is applied with `current_rst` set by the gate. Every `monitor_interval` batches the gate aggregates the buffer, computes `H_margin`, picks a new mode, and clears state.

**Tech Stack:** PyTorch, NA-CLIP ViT-L/14, Adam over visual-encoder LayerNorm γ/β only. No new dependencies.

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `adapt/cma_divgate_continual.py` | NEW | `class CMADivGateContinual` |
| `adapt/__init__.py` | MODIFY | Register the new method |
| `main_continual.py` | MODIFY | New `add_method_specific_args` branch |
| `bash/ACDC_10_round/cma_divgate_continual.sh` | NEW | Runnable experiment |
| `parse_acdc_results.py` | MODIFY | Append to `METHODS` |

---

### Task 1: Create the adapter class

**Files:** Create `adapt/cma_divgate_continual.py`.

- [ ] **Step 1: Write the full file** (full content shown — copy verbatim).

```python
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from ovss import load_ovss
from utils.misc import load_prompts_from_yaml, print_clip_parameters, print_optimizer_parameters

REFERENCE_PROMPT = 'a photo of a {}'

MODE_AGGRESSIVE = 'aggressive'
MODE_CAUTIOUS   = 'cautious'
MODE_BRAKE      = 'brake'


class CMADivGateContinual:
    """
    CMA-DivGate-Continual: Cross-Modal Alignment loss + diversity-gated
    stochastic restoration (CTTA, no reset).

    Every `monitor_interval` adapt batches, the marginal-class-entropy
    H_margin is computed from a buffer of per-batch marginal distributions
    and used to pick one of three modes:

        H_margin >= h_threshold              -> "aggressive" (rst = 0)
        h_warning <= H_margin < h_threshold  -> "cautious"   (rst = cautious_rst)
        H_margin < h_warning                  -> "brake"      (rst = brake_rst)

    The chosen mode persists until the next H_margin re-evaluation. The
    restoration is flat (single rate applied to every visual-encoder LN
    parameter), unlike the layer-stratified variant.

    See docs/cma_divgate_continual_spec.md for the full design.
    """

    def __init__(self, ovss_type, ovss_backbone, lr, classes, steps=1,
                 top_k_percent=0.2,
                 h_threshold=1.8, h_warning=1.2,
                 monitor_interval=50,
                 cautious_rst=0.005, brake_rst=0.05,
                 prompt_dir=None,
                 runtime_calculation=False, device='cpu'):
        self.ovss_type = ovss_type
        self.ovss_backbone = ovss_backbone
        self.lr = lr
        self.steps = steps
        self.top_k_percent = float(top_k_percent)
        if not (0.0 < self.top_k_percent <= 1.0):
            raise ValueError(f"top_k_percent must be in (0, 1], got {self.top_k_percent}")

        self.h_threshold = float(h_threshold)
        self.h_warning = float(h_warning)
        if self.h_warning > self.h_threshold:
            raise ValueError(
                f"h_warning ({self.h_warning}) must be <= h_threshold ({self.h_threshold})"
            )
        self.monitor_interval = int(monitor_interval)
        if self.monitor_interval < 1:
            raise ValueError(f"monitor_interval must be >= 1, got {self.monitor_interval}")
        self.cautious_rst = float(cautious_rst)
        self.brake_rst = float(brake_rst)
        for n, v in [('cautious_rst', self.cautious_rst), ('brake_rst', self.brake_rst)]:
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{n} must be in [0, 1], got {v}")

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
        print(f"+++ CMA-DivGate: thresholds=(warn {self.h_warning}, agg {self.h_threshold}), "
              f"monitor_interval={self.monitor_interval}, "
              f"rsts=(cautious {self.cautious_rst}, brake {self.brake_rst})")
        print(f"+++ Initial mode: {MODE_AGGRESSIVE} (rst=0); buffer fills before first re-evaluation")

        # ---------- Optimizer ----------
        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0)
        print_optimizer_parameters(self.optimizer, self.model)

        # ---------- Source state snapshot ----------
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(
            self.model, self.optimizer
        )

        # ---------- Text features ----------
        with torch.no_grad():
            self.text_x = self.extract_text_embeddings(
                self.classes, self.prompt_templates, average=False
            ).squeeze()

        # ---------- Gate state ----------
        self.marginal_buf = []          # list[Tensor (C,)]
        self.batch_count = 0            # batches in the current monitor window
        self.total_batches = 0          # cumulative, for diagnostic logs
        self.current_mode = MODE_AGGRESSIVE
        self.current_rst = 0.0

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
            if self.current_rst > 0.0:
                self._stochastic_restore_flat(self.current_rst)
        self.batch_count += 1
        self.total_batches += 1
        if self.batch_count >= self.monitor_interval:
            self._update_mode()
        if self.runtime:
            self.adapt_times.append(time.time() - t1)
        return loss_report

    def cma_loss(self, logits, image_features, text_features):
        """Identical to CMAContinual.cma_loss + buffer per-batch marginal."""
        avg_logits = logits.mean(dim=0)
        avg_text = text_features.mean(dim=0)
        avg_text = avg_text / avg_text.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        with torch.no_grad():
            probs = avg_logits.softmax(dim=1)
            confidence, pred_cls = probs.max(dim=1)
            n_total = confidence.numel()
            k = max(1, int(round(n_total * self.top_k_percent)))
            threshold = torch.topk(confidence.flatten(), k, sorted=True).values[-1]
            mask = confidence >= threshold

            # Buffer per-batch marginal class distribution for the gate.
            # Mean over (B, w, h) -> (C,). Detach + cpu to keep memory tiny.
            self.marginal_buf.append(probs.mean(dim=[0, 2, 3]).detach().float().cpu())

        B, _, w, h = avg_logits.shape
        patch_feats = image_features[:, 1:, :]
        D = patch_feats.shape[-1]
        vis_feat = patch_feats.reshape(B, w, h, D)

        text_target = avg_text[pred_cls]
        cos_sim = (vis_feat * text_target).sum(dim=-1)
        return -cos_sim[mask].mean()

    # ===========================================================
    # Diversity gate
    # ===========================================================

    @staticmethod
    def _pick_mode(h_margin, h_threshold, h_warning):
        if h_margin >= h_threshold:
            return MODE_AGGRESSIVE
        if h_margin >= h_warning:
            return MODE_CAUTIOUS
        return MODE_BRAKE

    def _mode_to_rst(self, mode):
        if mode == MODE_AGGRESSIVE:
            return 0.0
        if mode == MODE_CAUTIOUS:
            return self.cautious_rst
        if mode == MODE_BRAKE:
            return self.brake_rst
        raise ValueError(f"unknown mode {mode!r}")

    @torch.no_grad()
    def _update_mode(self):
        """Aggregate buffered marginals, compute H_margin, switch mode."""
        if len(self.marginal_buf) == 0:
            self.batch_count = 0
            return
        agg = torch.stack(self.marginal_buf, dim=0).mean(dim=0)        # (C,)
        agg = agg / agg.sum().clamp(min=1e-8)
        h_margin = -(agg * agg.clamp(min=1e-12).log()).sum().item()

        new_mode = self._pick_mode(h_margin, self.h_threshold, self.h_warning)
        if new_mode != self.current_mode:
            print(f"[DivGate] B{self.total_batches}: H_margin={h_margin:.3f}  "
                  f"{self.current_mode} -> {new_mode}")
        self.current_mode = new_mode
        self.current_rst = self._mode_to_rst(new_mode)

        self.marginal_buf.clear()
        self.batch_count = 0

    @torch.no_grad()
    def _stochastic_restore_flat(self, rst):
        """CoTTA-style flat stochastic restore on visual-encoder LN params."""
        for name, p in self.named_ln_params:
            mask = (torch.rand(p.shape, device=p.device) < rst).to(p.dtype)
            src = self.model_state[name].to(p.device, dtype=p.dtype)
            p.data.mul_(1.0 - mask).add_(src * mask)

    # ===========================================================
    # Shared helpers (verbatim from CMAContinual)
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

- [ ] **Step 2: Verify Python syntax**

Run: `python -c "import ast; ast.parse(open('adapt/cma_divgate_continual.py').read()); print('OK')"`
Expected: `OK`

---

### Task 2: Register the method in `adapt/__init__.py`

- [ ] **Step 1: Add the import** below the `cma_layered_continual` import line.

```python
from .cma_layered_continual import CMALayeredContinual
from .cma_divgate_continual import CMADivGateContinual
```

- [ ] **Step 2: Append the entry to `METHOD_CLASSES`** below the layered entry.

```python
    'cma_layered_continual': CMALayeredContinual,
    # Continual TTA with CMA + diversity-gated stochastic restoration (Direction B)
    'cma_divgate_continual': CMADivGateContinual,
}
```

- [ ] **Step 3: Verify registration loads**

Run (with MLMP env active): `python -c "from adapt import METHOD_CLASSES; print(METHOD_CLASSES['cma_divgate_continual'].__name__)"`
Expected: `CMADivGateContinual`

---

### Task 3: Add CLI args in `main_continual.py`

- [ ] **Step 1: Add the new branch** below the `cma_layered_continual` branch and above `dpcore`.

```python
    # --- CMA-DivGate-Continual (CMA + diversity gate, Direction B) ---
    elif method == 'cma_divgate_continual':
        parser.add_argument('--top_k_percent', type=float, default=0.2,
                            help='Top-K%% confidence mask for the CMA loss (default 0.2)')
        parser.add_argument('--h_threshold', type=float, default=1.8,
                            help='H_margin >= this -> aggressive mode (rst=0)')
        parser.add_argument('--h_warning', type=float, default=1.2,
                            help='h_warning <= H_margin < h_threshold -> cautious; '
                                 '< h_warning -> brake')
        parser.add_argument('--monitor_interval', type=int, default=50,
                            help='Batches between H_margin re-evaluations (default 50)')
        parser.add_argument('--cautious_rst', type=float, default=0.005,
                            help='Stochastic restore probability in cautious mode')
        parser.add_argument('--brake_rst', type=float, default=0.05,
                            help='Stochastic restore probability in brake mode')
```

- [ ] **Step 2: Verify parser**

Run:
```bash
python -c "
from main_continual import argparser, add_method_specific_args
p = argparser()
p = add_method_specific_args(p, 'cma_divgate_continual')
args, _ = p.parse_known_args(['--method','cma_divgate_continual','--corruptions_list','fog'])
print(args.h_threshold, args.h_warning, args.monitor_interval, args.cautious_rst, args.brake_rst, args.top_k_percent)
"
```
Expected: `1.8 1.2 50 0.005 0.05 0.2`

---

### Task 4: Pure-logic sanity checks

- [ ] **Step 1: Run the inline test**

```bash
python << 'EOF'
import math, torch

def H(p):
    p = p / p.sum().clamp(min=1e-8)
    return -(p * p.clamp(min=1e-12).log()).sum().item()

def pick(h, t=1.8, w=1.2):
    if h >= t: return 'aggressive'
    if h >= w: return 'cautious'
    return 'brake'

# 1. uniform 19-class
u = torch.full((19,), 1/19)
assert abs(H(u) - math.log(19)) < 1e-6, f"uniform: got {H(u)}, expected {math.log(19)}"
print(f"OK uniform-19          H={H(u):.4f} ~ log(19)={math.log(19):.4f}")

# 2. one-hot
oh = torch.zeros(19); oh[0] = 1.0
assert H(oh) < 1e-6, f"one-hot: got {H(oh)}, expected ~0"
print(f"OK one-hot             H={H(oh):.6f} ~ 0")

# 3. 50/50 two-class
half = torch.zeros(19); half[0] = 0.5; half[1] = 0.5
assert abs(H(half) - math.log(2)) < 1e-6, f"50/50: got {H(half)}, expected {math.log(2)}"
print(f"OK 50/50               H={H(half):.4f} ~ log(2)={math.log(2):.4f}")

# 4. mode picker
assert pick(2.0) == 'aggressive', f"mode 2.0: got {pick(2.0)}"
assert pick(1.5) == 'cautious',   f"mode 1.5: got {pick(1.5)}"
assert pick(0.5) == 'brake',      f"mode 0.5: got {pick(0.5)}"
assert pick(1.8) == 'aggressive', "boundary 1.8 -> aggressive"
assert pick(1.2) == 'cautious',   "boundary 1.2 -> cautious"
assert pick(1.19) == 'brake',     "just under 1.2 -> brake"
print("OK mode picker         all 6 boundary cases passed")

print("\nAll sanity cases passed.")
EOF
```
Expected: 4 `OK ...` lines + `All sanity cases passed.`

---

### Task 5: Create `bash/ACDC_10_round/cma_divgate_continual.sh`

- [ ] **Step 1: Write the script**

```bash
#!/bin/bash
# CMA-DivGate-Continual on ACDC (CTTA, 150 rounds).
# Cross-Modal Alignment loss + diversity-gated stochastic restoration:
# H_margin (marginal class entropy) is computed every MONITOR_INTERVAL
# batches; mode switches between aggressive (rst=0), cautious, brake.
# See docs/cma_divgate_continual_spec.md for the full design.

# GPU Configuration
GPU_ID=3

# Dataset Configuration
DATASET=ACDCDataset
DATA_DIR="data/ACDC/"
INIT_RESIZE="1120 560"
CONDITIONS="fog night rain snow"
WORKERS=4

# Method Configuration
METHOD="cma_divgate_continual"
OVSS_TYPE="naclip"
OVSS_BACKBONE="ViT-L/14"

# Hyperparameters (shared with CMA baseline for clean comparison)
BATCH_SIZE=1
LR=0.00001
STEPS=1
TOP_K_PERCENT=0.2

# Diversity gate (proposal_after_cma.md §2.3 defaults)
H_THRESHOLD=1.8       # H_margin >= this  -> aggressive (rst=0)
H_WARNING=1.2         # h_warning <= H < h_threshold -> cautious
MONITOR_INTERVAL=50   # batches between H_margin re-evaluations
CAUTIOUS_RST=0.005
BRAKE_RST=0.05

CONTINUAL_ROUNDS=150
SAVE_DIR="save/${DATASET}/${METHOD}_step_${STEPS}/"

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
                        --h_threshold $H_THRESHOLD \
                        --h_warning $H_WARNING \
                        --monitor_interval $MONITOR_INTERVAL \
                        --cautious_rst $CAUTIOUS_RST \
                        --brake_rst $BRAKE_RST \
                        \
                        --save_dir $SAVE_DIR \
                        --class_extensions
```

- [ ] **Step 2: chmod and bash syntax check**

```bash
chmod +x bash/ACDC_10_round/cma_divgate_continual.sh
bash -n bash/ACDC_10_round/cma_divgate_continual.sh && echo "Bash syntax OK"
```
Expected: `Bash syntax OK`

---

### Task 6: Update `parse_acdc_results.py`

- [ ] **Step 1: Append the entry** below the `CMA-Layered-continual` line.

```python
    ("CMA-Layered-continual", "cma_layered_continual_step_1", False),
    ("CMA-DivGate-continual", "cma_divgate_continual_step_1", False),
    ("MLMP (episodic)", "mlmp_batch_1",           True),
```

- [ ] **Step 2: Verify file parses**

Run: `python -c "import ast; ast.parse(open('parse_acdc_results.py').read()); print('OK')"`
Expected: `OK`

---

### Task 7: Smoke test

- [ ] **Step 1: Debug-mode run with `monitor_interval=5`** so the gate fires at least once.

```bash
CUDA_VISIBLE_DEVICES=3 python main_continual.py \
    --adapt \
    --method cma_divgate_continual \
    --ovss_type naclip --ovss_backbone ViT-L/14 \
    --dataset ACDCDataset --data_dir data/ACDC/ \
    --init_resize 1120 560 --patch_size 224 224 --patch_stride 112 \
    --corruptions_list fog night rain snow --workers 4 \
    --lr 0.00001 --steps 1 --batch_size 1 \
    --continual_rounds 1 --seed 0 \
    --top_k_percent 0.2 \
    --h_threshold 1.8 --h_warning 1.2 \
    --monitor_interval 5 \
    --cautious_rst 0.005 --brake_rst 0.05 \
    --save_dir save/_debug_cma_divgate/ \
    --class_extensions \
    --debug
```

Expected:
- `+++ CMA-DivGate: thresholds=...` printed once
- `+++ Initial mode: aggressive (rst=0); buffer fills before first re-evaluation`
- 4 `Rd 01 | <cond>: mIoU=...` lines with finite mIoU values
- `save/_debug_cma_divgate/results_all_rounds.txt` exists with one row
- May or may not see `[DivGate] B...` lines depending on H_margin values; absence of mode change just means H_margin stayed in the initial bucket — that's still a successful smoke test.

- [ ] **Step 2: Inspect output and clean up**

```bash
cat save/_debug_cma_divgate/results_all_rounds.txt
rm -rf save/_debug_cma_divgate/
```

---

### Task 8: Commit

- [ ] **Step 1: Verify git status**

`git status --short` should show:
```
 M adapt/__init__.py
 M main_continual.py
 M parse_acdc_results.py
?? adapt/cma_divgate_continual.py
?? bash/ACDC_10_round/cma_divgate_continual.sh
?? docs/cma_divgate_continual_plan.md
```

- [ ] **Step 2: Stage and commit**

```bash
git add adapt/cma_divgate_continual.py \
        adapt/__init__.py \
        main_continual.py \
        parse_acdc_results.py \
        bash/ACDC_10_round/cma_divgate_continual.sh \
        docs/cma_divgate_continual_plan.md
git commit -m "$(cat <<'EOF'
Implement CMA-DivGate-Continual (Direction B)

CMA loss + Buffer-N marginal-class-entropy gate. Every
monitor_interval (default 50) batches H_margin is computed from
buffered per-batch marginals; the gate switches between aggressive
(rst=0), cautious (rst=0.005), and brake (rst=0.05) modes. Mode
transitions are logged. Restoration is flat (CoTTA-style), not
layer-stratified.

Adds adapt/cma_divgate_continual.py, registers the method in
adapt/__init__.py and main_continual.py, ships
bash/ACDC_10_round/cma_divgate_continual.sh, and updates
parse_acdc_results.py.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review Checklist

- **Spec §2.1 (H_margin formula)** → Task 1 (`cma_loss` no_grad block + `_update_mode`); Task 4 (formula correctness)
- **Spec §2.2 (Buffer-N aggregation)** → Task 1 (`marginal_buf`, `_update_mode` aggregation step)
- **Spec §3 (3-tier mode)** → Task 1 (`_pick_mode`, `_mode_to_rst`, `current_mode`/`current_rst` state)
- **Spec §4 (adapt step ordering)** → Task 1 (`perform_adaptation`)
- **Spec §5 (public API)** → Task 1 (constructor + `adapt`/`continual_adapt`/`evaluate`/`reset`)
- **Spec §6 (hyperparameters)** → Task 3 (CLI defaults match), Task 5 (bash defaults match)
- **Spec §7 (CLI)** → Task 3
- **Spec §8 (bash)** → Task 5
- **Spec §9 (sanity checks)** → checks #1–#4 covered by Task 4 inline test; #5 (LN count) inherited from layered method (same `collect_ln_params`); #6 covered by Task 7 smoke run
- **Spec §10 (success criteria)** → not validated by plan; experiment-time concern
