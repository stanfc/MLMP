# GDG-PA — candidate mathematical derivations for the paper

Scope: GDG-PA on OVSS (NA-CLIP ViT-L/14, LayerNorm-only, no-reset CTTA).
Purpose: identify which parts of the method admit **real** derivations, and mark
clearly what must stay an empirical observation.

Notation used throughout:

| symbol | meaning |
|---|---|
| $C$ | number of semantic classes (19 on ACDC/Cityscapes) |
| $\theta_t$ | trainable LayerNorm parameters (visual encoder) at step $t$ |
| $\theta_a$ | restore anchor (shallow: lag-deque snapshot; deep: permanent best) |
| $g_t=\nabla_\theta L(\theta_t)$ | adaptation-loss gradient (DeYO+MLMP) |
| $\eta$ | learning rate; $r$ | per-element stochastic restore rate (`base_rst`) |
| $\bar p$ | window-averaged predicted class marginal, $\bar p\in\Delta^{C-1}$ |
| $H_{\mathrm{margin}}$ | $H(\bar p)=-\sum_c \bar p_c\log\bar p_c$ |

---

## A. $H_{\mathrm{margin}}$ certifies an upper bound on mIoU  *(provable — strongest)*

### A.1 Effective class count
The perplexity of the window-averaged marginal,
$$k_{\mathrm{eff}} \;:=\; \exp\!\big(H_{\mathrm{margin}}\big) \;=\; \exp\big(H(\bar p)\big),$$
is the standard *effective support size* of $\bar p$: it equals $C$ for a uniform
marginal and $1$ when all mass sits on a single class.

### A.2 The bound
Let $S=\{c:\bar p_c>0\}$ be the set of classes the model actually predicts over the
window. For any class $c\notin S$ the model emits no pixels of class $c$, so its
intersection is empty and $\mathrm{IoU}_c=0$. Hence

$$\mathrm{mIoU} \;=\; \frac1C\sum_{c=1}^{C}\mathrm{IoU}_c \;=\;\frac1C\sum_{c\in S}\mathrm{IoU}_c \;\le\; \frac{|S|}{C}.$$

Replacing the hard support $|S|$ by the entropy-based effective support gives the
usable form

$$\boxed{\;\mathrm{mIoU}\;\lesssim\;\frac{k_{\mathrm{eff}}}{C}\;=\;\frac{e^{H_{\mathrm{margin}}}}{C}\;}$$

### A.3 Why this matters for the paper
It upgrades $H_{\mathrm{margin}}$ from a heuristic monitor to a **performance
certificate**: a drop in $H_{\mathrm{margin}}$ *mathematically caps* attainable mIoU,
independent of how confident the model is. This is exactly the mode our figures show
(`p8_hmargin_collapse.png`: $H_{\mathrm{margin}}$ and mIoU fall together).

**Caveat to state honestly.** $|S|\le k_{\mathrm{eff}}$ is not an identity — $k_{\mathrm{eff}}$
is a soft (entropy) surrogate for $|S|$; equality holds for a uniform marginal on $S$.
Write the boxed form as "$\lesssim$" or state the exact bound with $|S|$ and use
$k_{\mathrm{eff}}$ as its smooth estimator.

---

## B. Stochastic restore contracts drift; $r$ caps the steady-state distance  *(provable — most actionable)*

### B.1 The update
GDG-PA alternates a gradient step with an element-wise stochastic restore,
$$\theta_{t+1} \;=\; (1-m_t)\odot(\theta_t-\eta g_t)\;+\;m_t\odot\theta_a,
\qquad m_t\stackrel{\text{iid}}{\sim}\mathrm{Bernoulli}(r).$$
Taking expectations over the mask,
$$\mathbb{E}[\theta_{t+1}] \;=\; (1-r)\,(\theta_t-\eta g_t)\;+\;r\,\theta_a .$$

### B.2 Contraction
Let $d_t:=\lVert\theta_t-\theta_a\rVert$ and assume bounded gradients $\lVert g_t\rVert\le G$.
Subtracting $\theta_a$ from both sides and applying the triangle inequality,
$$d_{t+1}\;\le\;(1-r)\,d_t\;+\;(1-r)\,\eta G\;\le\;(1-r)\,d_t+\eta G .$$
Unrolling the recursion,
$$d_t\;\le\;(1-r)^t d_0\;+\;\eta G\sum_{i=0}^{t-1}(1-r)^i,$$
and letting $t\to\infty$ (for $0<r\le1$) the geometric series converges:

$$\boxed{\;d_\infty\;\le\;\frac{\eta G}{r}\;}$$

### B.3 Three consequences to write up
1. **No restore diverges.** At $r=0$ the bound is vacuous ($d_\infty\le\infty$) and the
   recursion becomes $d_{t+1}\le d_t+\eta G$, i.e. drift can grow linearly in $t$.
   This is the formal statement of the baseline collapse we observe.
2. **$r$ is a drift dial.** To guarantee $d_\infty\le\delta$ it suffices to take
   $r\ge \eta G/\delta$ — a principled way to pick `base_rst` from the learning rate
   and the observed gradient scale (both are logged in `gate_log.csv`).
3. **Deep restore re-anchors the bound.** Switching $\theta_a$ to the permanent best
   (grad-norm minimum) state does not change the *form* of the bound; it changes the
   point the bound is centred on, so the guarantee is stated w.r.t. a *healthy* state
   rather than a drifted one. This is precisely the role of the PA (permanent anchor).

### B.4 This also explains the negative results
A loss-side regularizer (our `divloss`, `textalign`, `repel`, …) adds a term to $g_t$
but leaves the recursion $d_{t+1}\le d_t+\eta G'$ **unbounded**; it can shrink $G$ but
cannot introduce the contraction factor $(1-r)$. This is a clean formal account of why
every loss-form defence we tried failed while restore-based GDG-PA holds
(0702 report §7; `textalign`/`ratchet`/`repel`/`srcdistill`/`taconsensus` all ≈ base).

---

## C. Why anchor at the grad-norm minimum  *(motivation under an assumption — state as such)*

Assume the adaptation loss satisfies a local Polyak–Łojasiewicz condition with
constant $\mu>0$ on the healthy region,
$$L(\theta)-L^\star\;\le\;\frac{1}{2\mu}\lVert\nabla L(\theta)\rVert^2 .$$
Then the iterate with the smallest observed $\lVert\nabla L\rVert$ is the one with the
tightest suboptimality certificate for $L$ — i.e. the best-fit state *with respect to
the adaptation objective*.

Combined with the empirical fact that the entropy objective stops being aligned with
mIoU once collapse begins, the grad-norm minimum marks **the last iterate at which the
surrogate objective and the true metric still agree** — which is the state GDG-PA pins
as the permanent anchor.

> **Honesty note.** PL is an *assumption* here, not something we verify for NA-CLIP.
> Present §C as motivation ("under a local PL assumption…"), never as a theorem.

---

## D. What must stay empirical

| claim | status | evidence |
|---|---|---|
| $\mathrm{mIoU}\propto-\lVert\nabla L\rVert$ (Spearman $\rho\approx-0.9$) | **empirical observation** | `p9_gradnorm_vs_miou.png`; V20 $-0.84$, Cityscapes $-0.88$, ACDC $-0.90$, measured on the no-restore monitor runs |
| collapse regime threshold $0.9\,h_{\max}$ | tuned hyper-parameter | `h_drop_ratio` sweep |
| lag $\propto$ grad-slope (`lag_gain`) | heuristic controller | ablation only |

Do **not** dress the $\rho\approx-0.9$ correlation as a theorem — it is the paper's key
*observation*, and its value is that it is measured on un-gated runs (no circularity).

---

## E. Suggested paper layout

| section | content | strength |
|---|---|---|
| 3.1 Collapse, formally | §A — $H_{\mathrm{margin}}$ ⇒ mIoU ceiling | provable |
| 3.2 Why restoration works | §B — contraction, $d_\infty\le\eta G/r$; $r{=}0$ diverges | provable |
| 3.3 Where to anchor | §C — grad-norm minimum under local PL | assumption |
| 3.4 Observations | §D — $\rho\approx-0.9$, gate thresholds | empirical |

Writing §3.2 next to the failed loss-side variants (§B.4) is the strongest narrative:
it turns a pile of negative results into a single formal reason.
