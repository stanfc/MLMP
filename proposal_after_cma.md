# Proposal After CMA：下一步方向分析

**撰寫日期**：2026-04-25  
**背景**：CMA-continual 和 CMA-Proto-continual 實驗均失敗後的研究反思與新方向

---

## 0. 目前的問題總結

### 0.1 所有實驗結果一覽

| 方法 | Peak mIoU | 崩潰時間 | 穩定性 | 備註 |
|------|-----------|---------|--------|------|
| No Adaptation | 23.3 | 無 | ✅ 穩定 | baseline |
| CoTTA | 23.4 | 無 | ✅ 穩定 | 穩定但從不改善 |
| TENT-continual | 32.9 (R19) | ~R80 | ❌ | 進步快但崩潰 |
| MLMP-continual | 29.8 (R1) | 緩慢下降 | ❌ | |
| CMA-continual (k=0.2) | 26.8 (R16) | ~R32 | ❌ | 文字錨點不夠 |
| CMA-continual (k=0.5) | 27.4 (R18) | ~R40 | ❌ | k 越大崩得越晚 |
| CMA-Proto-continual | 26.1 (R16) | ~R54 | ❌ | 延遲但終究崩潰 |
| **MLMP episodic (step=1)** | **29.8** | 無（每次 reset） | N/A | **目標要超越這個** |
| **MLMP episodic (step=10)** | **30.6** | 無（每次 reset） | N/A | **終極目標** |

### 0.2 所有崩潰方法收斂到完全相同的 dead state

```
fog=1.23, night=1.05, rain=1.26, snow=1.27  (mean ≈ 1.20)
```

這個 dead state 是由 NA-CLIP 架構和 ACDC dataset 統計決定的——不是 loss 函數決定的。一旦進入這個 basin，無論哪種 loss 都無法脫出（梯度 ≈ 0）。

### 0.3 CMA-Proto 崩潰的根本原因（最重要的教訓）

我們原本認為「用 frozen source prototype 作為 external anchor 就能打破 confirmation bias loop」，但這個假設有一個關鍵缺陷：

**問題不在 target vector，在 class index selection（ĉ_i）。**

三項 loss 全部是：
```
L = -cos(v_i, target_X[ĉ_i])
```

其中 `ĉ_i = argmax(current_model(x_i))`。

即使 `p_src` 完全 frozen，feedback loop 依然存在：

```
Model 預測 pixel i 是 road
  → ĉ_i = road
  → L_CMA 把 v_i 拉向 t_road
  → L_src 把 v_i 拉向 p_src_road   ← p_src frozen，但還是在拉 road 方向
  → L_tgt 把 v_i 拉向 p_tgt_road
  → model 更傾向預測 road
  → loop 加速
```

**結論**：任何以「當前模型預測」作為 class index 的 loss，都不可避免地有 confirmation bias，無論 target vector 本身是否 frozen。

這是一個深層的設計缺陷，不是超參數調整能解決的。

### 0.4 要超越 episodic 的邏輯

Episodic MLMP 每個 sample 都從 source 重新出發，所以：
- 沒有累積 drift → 不崩潰
- 但也沒有跨 sample 的知識累積 → 每次都是白紙

**Continual 超越 Episodic 的唯一路徑**：
- 跨 sample 累積的知識必須是 **有益的**（net positive transfer）
- 這個知識必須 **不受 confirmation bias 污染**
- 這個知識的累積必須有 **防崩潰機制**

---

## 1. 方向 A：Layer-Stratified Adaptation（層次化差異化 Restoration）

### 1.1 核心直覺

ViT-L/14 的資訊處理有天然的層次結構：

```
Blocks  1-8:   低階特徵 — 顏色、邊緣、紋理、光照統計
Blocks  9-16:  中階特徵 — 區域形狀、部件
Blocks 17-24:  高階語義 — 物體類別、語義關係
```

Fog/night/rain/snow 的 domain shift 主要影響**低階特徵**（光照強度、顏色飽和度、noise pattern）。Class collapse（所有 pixel 都預測 road）是由**高階語義層的 drift** 造成的（model 的「什麼是 road」概念被污染）。

因此：讓早期 layers 自由適應 domain statistics，嚴格保護晚期 layers 的語義判別能力。

### 1.2 設計規格

**Loss 函數**：CMA loss（text-visual alignment），比 entropy 更好的方向信號：
```
L = -mean_{i ∈ TopK%} cos(v_i, t_{ĉ_i})
```

**Layer-stratified restoration**：
```python
layer_restoration_config = {
    'blocks_1_to_8':    rst=0.001,   # 幾乎自由，允許 domain 統計適應
    'blocks_9_to_16':   rst=0.01,    # 標準 CoTTA 水準
    'blocks_17_to_24':  rst=0.05,    # 強力保護，防止語義 drift
    # 可選：blocks_17_to_24 完全 frozen (rst=∞ 等效)
}
```

每個 gradient step 後，以上述 rst 做 stochastic restoration：
```python
for name, param in model.named_parameters():
    if 'LayerNorm' in name:
        layer_idx = extract_block_index(name)
        rst = get_rst_for_layer(layer_idx)
        if random.random() < rst:
            param.data = source_state[name].clone()
```

**Teacher/pseudo-label 生成**：CoTTA 風格的 EMA teacher + aug-averaged predictions，避免直接用 student model 生成 pseudo-labels。

### 1.3 為何可能超越 Episodic

Episodic MLMP 每次 reset **所有** layers，包括早期 layers 已學到的 domain 統計。在第二次看到 fog 時，它仍然從 source 的 early-layer representation 出發。

我們的 early layers 在 R2 看到 fog 時，已經具備了 R1 學到的 fog domain 統計知識。如果 early layer adaptation 是有益的（大機率是，因為 domain shift 主要在低階），累積的知識讓每次的 adaptation 從更好的起點出發。

理論上限：如果 early layers 完全適應到 test domain statistics，而 late layers 完全保持 source semantics，最終 performance 可能超過 episodic（因為 episodic 無法累積 early layer knowledge）。

### 1.4 關鍵超參數

| 參數 | 預設值 | 說明 |
|------|-------|------|
| `early_rst` | 0.001 | Blocks 1-8 的 restoration rate |
| `mid_rst` | 0.01 | Blocks 9-16 的 restoration rate |
| `late_rst` | 0.05 | Blocks 17-24 的 restoration rate |
| `early_cutoff` | 8 | Early/middle 的邊界（需要 ablation）|
| `late_cutoff` | 16 | Middle/late 的邊界（需要 ablation）|

### 1.5 Ablation 計畫

1. **只 freeze late layers（不做 early/mid 分層）**：驗證 late layer protection 是否足以防崩潰
2. **不同 cutoff 位置**：8/16 是猜測，可能 12/20 更好
3. **和 Diversity-Gate（方向 B）組合**：A+B 的組合可能是最強的

### 1.6 風險

- 最優的層邊界需要多次 ablation 才能確認
- 如果 domain shift 不只在早期 layers（研究表明 CLIP 的 domain sensitivity 分佈不均），這個假設可能有偏差
- ViT-L/14 有 24 個 blocks，layer-specific restoration 需要仔細 indexing

---

## 2. 方向 B：Diversity-Gated CMA（預測多樣性動態剎車）

### 2.1 核心直覺

既然：
- CoTTA 穩定但不進步（restoration 太強，阻止了所有有益更新）
- CMA 進步但崩潰（沒有任何保護機制）

能不能設計一個「有危險時自動剎車，安全時全速前進」的機制？

關鍵觀察：**Collapse 有可偵測的早期信號。**

當 confirmation bias 開始主導時，batch-level 的 class 分佈多樣性（prediction diversity）會下降——所有 pixels 開始預測相同的 1-2 個 class。這個信號：
- 不需要 ground truth label
- 在崩潰前 5-10 rounds 就已經可見
- 和 confirmation bias 直接相關

### 2.2 多樣性指標：Marginal Class Entropy

**不是** per-pixel entropy（這就是 TENT 在最小化的，是崩潰的 source）。

**是** batch 層級的 marginal class distribution entropy：

```python
# 對一個 batch 的所有 pixels 計算 marginal class 分佈
probs = softmax(avg_logits, dim=1)        # (B, C, w, h)
marginal = probs.mean(dim=[0, 2, 3])      # (C,) — 每個 class 的平均概率
H_margin = -(marginal * log(marginal)).sum()  # scalar，越高越好
```

對比：
- `H_margin ≈ log(19) ≈ 2.94`：完全均勻，每個 class 出現機率相等（理想態）
- `H_margin ≈ 0`：所有 pixels 預測同一個 class（collapse 的 dead state）
- `H_margin ≈ 1.5~2.0`：健康的預測分佈（有些 class 多，有些少，但都有）

### 2.3 動態剎車機制

```
每 N samples（預設 N=50）計算一次 H_margin：

if H_margin >= H_threshold:       # 健康狀態
    mode = "aggressive"
    restoration_rate = 0          # 完全停用 restoration
    loss = CMA (full gradient)

elif H_margin >= H_warning:       # 警告狀態
    mode = "cautious"
    restoration_rate = 0.005      # 輕度 restoration
    loss = CMA

else:                              # 危險狀態
    mode = "brake"
    restoration_rate = 0.05       # 強力 restoration（高於 CoTTA）
    loss = CMA (但 gradient 縮小)
```

**閥值建議**：
- `H_threshold = 1.8`（健康，可以激進）
- `H_warning = 1.2`（警告，開始保守）
- `H_danger = 0.8`（危險，強力 restoration）

這些值是估計值，實際需要從 R1 的 H_margin 基準（大約 1.6~2.0）來校準。

### 2.4 為何可能超越 Episodic

健康期間（H_margin 高），restoration 完全關閉 → 等同於沒有任何限制的 CMA 適應，adaptation 品質接近 episodic 甚至更好（因為 loss 方向更好）。

崩潰預警時才介入 → 不像 CoTTA 的 constant restoration 那樣「拖住」每一步更新。

如果 H_margin 能在整個 150 rounds 保持在 1.5 以上，model 就能在每個 round 都做接近 episodic 品質的適應，且不崩潰 → 有機會超越 episodic。

### 2.5 預期的 Performance 軌跡

```
R1-20:   H 高 → 激進模式 → 快速上升（類似 TENT/CMA 的早期）
R20-50:  H 開始下降警告 → 進入謹慎模式 → 上升變慢
R50+:    動態平衡 → H 在 threshold 附近震盪 → performance 維持在高位
```

最壞情況（brake 沒有及時啟動）：和 CMA 一樣崩潰，但有 restoration 作為 fallback。最好情況：performance 穩定在 28-32，不崩潰，超越 episodic。

### 2.6 關鍵超參數

| 參數 | 預設值 | 說明 |
|------|-------|------|
| `H_threshold` | 1.8 | 高於此值 → 激進模式 |
| `H_warning` | 1.2 | 低於此值 → 謹慎模式 |
| `monitor_interval` | 50 | 多少 samples 計算一次 H_margin |
| `brake_rst` | 0.05 | 危險模式的 restoration rate |

### 2.7 論文貢獻點

**H_margin 作為 label-free CTTA health metric** 這本身就是一個 contribution：
- 提出一個比 prediction confidence 更可靠的崩潰預測信號
- 可以用於 monitoring 任何 TTA 方法的狀態
- 實驗驗證：比較 H_margin 和真實 mIoU 的相關性曲線

### 2.8 風險

- Threshold 很敏感：太高 → 剎車太早，和 CoTTA 差不多；太低 → 等於沒剎車
- Performance 軌跡可能呈「鋸齒形」（激進 → 稍微崩一點 → 剎車 → 恢復 → 再激進），看起來不穩定
- H_margin 下降可能比 mIoU 下降慢（lag），導致反應不及時

---

## 3. 方向 C：Two-Timescale Meta-Adaptation（Fast-Slow 雙軌架構）

### 3.1 核心直覺：Meta-Learning 的 CTTA 翻譯

Meta-learning（尤其是 MAML）的核心思想是：**讓模型學會「如何快速適應」，而不只是學習適應某個特定任務**。

把這個思想翻譯到 CTTA：

- **Episodic MLMP 做的事**：每次從 source 出發，K-step 適應一個 sample
- **我們想做的事**：讓「出發點」本身隨著時間不斷進步

如果出發點從 source（23.3）進步到一個「融合了 test domain 知識的 meta-model」（假設 26-27），那麼同樣的 K-step episodic 適應就可以到達更高的終點。

### 3.2 架構設計

```
θ_slow (background model)：
  - 每個 sample 小幅更新（meta-gradient）
  - 代表「累積的 test domain 先驗知識」
  - 更新方向：θ_fast 做完 episodic 後的改善方向
  - Restoration：低但不為零（防止極端 drift）

θ_fast (per-sample model)：
  - 每個 sample 從 θ_slow 出發（不從 source 出發）
  - 做 K-step episodic 適應
  - 適應完成後，結果「反饋」給 θ_slow
  - 不保留狀態（每個 sample 後重置到 θ_slow）
```

**Prediction 使用 θ_fast**（適應過後的模型），不用 θ_slow。

### 3.3 θ_slow 的更新方式（Meta-Gradient）

有兩種選擇：

**Option 1（簡化版，First-Order MAML）**：

```python
# Step 1: Fast adaptation from θ_slow
θ_fast = θ_slow.copy()
for step in range(K):
    loss_fast = cma_loss(θ_fast, x)
    θ_fast = θ_fast - α_fast * gradient(loss_fast)

# Step 2: Update θ_slow in direction of θ_fast improvement
meta_direction = θ_fast - θ_slow  # first-order approximation
θ_slow = θ_slow + α_slow * meta_direction

# Step 3: Partial restoration of θ_slow toward source
θ_slow = (1 - rst_slow) * θ_slow + rst_slow * θ_source
```

**Option 2（MAML 完整版，Second-Order）**：
- 計算 `∇_θ_slow L(θ_fast(θ_slow))`，需要 second-order gradient
- 更準確但計算代價高（需要 hessian-vector product）
- 對 CTTA 可能過度複雜，Option 1 的近似通常夠用

### 3.4 為何可能超越 Episodic

```
Episodic 的軌跡：
  每個 sample：source (23.3) → K-step → ~30.6

方向 C 的預期軌跡：
  R1:  θ_slow = source (23.3) → K-step → ~30.6  (和 episodic 一樣)
  R10: θ_slow ≈ 24-25 (學到一些 fog/night 特性) → K-step → ~31-32?
  R50: θ_slow ≈ 26-27? → K-step → >32?
```

如果 θ_slow 的逐漸改善是真實的（non-trivial），每一個 episodic step 的起點都更好，最終可能達到比 episodic 更高的 performance。

### 3.5 這和現有方法的關係

| 方法 | 類比 |
|------|------|
| MAML outer loop | θ_slow 更新（meta-gradient） |
| MAML inner loop | θ_fast 的 K-step episodic |
| CoTTA restoration | θ_slow 的 partial restoration（防 meta-overfitting）|
| CMA loss | 更好的 inner loop loss，比 entropy 更穩定 |

本質上是 **Online MAML + CoTTA restoration + CMA inner loss**。

### 3.6 記憶體需求

- ViT-L/14 fp16 ≈ 1.7 GB
- 維護 θ_slow, θ_fast, θ_source 三份 LN 參數（不是整個 model）
- LN 參數數量遠小於整個 model，實際額外記憶體很小（< 50 MB）
- 只需要複製 LN 的 γ, β，不需要複製整個 ViT

### 3.7 關鍵超參數

| 參數 | 預設值 | 說明 |
|------|-------|------|
| `alpha_fast` | 1e-5 | θ_fast 的 inner loop learning rate |
| `alpha_slow` | 1e-6 | θ_slow 的 meta update rate（很小）|
| `K` | 1 或 3 | Per-sample fast adaptation steps |
| `rst_slow` | 0.005 | θ_slow 的 restoration rate（低但不為零）|
| `meta_momentum` | 0.9 | Meta-direction 的 EMA 平滑 |

### 3.8 風險

- Meta-gradient 信號的質量依賴 CMA loss 的可靠性；如果 CMA 本身有 bias，meta-gradient 也有 bias
- 計算代價：每個 sample 需要跑 K 次 forward + backward（比其他方法多 K 倍）
- θ_slow 的「進步」可能很慢，短期內看不出效果
- Debug 困難：雙軌架構的 bug 不容易發現

---

## 4. 建議的實驗順序

```
方向 B（Diversity-Gate）> 方向 A（Layer-Stratified）> 方向 C（Two-Timescale）
```

### 理由

**先做 B**：
- 直接解決 stability-plasticity tradeoff 的核心問題
- H_margin 作為 label-free metric 是新穎貢獻
- 快速實作（幾小時），快速驗證
- 如果 B 失敗，失敗的原因會告訴我們 A 和 C 需要調整什麼

**然後做 A**（不管 B 成敗）：
- Layer-stratified 的直覺很強，值得獨立驗證
- 如果 B 成功，A+B 組合可能是論文中的主要方法
- Ablation study：哪幾層保護 vs 哪幾層自由適應

**最後考慮 C**：
- 理論上限最高，但工程複雜度也最高
- 適合作為論文的「強化版本」（如果 B/A 證明了方向可行）

---

## 5. 成功標準

| 等級 | 標準 | 意義 |
|------|------|------|
| 基本成功 | 150 rounds 不崩潰（R150 ≥ 15 mIoU） | 打破「穩定就無法進步」的詛咒 |
| 目標 | 全程 mean mIoU ≥ 25，不崩潰 | 明顯優於 CoTTA（23.4）且穩定 |
| 理想 | 超越 No Adaptation 並持續上升 | 真正的 continual learning |
| 論文目標 | Mean ≥ 30.6（超越 episodic step=10） | Continual 超越 episodic 的理論優勢被實現 |

---

## 6. 這些方向和 CMA-Proto 的關係

CMA-Proto 失敗告訴我們，真正的問題是 **pseudo-label confirmation bias 在 class index selection 層面**。方向 A/B/C 從不同角度應對這個問題：

- **方向 A**：透過 layer protection 防止語義層被污染（結構性防禦）
- **方向 B**：透過 diversity monitoring 偵測到 bias 發生的早期信號，動態啟動防禦
- **方向 C**：透過 fast-slow 解耦讓 pseudo-label 來源（θ_slow）和被更新的 model（θ_fast）分離，直接打破 feedback loop

方向 C 在概念上最直接解決根本原因（解耦 pseudo-label 生成和 model 更新），但工程複雜度最高。方向 B 在實用性和論文貢獻的平衡上最佳。

---

*此文件記錄了 CMA/CMA-Proto 失敗後的系統性反思，作為後續實驗設計的理論基礎。*
