# Chapter 3 (Analysis) — 數學部分整理稿

**來源**：本檔案的定理/證明取自 [docs/gdgpa_math_derivations.md](gdgpa_math_derivations.md)（隊友針對
`deyo_mlmp_hmgate2_continual` / GDG-PA 寫的理論文件），依照論文 outline 重新分類、篩選、精簡證明長度。

**適用性說明**：這份理論是針對 GDG-PA 的機制（grad-norm 觸發 + H_margin 判斷 regime + permanent-best-anchor
還原）推導的。**CAGA（我們鎖定的方法，`deyo_mlmp_adagate_continual`）沿用同一套核心機制**——permanent
best anchor、stochastic restore、grad_norm trend + H_margin 的聯合判準——只是把觸發門檻和 lag 選擇換成
自我校準的統計量（`trend_stat=mad`, `lag_mode=ecdf`）。因此本檔案的定理**原則上直接適用於 CAGA**，不需要
重新證明；如果要在論文裡引用，方法名稱可以直接寫 CAGA，不用寫 GDG-PA。

**outline 對應**：

```
3. Analysis
   3.1 Why does collapse happen?              <- Part II（本檔案 §1）
   3.2 Signal: Grad norm   [Fig 2]             <- Part II 附帶 + Part IV（本檔案 §2）
   3.3 Signal: H margin    [Fig 3]             <- Part I（本檔案 §3）
```

---

## §1 — 3.1 Why does collapse happen?（數學）

### 1.0 OVSS 與一般 CTTA 的區別（待補——這段還沒有數學,是敘事骨架）

Outline 註記要求：「must include OVSS distinction, such as over-confidence single class, compared to
general CTTA, since there is MGP」。這一段**目前隊友的文件裡完全沒有涵蓋**，需要你們自己寫，我先把邏輯骨架
列出來，數學部分（§1.1-§1.3）可以直接接在後面當佐證：

- 一般 closed-set CTTA（例如 MGP 這類方法設計時假設的場景）的分類頭是固定、少量、離散的類別；collapse
  通常表現為「BatchNorm 統計量漂移」或「少數幾類主宰但類別數固定」。
- OVSS 用 pixel-wise softmax over free-form text embeddings，類別數可以很大（COCO-Stuff 171 類、
  ACDC/Cityscapes 19 類），而且文字端完全 frozen——**collapse 的表現是預測質量塌縮到 1-2 個類別
  （single-class over-confidence）**，這是 §1.1-§1.3 的 marginal-entropy 崩潰現象，數學上就是
  $H(\bar y) \to 0$（Theorem 1.2/1.3，見 §3）。
- MGP（`mgp_deyo_mlmp_continual`）作為一般 CTTA 方法搬進 OVSS 的代表，在我們的實驗裡展現的崩潰模式
  （V21 上表現優異但 Cityscapes-full 上從 24→2.3 急墜，見進度圖）可以當作「一般 CTTA 方法沒有針對
  open-vocabulary 的 marginal-collapse 設計，所以防不住這種特定崩潰模式」的實證對照。
- 這段寫完後，直接接 §1.1（Prop 2.1）作為「為什麼會塌縮到單一類」的數學解釋。

### 1.1 Entropy minimization 本身不排斥崩潰

> **【Proposition】（Mutual information 分解）** 設 $X$ 為輸入隨機變數、$Y$ 為預測標籤，
> $\bar y = \mathbb{E}_X[p_\theta(\cdot\mid X)]$。則
> $$I(X;Y) = H(\bar y) - \mathbb{E}_X\big[H(p_\theta(\cdot\mid X))\big]$$

**白話**：entropy minimization loss（DeYO/TENT 這類方法的核心項）只壓右式第二項（個別預測要果斷），
完全不管第一項 $H(\bar y)$（整體要不要用到多樣的類別）。這是「單一類別過度自信」這個 OVSS 特有崩潰模式
的第一層數學根源。

> **【Proposition】（退化解是全域最小解）** 設 $L_{\text{ent}}(\theta) = \mathbb{E}_X[H(p_\theta(\cdot\mid X))]$。
> 則 $\inf_\theta L_{\text{ent}} = 0$，且 constant map（所有輸入都預測同一類 $k$）屬於全域最小解集合，
> 此時 $H(\bar y) = 0$，即 $I(X;Y) = 0$（mutual information 的全域最小值）。

**白話**：把所有 pixel 判成同一類（例如 road/sky）不是優化失敗，是這個 loss 的滿分答案之一。這解釋了
"single class over-confidence" 為什麼會是 entropy-based CTTA 方法的必然吸引子，而不是意外。

### 1.2 核心結果：加 diversity loss 救不了（⭐ 本節最重要的定理）

> **【Theorem】（退化態是 gradient flow 的陷阱）** 考慮任意形如
> $$L_\lambda(\theta) = \mathbb{E}_X\big[H(p_\theta(\cdot\mid X))\big] - \lambda\,H(\bar y_\theta), \qquad \lambda \in \mathbb{R}$$
> 的目標函數。若參數序列 $\{\theta^{(s)}\}$ 使預測趨於退化（$p_{\theta^{(s)}}(\cdot\mid x) \to e_k$ 對幾乎所有
> $x$），則
> $$\lim_{s\to\infty}\nabla_\theta L_\lambda\big(\theta^{(s)}\big) = 0 \qquad \text{對每一個 } \lambda \text{ 都成立}$$

**證明關鍵步驟**（完整證明見來源文件 §Theorem 2.3）：對 logit 求導可得閉式
$$\frac{\partial H(p)}{\partial z_j} = -p_j\big(\log p_j + H(p)\big), \qquad
\frac{\partial H(\bar y)}{\partial z_j^{(n)}} = -\frac1N p_j^{(n)}\Big[\log\bar y_j - \sum_i p_i^{(n)}\log\bar y_i\Big]$$
兩個梯度都帶有 $p_j$（或 $p_j^{(n)}$）這個因子——這是 softmax 飽和的直接後果：模型越有自信，這個因子越趨近
0。**diversity 項的梯度跟 base loss 的梯度在同一個點上同時被乘成 0**，所以 $\lambda$ 開得再大也沒用
（$\lambda \times 0 = 0$）。

**白話**：這是整份理論裡分量最重的一條，因為它同時做到三件事——
1. 解釋了為什麼所有「改 loss 加 diversity/regularizer」的嘗試最終都會失敗；
2. 預測了「加了以後崩潰會延後但不會消失」這個具體的實驗結果；
3. 證成了 CAGA 的設計選擇必須是**非梯度**的 restoration，不能是 loss term。

> **【Corollary】（必須要有非梯度機制）** 由上述定理，任何純粹由 $\{H(p), H(\bar y)\}$ 構成的梯度下降，
> 一旦進入退化態鄰域即無法在有限時間內逃離。逃離必須依賴梯度以外的機制（即 CAGA 的 stochastic restore）。

**實驗驗證**（來源文件已有數據，可直接引用）：`deyo_mlmp_divreg_continual`（always-on 版 diversity loss）
15 組 × 150 rounds 的結果——ACDC 崩潰輪次從 R74 延後到 R110（$\lambda=3$）但**仍然崩潰**，R150 的 mIoU
只剩 6-10，而 gated 方法（GDG-PA/CAGA）150R 不崩、穩定在 31-32。**這正是「延後但不能阻止」的理論預測
被實驗證實**，可以直接當 theory-driven ablation 寫進論文。

---

## §2 — 3.2 Signal: Grad norm

### 2.1 為什麼 grad_norm 會下降到 0（跟 §1.2 是同一組數學）

上面 Theorem（§1.2）證明的閉式梯度 $\partial H(p)/\partial z_j = -p_j(\log p_j + H(p))$ 在 $p\to e_k$
時趨於 0——**這就是 grad_norm 在完全崩潰時歸零的數學原因**，直接對應 CLAUDE.md 描述的「Phase 4：
gradient ≈ 0、locked」。這段可以放在 [Fig 2] 之前，作為「grad_norm 為什麼是一個合理訊號」的理論鋪墊。

### 2.2 grad_norm 單獨使用的侷限（⭐ 解釋為什麼還需要 H_margin）

> **【Proposition】（駐點的二義性）** 令 $\mathcal{S} = \{\theta : \nabla_\theta L(\theta) = 0\}$。則
> $\mathcal{S}$ 同時包含：
> - **健康收斂態**：$H(\bar y_\theta)$ 較大（用到多樣類別）
> - **退化態**：$H(\bar y_\theta) \to 0$（由 §1.1 的 Prop）
>
> 因此 $\|\nabla L\|$（也就是 grad_norm）**單獨無法區分**這兩種狀態。

**白話**：這是 [Fig 2]（mIoU vs. grad_norm）這張圖背後真正要傳達的訊息——不是「grad_norm 低就是好」，
而是「grad_norm 低這件事本身有二義性：可能是收斂到好解，也可能是崩潰到壞解」。這對應到我們自己在
V20-full 上實測到的現象：**grad_norm 的歷史最低點在後期反而跟 h_margin 一起持續下降**（見
`docs/EXPERIMENT_STATUS.md` 或本次對話中的 gate_log.csv 分析），也就是 CAGA-hgate 想修正的那個問題——
用 grad_norm 最小值選 anchor 時，若不搭配 H_margin 健康檢查，選到的可能是退化態而非健康態。

**跟我們自己實驗的直接連結**（隊友文件裡沒寫，是我們自己的發現，但完全被這條 Proposition 預言了）：
隊友文件的 Corollary 4.2 甚至直接寫出了修正方案——

> 「只有在 H_margin 健康時，才允許更新 permanent best anchor」：
> ```python
> if g < self.g_min and not collapse_regime:
>     self.best_snapshot = self._win_buf[-1]
> ```

**這正是 CAGA-hgate 目前在跑的機制**（`anchor_h_min_ratio` 條件）。也就是說，我們在 V20-full 上憑實測
數據診斷出的問題、以及據此實作的修正，跟隊友完全獨立推導出的 Corollary 4.2 是同一件事。這個交叉驗證
（一邊是數學推導、一邊是實測診斷，結論一致）在論文裡是很強的佐證，建議在 3.2 或方法論的段落明確點出來。

### 2.3 [Fig 2] 建議畫法

隊友文件的 Prop 4.1 白話建議：畫一張 2D phase diagram，橫軸 grad_norm、縱軸 h_margin，把 150 round
的軌跡疊上去，健康區跟崩潰區會分成兩塊——`gate_log.csv` 已經同時記錄兩欄，資料現成。這跟我們之前在
V20-full CAGA 診斷時畫的 `v20_caga_gradnorm_hmargin_miou.png` 是同一個資料來源，可以直接重製成正式圖表。

---

## §3 — 3.3 Signal: H margin

### 3.1 H_margin 是到 uniform 分布的 KL divergence（核心定義性結果）

> **【Theorem】** 令 $u$ 為 $C$ 個類別上的均勻分布。對任意聚合 marginal $\bar y \in \Delta^{C-1}$，
> $$\boxed{H(\bar y) = \log C - D(\bar y\,\|\,u)}$$
> 為恆等式。等價地，以 effective class count $C_{\text{eff}} := e^{H(\bar y)}$ 表示，
> $C_{\text{eff}}/C = e^{-D(\bar y\|u)}$。

**白話**：監控 H_margin 就是監控「類別分布離均勻使用有多遠」，兩者只差一個常數 $\log C$。這給了
H_margin 一個比「熵」更具體的資訊幾何意義。

### 3.2 Min-entropy 下界（附帶重要的誠實但書）

> **【Theorem】** $H(\bar y) \ge -\log p^\star$，其中 $p^\star = \max_i \bar y_i$；此界 **tight**（不可能
> 再改進），等號於 $\bar y$ 在 support 上均勻時成立。

**⚠️ 誠實的限制（來源文件的重要提醒）**：這個界在操作點（ACDC 實測 $H\approx2.26$）上**幾乎沒有資訊量**——
代入只能得到 $p^\star \ge 10.4\%$，而一個在 9.6 個類別上均勻分布、完全沒崩潰的模型也滿足這個下界。
**不能用這條定理去 justify 0.9 這個 gate 門檻**，需要用下面的 Theorem 3.3。

### 3.3 操作點上真正有力道的定量保證

> **【Theorem】** $$\|\bar y - u\|_{\mathrm{TV}} \;\ge\; \frac12\sqrt{\frac{\log C - H(\bar y)}{C}}$$
> （這是 **reverse Pinsker** 方向，跟常見的 Pinsker 上界方向相反，容易寫錯，要特別注意。）

**代入 ACDC 操作點**（$C=19$，門檻 $H=2.26$）：$D = 0.684$ nat，
$$\|\bar y - u\|_{\mathrm{TV}} \ge \tfrac12\sqrt{0.684/19} \approx 9.5\%$$

**白話**：這條才是能在論文裡站得住腳的說法——「H_margin 低於門檻時，保證至少 9.5% 的預測質量已經
偏離均勻使用類別的狀態」，是非平凡的定量陳述，比 min-entropy 下界有力得多。

### 3.4 門檻的精確語意

> **【Corollary】** Gate 判準 $H_{\text{margin}} < \gamma\,h_{\max}$（我們的 `h_drop_ratio`）等價於
> $$C_{\text{eff}} < \big(C_{\text{eff}}^{\max}\big)^{\gamma}$$

**代入**（ACDC，$h_{\max}=2.51 \Rightarrow C_{\text{eff}}^{\max}=12.3$，$\gamma=0.9$）：門檻對應
$C_{\text{eff}} = 12.3^{0.9} = 9.6$。

**⚠️ 寫作陷阱**：熵掉 10%，有效類別數**不是**掉 10%、是掉 22%（12.3→9.6），因為熵是對數尺度。論文
若寫「熵下降 10% 觸發」會誤導讀者低估門檻的嚴格程度，建議改用 $C_{\text{eff}}$ 的語言。

### 3.5 H_margin 的單向性（誠實的限制，且能反過來當 motivation）

> **【Remark】** $H$ 高**不蘊含**健康——由 Lemma（$H(\bar y) \ge \frac1N\sum_n H(p^{(n)})$，等號僅在
> 所有樣本预测相同時成立）可知，即使每張圖各自都已經完全確定（$H(p^{(n)})=0$），只要不同圖崩到
> **不同**類別，$\bar y$ 仍可以維持高熵。

**這正是 VOC20 的實例**：$H_{\text{margin}}$ 全程維持 $\approx3.0$，mIoU 卻從 78.7 漂到 74.3
（uniform-degradation regime）。**這個限制不要藏起來，要主動寫成 §3.2（grad_norm）存在的理由**：
正因為 H_margin 對這種「均勻但整體品質下降」的模式是瞎的，才需要 grad_norm 的 trend 來補上這個盲區
（grad_norm 的 trend 在 VOC20 上仍然跟 mIoU 顯著相關，見 `docs/2026-06-18-contribution.md` §7.3 的
23-signal 分析：grad_norm 是唯一在 VOC20 上 trend 仍存活的訊號）。

### 3.6 [Fig 3] 建議畫法

同一張圖建議画 H_margin 隨 round 的軌跡，疊上 collapse_regime 判定區間（紅色 shading），跟我們在
`v20_caga_gradnorm_hmargin_miou.png` 已經做過的畫法一致，可以直接重製。

---

## 未收錄 / 待你們決定的部分

1. **§1.0（OVSS vs. 一般 CTTA 的區別）**：只給了骨架，沒有數學，需要你們自己補完整敘事，特別是
   MGP 的具體對照數據（V21 領先 vs. Cityscapes-full 崩潰，見我們自己的進度圖）。
2. **Part III（drift bound，Theorem 3.2/3.3/3.4/3.5）完全沒有放進這個檔案**——這組定理證明的是
   「restoration 之後參數漂移有上界、多久收斂、permanent anchor 為什麼比 rolling anchor 好」，內容
   很強（尤其 Theorem 3.5 直接證成 permanent-best-anchor 設計），但性質上更像是「方法論保證」而不是
   「3.1/3.2/3.3 要回答的問題」。建議你們決定：這組要不要放進 Analysis 的 3.4（如果 outline 有）、
   還是放進 Method 章節描述 CAGA 機制時引用。我沒有自作主張放進來。
3. **Corollary 4.2 的實作修正建議**（anchor 更新要檢查 `not collapse_regime`）：我已經在 §2.2 指出
   這跟 CAGA-hgate 現在在跑的機制是同一件事，但沒有把完整的程式碼片段搬進來，因為那是隊友對
   `gdgpa.py` 行號的描述，不是我們自己 `deyo_mlmp_adagate_hgate_continual.py` 的實作（雖然邏輯一致）。
   如果要在論文附錄放程式碼，建議用我們自己檔案的版本，不要直接抄隊友文件裡的行號引用。
4. **§7（原文件列的六項未證明事項）跟 §8.3（必改清單）我沒有整份搬過來**——那些是寫作提醒/正確性
   檢查清單，跟本檔案「整理三個小節的數學」的任務不直接相關，但很有用，需要的話我可以另外整理一份
   「寫作前檢查清單」。
