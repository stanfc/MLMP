# CAGA — 方法機制與實驗超參數

**最終方法 = CAGA**（原名 CAGA-hgate；程式中 `deyo_mlmp_adagate_hgate_continual`，類別 `DeYOMLMPAdaGateHGateContinual`，
檔案 `adapt/deyo_mlmp_adagate_hgate_continual.py`，繼承 `adapt/deyo_mlmp_adagate_continual.py`）。
本文件以此版本為主，並標明它與另外兩個版本的差異。所有數字皆從實際執行的
`save/*/cmd.sh`（main_continual.py 記錄的有效參數）與程式碼讀出，不是憑記憶。

## 0. 版本對照

| 名稱 | 程式 method 名 | `shallow_cap_mode` | `anchor_h_min_ratio` | 狀態 |
|---|---|---|---|---|
| **CAGA（最終；舊稱 CAGA-hgate）** | `deyo_mlmp_adagate_hgate_continual` | `growing_scaled` | 0.95 | 本文件主體 |
| **CAGA-gradgate**（不含 H 檢查；舊稱 CAGA） | `deyo_mlmp_adagate_continual` | `growing_scaled` | 無 | 消融 |
| CAGA-scale6 | `deyo_mlmp_adagate_continual` | `fixed`（cap=6） | 無 | 消融 |

三者**只有兩處不同**：anchor 更新時有沒有做 H_margin 健康檢查（§4.2），以及淺層還原的回溯上限怎麼算（§4.3）。
其餘全部相同。

---

## 1. 設定與協定

- **Backbone**：NA-CLIP ViT-L/14（`--ovss_type naclip`），文字端完全凍結。
- **可訓練參數**：visual encoder 的所有 LayerNorm affine（weight+bias）。實測 log 顯示 `LN params: 100` 個 tensor
  （24 blocks × 2 LN + `ln_pre` + `ln_post` = 50 個 LayerNorm × 2），共 50 × 1024 × 2 = **102,400 個純量**。
  注意：`--top_block_exclude 6` 有傳，但程式中 `_is_excluded()` 恆回傳 False（排除邏輯已移除），**這個參數目前不起作用**，全部 100 個 tensor 都在訓練。
- **Prompt**：`prompts.yaml` 的 7 個 template（itap of a {} / a bad photo of the {}. / a origami {}. / a photo of the large {}. / a {} in a video game. / art of the {}. / a photo of the small {}.）。
- **多層特徵（UAML）**：取最後 18 層（`--vision_outputs -1 … -18`）。
  - 適應時：7 個 prompt 各自出 logits（`vision_out_type="mean"`，18 層等權平均），得 shape `(T=7, B, C, h, w)`。
  - 評估時：用 7 個 prompt 平均後的文字向量，`vision_out_type="adaptive_weighted_mean"`（依各層熵加權）。
- **協定**：continual、**先評估後適應**（每張圖先用當前參數預測並計分，再更新）；`batch_size=1`、`steps=1`；
  連續 `continual_rounds=150` 輪，不做任何 per-sample reset；`--seed 0`。
- **Optimizer**：Adam，`lr=5e-6`，`betas=(0.9, 0.999)`，`weight_decay=0`。
- **輸入切 patch**：整張圖先 resize 到 `init_resize`，再以 224×224、stride 112 切 patch 個別前向後拼回（ACDC/Cityscapes 每張約 36 個 patch，其餘資料集 1 個）。

---

## 2. 單步適應（每個 batch）

繼承自 DeYO + MLMP，對每個 batch：

1. **前向**：得 7 個 prompt 的 pixel-wise logits，`ent = softmax entropy`（各 prompt、各 pixel）。
2. **記錄 marginal**：把 7 個 prompt 平均後的預測分布，對 batch/像素平均成 `(C,)`，放進 `marginal_buf`（CPU）。這是 H_margin 的原料。
3. **第一道 filter（熵）**：`mask_ent = ent < deyo_margin`，其中 `deyo_margin = 0.5 · ln C`（C = 類別數）。沒有任何 pixel 通過就**整個 batch 不更新**。
4. **第二道 filter（PLPD）**：對輸入做 patch shuffle（`aug_type=patch`、`patch_len=4`：切成 4×4 個區塊隨機重排）得 `x'`，
   前向得 `p'`。`plpd = p_orig[argmax] − p'[同一類別]`，只保留 `plpd > 0.2` 的 pixel。
5. **Loss**：對留下的 pixel，`loss = mean(ent × coeff)`，`coeff = 1/exp(ent − margin_e0) + 1/exp(−plpd)`，
   其中 `margin_e0 = 0.4 · ln C`（`reweight_ent=1`、`reweight_plpd=1`）。
6. **backward → 記錄 gradient norm → Adam step**。gradient norm 是所有 LN 參數梯度合併的 L2 norm，每個有做更新的 batch 記一個值進 `grad_buf`。
7. **還原（restoration）**：若目前狀態 `current_rst > 0` 且 `current_lag > 0`，立即對 LN 參數做一次 stochastic restore（§5）。
8. 每累積 `monitor_interval = 50` 個 batch，呼叫一次 `_update_gate()`（§3-§4）更新 gate 狀態，決定**下一個 window** 要不要還原、往哪還原。

> **注意**：第 6、7 步都在「filter 有放行」的分支裡。若整個 batch 被 filter 擋掉，既不更新也不還原，`grad_buf` 也不會增加；
> 若一個 window 內 `grad_buf` 為空，該 window 直接跳過（不更新 gate）。

---

## 3. 兩個訊號

**3.1 Gradient norm 的趨勢（optimization 側）**
1. window 平均：`g = mean(grad_buf)`（50 個 batch 的**不重疊區塊平均**，不是 EMA）。
2. 保留最近 `slope_window = 10` 個 window 的 `g`，做 OLS 線性回歸取**斜率** `slope`。
3. 標準化：`z = slope / (1.4826 · MAD(過去最多 50 個 slope))`（`trend_stat=mad`，`trend_hist=50`）。
   - MAD 需要至少 5 筆歷史，不足時若 slope 非 0 則 `z = ±10`（視為強趨勢），slope 為 0 則 `z = 0`。
   - `z` 只跟**之前**的 window 比較，當前 slope 在算完 z 之後才放進歷史。
4. 觸發條件：`z > trend_thr = 0.5`。

**3.2 Class-marginal entropy（prediction 側）**
1. window 內的 `marginal_buf` 取平均並正規化成 `ȳ`，`H_margin = −Σ ȳ_i ln ȳ_i`（自然對數）。
2. `h_max` = 迄今所有 window 的 H_margin 最大值（`h_max_mode=lifetime`，只增不減）。
3. `collapse_regime = H_margin < h_drop_ratio · h_max`，`h_drop_ratio = 0.9`。

---

## 4. 每個 window 的決策

### 4.1 演算法（依程式順序）

```
更新 gate（每 50 個有效 batch）：
  1. 存下當前 LN 參數快照 → win_buf（fp16，CPU；最多 max_windows+1 = 2001 個）
  2. g = mean(grad_buf)；加入 grad_hist（保留最近 10 個）
  3. 算 H_margin（§3.2）；清空 marginal_buf
  4. h_healthy = (H_margin >= anchor_h_min_ratio × h_max)   # h_max 為「不含本 window」的先前峰值；h_max=0（第一個 window）時視為 True
  5. anchor 更新（§4.2）
  6. 更新 h_max；windows_since_h_peak；collapse_regime = H_margin < 0.9 × h_max
  7. 算 slope、z（§3.1）
  8. 決策：
       if z <= 0.5 or windows_since_min < 1:      不還原（lag=0, rst=0）
       elif collapse_regime:                      DEEP：往 best_snapshot 還原，rst = base_rst
       else:                                      SHALLOW：往 lag 個 window 前的快照還原，rst = base_rst
  9. 寫 gate_log.csv；清空 grad_buf
```

### 4.2 Permanent best anchor（**hgate 與其他版本的差異在這裡**）

```
if g < g_min and h_healthy:          # hgate（CAGA 最終版）
    g_min = g;  windows_since_min = 0;  best_snapshot = 本 window 快照
else:
    windows_since_min += 1
```
- 不含 H 檢查的版本是 `if g < g_min:`，只看 gradient norm。
- `best_snapshot` **永不淘汰**（獨立於 `win_buf`），初始為初始權重。
- 設計動機：實測發現 grad_norm 的歷史最低點會跟著 H_margin 一起單調下降（模型滑向梯度變平但類別塌縮的區域），
  只看 grad_norm 選 anchor 會把 anchor 越釘越糟；加上 `H_margin ≥ 0.95 · h_max` 才允許更新，anchor 會停在「兩個訊號都健康」的最後一點。

### 4.3 淺層還原的回溯深度（`shallow_cap_mode = growing_scaled`）

```
u_sev = ECDF 排名：z 在 active_z_buf（最近 50 個「有觸發」的 z）中，嚴格小於 z 的比例
        （active_z_buf 不足 10 筆時 u_sev = 0.5）
cap   = min( len(win_buf) − 1,  ceil(windows_since_min × u_sev) )
lag   = max(1, ceil(cap × u_sev))            # lag_mode = ecdf：u 就是 u_sev
```
- 即：距離最佳 anchor 越久、當前趨勢越嚴重（z 排名越高），回溯範圍越深。
- `scale6` 版本改為 `cap = min(6, windows_since_min)`（固定最多 6 個 window）。
- DEEP 模式（collapse_regime）不使用 lag，直接對 `best_snapshot` 還原。

---

## 5. Restoration 操作

對每個 LN 參數 tensor `p`，逐元素獨立抽樣 `m ~ Bernoulli(base_rst = 0.01)`：

```
p ← (1 − m) · p + m · anchor
```
- anchor：DEEP 為 `best_snapshot`；SHALLOW 為 `win_buf[-1 − lag]`（超出緩衝範圍則取最舊一個）。
- 一次還原約 1% 的參數；在同一個 window 內，每個放行的 batch 都會再做一次（狀態沿用到下一次 `_update_gate` 更新為止）。
- 這是**不看梯度**的操作，也是方法必須是 restoration 而非 loss term 的原因（見 Analysis 章節 Prop. 1）。

---

## 6. 實作細節與已知行為

- **暖機**：第一個 window 只有 1 筆 `g`，slope=0、z=0，不會觸發；MAD 歷史不足 5 筆、ECDF 歷史不足 10 筆時用上面的退化值。
- **window 緩衝涵蓋範圍**：`win_buf` 上限 2001 個 window ≈ 100k 個 batch。每 round 的 batch 數依資料集而定
  （sub100：100 張 × 15 corruption = 1500 batch/round ≈ 30 window/round，緩衝約涵蓋 66 個 round；ACDC 約 406 batch/round ≈ 8 window/round，涵蓋超過 150 round）。
  超出者被淘汰，但 `best_snapshot` 不受影響。
- **記憶體**：每個快照約 102,400 × 2 bytes ≈ 0.2MB（fp16、CPU），2001 個約 0.4GB。
- **快照精度**：快照存 fp16 而參數是 fp32，anchor 因此有量化誤差（相對於參數量級很小，論文中若報數值界需註明）。
- **Checkpoint/resume**：`state_dict()` 存 LN 參數、Adam 狀態、以及全部 gate 狀態（`slope_buf`、`active_z_buf`、`marginal_buf`、`h_max`、`g_min`、`windows_since_min`、`best_snapshot`、`win_buf`…），
  每完成一個 (round, condition) 存一次。**不存 RNG 狀態**，所以 resume 後隨機遮罩與 patch shuffle 不會與不中斷時逐位元相同。
- **有傳但在這組設定下不起作用的參數**：
  `slope_deadzone`（只在 `trend_stat=abs` 時用）、`lag_gain`（只在 `lag_mode=gain`）、`lag_sat`（只在 `lag_mode=sat`）、
  `maxlag_shallow`（只在 `shallow_cap_mode=fixed`）、`h_max_decay`（只在 `h_max_mode=decaying`）、`top_block_exclude`（見 §1）、`ln_ckpt_every`（0=關閉）。
  論文中的「超參數」表只需列真正起作用的那幾個（見 §7.1 標 ✔ 的）。
- **輸出**：`gate_log.csv` 欄位 `total_batches, grad_norm, grad_slope, z, u, h_margin, collapse, windows_since_min, lag, rst, deep`。

---

## 7. 超參數

### 7.1 CAGA-hgate（全部 8 個資料集完全相同）

| 參數 | 值 | 起作用 | 說明 |
|---|---|---|---|
| `lr` | 5e-6 | ✔ | Adam |
| `batch_size` / `steps` | 1 / 1 | ✔ | |
| `continual_rounds` | 150 | ✔ | |
| `deyo_margin_factor` | 0.5 | ✔ | entropy filter 門檻 = 0.5·ln C |
| `deyo_margin_e0_factor` | 0.4 | ✔ | loss 重新加權的 e0 = 0.4·ln C |
| `plpd_threshold` | 0.2 | ✔ | PLPD filter |
| `aug_type` / `patch_len` | patch / 4 | ✔ | PLPD 用的 patch shuffle |
| `reweight_ent` / `reweight_plpd` | 1 / 1 | ✔ | |
| `monitor_interval` | 50 | ✔ | window 大小（batch 數） |
| `slope_window` | 10 | ✔ | 取斜率用的 window 數 |
| `trend_stat` | mad | ✔ | |
| `trend_thr` | 0.5 | ✔ | z 的觸發門檻（無單位） |
| `trend_hist` | 50 | ✔ | MAD 與 ECDF 的歷史長度 |
| `lag_mode` | ecdf | ✔ | |
| `shallow_cap_mode` | growing_scaled | ✔ | |
| `base_rst` | 0.01 | ✔ | 每次還原的逐元素機率 |
| `h_drop_ratio` | 0.9 | ✔ | collapse regime：H < 0.9·h_max |
| `h_max_mode` | lifetime | ✔ | |
| **`anchor_h_min_ratio`** | **0.95** | ✔ | **hgate 專屬**：H ≥ 0.95·h_max 才允許更新 anchor |
| `max_windows` | 2000 | ✔ | 快照緩衝 = 2001 個 window |
| `slope_deadzone` 0.002、`lag_gain` 1500、`lag_sat` 1.5、`maxlag_shallow` 6、`h_max_decay` 0.98、`top_block_exclude` 6 | — | ✘ | 見 §6 |

跨資料集只有「單一一組超參數、無 per-dataset 調參」：`trend_thr` 是無單位的 z 門檻（設計目標：跨資料集觸發率相近）。

### 7.2 各資料集的協定參數（皆 `naclip ViT-L/14`、severity 5、`split=val`、patch 224×224 / stride 112、seed 0）

| 資料集 | 類別數 C | data_dir | `init_resize` | 條件 | 每條件張數 | workers |
|---|---|---|---|---|---|---|
| ACDC（完整） | 19 | `.data/ACDC/` | 1120×560 | fog / night / rain / snow（真實天氣，不加 corruption） | 完整 val（約 406 張/round） | 1 |
| Cityscapes | 19 | `.data/cityscapes/` | 1120×560 | 15 種 ImageNet-C | 100（`subset_size=100`, `subset_seed=0`） | 0 |
| VOC20 | 20 | `.data/VOC2012/` | 224×224 | 15 種 ImageNet-C | 100 | 0 |
| VOC21 | 21 | `.data/VOC2012/` | 224×224 | 15 種 ImageNet-C | 100 | 0 |
| Pascal-Context 59 | 59 | `.data/VOC2010/` | 224×224 | 15 種 ImageNet-C | 100 | 0 |
| Pascal-Context 60 | 60 | `.data/VOC2010/` | 224×224 | 15 種 ImageNet-C | 100 | 0 |
| COCO-Stuff | 171 | `.data/coco_stuff164k/` | 224×224 | 15 種 ImageNet-C | 100 | 0 |
| COCO-Object | 81 | `.data/coco/` | 224×224 | 15 種 ImageNet-C | 100 | 0 |

- 15 種 ImageNet-C 依序：gaussian_noise, shot_noise, impulse_noise, defocus_blur, glass_blur, motion_blur, zoom_blur, snow, frost, fog, brightness, contrast, elastic_transform, pixelate, jpeg_compression；每個 round 依此順序各跑一遍，corruption 於線上施加。
- 每 round 圖片數：sub100 = 100 × 15 = 1500；ACDC ≈ 406。
- COCO-Object 的 GT 是另外由 `instances_val2017.json` 轉出（`scripts/make_coco_object_labels.py`，背景 = 0，80 個物件類別 = 1–80，面積大者先畫、小者覆蓋）。

### 7.3 Baseline 超參數

除下表列出的參數外，皆為該方法 constructor 的預設值。所有 baseline 的協定（batch=1、150 round、先評估後適應、7 prompt 之外的設定）與 §7.2 相同，
除非該列另有標示。`main_continual.py` 實際記錄的參數見各 `save/<dataset>/<run>/cmd.sh`。

| 方法（程式名） | lr | 其他起作用參數 |
|---|---|---|
| No-Adapt（`tent_continual`，不加 `--adapt`） | — | 只跑 1 round |
| MLMP-episodic（`main.py`） | 1e-3 | 每張圖 reset；`alpha_cls=1.0`、`steps=1`、`trials=1` |
| TENT（`tent_continual`） | 1e-5 | — |
| SAR（`sar_continual`） | 1e-5 | `e_margin=1.8`, `sam_rho=0.05`, `e_0=0.1`, `ema_factor=0.9`, `recovery_warmup=50` |
| DeYO（`deyo_continual`） | 1e-5 | `deyo_margin_factor=0.5`, `deyo_margin_e0_factor=0.4`, `plpd_threshold=0.2`, `aug_type=patch`, `patch_len=4`, `reweight_ent/plpd=1` |
| SPA（`spa_continual`） | 1e-5 | `noise_ratio=0.4`, `freq_mask_ratio=0.2`, `eraser_patch_size=14` |
| TRIBE（`tribe_continual`） | 1e-5 | `eta=0.005`, `gamma=0.0`, `lam=0.5`, `h0=0.05` |
| ViDA（`vida_continual`） | 1e-5 | `r=4`, `r2=16`, `ema=0.99`, `ema_vida=0.8`, `unc_n=10`, `unc_thr=0.2` |
| PTTA（`ptta_continual`） | 1e-5 | `e_margin=1.198`, `d_margin=0.05`, `loss2_weight=3.0`, `queue_size=1000`, `neighbor=1` |
| COME（`come_continual`） | 1e-5 | — |
| MGP-DeYO-MLMP（`mgp_deyo_mlmp_continual`） | 5e-6 | DeYO 參數同上 + `mgp_distill_freq=100`, `mgp_buffer_size=32`, `mgp_max_rank=32`, `mgp_residual_thr=0.75`, `mgp_collect_freq=40`；`vision_outputs` −1…−18 |
| DAF（`daf_mlmp_continual`） | 見下 | `optimizer=sgd`(momentum 0.9)、`alpha_cls=1.0`、`lamb_ent=1.0`、`lamb_div=2.0`、`lamb_cmac=0.5`、`alpha_safs=0.5`、`prompt_integration=loss`、`vision_outputs` −1…−18 |
| CoTTA（`cotta`） | 1e-5 | `mt=0.999`, `rst=0.01`, `ap=0.92`, `aug_n=32`；僅 ACDC 完整跑完，其他資料集已停止（不進表） |

### 7.4 各資料集之間**不一致**的地方（寫論文時必須處理或註明）

1. **DAF 有兩組設定**：
   - ACDC、V20、Cityscapes：`batch_size=1`、`lr=1.25e-4`、SGD（依 linear scaling 由官方 `batch=8, lr=1e-3` 換算）。
   - V21、P59、P60、COCO-Stuff、COCO-Object：`batch_size=8`、`lr=1e-3`、SGD（官方設定）。
2. **ViDA 的 ACDC 結果用 `lr=2e-7`**，其他資料集 `lr=1e-5`。
3. **SAR 的 ACDC 結果是較舊版本的執行**（cmd.sh 參數名為 `sar_rho`、`reset_constant_em=0.2`、`margin_e0_factor=0.4`，且沒有 `e_margin/e_0/ema_factor/recovery_warmup`），與其他資料集的 SAR 參數不同。
4. CAGA 系列在 ACDC 的 cmd.sh 少了 `h_max_mode` / `h_max_decay` / `ln_ckpt_every` 三個欄位，是當時尚未有這些參數；程式預設值（`lifetime` / 0.98 / 0）與其他資料集相同，**有效設定一致**。
5. 只有 ACDC 為完整資料集；其餘為 100 張子集 × 15 種 corruption。V20/Cityscapes 的完整版（不抽樣）曾跑過一部分但已停止，未納入。

### 7.5 三個 CAGA 版本的設定差異（其他全部相同）

| | CAGA-hgate（最終） | CAGA（不含 H 檢查） | CAGA-scale6 |
|---|---|---|---|
| method | `deyo_mlmp_adagate_hgate_continual` | `deyo_mlmp_adagate_continual` | `deyo_mlmp_adagate_continual` |
| `shallow_cap_mode` | growing_scaled | growing_scaled | fixed（`maxlag_shallow=6` 生效） |
| `anchor_h_min_ratio` | 0.95 | — | — |
| anchor 更新條件 | g 創新低 **且** H 健康 | g 創新低 | g 創新低 |
