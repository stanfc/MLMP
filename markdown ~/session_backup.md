 ▐▛███▜▌   Claude Code v2.1.116
▝▜█████▛▘  Sonnet 4.6 · Claude Pro
  ▘▘ ▝▝    ~/MLMP

❯ 請你完整了解一下我們專案的目標和目前最好的方法GDG-PA 請你幫我看一下GDG-PA的那 
  些超參數例如threshold等等是否有辦法讓他有一點adaptive的成分                   
  因為如果有但凡一點點adaptive的概念在裡面我的論文裡面敘述方法時會更好包裝      

● I'll start by understanding the project and GDG-PA's current implementation.

  Listed 1 directory (ctrl+o to expand)

● Bash(sed -n '1,80p' docs/EXPERIMENT_STATUS.md)
  ⎿  # Experiment Status — Research Arc

     **Last updated**: 2026-06-03
     … +106 lines (ctrl+o to expand)

  Searched for 2 patterns, read 1 file, listed 1 directory (ctrl+o to expand)

● Bash(sed -n '1,120p' docs/2026-06-18-contribution.md)
  ⎿  # 0618 Contribution — DeYO + MLMP + SmoothAnchor (gate ablation, 3 dat
     asets)
     … +85 lines (ctrl+o to expand)

● Bash(for f in bash/ACDC_10_round/deyo_mlmp_hmgate2_continual*.sh
      bash/cityscapes/deyo_mlmp_hmgate2_continual_5corr_sub100.sh
      bash/v20/deyo_mlmp_hmgate2_continual_5co…)
  ⎿  === bash/ACDC_10_round/deyo_mlmp_hmgate2_continual.sh
     GPU_ID=0
     DATASET=ACDCDataset
     … +85 lines (ctrl+o to expand)

● Bash(for d in save/ACDCDataset/deyo_mlmp_hmgate2_continual
      save/CityscapesDataset/deyo_mlmp_hmgate2_continual_5corr_sub100
      save/PascalVOC20Dataset/deyo_mlmp_hmgate2_…)
  ⎿  === save/ACDCDataset/deyo_mlmp_hmgate2_continual
     Traceback (most recent call last):
       File "<string>", line 4, in <module>
     … +15 lines (ctrl+o to expand)

  Searched for 2 patterns (ctrl+o to expand)

● Bash(python3 -c "
      import csv,statistics as st,glob…)
  ⎿  === ACDC  (600 windows)
       grad_norm  min=  4.4428 p10=  4.7801 med=  5.2459 p90=  5.5883 max=
      6.7907
     … +20 lines (ctrl+o to expand)

● Bash(python3 -c "
      import csv,statistics as st,os…)
  ⎿  === ACDC
       rel_slope (slope/g): p10=-0.00564 med=-0.00012 p90=+0.00543 max=+0.1
     5826
     … +11 lines (ctrl+o to expand)

  Searched for 1 pattern (ctrl+o to expand)

● Bash(sed -n '1111,1200p' docs/EXPERIMENT_STATUS.md)
  ⎿  ## 18. Phase O: grad_norm gate-mechanism campaign (2026-06-21 → 06-25)

     Goal: a DeYO+MLMP CTTA defense that is both HIGH and STABLE, beating t
     … +77 lines (ctrl+o to expand)

● Bash(for d in save/ACDCDataset/hmgate2_prompt_S0_baseline
      save/PascalVOC20Dataset/v20_acdc_matched/hmgate2_prompt_S0_baseline; do
      echo "== $d"; python3 -c "
      import s…)
  ⎿  == save/ACDCDataset/hmgate2_prompt_S0_baseline
     rounds=150 mean=31.06 peak=31.57@R150 last=31.57
     == save/PascalVOC20Dataset/v20_acdc_matched/hmgate2_prompt_S0_baseline
     rounds=150 mean=77.11 peak=78.99@R149 last=78.99

  Searched for 1 pattern (ctrl+o to expand)

❯ 供你參考 目前GDG-PA最好的表現應該是存在hmgate2_prompt_S0_baseline裡面！如果我沒有記錯的話 

● 我完整讀過了專案脈絡（CLAUDE.md、docs/EXPERIMENT_STATUS.md
  §16–18、docs/2026-06-18-contribution.md）和 GDG-PA 的實作
  adapt/deyo_mlmp_hmgate2_continual.py，並且直接去挖了你指的
  hmgate2_prompt_S0_baseline 的 gate_log.csv（ACDC 600 windows、VOC20 1500
  windows）來看那些門檻在跑的時候實際上落在什麼位置。結論比預期好很多：GDG-PA
  已經有兩個真正的 adaptive 成分，而另外兩個超參數其實是「死的」——把它們改成
  adaptive 等於刪掉 tunable，這是最強的包裝方式。

  ---
  1. 先確認基準

  ┌─────────────┬────────────────────┬─────────────────────┐
  │             │ ACDC (S0_baseline) │ VOC20 (S0_baseline) │
  ├─────────────┼────────────────────┼─────────────────────┤
  │ mean (150R) │ 31.06              │ 77.11               │
  ├─────────────┼────────────────────┼─────────────────────┤
  │ peak        │ 31.57 @ R150       │ 78.99 @ R149        │
  ├─────────────┼────────────────────┼─────────────────────┤
  │ last        │ 31.57              │ 78.99               │
  └─────────────┴────────────────────┴─────────────────────┘

  兩個資料集都是單調爬升到最後一輪、完全沒有衰退——這比 composite gate 的
  32.30(ACDC) 更值得寫，因為那是「高但會震盪」，GDG-PA 是「還在爬」。

  而且我確認了一件對論文很有價值的事：12 個 hmgate2 bash script（ACDC / 
  Cityscapes / VOC20 / VOC21 / COCO-Stuff / P59 / P60，full 和 
  sub100）用的是完全相同的 7 個 gate 超參數，一個都沒有 per-dataset 調過：

  SLOPE_WINDOW=10  SLOPE_DEADZONE=0.002  LAG_GAIN=1500  BASE_RST=0.01
  H_DROP_RATIO=0.9  MAXLAG_SHALLOW=6  MONITOR_INTERVAL=50   ← 12/12 一致

  這和 composite gate 形成強烈對比（那個的 conf_ceil
  必須每個資料集重新校準：ACDC 0.70 / VOC20 0.60 / Cityscapes 0.60，而且第一次
  sweep 因為 scale 搞錯導致 gate 完全沒觸發）。「零 per-dataset 校準」本身就是 
  adaptive 的實證，只是你目前沒有在論文裡主張它。

  ---
  2. 超參數審計：三類

  已經是 adaptive 的（可以直接寫進論文，不用改 code）

  參數: h_drop_ratio=0.9
  為什麼已經 adaptive: 判準是 H < 0.9 × 
    h_max(running)，相對於串流自己觀測到的最大  H，不是絕對門檻。這就是為什麼
    ACDC 的 H∈[2.17, 2.47] 和 VOC20 的 H∈[2.94, 3.08]
    兩個完全不同的尺度可以共用同一個數字。實測 collapse regime  觸發率：ACDC
    48.5%、VOC20 0.0%——同一條規則自動分出兩個 regime，這是最好的賣點。
  ────────────────────────────────────────
  參數: deyo_margin = 0.5·log(C)margin_e0 = 0.4·log(C)
  為什麼已經 adaptive: 已經正規化到 label space 大小，19 類 / 20 類 / 171
    類（COCO-Stuff）自動縮放。
  ────────────────────────────────────────
  參數: best_snapshot 錨點
  為什麼已經 adaptive: 錨點位置由 grad-norm 極小值資料驅動決定，不是「回到
    source」或「回到第 N 步」的固定設定。

  事實上是「死的」超參數 ← 這裡是最大機會

  我從 gate_log 算出來的：

  (a) lag_gain=1500 完全沒有作用。 lag = clamp(round(1500·slope), 1, cap)，cap =
   maxlag_shallow=6：

  ┌───────┬──────────────────────────────────┬───────────────────┐
  │       │ 活躍 window 的 1500·slope 中位數 │ 飽和到 cap 的比例 │
  ├───────┼──────────────────────────────────┼───────────────────┤
  │ ACDC  │ 25                               │ 95.7%             │
  ├───────┼──────────────────────────────────┼───────────────────┤
  │ VOC20 │ 276                              │ 99.9%             │
  └───────┴──────────────────────────────────┴───────────────────┘

  lag 的實際分布：ACDC {0: 322, 6: 162, 其他: 16}、VOC20 {0: 741, 6: 709, 其他: 
  50}。也就是說 lag 是二元的 {0, 6}，程式註解裡宣稱的「continuous / adaptive
  lag」實際上沒有被執行到。這如果被 reviewer 追問「lag_gain=1500
  怎麼調的」會很難回答。

  (b) slope_deadzone=0.002 幾乎等於「slope > 0」。 P(slope > 0.002) = ACDC
  0.463、VOC20 0.507，和 P(slope > 0) 差不多。同時它是絕對值門檻，但兩個資料集的
   slope 尺度差 10 倍（ACDC p10/p90 = −0.027/+0.030；VOC20 = −0.347/+0.299），所
  以它「跨資料集通用」是巧合（因為它太小、幾乎沒在過濾），不是設計。

  純固定、有改造空間的

  base_rst=0.01（不論塌縮多深都同一個力道）、plpd_threshold=0.2（絕對值）、monit
  or_interval=50、maxlag_shallow=6。

  ---
  3. 建議的 adaptive 改法（依 CP 值排序）

  ★ A. 把 slope 正規化 —— 風險最低、直接刪掉一個超參數

  把絕對 slope 換成相對自己近期噪音水準的標準化量：

  z = slope / (1.4826 · MAD(recent slope_window 個 slope) + ε)     # robust
  z-score
  觸發條件：z > z_thr（unitless，例如 1.0）

  - 語意變成「grad-norm 的上升超出它自己的噪音水準」——這是真正 self-calibrating
  的偵測器，跟資料集的 grad 尺度脫鉤。
  - 可證明是現況的嚴格推廣：把 z_thr 調到觸發率 ≈ 現在的 46–51%，行為幾乎不變 →
  幾乎零風險，但論述完全升級。
  - 論文寫法：slope_deadzone（絕對、需調）→ z_thr（無量綱、跨資料集固定）。

  ★★ B. 讓 lag 真的連續（順便刪掉 lag_gain）

  既然 lag_gain 是死的，直接用 A 的 z 把 lag 表達成「預算的比例」而不是「絕對
  window 數」：

  lag = ceil( maxlag_shallow · min(1, z / z_sat) )          # 1 … maxlag_shallow

  - lag_gain 這個 tunable 消失，中間 lag 值（2–5）第一次真的會被用到。
  - 這讓程式碼註解裡「adaptive-lag gate」的說法變成名副其實——目前如果 reviewer
  去看 lag 的直方圖會發現它是 bang-bang。

  ★★★ C. 讓 base_rst 隨塌縮深度連續縮放 —— 論文敘事最強的一項

  目前無論塌縮多深，restore 力道都是固定 0.01。改成：

  s = clamp( (h_max − H) / ((1 − h_drop_ratio) · h_max), 0, s_max )   #
  塌縮深度，無量綱
  rst = base_rst · ( ρ + (1 − ρ) · s )                                # ρ =
  保底比例（healthy regime 的溫和 restore）

  為什麼這是最好的包裝：

  - 兩個 regime 從「if/else 分支」變成「一條連續控制律的兩端」。VOC20 的 H
  從不下降 → s ≈ 0 → 自動退化成溫和 restore；ACDC 塌縮 → s 上升 → restore
  自動加深。你不再需要說「我們偵測 regime 然後切換策略」，而是說「restoration 
  strength is a continuous function of the observed marginal-entropy deficit; 
  the two regimes reported in prior work emerge as its two limits」——這句話在
  method section 的份量差很多。
  - 保底項 ρ 是必要的：contribution doc §2 顯示 VOC20 那個 always-on 的溫和
  restore（77.7 vs no-gate 77.4）正是它勝出的原因，ρ=0 會把它關掉。
  - 這也順便處理了你 §3 提到的「late-stage restore 力道不夠」的老問題。

  ★ D. h_max 從單邊 ratchet 改成穩健統計量

  目前 h_max = max(all history)，只增不減——一個早期的幸運 window
  會永久拉高參考點，長串流下會偏保守。改成長窗 p90 或慢速 EMA 上包絡：

  h_ref = running p90 over the last W windows      （或 EMA 的 upper envelope）

  好處：h_ref 會跟著串流走，論文可以說參考點是 online estimated 而非 all-time
  extremum。風險中等（會改變 ACDC 的觸發時序），要跑一次驗證。

  ☆ E. plpd_threshold 改成 batch 內分位數（真正動到 loss filter，不只是 gate）

  plpd_threshold = quantile_q( plpd[mask_ent] )        #
  保留固定「比例」而非固定「絕對值」

  語意：filter 自動適應該 domain 的 PLPD 分布，而不是假設 0.2 在所有 domain
  都是對的切點。這讓 adaptive 的成分同時出現在 loss 端和 defence 
  端，包裝上更完整。但風險最高——會改變每個 batch 的 pixel 接受率動態，需要 sweep
   q，我不建議跟 A/B/C 綁在一起做。

  monitor_interval / maxlag_shallow
  我建議不要動，它們是結構參數（時間解析度、預算上限），保持固定反而好解釋。

  ---
  4. 我的建議路徑

  先做 A + B + C，做成一個新 method（例如 
  deyo_mlmp_adagate_continual），並且保留「參數化退化成現況」的能力——設定 ρ=1, 
  z_thr 對齊觸發率、z_sat 夠小，就應該逼近現在的 S0_baseline
  曲線。這樣你有兩個東西可寫：

  1. 消融：adaptive 版 ≈ 或 > 固定版，但少了 2 個需要調的超參數（slope_deadzone,
   lag_gain）。
  2. 敘事：GDG-PA
  的三個控制量（觸發、深度、力道）全部無量綱、全部由串流自身統計量校準，跨 6
  個資料集（ACDC / Cityscapes / VOC20 / VOC21 / COCO-Stuff /
  P59-60）零重新校準——而 §18 的 composite gate 需要 per-dataset 校準
  conf_ceil，這正好是你現成的反例。

  有一點要先講清楚：A 和 B 大概率是「持平 + 論述變好」而不是漲分，因為它們替換掉
  的是本來就沒在作用的參數；真正可能改變數字的是 C（和
  D）。如果你的目標純粹是包裝，A+B 就夠且幾乎零風險；如果你想要 C
  帶來的敘事升級，就要接受一次 150R 的重跑驗證。

  要我把 A+B+C 實作成新 method（含 adapt/__init__.py 註冊、main_continual.py
  參數、三個資料集的 bash script，並附一個能退化回現況的 control
  設定）嗎？還是你想先只做 A+B 這個零風險的部分？

❯ 先跟我說一下A和B目前的超參數邏輯是什麼 我甚至連那個超參數是什麼作用都不知道   

● 讓我先把精確數字算出來，再從頭解釋機制。
● Bash(python3 -c "
      import csv,statistics as st…)
  ⎿  === ACDC  window數=600, grad_norm中位數=5.25
       正向 slope (>deadzone) 的分布: p10=0.0051 med=0.0168 p90=0.0392
       -> 要讓 lag 剛好鋪滿 1..6，lag_gain 需要 ~= 6/p90 = 153
     … +9 lines (ctrl+o to expand)

● 好，我從頭講。先建立骨架，兩個參數才會有意義。

  ---
  0. GDG-PA 的 gate 在做什麼

  整個 gate 的核心動作只有一個：每次 optimizer.step() 之後，把一小部分 LayerNorm
   參數「倒回」到過去某個時間點的值（deyo_mlmp_hmgate2_continual.py:377-388）：

  mask = (torch.rand(p.shape) < rst)      # 每個元素獨立擲骰子，機率 rst
  p.data.mul_(1-mask).add_(anchor[name]*mask)   # 中獎的元素被覆寫成 anchor 的值

  所以要決定三件事：

  ┌──────────────────────────────────────┬───────────────────────────┐
  │              要決定的事              │         由誰控制          │
  ├──────────────────────────────────────┼───────────────────────────┤
  │ 要不要倒回（開/關）                  │ slope_deadzone ← 這就是 A │
  ├──────────────────────────────────────┼───────────────────────────┤
  │ 倒回到多久以前（anchor 選哪個快照）  │ lag_gain → lag ← 這就是 B │
  ├──────────────────────────────────────┼───────────────────────────┤
  │ 倒回多少比例（每個元素被覆寫的機率） │ base_rst = 0.01，即 1%    │
  └──────────────────────────────────────┴───────────────────────────┘

  時間單位：window

  monitor_interval=50 定義了「1 個 window = 50 個 adapt batch」。每個 window
  結束時 _update_gate() 跑一次，做三件事：

  1. 存一張快照：把當下所有 LN 參數存進 _win_buf 這個 deque（L318）。所以
  _win_buf[-1] 是剛剛的狀態、_win_buf[-2] 是 50 個 batch 前、_win_buf[-7] 是 300
   個 batch 前……
  2. 算這個 window 的平均 grad_norm g（這 50 個 batch 的 LN 梯度 L2 norm
  平均）（L320）
  3. 決定下個 window 要用什麼 lag / rst

  為什麼盯 grad_norm？因為 §18 的發現：grad_norm 上升 ≈ 模型正在退化（grad_norm
  ∝ −mIoU）。它是個好的「偵測器」（但當 loss 用就爆炸，見 O-1 的負結果）。

  ---
  1. slope 是什麼

  grad_hist 只保留最近 slope_window=10 個 window 的平均 grad_norm，然後對這 10
  個點跑一條 OLS 直線，取斜率（_slope(), L171-180）：

  slope = 「grad_norm 每經過 1 個 window 上升多少」
  單位 = grad-norm 單位 / window

  - slope > 0 → 最近 500 個 batch（10 windows）grad_norm 在爬 → 疑似在退化
  - slope ≤ 0 → 平穩或下降 → 健康

  實測尺度（ACDC 的 grad_norm 中位數 5.25、VOC20 是 6.64）：

  ┌───────┬───────────┬───────────┐
  │       │ slope p10 │ slope p90 │
  ├───────┼───────────┼───────────┤
  │ ACDC  │ −0.027    │ +0.030    │
  ├───────┼───────────┼───────────┤
  │ VOC20 │ −0.347    │ +0.299    │
  └───────┴───────────┴───────────┘

  注意 VOC20 的 slope 尺度是 ACDC 的 10 倍——這件事等一下是關鍵。

  ---
  2. A：slope_deadzone = 0.002 的作用

  它是整個 restore 機制的總開關（L347-348）：

  if slope <= self.slope_deadzone or self.windows_since_min < 1:
      lag, rst = 0, 0.0          # ← 完全不 restore，自由學習

  意思是：「grad_norm 每個 window 至少要上升 0.002，我才認為它是真的在爬，而不是
  隨機噪音。低於這個就當作沒事，放它自由跑。」名字叫
  deadzone（死區）就是這個意思——控制論裡防止控制器對噪音過度反應的標準做法。

  （windows_since_min < 1 是第二個條件：如果現在就是史上 grad_norm
  最低點，那就沒有「過去」可以倒回，當然不 restore。）

  問題在哪：0.002 是絕對值。 換算一下：

  - ACDC：0.002 / 5.25 = 每個 window 0.038% 的漲幅
  - VOC20：0.002 / 6.64 = 每個 window 0.030% 的漲幅

  這遠遠小於兩邊的噪音水準（ACDC 的 slope 波動 ±0.03，是 deadzone 的 15
  倍；VOC20 ±0.3，是 150 倍）。所以實際效果是：

  P(slope > 0.002)  =  ACDC 0.463 ,  VOC20 0.507
  P(slope > 0)      ≈  ACDC 0.5   ,  VOC20 0.5

  這個「死區」幾乎沒有濾掉任何東西，它等價於「slope > 0」。
  你這個參數名義上是防噪音的門檻，實際上是個擲硬幣。

  而且它是絕對值，所以在兩個 slope 尺度差 10 倍的資料集上，它落在噪音分布的完全
  不同百分位——它能通用是因為它小到不起作用，不是因為它設計得好。

  ---
  3. B：lag_gain = 1500 的作用

  過了開關之後，healthy regime 走這條（L355-359）：

  cap = min(self.maxlag_shallow, self.windows_since_min)     # maxlag_shallow = 
  6
  lag = max(1, min(int(round(self.lag_gain * slope)), cap))  # ← lag_gain 在這

  設計意圖是「退化得越快，就倒得越遠」：

  lag = 1500 × slope   （單位：window）

  然後被兩個上限夾住：最多 6 個 window（maxlag_shallow），且不能超過「距離歷史最
  佳點多遠」（windows_since_min，因為倒回到比最佳點還早的狀態沒意義）。lag
  最後決定 anchor 是哪張快照（L383-384）：

  anchor = self._win_buf[-1 - lag]     # lag=1 → 50 batch 前；lag=6 → 300 batch 
  前

  問題在哪：1500 這個 gain 大到讓 lag 永遠飽和。

  要讓 lag 達到上限 6，只需要 slope > 6/1500 = 0.004。而實際的正向 slope：

  ┌───────┬────────────────┬────────────────┬───────────────────────────────┐
  │       │  正向 slope    │  超過 0.004    │ 若要讓 lag 真的鋪滿 1~6，gain │
  │       │     中位數     │     的比例     │             應該設            │
  ├───────┼────────────────┼────────────────┼───────────────────────────────┤
  │ ACDC  │ 0.0168         │ 93.2%          │ ~153                          │
  ├───────┼────────────────┼────────────────┼───────────────────────────────┤
  │ VOC20 │ 0.1838         │ 99.7%          │ ~17                           │
  └───────┴────────────────┴────────────────┴───────────────────────────────┘

  結果就是 lag 的實際直方圖：

  ACDC :  {0: 322,  6: 162,  其他(1~5,8,10,18): 16}
  VOC20:  {0: 741,  6: 709,  其他(1~5): 50}

  lag 是二元的 {0, 6}，中間值幾乎不存在。程式註解寫的 "adaptive-lag gate"、"lag
  隨 slope 連續縮放"，在實測上沒有被執行到——它退化成「不 restore，或倒回 300 個
  batch 前」的 bang-bang 開關。

  而且最後一欄說明了為什麼沒有任何一個絕對 gain 能通用：ACDC 需要 153、VOC20
  需要 17，差 9 倍。你之所以能在 12 個 script 用同一個
  1500，是因為它對兩邊都「太大」，兩邊都飽和——巧合，不是通用性。

  ---
  4. 總結這兩個參數現在的真實狀態

  ┌──────────────────────┬────────────────────────┬─────────────────────────┐
  │                      │      名義上的作用      │    實測上的真實行為     │
  ├──────────────────────┼────────────────────────┼─────────────────────────┤
  │ slope_deadzone=0.002 │ 濾掉噪音等級的 grad    │ ≈ slope >               │
  │                      │ 上升                   │ 0，幾乎不濾任何東西     │
  ├──────────────────────┼────────────────────────┼─────────────────────────┤
  │ lag_gain=1500        │ slope                  │ 永遠飽和 → lag ∈ {0,    │
  │                      │ 越大倒得越遠（連續）   │ 6}，二元                │
  └──────────────────────┴────────────────────────┴─────────────────────────┘

  所以 GDG-PA 現在實際在跑的規則，用白話講是：

  ▎ 每 50 個 batch 檢查一次。如果 grad_norm 
  ▎ 在爬（不管爬多快）且我們已經離開歷史最佳點 → 開始每步用 1% 機率把 LN 
  ▎ 參數倒回；塌縮 regime 倒回永久最佳快照，健康 regime 倒回300 batch 
  ▎ 前。否則完全不 restore。

  這個規則本身是有效的（ACDC/VOC20 都單調爬到 R150），但它比程式碼宣稱的簡單很多
  ，而且兩個「可調參數」實際上沒有在調任何東西。

  這就是為什麼我說 A 和 B 是零風險的改造：把 slope 除以它自己的噪音尺度（A）、把
   lag 表達成預算的比例而不是絕對 gain（B），你會同時得到「真的連續、真的自校準
  」的機制和少兩個超參數——而不是在既有效果上冒險。

  要我接著把 A、B
  改完後的具體公式和預期行為（含「如何驗證它退化回現況」）寫出來嗎？

❯ 那我知道了 請你幫我把目前的A B各自遇到的問題設計更好的方案然後做實驗下去跑    
  儘量多試幾種方法確保一定可以找到解決方案  目前GPU很空 請直接把實驗丟下去跑    
  企圖把這兩個問題解決 或是變成adaptive                                         

● Bash(nvidia-smi
      --query-gpu=index,name,memory.used,memory.total,utilization.gpu
      --format=csv; echo "=== running python"; ps aux | grep -c
      "[m]ain_continual.py")
  ⎿  index, name, memory.used [MiB], memory.total [MiB], utilization.gpu [%
     ]
     0, NVIDIA RTX PRO 6000 Blackwell Server Edition, 6379 MiB, 97887 MiB,
     … +11 lines (ctrl+o to expand)

  Read 1 file (ctrl+o to expand)

● GPU 3/4/5 有空間。先確認在跑的工作別動到，並抓取現有 run 的時間成本。

● Bash(ps aux | grep "[m]ain_continual.py" | sed 's/.*python/python/' | cut
      -c1-160; echo "=== nvidia procs"; nvidia-smi
      --query-compute-apps=pid,gpu_uuid,used_memory…)
  ⎿  python main_continual.py --adapt --method deyo_mlmp_hmgate2_continual
     --ovss_type naclip --ovss_backbone ViT-L/14 --prompt_dir prompts.yaml
     --dataset PascalVOC2
     … +6 lines (ctrl+o to expand)

  Listed 1 directory (ctrl+o to expand)

● 8 小時/run（ACDC sub50 150R）。先做離線重放校準——用已存的 gate_log.csv 的
  grad_norm 序列，把新統計量算出來，這樣門檻不用瞎猜。

● Bash(cat > /tmp/claude-1014/-home-tekai324-MLMP/38eb7291-b267-40f5-924c-e42393
      95f620/scratchpad/replay.py <<'EOF'
      import csv, math, statistics as st…)
  ⎿  === ACDC  (n=600, replay slope max-err vs log = 5.52e-07)
       current firing rate (abs>0.002) = 0.463
       rel   : p50=  -0.000 p75=  +0.003 p90=  +0.005 p95=  +0.007 | thr ma
     … +22 lines (ctrl+o to expand)

● Bash(cat bash/run_prompt_queue.sh 2>/dev/null | head -50; echo "=== VOC20
      duration"; grep -i duration
      save/PascalVOC20Dataset/v20_acdc_matched/hmgate2_prompt_S0_base…)
  ⎿  #!/bin/bash
     # GPU job QUEUE for the GDG-PA PROMPT-SET sweep (9 sets x {ACDC, VOC20
     } = 18 jobs).
     … +30 lines (ctrl+o to expand)

● GPU3 上是你自己跑了 1天9小時的 VOC20 job，我不會碰。GPU 4/5 全空。先把新的 lag
   規則也離線模擬一遍再寫 code。

● Bash(cat > /tmp/claude-1014/-home-tekai324-MLMP/38eb7291-b267-40f5-924c-e42393
      95f620/scratchpad/replay2.py <<'EOF'
      import csv, math, statistics as st…)
  ⎿  === ACDC
       mad0.0 ecdf: fire=0.497 deep=111 off=302  shallow-lag分布={1: 34, 2:
      31, 3: 31, 4: 37, 5: 28, 6: 26}
     … +15 lines (ctrl+o to expand)

● 校準完成——lag 分布從現在的 {0, 6} 二元變成真正鋪滿
  1~6，且兩個資料集用同一個無量綱門檻得到幾乎相同的觸發率（ACDC 0.305 / VOC20
  0.311）。這正是要的。開始實作。

● Bash(sed -n '1,60p' adapt/deyo_mlmp_hmgate2_continual.py > /dev/null && cp
      adapt/deyo_mlmp_hmgate2_continual.py adapt/deyo_mlmp_adagate_continual.py
      && echo copied)
  ⎿  copied

● Update(adapt/deyo_mlmp_adagate_continual.py)
  ⎿  Added 49 lines, removed 18 lines
       1  """
       2 -DeYOMLMPHMGate2Continual — HMGate with a PERMANENT best-state a
         -nchor for deep restore.                                        
       2 +DeYOMLMPAdaGateContinual — GDG-PA (hmgate2) with SELF-CALIBRATI
         +NG trigger and lag.                                            
       3  
       4 -Diagnosis fixed here: the original HMGate stores per-window LN 
         -snapshots in a deque                                           
       5 -of fixed length (max_windows=2000 windows = 100k batches). On l
         -ight runs (sub100,                                             
       6 -500 batch/round) that covers 200 rounds, so the healthy peak sn
         -apshot is always                                               
       7 -reachable and the curve looks flat. But on heavy runs (VOC20 fu
         -ll+15corr, 21735                                               
       8 -batch/round) the deque only covers ~4.6 rounds: the grad-min (p
         -eak) snapshot is                                               
       9 -EVICTED by ~R13, after which deep restore falls back to _win_bu
         -f[0] (the oldest                                               
      10 -still-held, already-degraded window). The reachable anchor then
         - slides forward and                                            
      11 -downward -> a residual ~-0.06/round decline (delay, not prevent
         -).                                                             
       4 +Motivation (measured on save/*/hmgate2_prompt_S0_baseline/gate_
         +log.csv):                                                      
       5 +Two of hmgate2's gate hyperparameters are nominally tunable but
         + empirically INERT,                                            
       6 +and both are expressed in absolute grad-norm units, so neither 
         +transfers across                                               
       7 +datasets by design -- only by accident.                        
       8  
      13 -Fix: keep a dedicated `best_snapshot` that is updated ONLY when
         - a new grad-norm                                               
      14 -minimum is found, and is NEVER evicted. In the COLLAPSE regime 
         -(deep restore),                                                
      15 -restore toward this permanent best instead of the rolling deque
         -, so the true                                                  
      16 -healthy peak is reachable no matter how many iterations have pa
         -ssed. The HEALTHY                                              
      17 -regime still uses the lag-deque shallow restore (so the slow cl
         -imb isn't                                                      
      18 -suppressed). Everything else (DeYO+MLMP loss, H_margin regime s
         -witch, grad slope                                              
      19 -trigger, base_rst stochastic restore) is identical to deyo_mlmp
         -_hmgate_continual.                                             
       9 +  (A) slope_deadzone=0.002 is meant to reject noise-level grad-
         +norm rises. But the                                            
      10 +      slope noise band is +-0.03 (ACDC) / +-0.35 (VOC20), i.e. 
         +15x / 150x larger,                                             
      11 +      so the "deadzone" degenerates to `slope > 0`: firing rate
         + 0.463 / 0.507                                                 
      12 +      vs P(slope>0) ~ 0.5. It filters nothing, and its meaning 
         +differs 10x between                                            
      13 +      the two datasets.                                        
      14  
      21 -gate_log.csv columns: total_batches, grad_norm, grad_slope, h_m
         -argin, collapse,                                               
      22 -windows_since_min, lag, rst, deep.                             
      15 +  (B) lag = round(lag_gain * slope) with lag_gain=1500 is meant
         + to scale restore                                              
      16 +      depth continuously with degradation speed. But lag satura
         +tes at the cap                                                 
      17 +      (maxlag_shallow=6) whenever slope > 6/1500 = 0.004, which
         + holds for 93.2%                                               
      18 +      (ACDC) / 99.7% (VOC20) of active windows. The realised la
         +g histogram is                                                 
      19 +      binary: ACDC {0:322, 6:162}, VOC20 {0:741, 6:709}. To act
         +ually spread lag                                               
      20 +      over 1..6 you would need lag_gain ~= 153 (ACDC) vs ~= 17 
         +(VOC20) -- a 9x                                                
      21 +      dataset-specific gap, i.e. no single absolute gain can wo
         +rk.                                                            
      22 +                                                               
      23 +Fix -- both decisions are re-expressed as UNITLESS quantities d
         +erived from the                                                
      24 +stream's own statistics:                                       
      25 +                                                               
      26 +  (A) trend_stat: the raw OLS slope is normalised before thresh
         +olding.                                                        
      27 +        'mad'   z = slope / (1.4826 * MAD(recent slopes))   <- 
         +robust z-score                                                 
      28 +        'tstat' z = slope / SE(slope)                        <-
         + OLS t-statistic                                               
      29 +        'rel'   z = slope / grad_norm                        <-
         + fractional growth                                             
      30 +        'abs'   z = slope                                    <-
         + hmgate2 behaviour                                             
      31 +      With 'mad' the two datasets' z distributions almost coinc
         +ide (p75 0.635 vs                                              
      32 +      0.656, p90 1.125 vs 1.114), so ONE unitless trend_thr giv
         +es the same firing                                             
      33 +      rate on both (thr=0.5 -> 0.305 / 0.311). `trend_thr` is t
         +hen a genuine                                                  
      34 +      signal-to-noise deadzone, not a scale-bound constant.    
      35 +                                                               
      36 +  (B) lag_mode: restore depth is a FRACTION of the available bu
         +dget.                                                          
      37 +        'ecdf' u = ECDF of z among recent ACTIVE z  -> lag = ce
         +il(cap * u)                                                    
      38 +               rank-based, hence invariant to any monotone resc
         +aling of the trend                                             
      39 +               statistic and free of tunables entirely (lag_gai
         +n is deleted).                                                 
      40 +        'sat'  u = clamp(z / lag_sat, 0, 1)         -> lag = ce
         +il(cap * u)                                                    
      41 +        'gain' lag = round(lag_gain * slope)         <- hmgate2
         + behaviour                                                     
      42 +      Offline replay of the S0_baseline stream gives a genuinel
         +y graded lag under                                             
      43 +      'ecdf': ACDC {1:34, 2:31, 3:31, 4:37, 5:28, 6:26} instead
         + of {0, 6}.                                                    
      44 +                                                               
      45 +Setting trend_stat='abs', trend_thr=slope_deadzone, lag_mode='g
         +ain' reproduces                                                
      46 +deyo_mlmp_hmgate2_continual exactly (no extra RNG is consumed),
         + which is the                                                  
      47 +`ctrl` arm of the sweep.                                       
      48 +                                                               
      49 +Everything else -- DeYO+MLMP loss, H_margin regime switch, perm
         +anent best-state                                               
      50 +anchor, base_rst stochastic restore -- is unchanged from hmgate
         +2.                                                             
      51 +                                                               
      52 +gate_log.csv columns: total_batches, grad_norm, grad_slope, z, 
         +u, h_margin,                                                   
      53 +collapse, windows_since_min, lag, rst, deep.                   
      54  CTTA hard-rule compliant: no per-sample reset; partial stochast
          ic restore only.
      55  """
      56  import os

● Update(adapt/deyo_mlmp_adagate_continual.py)
  ⎿  Added 1 line, removed 1 line
      72  REFERENCE_PROMPT = 'a photo of a {}'
      73  
      74  
      75 -class DeYOMLMPHMGate2Continual:                                
      75 +class DeYOMLMPAdaGateContinual:                                
      76      def __init__(self, ovss_type, ovss_backbone, lr, classes, s
          teps=1,
      77                   vision_outputs=tuple(range(-1, -19, -1)),
      78                   prompt_dir='prompts.yaml',

● Update(adapt/deyo_mlmp_adagate_continual.py)
  ⎿  Added 9 lines
      101                   h_drop_ratio=0.9,
      102                   maxlag_shallow=6,
      103                   monitor_interval=50,
      104 +                 # --- self-calibrating trigger (A) and lag (B
          +) ---                                                         
      105 +                 # trend_stat: how the raw OLS slope is normal
          +ised before the                                               
      106 +                 #   deadzone test. 'abs' + lag_mode='gain' ==
          + hmgate2.                                                     
      107 +                 trend_stat='mad',                            
      108 +                 trend_thr=0.5,                               
      109 +                 trend_hist=50,                               
      110 +                 # lag_mode: how restore depth is chosen insid
          +e the budget cap.                                             
      111 +                 lag_mode='ecdf',                             
      112 +                 lag_sat=1.5,                                 
      113                   save_dir=None,
      114                   runtime_calculation=False, device='cpu'):
      115          self.ovss_type = ovss_type

● Update(adapt/deyo_mlmp_adagate_continual.py)
  ⎿  Added 13 lines
      142          self.h_drop_ratio = float(h_drop_ratio)
      143          self.maxlag_shallow = int(maxlag_shallow)
      144          self.monitor_interval = int(monitor_interval)
      145 +        if trend_stat not in ('abs', 'rel', 'mad', 'tstat'):  
      146 +            raise ValueError(f"trend_stat must be abs|rel|mad|
          +tstat, got {trend_stat}")                                     
      147 +        if lag_mode not in ('gain', 'sat', 'ecdf'):           
      148 +            raise ValueError(f"lag_mode must be gain|sat|ecdf,
          + got {lag_mode}")                                             
      149 +        self.trend_stat = trend_stat                          
      150 +        # 'abs' keeps hmgate2's semantics, where the deadzone 
          +IS slope_deadzone                                             
      151 +        self.trend_thr = float(slope_deadzone) if trend_stat =
          += 'abs' else float(trend_thr)                                 
      152 +        self.trend_hist = int(trend_hist)                     
      153 +        self.lag_mode = lag_mode                              
      154 +        self.lag_sat = float(lag_sat)                         
      155 +        # slope history (for the MAD scale) and active-z histo
          +ry (for the ECDF rank)                                        
      156 +        self.slope_buf = deque(maxlen=self.trend_hist)        
      157 +        self.active_z_buf = deque(maxlen=self.trend_hist)     
      158          # H_margin regime state
      159          self.marginal_buf = []
      160          self.h_max = 0.0

● Now the trend statistic and lag helpers, replacing the bare _slope:

● Update(adapt/deyo_mlmp_adagate_continual.py)
  ⎿  Added 58 lines, removed 1 line
      220              self.adapt_times = []
      221              self.eval_times = []
      222  
      223 -    # ---------- slope ----------                             
      223 +    # ---------- trend statistics ----------                  
      224      @staticmethod
      225      def _slope(ys):
      226 +        """OLS slope of ys against 0..n-1 (identical to hmgate
          +2's _slope)."""                                               
      227          n = len(ys)
      228          if n < 2:
      229              return 0.0
     ...
      233          den = sum((i - xm) ** 2 for i in range(n))
      234          return num / den if den > 0 else 0.0
      235  
      236 +    @classmethod                                              
      237 +    def _slope_tstat(cls, ys):                                
      238 +        """OLS slope together with its t-statistic slope / SE(
          +slope)."""                                                    
      239 +        n = len(ys)                                           
      240 +        b = cls._slope(ys)                                    
      241 +        if n < 3:                                             
      242 +            return b, 0.0                                     
      243 +        xm = (n - 1) / 2.0                                    
      244 +        ym = sum(ys) / n                                      
      245 +        Sxx = sum((i - xm) ** 2 for i in range(n))            
      246 +        if Sxx <= 0:                                          
      247 +            return b, 0.0                                     
      248 +        resid = [ys[i] - (ym + b * (i - xm)) for i in range(n)
          +]                                                             
      249 +        se = math.sqrt(sum(r * r for r in resid) / (n - 2)) / 
          +math.sqrt(Sxx)                                                
      250 +        if se <= 1e-12:                                       
      251 +            return b, (0.0 if abs(b) < 1e-12 else math.copysig
          +n(_Z_DEGENERATE, b))                                          
      252 +        return b, b / se                                      
      253 +                                                              
      254 +    @staticmethod                                             
      255 +    def _median(xs):                                          
      256 +        s = sorted(xs)                                        
      257 +        n = len(s)                                            
      258 +        return 0.5 * (s[(n - 1) // 2] + s[n // 2])            
      259 +                                                              
      260 +    def _trend_z(self, slope, tstat, g):                      
      261 +        """Normalise the raw slope into a unitless trend stren
          +gth z.                                                        
      262 +                                                              
      263 +        The slope buffer is appended by the caller AFTER this 
          +call so that z is                                             
      264 +        always scored against the PRECEDING windows, never aga
          +inst itself.                                                  
      265 +        """                                                   
      266 +        if self.trend_stat == 'abs':                          
      267 +            return slope                                      
      268 +        if self.trend_stat == 'tstat':                        
      269 +            return tstat                                      
      270 +        if self.trend_stat == 'rel':                          
      271 +            return slope / g if g > 1e-9 else 0.0             
      272 +        # 'mad': robust z-score of the slope against its own r
          +ecent spread                                                  
      273 +        if len(self.slope_buf) < _MAD_MIN_HIST:               
      274 +            scale = 0.0                                       
      275 +        else:                                                 
      276 +            med = self._median(self.slope_buf)                
      277 +            scale = 1.4826 * self._median([abs(x - med) for x 
          +in self.slope_buf])                                           
      278 +        if scale > 1e-9:                                      
      279 +            return slope / scale                              
      280 +        return 0.0 if abs(slope) < 1e-12 else math.copysign(_Z
          +_DEGENERATE, slope)                                           
      281 +                                                              
      282 +    def _lag_fraction(self, z):                               
      283 +        """Fraction u in [0, 1] of the lag budget to use for t
          +his window."""                                                
      284 +        if self.lag_mode == 'sat':                            
      285 +            return min(1.0, max(0.0, z / self.lag_sat)) if sel
          +f.lag_sat > 1e-9 else 1.0                                     
      286 +        # 'ecdf': rank of z among recent ACTIVE z. Rank-based,
          + so invariant to any                                          
      287 +        # monotone rescaling of the trend statistic and free o
          +f tunables.                                                   
      288 +        if len(self.active_z_buf) < _ECDF_MIN_HIST:           
      289 +            return 0.5                                        
      290 +        return sum(1 for x in self.active_z_buf if x < z) / le
          +n(self.active_z_buf)                                          
      291 +                                                              
      292      @torch.no_grad()
      293      def _snapshot_ln_weights(self):
      294          return {name: p.detach().cpu().to(torch.float16).clone
           ()
● Update(adapt/deyo_mlmp_adagate_continual.py)
  ⎿  Added 7 lines
      71  
      72  REFERENCE_PROMPT = 'a photo of a {}'
      73  
      74 +# z assigned when the normalising scale is degenerate (zero spr
         +ead) but the slope                                             
      75 +# is non-zero: treat as an unambiguously strong trend.         
      76 +_Z_DEGENERATE = 10.0                                           
      77 +# minimum history before the MAD scale / ECDF rank are consider
         +ed meaningful                                                  
      78 +_MAD_MIN_HIST = 5                                              
      79 +_ECDF_MIN_HIST = 10                                            
      80  
      81 +                                                               
      82  class DeYOMLMPAdaGateContinual:
      83      def __init__(self, ovss_type, ovss_backbone, lr, classes, s
          teps=1,
      84                   vision_outputs=tuple(range(-1, -19, -1)),

● Now the decision block in _update_gate:

● Update(adapt/deyo_mlmp_adagate_continual.py)
  ⎿  Added 20 lines, removed 7 lines
      459          # collapse regime when H_margin has dropped below a fr
           action of its peak
      460          collapse_regime = h_margin < self.h_drop_ratio * self.
           h_max
      461  
      462 -        slope = self._slope(self.grad_hist)                   
      462 +        slope, tstat = self._slope_tstat(self.grad_hist)      
      463 +        # (A) unitless trend strength, scored against the PREC
          +EDING windows                                                 
      464 +        z = self._trend_z(slope, tstat, g)                    
      465 +        self.slope_buf.append(slope)                          
      466 +                                                              
      467          deep = False
      464 -        if slope <= self.slope_deadzone or self.windows_since_
          -min < 1:                                                      
      468 +        u = 0.0                                               
      469 +        if z <= self.trend_thr or self.windows_since_min < 1: 
      470              lag, rst = 0, 0.0
      471          elif collapse_regime:
      472              # DEEP: restore toward the PERMANENT best anchor (
           never evicted), so the
     ...
      474              # lag is irrelevant here (anchor is best_snapshot,
            not the deque).
      475              deep = True
      476              lag, rst = self.windows_since_min, self.base_rst
      477 +            self.active_z_buf.append(z)                       
      478          else:
      479              # SHALLOW: lag-deque restore so the slow climb (VO
           C20) is never suppressed.
      480              cap = min(self.maxlag_shallow, self.windows_since_
           min)
      475 -            lag = max(1, min(int(round(self.lag_gain * slope))
          -, cap))                                                       
      481 +            if self.lag_mode == 'gain':                       
      482 +                lag = max(1, min(int(round(self.lag_gain * slo
          +pe)), cap))                                                   
      483 +            else:                                             
      484 +                # (B) restore depth as a FRACTION of the avail
          +able budget                                                   
      485 +                u = self._lag_fraction(z)                     
      486 +                lag = max(1, math.ceil(cap * u))              
      487 +            self.active_z_buf.append(z)                       
      488              rst = self.base_rst
      489  
      478 -        print(f"[DeYO-MLMP-HMGate2] B{self.total_batches}: gra
          -d={g:.3f} "                                                   
      479 -              f"slope={slope:+.4f} H={h_margin:.3f}/{self.h_ma
          -x:.3f} "                                                      
      490 +        print(f"[DeYO-MLMP-AdaGate] B{self.total_batches}: gra
          +d={g:.3f} "                                                   
      491 +              f"slope={slope:+.4f} z({self.trend_stat})={z:+.3
          +f}/{self.trend_thr:.3f} "                                     
      492 +              f"H={h_margin:.3f}/{self.h_max:.3f} "           
      493                f"{'COLLAPSE' if collapse_regime else 'healthy'}
            "
      481 -              f"since_min={self.windows_since_min} lag={lag}w 
          -rst={rst:.4f} "                                               
      494 +              f"since_min={self.windows_since_min} u={u:.2f} l
          +ag={lag}w rst={rst:.4f} "                                     
      495                f"{'DEEP->best' if deep else 'shallow'}")
      496          if self.gate_log_path is not None:
      497              with open(self.gate_log_path, 'a') as _f:
      485 -                _f.write(f"{self.total_batches},{g:.6f},{slope
          -:.6f},"                                                       
      498 +                _f.write(f"{self.total_batches},{g:.6f},{slope
          +:.6f},{z:.6f},{u:.6f},"                                       
      499                           f"{h_margin:.6f},{int(collapse_regime
           )},"
      500                           f"{self.windows_since_min},{lag},{rst
           :.6f},{int(deep)}\n")
      501          self.current_lag = lag

● Now the log header, banner, and reset():

● Update(adapt/deyo_mlmp_adagate_continual.py)
  ⎿  Added 1 line, removed 1 line
      216              os.makedirs(save_dir, exist_ok=True)
      217              self.gate_log_path = os.path.join(save_dir, "gate_
           log.csv")
      218              with open(self.gate_log_path, 'w') as _f:
      219 -                _f.write("total_batches,grad_norm,grad_slope,h
          -_margin,collapse,"                                            
      219 +                _f.write("total_batches,grad_norm,grad_slope,z
          +,u,h_margin,collapse,"                                        
      220                           "windows_since_min,lag,rst,deep\n")
      221  
      222          with torch.no_grad():

● Update(adapt/deyo_mlmp_adagate_continual.py)
  ⎿  Added 7 lines, removed 3 lines
      192          self.named_ln_params = list(zip(names, params))
      193  
      194          print_clip_parameters(self.model)
      195 -        print(f"+++ DeYO-MLMP-HMGate2 permanent-best-anchor (T
          -={len(self.prompt_templates)}): "                             
      195 +        lag_desc = (f"gain={self.lag_gain}" if self.lag_mode =
          += 'gain'                                                      
      196 +                    else (f"sat={self.lag_sat}" if self.lag_mo
          +de == 'sat' else "rank-based"))                               
      197 +        print(f"+++ DeYO-MLMP-AdaGate self-calibrating (T={len
          +(self.prompt_templates)}): "                                  
      198                f"deyo_margin={self.deyo_margin:.3f}, plpd_thr={
           self.plpd_threshold}, "
      199                f"UAML={len(self.vision_outputs)} | "
      198 -              f"slope_window={self.slope_window}, deadzone={se
          -lf.slope_deadzone}, "                                         
      199 -              f"lag_gain={self.lag_gain}(win), base_rst={self.
          -base_rst}, "                                                  
      200 +              f"trend_stat={self.trend_stat}, trend_thr={self.
          +trend_thr}, "                                                 
      201 +              f"trend_hist={self.trend_hist}, lag_mode={self.l
          +ag_mode}({lag_desc}), "                                       
      202 +              f"slope_window={self.slope_window}, base_rst={se
          +lf.base_rst}, "                                               
      203 +              f"maxlag_shallow={self.maxlag_shallow}, h_drop_r
          +atio={self.h_drop_ratio}, "                                   
      204                f"max_windows={self.max_windows}, monitor={self.
           monitor_interval} "
      205                f"-> LN params: {len(params)}")
      206  

● Update(adapt/deyo_mlmp_adagate_continual.py)
  ⎿  Added 2 lines
      337          self.grad_buf.clear()
      338          self.grad_hist.clear()
      339          self.marginal_buf.clear()
      340 +        self.slope_buf.clear()                                
      341 +        self.active_z_buf.clear()                             
      342          self.h_max = 0.0
      343  
      344      def _adapt_forward(self, x):

● Now register the method and its CLI args:

● Bash(sed -n '195,205p' adapt/__init__.py)
  ⎿      # DeYO + MLMP + H-MARGIN-regime gate (method1 deep cap unlocked on
     ly when H_margin drops)
         'deyo_mlmp_hmgate_continual': DeYOMLMPHMGateContinual,
     … +15 lines (ctrl+o to expand)

● Bash(sed -n '695,740p' main_continual.py)
  ⎿      elif method in ('deyo_mlmp_hmgate2_continual', 'deyo_mlmp_promptw_
     hmgate2_continual',
                         'deyo_mlmp_textres_hmgate2_continual'):
     … +46 lines (ctrl+o to expand)

● Update(main_continual.py)
  ⎿  Added 1 line, removed 1 line
      693  
      694      # --- DeYO+MLMP HMGate2: deep restore -> PERMANENT best an
           chor (never evicted) ---
      695      elif method in ('deyo_mlmp_hmgate2_continual', 'deyo_mlmp_
           promptw_hmgate2_continual',
      696 -                    'deyo_mlmp_textres_hmgate2_continual'):   
      696 +                    'deyo_mlmp_textres_hmgate2_continual', 'de
          +yo_mlmp_adagate_continual'):                                  
      697          parser.add_argument('--vision_outputs', nargs='+', typ
           e=int,
      698                              default=tuple(range(-1, -19, -1)))
      699          parser.add_argument('--deyo_margin_factor', type=float
           , default=0.5)

● Update(main_continual.py)
  ⎿  Added 23 lines
      715                                   '-> deep restore toward PERMA
           NENT best anchor')
      716          parser.add_argument('--maxlag_shallow', type=int, defa
           ult=6)
      717          parser.add_argument('--monitor_interval', type=int, de
           fault=50)
      718 +        if method == 'deyo_mlmp_adagate_continual':           
      719 +            # self-calibrating trigger (A) and lag (B); see th
          +e module docstring for                                        
      720 +            # why the absolute slope_deadzone / lag_gain are i
          +nert in hmgate2.                                              
      721 +            parser.add_argument('--trend_stat', type=str, defa
          +ult='mad',                                                    
      722 +                                choices=['abs', 'rel', 'mad', 
          +'tstat'],                                                     
      723 +                                help="normalisation of the gra
          +d-norm OLS slope before the "                                 
      724 +                                     "deadzone test. 'abs' = h
          +mgate2 (raw slope); 'mad' = "                                 
      725 +                                     "robust z vs recent slope
          + spread; 'tstat' = slope/SE; "                                
      726 +                                     "'rel' = slope/grad_norm.
          +")                                                            
      727 +            parser.add_argument('--trend_thr', type=float, def
          +ault=0.5,                                                     
      728 +                                help='unitless deadzone on the
          + normalised trend z. Ignored '                                
      729 +                                     'for trend_stat=abs, whic
          +h uses --slope_deadzone.')                                    
      730 +            parser.add_argument('--trend_hist', type=int, defa
          +ult=50,                                                       
      731 +                                help='window count for the MAD
          + scale and the ECDF rank.')                                   
      732 +            parser.add_argument('--lag_mode', type=str, defaul
          +t='ecdf',                                                     
      733 +                                choices=['gain', 'sat', 'ecdf'
          +],                                                            
      734 +                                help="shallow-restore depth. '
          +gain' = hmgate2 "                                             
      735 +                                     "(round(lag_gain*slope), 
          +saturates in practice); "                                     
      736 +                                     "'sat' = ceil(cap*clamp(z
          +/lag_sat)); 'ecdf' = "                                        
      737 +                                     "ceil(cap*rank(z)), tunab
          +le-free and scale-invariant.")                                
      738 +            parser.add_argument('--lag_sat', type=float, defau
          +lt=1.5,                                                       
      739 +                                help='z at which lag_mode=sat 
          +reaches the full budget.')                                    
      740 +                                                              
      741          if method == 'deyo_mlmp_promptw_hmgate2_continual':
      742              # entropy-weighted prompt aggregation (prompt/text
           -template axis)
      743              parser.add_argument('--prompt_weight_beta', type=f
           loat, default=0.0,

● Update(adapt/__init__.py)
  ⎿  Added 1 line
      72  from .deyo_mlmp_hmgate2_continual import DeYOMLMPHMGate2Continu
          al
      73  from .deyo_mlmp_promptw_hmgate2_continual import DeYOMLMPPrompt
          WHMGate2Continual
      74  from .deyo_mlmp_textres_hmgate2_continual import DeYOMLMPTextRe
          sHMGate2Continual
      75 +from .deyo_mlmp_adagate_continual import DeYOMLMPAdaGateContinu
         +al                                                             
      76  from .mlmp_divgate_continual import MLMPDivGateContinual
      77  from .sar_mlmp_smooth_anchor_continual import SARMLMPSmoothAnch
          orContinual
      78  

● Update(adapt/__init__.py)
  ⎿  Added 2 lines
      200      'deyo_mlmp_promptw_hmgate2_continual': DeYOMLMPPromptWHMGa
           te2Continual,
      201      # GDG-PA + learnable text-embedding residual + orthogonali
           ty regularizer
      202      'deyo_mlmp_textres_hmgate2_continual': DeYOMLMPTextResHMGa
           te2Continual,
      203 +    # GDG-PA with self-calibrating trigger (normalised trend) 
          ++ lag (budget fraction)                                       
      204 +    'deyo_mlmp_adagate_continual': DeYOMLMPAdaGateContinual,  
      205      # Continual TTA with MLMP loss + diversity-gated stochasti
           c restoration
      206      'mlmp_divgate_continual': MLMPDivGateContinual,
      207      # Continual TTA with SAR (SAM + reliable filter) + diversi
           ty-gated restore (DivGate replaces SAR recovery)

  Searched for 2 patterns (ctrl+o to expand)

● Bash(sed -n '163,170p' main_continual.py)
  ⎿      parser.add_argument('--method', type=str, default='cotta',
                             help='Adaptation method name')
         parser.add_argument('--log_signals', action='store_true',
     … +9 lines (ctrl+o to expand)

● Bash(conda run -n MLMP python -c "
      import ast,sys…)
  ⎿  ast.parse(src); print('syntax OK')
     from adapt import METHODS if False else None
     ` failed. (See above for error)
     … +8 lines (ctrl+o to expand)

  Read 1 file (ctrl+o to expand)

● Gate 從 window 2 就開始 restore，所以 3 輪的 ctrl 就足以驗證共用路徑。先寫
  sweep 腳本：

● Write(bash/sweep_adagate.sh)
  ⎿  Wrote 108 lines to bash/sweep_adagate.sh
       2 # AdaGate sweep: make GDG-PA's two INERT absolute hyperparameter
       3 # self-calibrating.
       4 # Everything except the gate's trigger rule (A) and shallow-lag 
       5 # to GDG-PA's best config (= save/*/hmgate2_prompt_S0_baseline),
          so any mIoU delta
       7 # is attributable to A/B alone.
          measurement shows/trend_thr replace slope_deadzone=0.002, which
       9 #       degenerates to `slope > 0` (fires 46%/51%, vs P(slope>0)
      10 #   (B) lag_mode replaces lag_gain=1500, which saturates the cap
          in 93%/99.7% of
     … +135 lines (ctrl+o to expand)

● Now the equivalence smoke test — 3 rounds of ctrl, which must reproduce GDG-PA
   exactly:

● Bash(GPU=4 ROUNDS=3 SEQ=1 TAG=_smoke timeout 3000 bash bash/sweep_adagate.sh
      acdc ctrl 2>&1 | tail -5)  ⎿  [23:28] START acdc arm=ctrl (stat=abs thr=0.002 lag=gain sat=1.5) gpu=
     4 rounds=3 -> save/ACDCDataset/adagate_ctrl_smoke/
     [23:33] DONE  acdc arm=ctrl (exit 0)
     [23:33] acdc adagate sweep COMPLETE.

● Bash(echo "=== mIoU (ctrl vs GDG-PA S0)"; paste <(head -4
      save/ACDCDataset/adagate_ctrl_smoke/results_all_rounds.txt) <(head -4
      save/ACDCDataset/hmgate2_prompt_S0_ba…)
  ⎿  === mIoU (ctrl vs GDG-PA S0)
     Round, fog, night, rain, snow, Mean_mIoU     Round, fog, night, rain,
     snow, Mean_mIoU
     … +13 lines (ctrl+o to expand)

● ctrl 完全等價——3 輪 mIoU 每個小數位都相同，12 個 gate window 的
  lag/rst/collapse 全數吻合，z == grad_slope。重構是安全的。現在丟實驗。

● Bash(rm -rf save/ACDCDataset/adagate_ctrl_smoke && \
      nohup env GPU=4 ROUNDS=150 STAGGER=60 bash bash/sweep_adagate.sh acdc
      "Amad0 Amad05 Amad10 Atstat10" > save/_swe…)
  ⎿  launched GPU4 (ACDC, A-only ablation)
     launched GPU5 (ACDC, B-only + A+B)
     launched GPU3 (VOC20, B-focused)

● Bash(sleep 420; nvidia-smi --query-gpu=index,memory.used,utilization.gpu
      --format=csv,noheader; echo "=== jobs up:"; ps aux | grep
      "[m]ain_continual.py --adapt --met…)
  ⎿  Error: Blocked: sleep 420 followed by: nvidia-smi 
     --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader echo 
     "=== jobs up:" ps aux grep "[m]ain_continual.py --adapt --method 
     deyo_mlmp_adagate" wc -l echo "=== per-arm first gate lines:" for f in 
     save/_sweep_logs/adagate_{acdc,v20}_*.log; do printf "%-40s " "$(basename 
     $f)"; grep -m1 "AdaGate\] B" $f 2>/dev/null | cut -c1-125 || echo "(no gate
      line yet)"; done. To wait for a condition, use Monitor with an until-loop 
     (e.g. `until <check>; do sleep 2; done`). To wait for a command you 
     started, use run_in_background: true. Do not chain shorter sleeps to work 
     around this block.

● Bash(until [ "$(ls save/ACDCDataset/adagate_*/gate_log.csv
      save/PascalVOC20Dataset/v20_acdc_matched/adagate_*/gate_log.csv
      2>/dev/null | wc -l)" -ge 12 ]; do sleep 3…)
  ⎿  Running in the background (↓ to manage)

● 趁著跑起來，我先寫好結果彙整工具。

● Write(scripts/collect_adagate.py)
  ⎿  Wrote 103 lines to scripts/collect_adagate.py
       2 """Collect the AdaGate sweep against the GDG-PA (hmgate2) refere
         nce.
       4 For each arm it reports the mIoU summary plus the two gate diagn
         ostics the sweep
         llow-lag spread the trigger firing rate (A) and the realised sha
       6 (B). GDG-PA's lag histogram is binary {0, maxlag_shallow}; a wor
       7 mass on the intermediate lags.
       8 
      10     python scripts/collect_adagate.py                # both data
         sets
     … +111 lines (ctrl+o to expand)

● Bash(nvidia-smi --query-gpu=index,memory.used,utilization.gpu
      --format=csv,noheader; echo "--- adagate procs:"; ps aux | grep -c
      "[m]ain_continual.py --adapt --metho…)
  ⎿  0, 6379 MiB, 99 %
     1, 97041 MiB, 100 %
     2, 27377 MiB, 100 %
     … +6 lines (ctrl+o to expand)

● Bash(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader |
      while read pid mem rest; do p=${pid%,}; g=$(nvidia-smi -q -x 2>/dev/null |
       grep -c x 2>/…)
  ⎿  289735  6370 MiB  other
     3162616  97032 MiB  other
     474046  27366 MiB  other
     … +5 lines (ctrl+o to expand)

● Bash(for p in 3464922 3489079 3489538 3489984 3162616 474046 289735 955391; do
       a=$(ps -p $p -o args= 2>/dev/null); if [ -n "$a" ]; then echo "PID $p :
      $(echo $a | gr…)
  ⎿  PID 3464922 : VLLM::EngineCore
     PID 3489079 : --save_dir save/ACDCDataset/adagate_Amad0/
     PID 3489538 : --save_dir save/ACDCDataset/adagate_Becdf/
     … +17 lines (ctrl+o to expand)

● ⚠️ 另一位使用者的 VLLM engine 在我啟動後佔走了 GPU 4 的 82
  GB。先確認損害並改道。

● Bash(echo "=== GPU4 launcher log:"; cat
      save/_sweep_logs/adagate_wave1_gpu4.log; echo; echo "=== OOM in arm
      logs?"; for a in Amad0 Amad05 Amad10 Atstat10; do f=save/…)
  ⎿  === GPU4 launcher log:
     [23:33] START acdc arm=Amad0 (stat=mad thr=0.0 lag=gain sat=1.5) gpu=4
      rounds=150 -> save/ACDCDataset/adagate_Amad0/
     … +16 lines (ctrl+o to expand)

● Amad05 已被 OOM 殺掉。GPU 4 現在只剩 1.6 GB，不可用——把 A 組全部改道到 GPU 3。

● Bash(pkill -f "bash bash/sweep_adagate.sh acdc Amad0 Amad05 Amad10 Atstat10"
      2>/dev/null; sleep 1
      # kill only the GPU4-resident adagate run (Amad0), leave every othe…)
  ⎿  Error: Exit code 144

● Bash(sleep 6; pgrep -af "deyo_mlmp_adagate" | grep -o "\--save_dir [^ ]*";
      echo "--- launcher procs:"; pgrep -af "sweep_adagate.sh" | cut -c1-100;
      echo "--- gpu:"; n…)
  ⎿  --save_dir save/ACDCDataset/adagate_Amad0/
     --save_dir save/ACDCDataset/adagate_Amad0/
     --save_dir save/ACDCDataset/adagate_Becdf/
     … +23 lines (ctrl+o to expand)

● Bash(for p in $(pgrep -f "adagate_Amad0/"); do kill -9 $p 2>/dev/null && echo
      "killed $p"; done; sleep 8; rm -rf save/ACDCDataset/adagate_Amad0; pgrep
      -f "adagate_Am…)
  ⎿  Error: Exit code 1
     killed 3488681
     killed 3489079

● Bash(sleep 8; rm -rf save/ACDCDataset/adagate_Amad0; pgrep -f "adagate_Amad0/"
       >/dev/null && echo "STILL ALIVE" || echo "GPU4 clear of my jobs";
      nvidia-smi --query-g…)
  ⎿  STILL ALIVE
     3, 11785 MiB, 85468 MiB
     4, 82699 MiB, 14554 MiB
     5, 38775 MiB, 58477 MiB
● GPU 4 記憶體已從 95.6 GB 降回 82.7 GB（只剩別人的 VLLM）。把 A 組改丟 GPU 3：

● Bash(nohup env GPU=3 ROUNDS=150 STAGGER=90 bash bash/sweep_adagate.sh acdc
      "Amad0 Amad05 Amad10" > save/_sweep_logs/adagate_wave1b_gpu3.log 2>&1 &
      echo "relaunched A…)
  ⎿  relaunched A-arms on GPU3

● Bash(until [ "$(pgrep -cf 'deyo_mlmp_adagate_continual')" -ge 22 ] || [ "$(ls
      -d save/ACDCDataset/adagate_*
      save/PascalVOC20Dataset/v20_acdc_matched/adagate_* 2>/dev…)
  ⎿  Running in the background (↓ to manage)

  Read 1 file (ctrl+o to expand)

● While the runs settle, I'll write the spec doc that the project convention
  expects.

● Write(docs/adagate_continual_spec.md)
  ⎿  Wrote 179 lines to docs/adagate_continual_spec.md
       2 # AdaGate — self-calibrating trigger and lag for GDG-PA
         ate_continual.py](../adapt/deyo_mlmp_adagate_continual.py))_adag
       4 **Base**: GDG-PA = `deyo_mlmp_hmgate2_continual` (current best; 
       5 **Sweep**: [bash/sweep_adagate.sh](../bash/sweep_adagate.sh) · *
         *Collector**: [scripts/collect_adagate.py](../scripts/collect_ad
       6 **Status**: implemented, `ctrl` equivalence verified bit-exact, 
       7 11-arm sweep launched 2026-08-02.
       8 ---
      10 ## 1. Problem — two hyperparameters that do not do what they cla
         im
     … +241 lines (ctrl+o to expand)

  Read 1 file (ctrl+o to expand)

● Bash(nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu
      --format=csv,noheader -i 3,5; echo "--- my runs:"; pgrep -af
      deyo_mlmp_adagate | grep -o "\…)
  ⎿  3, 27655 MiB, 69597 MiB, 100 %
     5, 51699 MiB, 45554 MiB, 100 %
     --- my runs:
     … +14 lines (ctrl+o to expand)

  Read 1 file (ctrl+o to expand)

● Bash(for a in Amad0 Amad05 Amad10; do printf "%-8s " $a; if pgrep -f
      "adagate_$a/" >/dev/null 2>&1; then printf "RUNNING  "; else printf
      "not-running  "; fi; grep -c…)
  ⎿  Amad0    RUNNING  0 gate-windows;
     Amad05   RUNNING  0 gate-windows;
     Amad10   not-running  0 gate-windows;  CUDA out of memory Traceback
     … +5 lines (ctrl+o to expand)

● Amad10 的 OOM 是 GPU 4 留下的舊 log；GPU 3 的 launcher 還沒排到它。等全部 11
  個 arm 產出 gate window 再確認。

● Bash(cd /home/tekai324/MLMP; until [ "$(find save/ACDCDataset/adagate_*
      save/PascalVOC20Dataset/v20_acdc_matched/adagate_* -name gate_log.csv
      2>/dev/null | xargs -r…)
  ⎿  Running in the background (↓ to manage)

  Read 1 file (ctrl+o to expand)

● Bash(python3 scripts/collect_adagate.py --dataset acdc 2>&1 | head -40)
  ⎿  ===== ACDC =====
       GDG-PA (hmgate2_prompt_S0_) R150 mean= 31.06 (+0.00)  peak= 31.57@R1
     50 last= 31.57  std(tail30)=0.06
     … +15 lines (ctrl+o to expand)

● 全部 11 個 arm 都在跑並產出正常的 R1（29.2，與 GDG-PA 的 29.21
  一致）。掛一個失敗監看：

● Monitor(AdaGate sweep failures + hourly heartbeat)
  ⎿  Monitor started · task badl14lnw · persistent

● 已經跑起來了。總結我做了什麼：

  診斷 → 方案

  Column 1: A 觸發
  現在的問題（實測）: slope_deadzone=0.002 是絕對值，但 slope 噪音帶是它的
    15×(ACDC) / 150×(VOC20) → 退化成 slope>0，觸發率 0.463/0.507 ≈ 擲硬幣
  新方案: z = slope / (1.4826·MAD(近 50 window 的 
    slope))，門檻變成無量綱的信噪比。離線重放證實兩個資料集的 z
  分布幾乎重合（p75
     0.635 vs 0.656、p90 1.125 vs 1.114），同一個 thr=0.5 給出 0.305/0.311
    的觸發率
  ────────────────────────────────────────
  Column 1: B 深度
  現在的問題（實測）: lag = round(1500·slope)，只要 slope>0.004 就飽和 →
    93.2%/99.7% 的活躍 window 頂到上限，lag 實際是 {0, 6} 二元。要真的鋪開需要
    gain≈153(ACDC) vs ≈17(VOC20)，差 9 倍
  新方案: lag = ceil(cap · u)，u = z 在近期活躍 z 中的 ECDF 
    排名。排名法對任何單調變換不變 → 天生尺度無關，而且直接刪掉 
  lag_gain。重放後
     lag 變成 {1:20, 2:24, 3:26, 4:21, 5:15, 6:12}

  順帶一提：我也試了你可能會直覺想到的
  slope/grad_norm（相對成長率）——它不夠，兩個資料集還是差 8 倍，因為它正規化了
  slope 的大小而非噪音水準。這個負結果本身在論文裡值得寫一句。

  已完成

  - 新方法 adapt/deyo_mlmp_adagate_continual.py，含 4 種 trend
  統計量（abs/rel/mad/tstat）× 3 種 lag 規則（gain/sat/ecdf）
  - 等價性已驗證：ctrl arm（abs+gain）3 輪 ACDC 與 hmgate2_prompt_S0_baseline
  每個 mIoU 小數位、12 個 gate window 的 lag/rst/collapse 全部相同，z == 
  grad_slope。所以任何差異都歸因於 A/B
  - 註冊到 adapt/__init__.py / main_continual.py，sweep 腳本
  bash/sweep_adagate.sh，彙整工具 scripts/collect_adagate.py，spec
  docs/adagate_continual_spec.md

  正在跑（11 個 arm × 150R）

  - GPU 3（ACDC A 組隔離）：Amad0 Amad05 Amad10 ＋（VOC20）Becdf ABmad0_ecdf
  ABmad05_ecdf ABmad05_sat
  - GPU 5（ACDC B 組 + A+B）：Becdf ABmad05_ecdf ABmad10_ecdf ABtstat10_ecdf

  VOC20 是純 lag 測試台（它的 collapse regime 觸發率是 0，100% 的 restore 都走 
  lag 路徑），ACDC 則兩條路徑都走。

  兩件要說清楚的：(1) 我啟動後別人的 VLLM engine 佔走了 GPU 4 的 82 GB，害
  Amad05 OOM，我把 ACDC 組全部改道到 GPU 3 並清掉了殘骸（沒有動任何別人的
  job，包含你在 GPU 3 上跑了 1 天 9 小時的 VOC20 full_15corr）。(2) GPU 3 現在有
   7 個我的 job，競爭下 150R 可能要 30h 以上；等我量到實際的 round/hr
  會視情況再平衡。Atstat10 等 4 個 arm 排在第二波。

  失敗監看已掛上，掛掉會通知我。

● Monitor event: "AdaGate sweep failures + hourly heartbeat"

● Bash(cd /home/tekai324/MLMP; for a in Amad0 Amad05 Amad10; do printf "%-8s "
      $a; pgrep -f "adagate_$a/" >/dev/null && printf "RUNNING " || printf "DEAD
          "; f=save/…)
  ⎿  Amad0    RUNNING log_mtime=23:36 oom=0 gate_windows=0
     Amad05   RUNNING log_mtime=23:38 oom=0 gate_windows=0
     Amad10   RUNNING log_mtime=23:39 oom=0 gate_windows=0
     … +18 lines (ctrl+o to expand)

● 誤報，已查證：那個 OOM 是 GPU 4 時期的舊 log，被 GPU 3
  重啟時覆寫前剛好被監看器讀到。Amad10 現在是 RUNNING（log
  已重寫、oom=0，launcher 也記錄了 23:39 的 START）。

  11 個 arm 全部確認在跑：

  - ACDC (7)：Amad0 Amad05 Amad10 Becdf ABmad05_ecdf ABmad10_ecdf ABtstat10_ecdf
  - VOC20 (4)：Becdf ABmad0_ecdf ABmad05_ecdf ABmad05_sat

  監看繼續掛著。等累積到足夠輪數（大約幾小時後能看出 gate 行為、20+
  輪後能看出趨勢）我再回報，重點會看兩件事：graded 有沒有真的脫離 GDG-PA 的
  0.07（代表 B 修好了），以及提高 trend_thr 讓觸發率從 ~50% 降到 30%/14% 之後
  mIoU 往哪走。

● Background command "Wait until all 11 arms produce gate windows" completed
(exit code 0)

● Bash(python3 scripts/collect_adagate.py 2>&1)  ⎿  ===== ACDC =====
       GDG-PA (hmgate2_prompt_S0_) R150 mean= 31.06 (+0.00)  peak= 31.57@R1
     50 last= 31.57  std(tail30)=0.06
     … +31 lines (ctrl+o to expand)

● Bash(cd /home/tekai324/MLMP; python3 -c "
      import glob,os…)
  ⎿  === ACDC  (round-matched vs GDG-PA)
       adagate_ABmad05_ecdf   R2:  29.29  GDG-PA R2:  29.25   delta=+0.040
       adagate_ABmad10_ecdf   R1:  29.22  GDG-PA R1:  29.21   delta=+0.010
     … +8 lines (ctrl+o to expand)

● Background command "Wait for all arms to settle, then check for OOM" completed (exit code 0)

● 第一批 gate 診斷出來了，機制層面 B 已經確認修好：

  ┌────────────────────────────────┬───────────────┬───────────────────────┐
  │                                │ GDG-PA (150R) │    新 arm（早期）     │
  ├────────────────────────────────┼───────────────┼───────────────────────┤
  │                                │               │ VOC20 Becdf           │
  │ graded（shallow restore        │ ACDC 0.07 /   │ 0.84、ABmad0_ecdf     │
  │ 落在預算內部、非飽和的比例）   │ VOC20 0.06    │ 0.83；ACDC            │
  │                                │               │ ABmad05_ecdf 0.33     │
  ├────────────────────────────────┼───────────────┼───────────────────────┤
  │                                │ VOC20 [1:1    │                       │
  │ shallow-lag 直方圖             │ 2:3 3:11 4:16 │ VOC20 Becdf [1:1 2:4  │
  │                                │  5:19         │ 3:9 4:2 5:1 6:2]      │
  │                                │ **6:709**]    │                       │
  └────────────────────────────────┴───────────────┴───────────────────────┘

  ECDF 的 lag 不再擠在上限，而是真的分散在 1~6。這正是 B 要解決的問題。

  也可以看到 A 的觸發率確實被 trend_thr 拉開了：VOC20 上 Becdf(thr=0) 0.413 →
  ABmad0_ecdf 0.353 → ABmad05_ecdf 0.130，門檻現在是有作用的，不像原本的 0.002
  那樣等於擲硬幣。

  mIoU 目前還不能解讀。同輪次對比雖然全部略高於 GDG-PA（ACDC +0.01~+0.04、VOC20
  +0.00~+0.20），但這是第 1~4 輪，遠在噪音範圍內；GDG-PA 的曲線要到 R150 才爬到
  31.57，勝負得等幾十輪後才看得出來。（collect_adagate.py 預設印的 delta 是對
  GDG-PA 的 150 輪平均，早期會虛假地顯示 −1.8/−4.5，別被那欄誤導。）

  繼續跑，有實質進展再回報。

❯ A 的新方案我還是完全看不懂 你講的太抽像了 稍微解釋清楚一點                    

● 讓我用你資料裡的真實數字，一步一步走一遍。

● Bash(cd /home/tekai324/MLMP; python3 -c "
      import csv,statistics as st…)
  ⎿  === ACDC  (第 201 個 window)
       前 50 個 window 的 slope，取樣: [-0.012, 0.006, -0.02, -0.011, -0.02
     2, -0.01, -0.04, -0.006] ...
     … +18 lines (ctrl+o to expand)

● 先講 A 要解決的問題到底是什麼

  Gate 每個 window 要回答一個問題：「grad_norm 現在是在上升嗎？」

  它手上只有一個數字：slope（最近 10 個 window 的 grad_norm
  擬合直線的斜率）。舊規則是：

  slope > 0.002  →  算「在上升」  →  開始 restore

  問題是：slope = +0.0035 到底算不算「在上升」？
  這句話單獨看是沒有意義的。要回答它，你必須先知道這個 slope 
  平常自己會亂跳多少。

  ▎ 類比：有人跟你說「今天比平均高了 2 度」，這算異常嗎？
  ▎ 在台北（日溫差約 3 度）→ 2 度很明顯，是異常。
  ▎ 在沙漠（日溫差約 20 度）→ 2 度是噪音，什麼都不代表。
  ▎
  ▎ 同一個絕對數字，在不同的波動環境下意義完全不同。 0.002 就是那個「2 
  ▎ 度」——它被寫死了，但 ACDC 和 VOC20 是兩個溫差差 10 倍的地方。

  ---
  新方案：先量「平常會亂跳多少」，再拿它當尺

  新規則不直接看
  slope，而是先問這條串流「你平常波動多大」，再用那個波動當單位去量。三步：

  第 1 步：收集最近 50 個 window 的 slope

  真實資料（ACDC 第 201 個 window 之前的 50 筆）：
  -0.012,  +0.006,  -0.020,  -0.011,  -0.022,  -0.010,  -0.040,  -0.006, ...

  第 2 步：算出「這條串流的噪音尺度」

  用 MAD（median absolute deviation，中位絕對離差），就三個動作：

  1. 取這 50 筆的中位數：            med = -0.0036
  2. 算每一筆離 med 多遠，取絕對值：  |-0.012-(-0.0036)| = 0.0084,
  |0.006-(-0.0036)| = 0.0096, ...
  3. 再取這些距離的中位數：           MAD = 0.0144      ← 「典型偏離量」

  再乘上 1.4826 換算成跟標準差同一個刻度（這是 MAD
  的標準換算常數，讓它在常態分布下等於 σ）：

  噪音尺度 = 1.4826 × 0.0144 = 0.0213

  白話：「這條串流的 slope 平常大概就在 ±0.021 之間亂晃。」

  ▎ 為什麼用 MAD 而不是直接算標準差？ 因為我們要偵測的就是「暴衝」。ACDC 的 
  ▎ slope 最大值出現過 0.88，是典型值的 40 
  ▎ 倍。標準差會被這種暴衝拉大，導致「尺」自己被要測的東西污染，偵測器等於自我失
  ▎ 明。中位數類的統計量不受少數極端值影響。

  第 3 步：把 slope 換算成「幾個噪音單位」

  z = slope / 噪音尺度 = +0.0035 / 0.0213 = +0.16

  z 讀作：「這次的上升是平常波動的 0.16 倍。」 這是一個沒有單位的數字——不管你的
  grad_norm 是 5 還是 6000，不管 slope 是 0.003 還是 0.3，z 都是同一個意思。

  ---
  同一個 window，兩條規則的差別

  ACDC 第 201 個 window，slope = +0.0035

  舊規則:  +0.0035 > 0.002  ?  →  觸發 restore   ← 但這只是平常波動的
  16%，根本是噪音
  新規則:  z = +0.16 > 0.5  ?  →  不觸發          ← 正確地忽略了它

  這就是為什麼舊的「死區」等於沒有死區：0.002 只有 ACDC
  噪音尺度（0.0213）的十分之一，幾乎任何正的 slope 都會通過，所以觸發率 46%（≈
  擲硬幣）。

  而 VOC20 的噪音尺度是 0.2941，是 ACDC 的 14 倍。同一個 0.002 在那裡是噪音的
  0.7%，更加沒有過濾能力。但兩邊的 z 是同一把尺——z=0.5
  在兩個資料集都是「上升幅度達到平常波動的一半」。

  trend_thr 於是有了可以用中文講清楚的意思：

  ┌───────────┬──────────────────────────────┬───────────────────────────┐
  │ trend_thr │             白話             │ 實測觸發率 (ACDC / VOC20) │
  ├───────────┼──────────────────────────────┼───────────────────────────┤
  │ 0.0       │ 只要在上升就算（≈ 舊行為）   │ 0.50 / 0.51               │
  ├───────────┼──────────────────────────────┼───────────────────────────┤
  │ 0.5       │ 上升幅度要超過平常波動的一半 │ 0.305 / 0.311             │
  ├───────────┼──────────────────────────────┼───────────────────────────┤
  │ 1.0       │ 上升幅度要超過平常波動本身   │ 0.135 / 0.139             │
  └───────────┴──────────────────────────────┴───────────────────────────┘

  注意最右欄：兩個資料集的觸發率幾乎一模一樣。這不是我調出來的，是同一個無量綱門
  檻自動達成的——這就是「adaptive」的實證。

  ---
  補充：為什麼不能用更直覺的 slope / grad_norm

  你可能會想：既然要正規化，除以 grad_norm 不就好了？我試過（rel 模式），不行：

  ┌──────────────────────┬──────────┬───────────┬─────────┐
  │                      │ ACDC p90 │ VOC20 p90 │  差距   │
  ├──────────────────────┼──────────┼───────────┼─────────┤
  │ slope / grad_norm    │ 0.005    │ 0.042     │ 8 倍    │
  ├──────────────────────┼──────────┼───────────┼─────────┤
  │ slope / (1.4826·MAD) │ 1.125    │ 1.114     │ 1.01 倍 │
  └──────────────────────┴──────────┴───────────┴─────────┘

  原因：除以 grad_norm 只正規化了訊號的大小（5.25 vs 6.64，只差 1.3
  倍），但兩個資料集真正差 14
  倍的是噪音水準。除錯了東西，所以沒解決問題。這個負結果我也留在
  docs/adagate_continual_spec.md §2 裡，論文可以拿來當一句對照。

● Monitor event: "AdaGate sweep failures + hourly heartbeat"

● Bash(cd /home/tekai324/MLMP; for d in save/ACDCDataset/adagate_*
      save/PascalVOC20Dataset/v20_acdc_matched/adagate_*; do
      f=$d/results_all_rounds.txt; [ -f "$f" ] || c…)
  ⎿  Error: Tool permission request failed: AbortError: Tool permission stream 
     closed before response received

● Monitor event: "AdaGate sweep failures + hourly heartbeat"

● Monitor event: "AdaGate sweep failures + hourly heartbeat"

● Monitor event: "AdaGate sweep failures + hourly heartbeat"

● Monitor event: "AdaGate sweep failures + hourly heartbeat"

● Monitor event: "AdaGate sweep failures + hourly heartbeat"
  ⎿  Interrupted · What should Claude do instead?

● 2 background shell command task(s) from the previous session have no
completion record. They may have been stopped (via the UI, Monitor timeout, or
agent teardown — these leave no transcript marker), or they may have been
running when the previous Claude Code process exited. They have been marked
stopped. Task ids: bcw1f1r74, badl14lnw. Task ids in this notification beginning
 with "__orphan_summary" are internal scan markers, not tasks.

❯ 幫我看一下現在跑的怎麼樣                                                      

● Bash(cd /home/tekai324/MLMP; echo "=== alive:"; pgrep -cf deyo_mlmp_adagate;
      echo "=== GPU:"; nvidia-smi
      --query-gpu=index,memory.used,memory.free,utilization.gpu --…)
  ⎿  === alive:
     1
     === GPU:
     … +17 lines (ctrl+o to expand)

● 全部 11 個 arm 都跑完 150 輪了。

● Bash(cd /home/tekai324/MLMP; python3 scripts/collect_adagate.py --dataset
      acdc; python3 scripts/collect_adagate.py --dataset v20)
  ⎿  ===== ACDC =====
       GDG-PA (hmgate2_prompt_S0_) R150 mean= 31.06 (+0.00)  peak= 31.57@R1
     50 last= 31.57  std(tail30)=0.06
     … +37 lines (ctrl+o to expand)

● 全部跑完了，結果很好——每個 arm 在兩個資料集上都 ≥ 
  GDG-PA。先看趨勢穩定性再下結論：

● Bash(cd /home/tekai324/MLMP; python3 -c "
      import os…)
  ⎿  ACDC — 每 25 輪取樣 + 後段趨勢
     arm             R1     R25    R50    R75    R100   R125   R150     R10
     0->150
     … +22 lines (ctrl+o to expand)

● 全部穩定，沒有任何 arm 崩潰。現在跑第二波補關鍵缺口。先確認 Cityscapes
  參考點：
● Bash(cd /home/tekai324/MLMP; for d in
      save/CityscapesDataset/deyo_mlmp_hmgate2_continual_5corr_sub100
      save/CityscapesDataset/deyo_mlmp_hmgate2_continual; do echo "==…)
  ⎿  == save/CityscapesDataset/deyo_mlmp_hmgate2_continual_5corr_sub100
     missing
     == save/CityscapesDataset/deyo_mlmp_hmgate2_continual
     … +8 lines (ctrl+o to expand)

● GPU 3/4 全空。加兩個更嚴的門檻 arm，然後跑第二波：

● Update(bash/sweep_adagate.sh)
  ⎿  Added 2 lines
      71      ABmad0_ecdf)    echo "mad   0.0   ecdf 1.5" ;;
      72      ABmad05_ecdf)   echo "mad   0.5   ecdf 1.5" ;;
      73      ABmad10_ecdf)   echo "mad   1.0   ecdf 1.5" ;;
      74 +    ABmad15_ecdf)   echo "mad   1.5   ecdf 1.5" ;;             
      75 +    ABmad20_ecdf)   echo "mad   2.0   ecdf 1.5" ;;             
      76      ABtstat10_ecdf) echo "tstat 1.0   ecdf 1.5" ;;
      77      ABmad0_sat)     echo "mad   0.0   sat  1.5" ;;
      78      ABmad05_sat)    echo "mad   0.5   sat  1.5" ;;

● Bash(cd /home/tekai324/MLMP
      nohup env GPU=3 ROUNDS=150 STAGGER=90 bash bash/sweep_adagate.sh v20
      "ABmad10_ecdf ABtstat10_ecdf" > save/_sweep_logs/adagate_w2_v20.log…)
  ⎿  wave2 launched: GPU3 = VOC20 transfer + ACDC bracket, GPU4 = Cityscape
     s (incl. ctrl = GDG-PA reference)

● Monitor(Wave-2 failures + heartbeat)
  ⎿  Monitor started · task bp3p1m9k9 · persistent

● 第一波 11 個 arm 全部完成 150 輪。結論：兩個問題都解決了，而且不只是持平——每個
   arm 在兩個資料集上都 ≥ GDG-PA。

  ACDC（150R）

  ┌───────────────┬────────────────┬───────┬──────┬───────┬──────┬───────┐
  │      arm      │     修的是     │ 觸發  │ mean │ vs GD │ peak │ grade │
  │               │                │  率   │      │ G-PA  │      │   d   │
  ├───────────────┼────────────────┼───────┼──────┼───────┼──────┼───────┤
  │ GDG-PA        │ —              │ 0.463 │ 31.0 │ —     │ 31.5 │ 0.07  │
  │               │                │       │ 6    │       │ 7    │       │
  ├───────────────┼────────────────┼───────┼──────┼───────┼──────┼───────┤
  │ Amad0         │ A（觸發率對齊  │ 0.512 │ 31.0 │ −0.03 │ 31.5 │ 0.17  │
  │               │ ）             │       │ 3    │       │ 0    │       │
  ├───────────────┼────────────────┼───────┼──────┼───────┼──────┼───────┤
  │ Becdf         │ B only         │ 0.488 │ 31.0 │ −0.04 │ 31.5 │ 0.69  │
  │               │                │       │ 2    │       │ 8    │       │
  ├───────────────┼────────────────┼───────┼──────┼───────┼──────┼───────┤
  │ ABmad05_ecdf  │ A+B            │ 0.292 │ 31.3 │ +0.25 │ 32.0 │ 0.71  │
  │               │                │       │ 1    │       │ 5    │       │
  ├───────────────┼────────────────┼───────┼──────┼───────┼──────┼───────┤
  │ Amad10        │ A only         │ 0.155 │ 31.6 │ +0.56 │ 32.3 │ 0.13  │
  │               │                │       │ 2    │       │ 2    │       │
  ├───────────────┼────────────────┼───────┼──────┼───────┼──────┼───────┤
  │ ABmad10_ecdf  │ A+B            │ 0.167 │ 31.6 │ +0.62 │ 32.3 │ 0.69  │
  │               │                │       │ 8    │       │ 8    │       │
  ├───────────────┼────────────────┼───────┼──────┼───────┼──────┼───────┤
  │ ABtstat10_ecd │ A+B            │ 0.037 │ 32.0 │ +1.01 │ 33.1 │ 0.00  │
  │ f             │                │       │ 7    │       │ 6    │       │
  └───────────────┴────────────────┴───────┴──────┴───────┴──────┴───────┘

  VOC20（150R）

  ┌─────────────────┬────────┬───────┬───────────┬────────┐
  │       arm       │ 觸發率 │ mean  │ vs GDG-PA │ graded │
  ├─────────────────┼────────┼───────┼───────────┼────────┤
  │ GDG-PA          │ 0.506  │ 77.11 │ —         │ 0.06   │
  ├─────────────────┼────────┼───────┼───────────┼────────┤
  │ ABmad0_ecdf     │ 0.519  │ 77.54 │ +0.43     │ 0.67   │
  ├─────────────────┼────────┼───────┼───────────┼────────┤
  │ Becdf（B only） │ 0.528  │ 77.56 │ +0.45     │ 0.67   │
  ├─────────────────┼────────┼───────┼───────────┼────────┤
  │ ABmad05_sat     │ 0.321  │ 77.79 │ +0.68     │ 0.74   │
  ├─────────────────┼────────┼───────┼───────────┼────────┤
  │ ABmad05_ecdf    │ 0.315  │ 77.80 │ +0.69     │ 0.67   │
  └─────────────────┴────────┴───────┴───────────┴────────┘

  ---
  三個實質發現

  1. B 確實修好了，而且在 VOC20 單獨就有效。 graded 從 GDG-PA 的 0.06/0.07 跳到
  0.67~0.74，lag 直方圖從 […6:709] 變成 [1:147 2:130 3:151 4:129 5:124 
  6:111]。關鍵是 Becdf——只改 B、觸發規則完全不動——在 VOC20 拿到 +0.45。因為
  VOC20 從不進入 collapse regime，100% 的 restore 都走 lag 路徑，所以這個 +0.45
  完全歸因於 B。

  2. A 是 ACDC 的主要槓桿，而且揭露了 GDG-PA 一直在「過度 restore」。
  劑量反應乾淨且單調：

  觸發率 0.51 → 31.03      GDG-PA 的退化死區等於擲硬幣
  觸發率 0.32 → 31.16
  觸發率 0.16 → 31.62
  觸發率 0.04 → 32.07      peak 衝到 33.16（GDG-PA 只有 31.57）

  原本的 0.002 讓 gate 在 46% 的 window
  都在拉回模型，把爬升壓住了。給它一個真的死區，模型才爬得上去。

  3. 兩個修正各自對應一個 regime——這正好是論文要的敘事。 ACDC（塌縮型）靠
  A、VOC20（均勻退化型）靠 B；Becdf 在 ACDC 是 −0.04（無效），Amad0 在 ACDC 也是
   −0.03（無效）。必須兩個都修，而且它們互補而非重疊。

  穩定性：所有 arm R100→R150 的變化都在 −0.34 ~ +0.21，無一崩潰。

  ---
  要留意的一點

  ABtstat10_ecdf（ACDC 最高的 32.07）觸發率只有 3.7%，整場 150 輪只做了 9 次
  shallow restore——它基本上是「幾乎不 restore」。而且它的 std(tail30)=0.25
  是其他 arm 的 4 倍，peak 在 R91 之後掉了 0.6。DeYO+MLMP 完全不加 gate 在 ACDC
  是會崩的（33.5→6.0），所以最佳點一定在 0~4% 之間某處，tstat 1.0
  可能已經站在懸崖邊。我不會直接把它當結論。

  另外 tstat 在我的離線重放裡跨資料集差 2.6 倍（mad 只差 1.01 倍），所以它的
  ACDC 勝利未必能轉移——這正是第二波要驗的。

  第二波（已丟下去）

  - GPU 3：VOC20 ABmad10_ecdf + ABtstat10_ecdf（轉移性測試：ACDC
  上勝出的嚴格門檻在 VOC20 是否也贏？如果 VOC20
  反而變差，「單一無量綱門檻通用」的主張就要收斂）；ACDC ABmad15_ecdf +
  ABmad20_ecdf（夾出最佳點，看反 U 型在哪裡轉）
  - GPU 4：Cityscapes ctrl（= GDG-PA 參考，這台機器上沒有現成的）+ ABmad05_ecdf
  + ABmad10_ecdf + ABtstat10_ecdf（第三個資料集）

  Spec 已寫在 docs/adagate_continual_spec.md，等第二波回來我再更新結果章節。

● Monitor event: "Wave-2 failures + heartbeat"

● Bash(cd /home/tekai324/MLMP; date +%H:%M; echo "=== wave2 launchers:"; cat
      save/_sweep_logs/adagate_w2_*.log; echo "=== running:"; pgrep -af
      deyo_mlmp_adagate | grep…)
  ⎿  19:50
     === wave2 launchers:
     [19:49] START acdc arm=ABmad15_ecdf (stat=mad thr=1.5 lag=ecdf sat=1.5
     … +12 lines (ctrl+o to expand)

● 正常——第二波是 19:49 剛啟動，90 秒 stagger，目前 3 個已起，其餘 5
  個會陸續進來。無需處理。

❯ 我明天要做這幾個改動的簡報報告給教授聽 請你幫我：                             
  1. 詳細說明A和B分別遇到什麼問題 並且應該怎麼解決 白話一點解釋                 
  2. 告訴我你做的實驗那麼多組每一組分別做了什麼、表現怎麼樣                     
  3. 視覺化呈現幾張圖來說明實驗結果                                             

● 明天要報告，我做一份完整的視覺化簡報。先確認第二波進度：

● Bash(cd /home/tekai324/MLMP; date "+%m-%d %H:%M"; for d in
      save/ACDCDataset/adagate_*
      save/PascalVOC20Dataset/v20_acdc_matched/adagate_*
      save/CityscapesDataset/adaga…)
  ⎿  08-04 19:54
     ACDCDataset/adagate_ABmad05_ecdf                         UP    R150
     ACDCDataset/adagate_ABmad10_ecdf                         UP    R150
     … +16 lines (ctrl+o to expand)

● 第二波才剛開始（R1~R2），明天報告會以第一波完整的 11 個 arm
  為主。先把資料整理出來：

● Bash(cd /home/tekai324/MLMP; python3 -c "
      import csv,json,os,statistics as st…)
  ⎿  acdc {'GDG-PA': (31.061, 0.4633333333333333, 0.07), 'Amad0': (31.029,
     0.5116666666666667, 0.17), 'Amad05': (31.157, 0.31666666666666665, 0.0
     6), 'Amad10': (31.62, 0.155, 0.13), 'Becdf': (31.023, 0.48833333333333
     … +7 lines (ctrl+o to expand)

● Skill(artifact-design)
  ⎿  Successfully loaded skill

● Skill(dataviz)
  ⎿  Successfully loaded skill

  Read 1 file (ctrl+o to expand)

● Bash(cd /home/tekai324/MLMP; python3 -c "
      import csv,statistics as st…)
  ⎿  ACDC: 噪音尺度中位數=0.0238   0.002 落在 0.084 個噪音單位   (n=550)
     VOC20: 噪音尺度中位數=0.2690   0.002 落在 0.007 個噪音單位   (n=1450)

● Bash(cd /tmp/claude-1014/bundled-skills/2.1.220/59c4c4a19e15df9603b47d5d3ac77c
      0a/dataviz; echo "--- 3-series (lines), light:"; node
      scripts/validate_palette.js "#2a7…)
  ⎿  --- 3-series (lines), light:
     /bin/bash: line 1: node: command not found
     --- 3-series, dark:
     /bin/bash: line 1: node: command not found
  ⎿  Shell cwd was reset to /home/tekai324/MLMP

● Bash(command -v nodejs bun deno 2>/dev/null; ls /usr/lib/node_modules
      2>/dev/null | head -3; conda run -n MLMP node --version 2>&1 | head -2; ls
       ~/.nvm/versions/node…)
  ⎿  Error: Exit code 2
     /tmp/tmpkoaojn5d: line 21: node: command not found
     ERROR conda.cli.main_run:execute(142): `conda run node --version` failed. 
     (See above for error)
● 沒有 Node runtime，我用文件記載的驗證過的 surface，並在 Python 裡自行複驗與
  surface 相關的對比度檢查：

● Bash(python3 -c "
      def lin(c):…)
  ⎿  LIGHT surface #fcfcfb
       series-1 blue     #2a78d6   4.30:1  OK
       series-2 orange   #eb6834   3.12:1  OK
     … +13 lines (ctrl+o to expand)

● Bash(cd /home/tekai324/MLMP; python3 -c "
      import json…)
  ⎿  bytes 13670

● Bash(cd /home/tekai324/MLMP; python3 - <<'PY'
      import json, io, os…)
  ⎿  part 1 written

● Bash(cd /home/tekai324/MLMP; cat >> /tmp/claude-1014/-home-tekai324-MLMP/38eb7
      291-b267-40f5-924c-e4239395f620/scratchpad/adagate_report.html <<'JSEOF'
      const NS="http…)
  ⎿  written: 54288 bytes

● 簡報做好了：https://claude.ai/code/artifact/526949a9-3ccd-435d-a27a-eb5833788f
  91

  單頁、可捲、明暗主題都有，軌跡圖可以滑鼠 hover
  看任一輪的數值。內容依你的三個要求編排：

  1. 問題與解法（白話） — §01–§03
  - A 用「台北 vs 沙漠的日溫差」類比解釋為什麼絕對門檻沒意義，配一張圖顯示 0.002
   在 ACDC 落在 0.084 個噪音單位、VOC20 落在 0.007（差 12
  倍），而新門檻依定義落在同一位置
  - B 用實測 lag 直方圖證明「連續 lag」是假的
  - 解法用 ACDC 第 201 個 window 的真實數字逐步走一遍 MAD → z
  - 含等價性驗證那段（這是教授最可能追問的：你怎麼知道沒改壞）

  2. 每組實驗做什麼 + 表現 — §04（設計表，每組「在問什麼」）＋ §05（ACDC/VOC20
  兩張完整結果表，含 mean/delta/peak/last/std/graded）

  3. 四張圖
  - 門檻在噪音單位軸上的位置（診斷 A）
  - lag 分布正規化比較（診斷 B，93% 擠在上限 vs 鋪滿 1–6）
  - 劑量反應圖（觸發率 vs mIoU，單調下降 → GDG-PA 一直在過度 restore）—
  我認為這是最有說服力的一張
  - 150 輪雙面板軌跡（可 hover）

  兩件事先提醒你，免得報告時被問倒：

  主推哪一組。 ACDC 最高的 ABtstat10_ecdf（+1.01）觸發率只有 3.7%，150 輪只做了
  9 次 restore，後段 std 是別人的 4 倍，而且 tstat 跨資料集差 2.6 倍（mad 只差
  1.01 倍）。我在 §06 建議主推 ABmad10_ecdf（+0.62、std 0.06），把 tstat
  當「上界探針」講。這樣被問「那你怎麼知道它不會在 300 輪崩掉」時你有話可答。

  第二波還沒結果。 轉移性測試（VOC20 的嚴格門檻）、最佳點夾擊、Cityscapes 昨晚
  19:49 才啟動，目前才 R1–R2，明天報告時大概率仍未完成。我在 §07
  把它們列成「還在跑」，沒有預設結論。

❯ 再跟我說明一下解決B用的那個ECDF排名方法是什麼 解釋清楚一點                    
  另外我發現tstat的組別表現相當好 可以跟我說明一下這是什麼嗎？                  

● Bash(cd /home/tekai324/MLMP; python3 -c "
      import csv,math,statistics as st…)
  ⎿  --- 雜訊大、t 小  (window 529)
         10 個 grad_norm: [5.59, 4.92, 5.14, 5.46, 5.43, 4.75, 5.16, 6.55,
     5.86, 4.69]
     … +7 lines (ctrl+o to expand)

● Bash(cd /home/tekai324/MLMP; python3 -c "
      import csv…)
  ⎿  ABmad05_ecdf (VOC20) 真實的 (z, u, lag) —— 從第 600 個 window 附近取 1
     2 筆有觸發的:
      window       z     u=ECDF排名   cap   lag
     … +14 lines (ctrl+o to expand)

● B：ECDF 排名

  要解決的轉換問題

  B 要把「趨勢強度」轉成「倒回幾個 window」。任何用數值做的映射（lag = 倍率 × 
  強度）都會遇到同一件事：倍率必須配合強度的尺度，而尺度每個資料集不同（這就是
  lag_gain 需要 153 vs 17 的原因）。

  ECDF 的想法是完全不看數值，只看排名：

  ▎ 不問「這次的 z 有多大」，而問「這次的 z，在我最近觸發過的那些當中排第幾」。

  演算法（三步）

  1. 維護一個「最近 50 次有觸發的 z」的緩衝區    ← 只存有觸發的，不存沒觸發的
  2. u = 緩衝區裡「比現在這個 z 還小」的比例      ← 0 到 1 之間的百分位
  3. lag = max(1, ceil(cap × u))                 cap = min(6, 距離歷史最佳點)

  u 讀作「這次的退化訊號，在我最近的經驗裡屬於第幾個百分位」。u=0.9
  就是「這是最近少見的強訊號」→ 用掉 90% 的預算深度；u=0.1
  就是「這在最近算輕微」→ 只倒回 1 格。

  真實運作（VOC20 ABmad05_ecdf，第 603–643 個 window）

  ┌────────┬──────┬───────────┬─────┬───────┐
  │ window │  z   │ u（排名） │ cap │ → lag │
  ├────────┼──────┼───────────┼─────┼───────┤
  │    603 │ 0.88 │      0.34 │   6 │     3 │
  ├────────┼──────┼───────────┼─────┼───────┤
  │    604 │ 1.21 │      0.74 │   6 │     5 │
  ├────────┼──────┼───────────┼─────┼───────┤
  │    613 │ 0.59 │      0.08 │   6 │     1 │
  ├────────┼──────┼───────────┼─────┼───────┤
  │    615 │ 1.02 │      0.60 │   6 │     4 │
  ├────────┼──────┼───────────┼─────┼───────┤
  │    634 │ 1.59 │      0.98 │   6 │     6 │
  ├────────┼──────┼───────────┼─────┼───────┤
  │    643 │ 0.54 │      0.02 │   6 │     1 │
  └────────┴──────┴───────────┴─────┴───────┘

  那 473 次觸發的 z 分布是 p10=0.57 / p50=0.97 / p90=1.55 —— 全部擠在 0.5~1.6 
  這個窄帶裡。這正是 lag_gain 失敗的根源：任何固定倍率乘上這麼窄的一段值，不是全
  部塞到下限就是全部飽和到上限。但排名會自動把這個窄帶拉開鋪滿 0~1。

  關鍵性質：單調不變性

  排名只依賴「誰比誰大」，不依賴「大多少」。所以：

  - 把所有 z 乘以 1000 → 排名完全不變 → lag 完全不變
  - 換一種 trend 統計量（mad → tstat → 原始 slope）→ 只要它們對同一批 window
  的排序一致，lag 就一致

  所以 ECDF 天生就是尺度無關的，不需要任何校準常數 —— lag_gain 
  直接消失，而且沒有新的超參數取代它。 這是 sat 模式做不到的（它是 u = 
  clamp(z/lag_sat)，還是需要 lag_sat
  這個數值門檻）。實測上兩者表現幾乎一樣（VOC20 77.80 vs 77.79），所以選 ECDF 
  純粹是因為它少一個超參數。

  （緩衝區不足 10 筆時 u 退回 0.5，這只影響開頭幾個 window。）

  ---
  tstat 是什麼

  它問的是跟 MAD 完全不同的問題

  ┌─────┬──────────────┬───────────────────────────────────────────────────┐
  │     │ 拿什麼當比較 │                       在問                        │
  │     │     基準     │                                                   │
  ├─────┼──────────────┼───────────────────────────────────────────────────┤
  │     │ 過去 50 個   │ 這次的 slope，跟我以前的 slope                    │
  │ mad │ window 的    │ 比，算大嗎？（跨時間）                            │
  │     │ slope        │                                                   │
  ├─────┼──────────────┼───────────────────────────────────────────────────┤
  │ tst │ 這 10 個點離 │ 這 10 個點真的在一條上升的線上嗎，還是只是亂跳、  │
  │ at  │ 擬合線有多散 │ 被硬擬合出一個斜率？（窗內）                      │
  └─────┴──────────────┴───────────────────────────────────────────────────┘

  公式

  OLS 除了給你斜率，還能告訴你這個斜率估得有多準：

  SE(slope) = 殘差標準差 / √Sxx        ← 斜率本身的不確定度
  t = slope / SE(slope)                ← 斜率是它自身不確定度的幾倍

  真實對照（ACDC，兩個 slope 相近的 window）

  window 529   grad_norm = [5.59, 4.92, 5.14, 5.46, 5.43, 4.75, 5.16, 6.55,
  5.86, 4.69]
               slope = +0.0237   殘差標準差 = 0.590   SE = 0.0649   →  t = +0.37
               ↑ 上上下下亂跳，那條「上升線」根本不可信

  window 26    grad_norm = [4.96, 5.32, 5.12, 5.17, 5.16, 5.52, 5.37, 5.64,
  5.18, 5.44]
               slope = +0.0421   殘差標準差 = 0.173   SE = 0.0191   →  t = +2.21
               ↑ 點乖乖貼著線走，上升是真的

  兩個 window 的 slope 只差不到 2 倍，t 卻差了 6 倍。 舊規則（slope > 
  0.002）兩個都會觸發；tstat > 1.0 只會觸發後者。

  為什麼它在 ACDC 表現好

  ACDC 的 grad_norm 序列雜訊很大（殘差標準差常常 0.5 以上，而典型 slope 只有
  0.02）。所以 ACDC 有大量「斜率不小、但其實是亂跳」的假警報，tstat
  正好專門擋這一類。實測 P(t > 1.0) = 0.064，觸發率降到 3.7% —— 而我們已經知道
  ACDC 上觸發率越低表現越好，所以它拿到最高分。

  三個要小心的地方

  1. 它不是真正的顯著性檢定，別在論文裡這樣寫。 n=10、df=8 時 t=1.0 只對應單尾
  p≈0.17，一點都不嚴格。更根本的是：相鄰 window 的 slope 共用 10 個點裡的 9
  個，序列高度自相關，t 分布的名目 p 值在這裡不成立。正確的說法是「以窗內殘差為
  尺度的無量綱趨勢強度」，不是「統計檢定」。（我在簡報 §04 的表格裡寫成「OLS
  顯著性檢定」，這句要改掉——教授很可能會抓這點。）

  2. 它跨資料集不穩定。 離線重放：ACDC 的 t 分布 p90 = 0.85，VOC20 是 2.25 —— 差
   2.6 倍（相比之下 mad 只差 1.01 倍）。所以同一個 t=1.0 在 VOC20 會鬆很多，ACDC
   的勝利未必轉移。這正是第二波在測的（VOC20 ABtstat10_ecdf 昨晚啟動）。

  3. 它贏的方式讓 B 幾乎沒被啟用。 這組的 graded = 0.00，因為整場 150 輪只做了 9
   次 shallow restore。換句話說 ABtstat10_ecdf 的 +1.01 主要是「幾乎不 
  restore」換來的，不是 A+B 機制運作良好換來的。 加上它後 30 輪 std =
  0.25（別組的 4 倍）、peak 在 R91 之後掉了 0.6，它可能站在「restore
  太少會漂移」的懸崖邊。

  所以我的建議不變：主推 ABmad10_ecdf（+0.62、std 0.06、graded 0.69），tstat 
  當上界探針講。 如果第二波顯示 tstat 在 VOC20 也贏，那再升格成主推也不遲。

❯ 我想問一下mad的方法需要前面50個window的slope數值                              
  那剛開始跑不到50個windows食藥怎麼辦                                           

● Bash(cd /home/tekai324/MLMP; python3 -c "
      import csv…)
  ⎿  ACDC ABmad05_ecdf (trend_thr=0.5) 的前 14 個 window —— 暖身期實際行為
      win     slope        z  since_min  lag     rst  說明
        1   +0.0000    +0.00          0    0   0.000  buf<5 → scale=0 → z=±
     … +14 lines (ctrl+o to expand)

● 好問題——這是實作上必須處理的邊界情況。答案是：不是「等滿 50 
  才開始」，而是滾動緩衝區，有幾筆就用幾筆。

  實際的處理

  deyo_mlmp_adagate_continual.py 裡是一個 deque(maxlen=50)，每個 window 結束後把
   slope 推進去。算 MAD 時用的是當下緩衝區裡有的全部，只有一個下限：

  _MAD_MIN_HIST = 5

  if len(self.slope_buf) < _MAD_MIN_HIST:      # 不到 5 筆
      scale = 0.0
  else:
      med = self._median(self.slope_buf)        # 用「現有的」不管幾筆
      scale = 1.4826 * self._median([abs(x - med) for x in self.slope_buf])

  if scale > 1e-9:
      return slope / scale
  return 0.0 if abs(slope) < 1e-12 else math.copysign(10.0, slope)   # 退化 
  fallback

  所以有三個階段：

  ┌────────────┬─────────────┬──────────────────────────────────────────────┐
  │    階段    │   緩衝區    │                     行為                     │
  ├────────────┼─────────────┼──────────────────────────────────────────────┤
  │ window 1–5 │ < 5 筆      │ scale=0 → z = ±10 → 退回舊行為（slope>0      │
  │            │             │ 就觸發）                                     │
  ├────────────┼─────────────┼──────────────────────────────────────────────┤
  │ window     │ 5~49 筆     │ 真的 MAD，只是樣本少、估得粗                 │
  │ 6–50       │             │                                              │
  ├────────────┼─────────────┼──────────────────────────────────────────────┤
  │ window 51+ │ 滿 50       │ 完整估計                                     │
  │            │ 筆滾動      │                                              │
  └────────────┴─────────────┴──────────────────────────────────────────────┘

  真實跑出來的樣子（ACDC ABmad05_ecdf，thr=0.5）

  win   slope        z   since_min  lag    rst
    1  +0.0000    +0.00       0      0   0.000   ← 第一個 window
  本來就沒有過去可倒回
    2  +0.8803   +10.00       1      1   0.010   ┐
    3  +0.3407   +10.00       2      1   0.010   │ 暖身：z 被鎖在
  +10，等於舊規則
    4  +0.1146   +10.00       3      2   0.010   │
    5  +0.0381   +10.00       4      2   0.010   ┘
    6  +0.0694    +0.41       5      0   0.000   ← 第一個真正的 z，立刻開始過濾
    7  +0.0725    +0.67       6      3   0.010
    9  +0.0274    +0.48       8      0   0.000   ← 0.48 < 0.5，正確地擋掉
   11  +0.0037    +0.08      10      0   0.000

  Window 6 一有 5 筆歷史，z 馬上落回合理範圍（0.08~0.82），沒有出現失控值。

  影響有多大

  - 暖身期（前 5 個 window）= ACDC 的 0.8%、VOC20 的 0.3%
  - 緩衝區滿 50 之前 = ACDC 的 8.3%、VOC20 的 3.3%

  而且暖身期正好在模型還幾乎等於 source 狀態的時候：此時「倒回幾個 window
  前」的快照跟現在的權重差異極小，restore 幾乎是 no-op。加上 cap = min(6, 
  windows_since_min) 這個天然限制——第 2 個 window 最多只能倒回 1
  格——所以就算暖身期判斷偏鬆，實際造成的擾動也很小（上表 lag 只有 1、2）。

  為什麼選「暖身偏寬鬆」而不是「暖身不觸發」

  兩種設計都合理，我選了寬鬆的，理由有二：

  1. 它退化成 GDG-PA 的行為，所以 ctrl arm 才能做到 bit-identical
  驗證。如果暖身期改成「不觸發」，abs
  模式就不會等價於原方法，等價性驗證這個工具就沒了。
  2. 保守版有失效風險：如果暖身期完全不 restore，而某個資料集在前 250 個 batch
  就開始退化，那就等於把防護關掉。寬鬆版最壞情況只是多做幾次 1% 的 restore。

  兩個可以在論文裡先講清楚的點

  _MAD_MIN_HIST = 5 和 trend_hist = 50
  都是新引入的結構參數——雖然它們是無量綱的「樣本數」而不是尺度綁定的閾值（這是關
  鍵差別，樣本數不需要跨資料集校準），但嚴格說仍然是兩個新常數。誠實的說法是：

  ▎ 我們刪掉了 2 個尺度綁定的閾值（slope_deadzone、lag_gain），代價是引入 1 
  ▎ 個尺度無關的門檻（trend_thr）加上估計視窗長度。

  --trend_hist 是暴露在 CLI 的，如果教授問「50
  是怎麼來的」，目前的誠實答案是沒有掃過——我是取「夠長到讓 MAD
  穩定、又短到能跟上串流變化」的常見預設。要補的話這是個乾淨的敏感度實驗（20 /
  50 / 100 /
  200），不過我建議排在第二波那三個問題（轉移性、最佳點、Cityscapes）之後。

● Monitor event: "Wave-2 failures + heartbeat"

● All 8 wave-2 runs are up and healthy — nothing to act on.

● Monitor event: "Wave-2 failures + heartbeat"

● Routine heartbeat — 8 runs still alive, no action needed.

● Monitor event: "Wave-2 failures + heartbeat"

● Bash(cd /home/tekai324/MLMP; python3 -c "
      import os…)
  ⎿  Error: Tool permission request failed: AbortError: Tool permission stream 
     closed before response received

● Monitor event: "Wave-2 failures + heartbeat"
  ⎿  Interrupted · What should Claude do instead?
● No completion record was found for this background shell command from the
previous session. It may have been stopped (via the UI, Monitor timeout, or
agent teardown — these leave no transcript marker), or it may have been running
when the previous Claude Code process exited. Check the output file for partial
results before assuming it completed.
  ⎿  1 skill available

❯ 請問結果跑出來怎麼樣                                                          

● Bash(cd /home/tekai324/MLMP; date "+%m-%d %H:%M"; echo "alive: $(pgrep -cf
      deyo_mlmp_adagate)"; for d in save/ACDCDataset/adagate_ABmad15_ecdf
      save/ACDCDataset/adaga…)
  ⎿  08-05 20:51
     alive: 1
     ACDCDataset/adagate_ABmad15_ecdf                           R150
     … +7 lines (ctrl+o to expand)

● 第二波 8 組也全部跑完 150 輪了。

● Bash(cd /home/tekai324/MLMP; python3 -c "
      import csv,os,statistics as st…)
  ⎿  ===== ACDC =====
       組別              觸發率     mean   vs基準    peak        last   tai
     l_sd  graded
     … +21 lines (ctrl+o to expand)

● Bash(cd /home/tekai324/MLMP; python3 -c "
      import os…)
  ⎿  後段趨勢（peak 之後有沒有掉）
     --- ACDC
       GDG-PA            R100= 31.36 R125= 31.36 R150= 31.57   R100→150  +0
     … +19 lines (ctrl+o to expand)

● 結果出來了，而且推翻了我先前給你的建議。先更新資料：

● Bash(cd /home/tekai324/MLMP; python3 -c "
      import csv,json,os,statistics as st…)
  ⎿  bytes 22120
     acdc ['GDG-PA', 'Amad0', 'Amad05', 'Amad10', 'Becdf', 'ABmad05_ecdf',
     'ABmad10_ecdf', 'ABmad15_ecdf', 'ABmad20_ecdf', 'ABtstat10_ecdf']
     … +3 lines (ctrl+o to expand)

● Now updating the report with wave-2 results and the corrected recommendation:

● Bash(cd /home/tekai324/MLMP; F=/tmp/claude-1014/-home-tekai324-MLMP/38eb7291-b
      267-40f5-924c-e4239395f620/scratchpad/adagate_report.html
      python3 - "$F" <<'PY'…)
  ⎿  data + masthead updated

● Bash(cd /home/tekai324/MLMP; F=/tmp/claude-1014/-home-tekai324-MLMP/38eb7291-b
      267-40f5-924c-e4239395f620/scratchpad/adagate_report.html
      python3 - "$F" <<'PY'…)
  ⎿  cityscapes table + transferability figure inserted

● Bash(cd /home/tekai324/MLMP; F=/tmp/claude-1014/-home-tekai324-MLMP/38eb7291-b
      267-40f5-924c-e4239395f620/scratchpad/adagate_report.html
      python3 - "$F" <<'PY'…)
  ⎿  sections 6 and 7 rewritten

❯ 請繼續                                                                        

● Bash(cd /home/tekai324/MLMP; F=/tmp/claude-1014/-home-tekai324-MLMP/38eb7291-b
      267-40f5-924c-e4239395f620/scratchpad/adagate_report.html
      python3 - "$F" <<'PY'…)
  ⎿  JS updated: 63032 bytes

● Bash(cd /home/tekai324/MLMP; F=/tmp/claude-1014/-home-tekai324-MLMP/38eb7291-b
      267-40f5-924c-e4239395f620/scratchpad/adagate_report.html
      python3 - "$F" <<'PY'…)
  ⎿  OK — anchors, functions, and data all verified
     arms: {'acdc': 10, 'v20': 7, 'city': 4}
     city fire: {'GDG-PA': 0.479, 'ABmad05_ecdf': 0.299, 'ABmad10_ecdf': 0.
     105, 'ABtstat10_ecdf': 0.296}

● Update(docs/adagate_continual_spec.md)
  ⎿  Added 49 lines
      146  
      147  ---
      148  
      149 +## 3b. RESULTS — both waves complete (19 runs × 150R, 3 datase
          +ts)                                                           
      150 +                                                              
      151 +Reference on ACDC/VOC20 is `hmgate2_prompt_S0_baseline`; on Ci
          +tyscapes it is the `ctrl`                                     
      152 +arm (bit-equivalent to GDG-PA), since no same-machine GDG-PA C
          +ityscapes run existed.                                        
      153 +                                                              
      154 +| arm | ACDC mean (Δ) | VOC20 mean (Δ) | Cityscapes mean (Δ) |
      155 +|---|---|---|---|                                             
      156 +| GDG-PA / ctrl | 31.06 | 77.11 | 23.85 |                     
      157 +| `Becdf` (B only) | 31.02 (−0.04) | **77.56 (+0.45)** | — |  
      158 +| `Amad0` (A only, firing matched) | 31.03 (−0.03) | — | — |  
      159 +| **`ABmad05_ecdf`** | **31.31 (+0.25)** | **77.80 (+0.69)** |
          + **23.97 (+0.12)** |                                          
      160 +| `ABmad10_ecdf` | 31.68 (+0.62) | 77.90 (+0.79) | **23.74 (−0
          +.10)** |                                                      
      161 +| `ABmad15_ecdf` | 31.81 (+0.74) | — | — |                    
      162 +| `ABmad20_ecdf` | 32.12 (+1.06) | — | — |                    
      163 +| `ABtstat10_ecdf` | 32.07 (+1.01) | 77.84 (+0.73) | 23.91 (+0
          +.07) |                                                        
      164 +                                                              
      165 +**(B) is fixed.** `graded` goes from 0.02–0.07 (all three refe
          +rences) to 0.60–0.83. `Becdf`                                 
      166 +— B alone, trigger untouched — earns +0.45 on VOC20, where 100
          + % of restores take the lag                                   
      167 +path, so the gain is attributable to B alone. On ACDC, B alone
          + does nothing (−0.04) and A                                   
      168 +alone at matched firing does nothing (−0.03): **the two fixes 
          +are complementary, one per                                    
      169 +regime**, matching the project's existing collapse-vs-uniform-
          +degradation split.                                            
      170 +                                                              
      171 +**(A) shows the base method was over-restoring.** All three re
          +ferences fire in 0.46–0.51 of                                 
      172 +windows with `graded` ≈ 0 — the same pathology reproduces inde
          +pendently on a third dataset.                                 
      173 +                                                              
      174 +**Transferability — the decisive result.** Firing rate under o
          +ne unitless threshold:                                        
      175 +                                                              
      176 +| threshold | ACDC | VOC20 | Cityscapes | spread | verdict |  
      177 +|---|---|---|---|---|---|                                     
      178 +| `mad 0.5` | 0.292 | 0.315 | 0.299 | **1.1×** | transfers |  
      179 +| `mad 1.0` | 0.167 | 0.155 | 0.105 | 1.6× | marginal |       
      180 +| `tstat 1.0` | 0.037 | 0.251 | 0.296 | **8.0×** | **does not 
          +transfer** |                                                  
      181 +                                                              
      182 +`tstat`'s ACDC win (+1.01) came from *accidentally firing at 3
          +.7 % on ACDC*, not from the                                   
      183 +statistic being better: once VOC20/Cityscapes push its firing 
          +to 0.25–0.30, it performs the                                 
      184 +same as `mad 0.5` (77.84 vs 77.80; 23.91 vs 23.97). **The stat
          +istic choice is not the lever                                 
      185 +— cross-dataset firing-rate stability is.**                   
      186 +                                                              
      187 +**`ABmad05_ecdf` is the recommended configuration** — the only
          + arm positive on all three                                    
      188 +datasets with stable tails (peak→last −0.25 / −0.35 / −0.05). 
          +`mad 1.0` scores higher on                                    
      189 +ACDC and VOC20 *mean* but is **negative on Cityscapes** (−0.10
          +) and decays −1.19 from peak                                  
      190 +on VOC20 (78.89@R84 → 77.70@R150). The ACDC-only monotone "low
          +er firing is better" trend                                    
      191 +(down to 3.5 % / +1.06) is **dataset-specific**: the inverted-
          +U turns much earlier on                                       
      192 +Cityscapes.                                                   
      193 +                                                              
      194 +**Not yet done**: all runs are seed=0; `trend_hist` has never 
          +been swept.                                                   
      195 +                                                              
      196 +---                                                           
      197 +                                                              
      198  ## 4. What counts as success
      199  
      200  The primary claim is **methodological**, not a leaderboard del
           ta: two absolute, dataset-bound

● 兩波共 19
  組全部跑完，涵蓋三個資料集。結果推翻了我上次給你的主推建議，先講這件事。

  轉移性測試 —— 這波最關鍵的結果

  同一個無量綱門檻，在三個資料集上的觸發率：

  ┌──────────┬───────┬───────┬────────────┬──────┬───────┬────────┬───────┐
  │   門檻   │ ACDC  │ VOC20 │ Cityscapes │ 極差 │ ACDC  │ VOC20  │ City  │
  │          │       │       │            │      │   Δ   │   Δ    │   Δ   │
  ├──────────┼───────┼───────┼────────────┼──────┼───────┼────────┼───────┤
  │ mad 0.5  │ 0.292 │ 0.315 │ 0.299      │ 1.1× │ +0.25 │ +0.69  │ +0.12 │
  ├──────────┼───────┼───────┼────────────┼──────┼───────┼────────┼───────┤
  │ mad 1.0  │ 0.167 │ 0.155 │ 0.105      │ 1.6× │ +0.62 │ +0.79  │ −0.10 │
  ├──────────┼───────┼───────┼────────────┼──────┼───────┼────────┼───────┤
  │ tstat    │ 0.037 │ 0.251 │ 0.296      │ 8.0× │ +1.01 │ +0.73  │ +0.07 │
  │ 1.0      │       │       │            │      │       │        │       │
  └──────────┴───────┴───────┴────────────┴──────┴───────┴────────┴───────┘

  tstat 的 ACDC 勝利是假象。 它在 ACDC 拿 +1.01，是因為它剛好在 ACDC 只觸發
  3.7%；到了 VOC20 和 Cityscapes，觸發率被拉到 0.25–0.30，表現就退回跟 mad 0.5
  幾乎一樣（77.84 vs 77.80、23.91 vs 23.97）。

  這帶出一個比原本更好的論文結論：真正的槓桿不是選哪個統計量，而是觸發率能不能跨
  資料集保持穩定。 mad 的極差是 1.1 倍，tstat 是 8 倍。

  更正：主推應該是 ABmad05_ecdf，不是 ABmad10_ecdf

  我上次建議 mad 1.0。第二波顯示它在 Cityscapes 是 −0.10（劣於基準），而且在
  VOC20 從 peak 78.89@R84 一路掉到 77.70@R150（−1.19）。

  ABmad05_ecdf 是唯一三個資料集全正、且尾段全部穩定的組別（peak→last 只有 −0.25
  / −0.35 / −0.05）。

  其他確認的事

  ACDC 的單調下降繼續，但那是 ACDC 專屬的。 mad 1.5（觸發 7.0%）+0.74、mad 
  2.0（3.5%）+1.06，成為 ACDC 最高分。但同樣的嚴格化在 Cityscapes 早就轉負了 ——
  反 U 型存在，只是每個資料集轉折點不同。所以「一直調低觸發率」不能寫成通則。

  第三個資料集獨立重現了病灶。 Cityscapes 的 ctrl 觸發率 0.479、graded 0.02 ——
  跟 ACDC（0.463 / 0.07）、VOC20（0.506 /
  0.06）完全一致。三個資料集都證實基準在過度 restore 且 lag 是死的。

  B 在三個資料集都修好了：graded 從 0.02–0.07 → 0.60–0.83。

  ---
  簡報已更新（同一個網址）：https://claude.ai/code/artifact/526949a9-3ccd-435d-a
  27a-eb5833788f91

  新增了 Cityscapes 結果表、轉移性長條圖（三根等高 vs 三根落差 8
  倍，一眼就看得懂）、三面板軌跡圖（展示 mad 1.0 在 VOC20 衰退、在 Cityscapes
  轉負），§06 和 §07
  重寫，並明確標示了對先前建議的更正。docs/adagate_continual_spec.md
  也補上了結果章節。

  還沒做的兩件事我也寫進去了：全部都是 seed=0（Cityscapes 的 +0.12
  幅度小，需要多 seed 才能確認不是噪音），以及 trend_hist=50 從未掃過。

❯ 幫我把這個session的發現寫進docs/裡面還有專案記憶裡面                          
  希望之後的session可以熟悉今天的發現                                           

  Listed 1 directory (ctrl+o to expand)

● Bash(cd /home/tekai324/MLMP; sed -n '1195,1215p' docs/EXPERIMENT_STATUS.md;
      echo "=== tail:"; tail -6 docs/EXPERIMENT_STATUS.md; echo "=== total
      lines:"; wc -l docs/…)  ⎿  === tail:
     - Catalog: `docs/gate_mechanisms_catalog.md` (EN) + `_zh.md` (詳細中文
     ，逐術語).
     … +9 lines (ctrl+o to expand)

● Update(docs/EXPERIMENT_STATUS.md)
  ⎿  Added 101 lines
      1184  
      1185  ---
      1186  
      1187 +## 19. Phase S: AdaGate — GDG-PA's two inert hyperparameters 
           +made self-calibrating (2026-08-02 → 08-05)                   
      1188 +                                                             
      1189 +Spec: **[docs/adagate_continual_spec.md](adagate_continual_sp
           +ec.md)**. Method                                             
      1190 +`adapt/deyo_mlmp_adagate_continual.py`; sweep `bash/sweep_ada
           +gate.sh`; collector                                          
      1191 +`scripts/collect_adagate.py`. Report artifact (updated, 3 dat
           +asets):                                                      
      1192 +https://claude.ai/code/artifact/526949a9-3ccd-435d-a27a-eb583
           +3788f91                                                      
      1193 +                                                             
      1194 +### S-1. The diagnosis — two hyperparameters that measurably 
           +do nothing                                                   
      1195 +                                                             
      1196 +Motivating question from the user: can GDG-PA's thresholds be
           + made *adaptive*, for a better                               
      1197 +paper framing? Measurement on `save/*/hmgate2_prompt_S0_basel
           +ine/gate_log.csv` found that                                 
      1198 +**two of them were already inert**, so making them adaptive *
           +removes* tunables rather than                                
      1199 +adding any:                                                  
      1200 +                                                             
      1201 +- **(A) `slope_deadzone = 0.002`** is meant to reject noise-l
           +evel grad-norm rises, but the                                
      1202 +  slope noise band is ±0.03 (ACDC) / ±0.35 (VOC20) — 15× / 15
           +0× larger. It degenerates to                                 
      1203 +  `slope > 0`: firing 0.463 / 0.507 vs `P(slope>0)` ≈ 0.5. It
           + filters nothing, and it is an                               
      1204 +  absolute constant applied to two slope distributions 10× ap
           +art in scale.                                                
      1205 +- **(B) `lag_gain = 1500`** is meant to scale restore depth c
           +ontinuously with degradation                                 
      1206 +  speed, but saturates the cap whenever `slope > 6/1500 = 0.0
           +04` — true in **93.2 %** (ACDC)                              
      1207 +  / **99.7 %** (VOC20) of active windows. Realised lag is bin
           +ary: ACDC `{0:322, 6:162}`,                                  
      1208 +  VOC20 `{0:741, 6:709}`. Spreading lag over 1..6 would need 
           +`lag_gain ≈ 153` (ACDC) vs                                   
      1209 +  `≈ 17` (VOC20) — a 9× gap, so **no single absolute gain can
           + work**.                                                     
      1210 +                                                             
      1211 +Both pathologies reproduce independently on **Cityscapes** (`
           +ctrl` fires 0.479, `graded` 0.02).                           
      1212 +                                                             
      1213 +### S-2. The fix                                             
      1214 +                                                             
      1215 +- **(A) `trend_stat`**: normalise the OLS slope before thresh
           +olding. `mad` = robust z-score                               
      1216 +  `slope / (1.4826·MAD(last 50 slopes))`; also `tstat` = `slo
           +pe/SE(slope)`, `rel` =                                       
      1217 +  `slope/grad_norm`, `abs` = hmgate2. `trend_thr` becomes a u
           +nitless SNR deadzone.                                        
      1218 +- **(B) `lag_mode`**: restore depth as a **fraction of the bu
           +dget**, `lag = ceil(cap·u)`.                                 
      1219 +  `ecdf` sets `u` = rank of z among the last 50 *active* z — 
           +rank-based, hence invariant to                               
      1220 +  any monotone rescaling of the trend statistic and **free of
           + tunables** (`lag_gain` deleted).                            
      1221 +  `sat` uses `u = clamp(z/lag_sat)`; `gain` is hmgate2.      
      1222 +                                                             
      1223 +**Equivalence verified**: `trend_stat=abs, lag_mode=gain` con
           +sumes no extra RNG and reproduced                            
      1224 +`hmgate2_prompt_S0_baseline` to every mIoU digit over 3 ACDC 
           +rounds and on all 12 gate windows.                           
      1225 +This is what licenses attributing every delta to A/B.        
      1226 +                                                             
      1227 +**Negative result worth reporting**: the intuitive `slope/gra
           +d_norm` normalisation is                                     
      1228 +**insufficient** — ACDC p90 0.005 vs VOC20 0.042, still 8× ap
           +art, because it normalises the                               
      1229 +signal's *magnitude* (5.25 vs 6.64, 1.3×) not its *noise leve
           +l* (11×). `mad` gives 1.125 vs                               
      1230 +1.114 (1.01×).                                               
      1231 +                                                             
      1232 +### S-3. Results — 19 runs × 150R × 3 datasets               
      1233 +                                                             
      1234 +| arm | ACDC (Δ) | VOC20 (Δ) | Cityscapes (Δ) |              
      1235 +|---|---|---|---|                                            
      1236 +| GDG-PA / `ctrl` | 31.06 | 77.11 | 23.85 |                  
      1237 +| `Becdf` (B only) | 31.02 (−0.04) | **77.56 (+0.45)** | — | 
      1238 +| `Amad0` (A only, matched firing) | 31.03 (−0.03) | — | — | 
      1239 +| **`ABmad05_ecdf`** ← **recommended** | **31.31 (+0.25)** | 
           +**77.80 (+0.69)** | **23.97 (+0.12)** |                      
      1240 +| `ABmad10_ecdf` | 31.68 (+0.62) | 77.90 (+0.79) | **23.74 (−
           +0.10)** |                                                    
      1241 +| `ABmad15_ecdf` | 31.81 (+0.74) | — | — |                   
      1242 +| `ABmad20_ecdf` | 32.12 (+1.06) | — | — |                   
      1243 +| `ABtstat10_ecdf` | 32.07 (+1.01) | 77.84 (+0.73) | 23.91 (+
           +0.07) |                                                      
      1244 +                                                             
      1245 +1. **B is fixed and matters on its own in the uniform-drift r
           +egime.** `graded` goes 0.02–0.07                             
      1246 +   → 0.60–0.83. `Becdf` (B alone, trigger untouched) earns **
           ++0.45 on VOC20**, where the                                  
      1247 +   collapse regime never fires so 100 % of restores take the 
           +lag path.                                                    
      1248 +2. **A is the ACDC lever and shows GDG-PA was over-restoring.
           +** Firing 0.51→31.03,                                        
      1249 +   0.32→31.16, 0.16→31.62, 0.035→32.12.                      
      1250 +3. **The two fixes are complementary, one per regime.** On AC
           +DC, B alone does nothing                                     
      1251 +   (−0.04) and A-at-matched-firing does nothing (−0.03). Maps
           + onto the existing                                           
      1252 +   collapse-vs-uniform-degradation split (§4/§5, contribution
           + doc §4).                                                    
      1253 +4. **Transferability is the decisive result** — firing rate u
           +nder one unitless threshold:                                 
      1254 +                                                             
      1255 +| threshold | ACDC | VOC20 | Cityscapes | spread | verdict | 
      1256 +|---|---|---|---|---|---|                                    
      1257 +| `mad 0.5` | 0.292 | 0.315 | 0.299 | **1.1×** | transfers | 
      1258 +| `mad 1.0` | 0.167 | 0.155 | 0.105 | 1.6× | marginal |      
      1259 +| `tstat 1.0` | 0.037 | 0.251 | 0.296 | **8.0×** | **does not
           + transfer** |                                                
      1260 +                                                             
      1261 +**`tstat`'s ACDC win was an artifact of accidentally firing a
           +t 3.7 % there**, not of the                                  
      1262 +statistic being better: once VOC20/Cityscapes push its firing
           + to 0.25–0.30 it matches                                     
      1263 +`mad 0.5` (77.84 vs 77.80; 23.91 vs 23.97). **The statistic c
           +hoice is not the lever;                                      
      1264 +cross-dataset firing-rate stability is.** Do NOT frame `tstat
           +` as a "significance test" in                                
      1265 +the paper — with a 10-point sliding window sharing 9/10 point
           +s, the series is heavily                                     
      1266 +autocorrelated and the nominal t-distribution does not apply;
           + call it a residual-scaled                                   
      1267 +unitless trend strength.                                     
      1268 +                                                             
      1269 +### S-4. Standing recommendation and open items              
      1270 +                                                             
      1271 +**`ABmad05_ecdf`** (`trend_stat=mad, trend_thr=0.5, lag_mode=
           +ecdf`) is the configuration to                               
      1272 +carry forward — the only arm positive on all three datasets w
           +ith stable tails (peak→last                                  
      1273 +−0.25 / −0.35 / −0.05). `mad 1.0` scores higher on ACDC/VOC20
           + *mean* but is **negative on                                 
      1274 +Cityscapes** and decays −1.19 from peak on VOC20 (78.89@R84 →
           + 77.70@R150). The ACDC-only                                  
      1275 +monotone "lower firing is better" trend is **dataset-specific
           +** — the inverted-U turns much                               
      1276 +earlier on Cityscapes.                                       
      1277 +                                                             
      1278 +Open: **all runs are seed=0** (Cityscapes' +0.12 needs multi-
           +seed); **`trend_hist=50` was                                 
      1279 +never swept**; Cityscapes threshold bracket (mad 0.3 / 0.7) w
           +ould locate its earlier turn.                                
      1280 +                                                             
      1281 +Honest cost accounting for the paper: 2 *scale-bound* thresho
           +lds deleted                                                  
      1282 +(`slope_deadzone`, `lag_gain`), 1 *scale-free* threshold intr
           +oduced (`trend_thr`) plus 2                                  
      1283 +structural sample-count parameters (`trend_hist=50`, MAD warm
           +-up min 5) that do not need                                  
      1284 +per-dataset calibration.                                     
      1285 +                                                             
      1286 +---                                                          
      1287 +                                                             
      1288  *End of research-arc document. For any detail not covered her
            e, follow the reading map in §13.*

● Now CLAUDE.md — the header pointer and method table:

● Update(CLAUDE.md)
  ⎿  Added 16 lines
       4  
       5  # MLMP — Project Guide for Claude
       6  
       7 +> **🟩 Newest (2026-08-05, Phase S): `deyo_mlmp_adagate_continu
         +al` supersedes GDG-PA as the                                   
       8 +> configuration to carry forward.** GDG-PA's `slope_deadzone=0.
         +002` and `lag_gain=1500` were                                  
       9 +> measured to be **inert** — the deadzone degenerates to `slope
         +>0` (fires 46–51 % on all three                                
      10 +> datasets), and lag saturates the cap in 93–99.7 % of active w
         +indows (realised lag is binary                                 
      11 +> `{0,6}`). AdaGate replaces both with unitless statistics: `tr
         +end_stat=mad` (robust z of the                                 
      12 +> slope vs its own recent spread) and `lag_mode=ecdf` (lag = ra
         +nk of z within budget, which                                   
      13 +> **deletes `lag_gain` outright**). **Recommended config `ABmad
         +05_ecdf`**                                                     
      14 +> (`--trend_stat mad --trend_thr 0.5 --lag_mode ecdf`): ACDC **
         +31.31 (+0.25)**, VOC20                                         
      15 +> **77.80 (+0.69)**, Cityscapes **23.97 (+0.12)** — the only ar
         +m positive on all three with                                   
      16 +> stable tails. `trend_thr=0.5` self-calibrates to firing 0.292
         +/0.315/0.299 (spread 1.1×);                                    
      17 +> `tstat` does NOT transfer (0.037/0.251/0.296, spread 8.0×) so
         + its ACDC +1.01 is an artifact.                                
      18 +> `abs`+`gain` reproduces GDG-PA **bit-identically** (verified)
         + — use it as the control arm.                                  
      19 +> Full arc: **[docs/EXPERIMENT_STATUS.md](docs/EXPERIMENT_STATU
         +S.md) §19**, spec                                              
      20 +> [docs/adagate_continual_spec.md](docs/adagate_continual_spec.
         +md). Caveats: all seed=0;                                      
      21 +> `trend_hist=50` never swept.                                 
      22 +>                                                              
      23  > **🔀 Merged `origin/2026.06.04` (學長) on 2026-06-04**: broug
          ht in **36 new methods** (deyo/rotta/dat/kff baselines, CLIPArT
          T variants, MLMP topk/minprompt/smooth-anchor variants, TENT si
          ggate/contgate/layered/anchor variants), a **2nd backbone CAT-S
          eg** (`ovss/catseg/`), **ACDCMerged3/6/10** class-merging datas
          ets, universal `entropy_log.csv`, and new CLI (`--split train+v
          al`, `--subset_size`, `--acdc_overlay_corruptions`, `--corrupti
          on_severity`). **None are in our research narrative yet — they 
          are an unvalidated toolbox.** Full catalog + smooth-anchor expl
          ainer: **[docs/merged_2026-06-04_inventory.md](docs/merged_2026
          -06-04_inventory.md)**. Gotchas: needs `einops`; our old `--sev
          erity` is gone (use `--corruption_severity`); 55 methods / 293 
          scripts.
      24  >
      25  > **🟢 Current status (2026-05-25)**:

● Update(CLAUDE.md)
  ⎿  Added 1 line
      133  | **`tent_divgate_continual`** | `adapt/tent_divgate_continual
           .py` | [tent_divgate_continual_spec.md](docs/tent_divgate_cont
           inual_spec.md) | **Best confirmed method. All 150R complete.**
            Baseline (h_thr=1.8): mean=30.14, R150=29.02. After cautious_
           rst sweep + threshold tune (h_thr=1.6, h_warn=1.4, cau_rst=0.0
           1): **mean=31.59, peak=32.96@R27, R150=31.34**. Beats MLMP-epi
           sodic by +1.0 mIoU, stable to R150. |
      134  | `sar_continual` | `adapt/sar_continual.py` (+ `adapt/sam.py`
           ) | [2026-05-17-sar-eata-v20-design.md](docs/2026-05-17-sar-ea
           ta-v20-design.md) | **Ran, partial win.** ACDC (post-bug-fix n
           ot yet rerun; existing data is pre-fix `>` direction): mean=30
           .32, peak=33.38@R15, R150=25.31 — **higher peak than TENT-DivG
           ate but W-shape dips at R50/R100/R150**. VOC20 weather: ~70.9 
           ≈ No-Adapt 70.79 (headroom-limited). **Direction bug**: recove
           ry was `>` should be `<` per paper; `E_0` rescaled `0.7→0.1`. 
           |
      135  | `eata_continual` | `adapt/eata_continual.py` | [2026-05-17-s
           ar-eata-v20-design.md](docs/2026-05-17-sar-eata-v20-design.md)
            | **Implemented, never run** (deferred behind SAR results). P
           re-stream Fisher EWC + reliable+non-redundant filter. |
      136 +| **`deyo_mlmp_adagate_continual`** | `adapt/deyo_mlmp_adagate
          +_continual.py` | [adagate_continual_spec.md](docs/adagate_cont
          +inual_spec.md) | **Phase S — current recommendation.** GDG-PA 
          +with a self-calibrating trigger (`trend_stat`/`trend_thr`) and
          + lag (`lag_mode`). Best arm `ABmad05_ecdf`: ACDC 31.31 / VOC20
          + 77.80 / Cityscapes 23.97, positive on all three. `abs`+`gain`
          + = bit-identical GDG-PA. |                                    
      137  | **`sar_divgate_continual`** | `adapt/sar_divgate_continual.p
           y` | [sar_divgate_continual_spec.md](docs/sar_divgate_continua
           l_spec.md) | **Implemented, NOT YET RUN (Phase J, next priorit
           y).** Hybrid: SAR's SAM + reliable filter, with DivGate's 3-ti
           er graded stochastic restore replacing SAR's hard recovery. Hy
           pothesis: SAM peak (33.38) + DivGate stability (R150 31.34) → 
           ideal mean ≥ 32.5 on ACDC. |
      138  
      139  The original three-direction reframing is in [proposal_after_c
           ma.md](proposal_after_cma.md) (written after CMA-Proto failed)
           . Direction A = layered restoration; Direction B = diversity g
           ate; Direction C (two-timescale meta-adapt) is still deferred.

● Now the project memory:

  Recalled 1 memory, wrote 2 memories (ctrl+o to expand)

● Now marking the superseded claim in the older memory so a future session
  doesn't act on it:

  Wrote 1 memory (ctrl+o to expand)

● Bash(cd /home/tekai324/MLMP; echo "=== docs 變更:"; git status --short docs/
      CLAUDE.md adapt/deyo_mlmp_adagate_continual.py bash/sweep_adagate.sh
      scripts/collect_adaga…)
  ⎿  === docs 變更:
      M CLAUDE.md
      M adapt/__init__.py
     … +12 lines (ctrl+o to expand)

● 都寫好了。

  docs/

  docs/adagate_continual_spec.md（新增）—
  完整方法規格：兩個缺陷的量測證據、修正設計、離線校準數字、19
  組實驗結果、以及等價性驗證怎麼做的。

  docs/EXPERIMENT_STATUS.md §19（新增章節）— 接在 Phase O 之後的「Phase S:
  AdaGate」，依專案既有格式寫成 S-1 診斷 / S-2 修正 / S-3 結果 / S-4
  建議與待辦。這是專案的主要入口文件，之後的 session 讀 §19 就能接手。

  CLAUDE.md（更新）— 最上方加了一段 🟩 header pointer（新 session
  一開始就會讀到），並在 Phase 2 方法表加入 deyo_mlmp_adagate_continual 一列。

  專案記憶

  adagate_selfcalibrating_gate.md（新增）+ MEMORY.md
  索引第一行。裡面除了結果，我特別寫進了四個容易踩的坑，因為這些是這個 session
  真正花代價學到的：

  - 不要推薦 mad 1.0 —— 它 ACDC/VOC20 的 mean 比較高，但 Cityscapes
  是負的、VOC20 尾段掉
  1.19。我第一波就是這樣建議然後被第二波推翻，寫進去避免下次重蹈。
  - ACDC 的「觸發率越低越好」是資料集專屬的，不能當通則。
  - 不要把 tstat 講成顯著性檢定 —— 滑動視窗共用 9/10 個點，自相關嚴重，名目 t
  分布不成立。
  - slope/grad_norm 是乾淨的負結果，值得在論文寫一句。

  也在舊的 gdgpa_prompt_sweep.md 頂端加了 SUPERSEDED 標註並連到新記憶 ——
  那份記憶說「GDG-PA 是 current best」，現在已經過期，不標的話未來 session
  會照著錯的前提走。（該檔關於 prompt 軸的結論仍然有效，因為這次只動了 gate 的
  trigger/lag，我也註明了。）

  另外記了兩件操作面的事：collect_adagate.py 的 Δ 欄位是跟參考的 150
  輪平均比，所以partial run 會顯示假的負值；以及這台機器是共用的（別人的 VLLM
  中途佔走 GPU 4 的 82GB 害一組 OOM），下次放 job 前要先看剩餘
  VRAM、不要殺不是自己的 process。

  程式碼本身（adapt/deyo_mlmp_adagate_continual.py、bash/sweep_adagate.sh、scrip
  ts/collect_adagate.py 及 adapt/__init__.py / main_continual.py
  的註冊）目前都還是 uncommitted。要我幫你開一個 branch commit 起來嗎？

❯ 現在請幫我把3個datasets的mad那個方法的參數多跑一些實驗看看能不能找到真的在三  
  個組別都表現得很好的 例如確認3個datasets都有mad05 mad07 mad10 mad13等等       
                                                                                
  另外我還想要針對mad05做一個實驗在三個datasets上面：把shallow restore的6       
  lag上線改掉 換成10, 15, 20等等不同的超參數                                    
  看看可不可以讓V20最後不要跌落GDG-PA之下                                       
                                                                                
  請幫我把這些實驗丟下去跑 儘量跑在GPU 3 4 5上面 原則上都可以儘量用             

● Bash(cd /home/tekai324/MLMP; nvidia-smi
      --query-gpu=index,memory.used,memory.free,utilization.gpu
      --format=csv,noheader; echo "=== 現有 mad arms:"; for r in save/ACDCD…)
  ⎿  0, 0 MiB, 97252 MiB, 0 %
     1, 0 MiB, 97252 MiB, 0 %
     2, 51687 MiB, 45566 MiB, 100 %
     … +12 lines (ctrl+o to expand)

● GPU 3/4/5 全空。先擴充 sweep 腳本支援 maxlag_shallow 這個軸：

● Update(bash/sweep_adagate.sh)
  ⎿  Added 24 lines, removed 14 lines
      60  H_DROP_RATIO=0.9; MAXLAG_SHALLOW=6; MONITOR_INTERVAL=50; TREND_
          HIST=50
      61  mkdir -p save/_sweep_logs
      62  
      63 +# fields: trend_stat  trend_thr  lag_mode  lag_sat  maxlag_shal
         +low                                                            
      64  arm_cfg() {
      65    case "$1" in
      65 -    ctrl)           echo "abs   0.002 gain 1.5" ;;             
      66 -    Amad0)          echo "mad   0.0   gain 1.5" ;;             
      67 -    Amad05)         echo "mad   0.5   gain 1.5" ;;             
      68 -    Amad10)         echo "mad   1.0   gain 1.5" ;;             
      69 -    Atstat10)       echo "tstat 1.0   gain 1.5" ;;             
      70 -    Becdf)          echo "abs   0.002 ecdf 1.5" ;;             
      71 -    ABmad0_ecdf)    echo "mad   0.0   ecdf 1.5" ;;             
      72 -    ABmad05_ecdf)   echo "mad   0.5   ecdf 1.5" ;;             
      73 -    ABmad10_ecdf)   echo "mad   1.0   ecdf 1.5" ;;             
      74 -    ABmad15_ecdf)   echo "mad   1.5   ecdf 1.5" ;;             
      75 -    ABmad20_ecdf)   echo "mad   2.0   ecdf 1.5" ;;             
      76 -    ABtstat10_ecdf) echo "tstat 1.0   ecdf 1.5" ;;             
      77 -    ABmad0_sat)     echo "mad   0.0   sat  1.5" ;;             
      78 -    ABmad05_sat)    echo "mad   0.5   sat  1.5" ;;             
      66 +    ctrl)           echo "abs   0.002 gain 1.5  6" ;;          
      67 +    Amad0)          echo "mad   0.0   gain 1.5  6" ;;          
      68 +    Amad05)         echo "mad   0.5   gain 1.5  6" ;;          
      69 +    Amad10)         echo "mad   1.0   gain 1.5  6" ;;          
      70 +    Atstat10)       echo "tstat 1.0   gain 1.5  6" ;;          
      71 +    Becdf)          echo "abs   0.002 ecdf 1.5  6" ;;          
      72 +    ABmad0_ecdf)    echo "mad   0.0   ecdf 1.5  6" ;;          
      73 +    ABmad03_ecdf)   echo "mad   0.3   ecdf 1.5  6" ;;          
      74 +    ABmad05_ecdf)   echo "mad   0.5   ecdf 1.5  6" ;;          
      75 +    ABmad07_ecdf)   echo "mad   0.7   ecdf 1.5  6" ;;          
      76 +    ABmad10_ecdf)   echo "mad   1.0   ecdf 1.5  6" ;;          
      77 +    ABmad13_ecdf)   echo "mad   1.3   ecdf 1.5  6" ;;          
      78 +    ABmad15_ecdf)   echo "mad   1.5   ecdf 1.5  6" ;;          
      79 +    ABmad20_ecdf)   echo "mad   2.0   ecdf 1.5  6" ;;          
      80 +    ABtstat10_ecdf) echo "tstat 1.0   ecdf 1.5  6" ;;          
      81 +    ABmad0_sat)     echo "mad   0.0   sat  1.5  6" ;;          
      82 +    ABmad05_sat)    echo "mad   0.5   sat  1.5  6" ;;          
      83 +    # --- maxlag_shallow axis, pinned at the recommended mad 0.
         +5 + ecdf ---                                                   
      84 +    # cap = min(maxlag_shallow, windows_since_min); ECDF then s
         +preads lag over 1..cap,                                        
      85 +    # so raising it deepens the reachable shallow anchor withou
         +t changing the trigger.                                        
      86 +    ABmad05_ecdf_ml10) echo "mad 0.5 ecdf 1.5 10" ;;           
      87 +    ABmad05_ecdf_ml15) echo "mad 0.5 ecdf 1.5 15" ;;           
      88 +    ABmad05_ecdf_ml20) echo "mad 0.5 ecdf 1.5 20" ;;           
      89      *) return 1 ;;
      90    esac
      91  }

● Update(bash/sweep_adagate.sh)
  ⎿  Added 1 line, removed 1 line
      93  run_one() {
      94    local arm="$1"
      95    local cfg; cfg="$(arm_cfg "$arm")" || { echo "unknown arm: $a
          rm"; return 1; }
      96 -  read -r TSTAT TTHR LMODE LSAT <<< "$cfg"                     
      96 +  read -r TSTAT TTHR LMODE LSAT MAXLAG <<< "$cfg"              
      97    local SAVE="save/${DATASET}/${SUBDIR}adagate_${arm}${TAG}/"
      98    local LOG="save/_sweep_logs/adagate_${DS}_${arm}${TAG}.log"
      99    echo "[$(date +%H:%M)] START $DS arm=$arm (stat=$TSTAT thr=$T
          THR lag=$LMODE sat=$LSAT) gpu=$GPU rounds=$ROUNDS -> $SAVE"

● Bash(cd /home/tekai324/MLMP; python3 - <<'PY'
      p='bash/sweep_adagate.sh'; s=open(p).read()…)
  ⎿  11:#       active windows, making lag binary {0, maxlag_shallow}.
     60:H_DROP_RATIO=0.9; MAXLAG_SHALLOW=6; MONITOR_INTERVAL=50; TREND_HIST
     =50
     … +10 lines (ctrl+o to expand)

● Now a memory-gated queue so 20 jobs fill GPU 3/4/5 as slots free, without
  OOM-ing anyone:

● Write(bash/run_adagate_queue.sh)
  ⎿  Wrote 92 lines to bash/run_adagate_queue.sh
       2 # GPU job QUEUE for the AdaGate Phase-S follow-up (20 jobs on GP
       3 #s 3/4/5).
         s per job:, both pinned to lag_mode=ecdf so only ONE thing varie
         capes}) trend_thr grid  {0.3, 0.7, 1.3, 1.5} x {acdc, v20, citys
       6 #       -> completes a 0.3/0.5/0.7/1.0/1.3/1.5 curve on all thre
       7 #          (0.5 and 1.0 already exist everywhere; ACDC also has 
       8 #          Goal: find a threshold that is good on ALL THREE, sin
       9 #          negative on Cityscapes and mad 0.5 is currently the o
      10 #   (2) maxlag_shallow {10, 15, 20} at mad 0.5 x all three datas
         ets
     … +114 lines (ctrl+o to expand)

● Bash(cd /home/tekai324/MLMP; bash -n bash/sweep_adagate.sh && echo "sweep
      syntax OK"; bash -n bash/run_adagate_queue.sh && echo "queue syntax OK"
      echo "=== arm 設定檢查:…)
  ⎿  sweep syntax OK
     queue syntax OK
     === arm 設定檢查:
     … +8 lines (ctrl+o to expand)

● All arms resolve correctly. Launching the queue:

● Bash(cd /home/tekai324/MLMP; nohup bash bash/run_adagate_queue.sh >
      save/_sweep_logs/adagate_queue_main.log 2>&1 & echo "queue PID $!"; sleep
      20; head -20 save/_swee…)
  ⎿  queue PID 1000555
     [aq] 08-09 11:05 start: 20 jobs, GPUs: 3 4 5, rounds=150, max 4/GPU
     [aq] 11:05 PLACE acdc ABmad03_ecdf (need 15000+4000) -> GPU3 (free 972
     52, reserved 0, slots 0/4)

● Bash(cd /home/tekai324/MLMP; until [ "$(grep -c 'PLACE'
      save/_sweep_logs/adagate_queue_main.log)" -ge 12 ]; do sleep 60; done;
      echo "12+ placed:"; grep PLACE save/_s…)
  ⎿  Running in the background (↓ to manage)

● Monitor(AdaGate follow-up queue: failures + progress)
  ⎿  Monitor started · task b0fkfex68 · persistent
● 20 個 job 已經進佇列開始跑了。

  實驗一：trend_thr 網格（找三個資料集都好的門檻）

  補齊之後三個資料集都會有 0.3 / 0.5 / 0.7 / 1.0 / 1.3 / 1.5 的完整曲線：

  ┌────────────┬────────────────────┬────────────────────┐
  │            │        已有        │       這次補       │
  ├────────────┼────────────────────┼────────────────────┤
  │ ACDC       │ 0.5, 1.0, 1.5, 2.0 │ 0.3, 0.7, 1.3      │
  ├────────────┼────────────────────┼────────────────────┤
  │ VOC20      │ 0.5, 1.0           │ 0.3, 0.7, 1.3, 1.5 │
  ├────────────┼────────────────────┼────────────────────┤
  │ Cityscapes │ 0.5, 1.0           │ 0.3, 0.7, 1.3, 1.5 │
  └────────────┴────────────────────┴────────────────────┘

  我多加了 0.3（你沒提到但很關鍵）：Cityscapes 在 1.0 就已經轉負，代表它的反 U
  頂點比另外兩個早，很可能落在 0.5 以下。只往上掃會錯過它的最佳點。

  實驗二：maxlag_shallow = 10 / 15 / 20（三個資料集，都固定 mad 0.5 + ecdf）

  機制上這個改動很乾淨：cap = min(maxlag_shallow, windows_since_min)，ECDF 再把
  lag 鋪滿 1..cap。所以放大上限只加深「可到達的淺層錨點」，完全不動觸發規則——你
  的假設（VOC20 尾段掉到 GDG-PA 之下是因為倒得不夠遠）可以被單獨檢驗。

  不過有一點要先講，免得結果出來時意外：這個方向也可能反效果。VOC20
  是均勻退化型，GDG-PA 到 R150 都還在爬（78.99），而 mad05
  是提早到頂後緩降（peak 78.95@R102 → 78.60）。倒得更遠等於把模型拉回更早的狀態
  ，可能反而壓抑爬升、讓尾段更低。10/15/20 三個點應該足以看出是單調改善還是反 U
  —— 如果三個都更差，那就代表 VOC20
  的尾段問題不在「倒得不夠遠」，而在「根本不該在那個階段
  restore」，那下一步該調的是 base_rst 或讓觸發率隨時間衰減，而不是 lag 上限。

  執行方式

  用了記憶體閘控佇列 bash/run_adagate_queue.sh（沿用專案既有的
  run_prompt_queue.sh 模式），每張 GPU 最多 4 個並行、每次放置前查即時剩餘 VRAM
  並保留 4 GB buffer。這是因為上次有別人的 VLLM engine 中途佔走 GPU 4 的 82 GB
  害一組 OOM——這個佇列會避開，也不會擠掉別人的 job。

  只用 GPU 3/4/5（你指定的）。20 個 job × 約 8 GPU-小時 ÷ 3 張卡 ≈ 50 
  小時以上才會全部跑完，但 results_all_rounds.txt
  每輪都會寫，所以中途隨時可以看趨勢。失敗監看已掛上。

  bash/sweep_adagate.sh 我加了第五個欄位 maxlag_shallow，既有 arm 全部維持 6
  不變，所以先前的結果仍然可比。

● Monitor event: "AdaGate follow-up queue: failures + progress"

● Expected — the queue settles 120s between placements. Nothing to act on.

❯ 我希望所有job平行在這些GPU上面跑！儘量把VRAM填滿沒關係                        

● Bash(cd /home/tekai324/MLMP; echo "=== 已放置:"; grep PLACE
      save/_sweep_logs/adagate_queue_main.log | sed 's/.*PLACE //;s/ (need.*-> /
       -> /'; echo; echo "=== queue driv…)
  ⎿  === 已放置:
     acdc ABmad03_ecdf -> GPU3 (free 97252, reserved 0, slots 0/4)
     … +15 lines (ctrl+o to expand)

● 只放置了 1 個。停掉佇列驅動（保留已在跑的那個 job），改成全部平行：

● Bash(cd /home/tekai324/MLMP; kill 1000555 2>/dev/null; sleep 3
      pgrep -f "run_adagate_queue.sh" >/dev/null && echo "driver 仍在" || echo
      "driver 已停"…)
  ⎿  driver 仍在
     acdc ABmad03 仍在跑 (保留)

● Bash(cd /home/tekai324/MLMP; kill -9 1000555 2>/dev/null; sleep 3
      pgrep -f "bash bash/run_adagate_queue.sh" >/dev/null && echo "driver 仍在"
       || echo "driver 已停"…)
  ⎿  driver 仍在
     acdc ABmad03 保留中

● Bash(cd /home/tekai324/MLMP; pgrep -af "run_adagate_queue" | grep -v
      "pgrep\|grep" | awk '{print $1}' | while read p; do ps -p $p -o args= |
      grep -q "^bash bash/run_…)
  ⎿  2

● Bash(cd /home/tekai324/MLMP; ps aux | grep "[r]un_adagate_queue.sh" | awk
      '{print $2, $11, $12, $13}')
  ⎿  (No output)

● Driver 已清除。現在把剩下 19 個全部平行丟上去：

● Bash(cd /home/tekai324/MLMP; pgrep -f "adagate_ABmad03_ecdf/" >/dev/null &&
      echo "acdc ABmad03 存活" || echo "acdc ABmad03 掉了"
      # GPU3: +3 heavy(acdc) +3 light(v20)  ->…)
  ⎿  acdc ABmad03 存活
     launched 19 jobs across GPU 3/4/5

● Bash(cd /home/tekai324/MLMP; until [ "$(pgrep -cf 'main_continual.py --adapt
      --method deyo_mlmp_adagate')" -ge 20 ]; do sleep 45; done; sleep 300; echo
       "settled at $…)
  ⎿  Running in the background (↓ to manage)

● 全部 20 個改成同時平行跑，配置如下（依 VRAM 佔用平衡，重型 ACDC/Cityscapes ≈13
   GB、輕型 VOC20 ≈3 GB）：

  ┌─────┬───────────────────────────────────┬──────────────────┬───────────┐
  │ GPU │               重型                │       輕型       │ 預估 VRAM │
  ├─────┼───────────────────────────────────┼──────────────────┼───────────┤
  │ 3   │ acdc mad03(續跑) mad07 mad13 ml10 │ v20 ml10 ml15    │ ~61 GB /  │
  │     │                                   │ ml20             │ 97        │
  ├─────┼───────────────────────────────────┼──────────────────┼───────────┤
  │ 4   │ acdc ml15 ml20；city mad03 mad07  │ v20 mad03 mad07  │ ~71 GB /  │
  │     │ mad13                             │                  │ 97        │
  ├─────┼───────────────────────────────────┼──────────────────┼───────────┤
  │ 5   │ city mad15 ml10 ml15 ml20         │ v20 mad13 mad15  │ ~58 GB /  │
  │     │                                   │                  │ 97        │
  └─────┴───────────────────────────────────┴──────────────────┴───────────┘

  先前那個 acdc ABmad03_ecdf
  我保留了沒重啟（已經跑了一段），只殺掉佇列驅動程序本身。啟動間隔 30 秒，避免
  20 個同時載入 ViT-L/14 造成瞬間 VRAM 尖峰和磁碟爭用。

  有一點先講清楚：全部平行不會縮短總完成時間——GPU 總工作量固定（20 × 約 8
  GPU-小時 ÷ 3 張卡 ≈ 53
  小時），平行只是讓它們同時推進而不是分兩批。差別在於：分批的話前 12 個大約 20
  小時就能看結果，全平行則是所有結果都在約 50 小時後才接近完成。不過因為
  results_all_rounds.txt 每輪都寫，趨勢隨時可讀，而且這些 run 的 dataloader
  和每輪評估有不少 CPU/IO 空檔，塞滿確實能吃到一些重疊增益。

  我掛了背景檢查，20 個都起來並穩定後會回報實際 VRAM 佔用和有無 OOM。

● Background command "Wait for all 20 to settle then check VRAM and OOM"
completed (exit code 0)

  Read 1 file (ctrl+o to expand)

● Bash(cd /home/tekai324/MLMP; date +%H:%M; nvidia-smi
      --query-gpu=index,memory.used,memory.free,utilization.gpu
      --format=csv,noheader -i 3,4,5; echo "runs: $(pgrep -c…)
  ⎿  11:24
     3, 73460 MiB, 23793 MiB, 100 %
     4, 70515 MiB, 26738 MiB, 100 %
     … +3 lines (ctrl+o to expand)

● Bash(cd /home/tekai324/MLMP; pgrep -af "main_continual.py --adapt --method
      deyo_mlmp_adagate" | grep -o "\--save_dir [^ ]*" | sed 's|--save_dir
      save/||' | sort -u |…)
  ⎿       1     ||'"'"'
          2     ACDCDataset/adagate_ABmad03_ecdf/
          3     ACDCDataset/adagate_ABmad05_ecdf_ml10/
     … +15 lines (ctrl+o to expand)

● 20 個 arm 全部平行跑起來了，沒有 OOM。

  ┌─────┬───────────┬─────────┬────────┐
  │ GPU │ VRAM 佔用 │  剩餘   │ 使用率 │
  ├─────┼───────────┼─────────┼────────┤
  │ 3   │ 73.5 GB   │ 23.8 GB │ 100%   │
  ├─────┼───────────┼─────────┼────────┤
  │ 4   │ 70.5 GB   │ 26.7 GB │ 100%   │
  ├─────┼───────────┼─────────┼────────┤
  │ 5   │ 57.6 GB   │ 39.7 GB │ 100%   │
  └─────┴───────────┴─────────┴────────┘

  實際佔用比我估的高（重型 run 約 18–20 GB 而非 13 GB），但三張卡都還有 24–40 GB
   餘裕，安全。

  跑起來的 20 個：

  - trend_thr 網格（11 個）：ACDC 0.3/0.7/1.3、VOC20 0.3/0.7/1.3/1.5、Cityscapes
   0.3/0.7/1.3/1.5 — 補齊之後三個資料集都會有 0.3/0.5/0.7/1.0/1.3/1.5 的完整曲線
  - maxlag_shallow 軸（9 個）：三個資料集各跑 ml10/ml15/ml20，全部固定在 mad 0.5
   + ecdf

  失敗監看已掛著。等累積到 30–50 輪能看出趨勢時我再回報，重點會看：(1)
  三個資料集的反 U 頂點是否收斂到同一個 trend_thr；(2) VOC20 的尾段隨
  maxlag_shallow 是單調改善還是反效果。

● 2 background shell command task(s) from the previous session have no
completion record. They may have been stopped (via the UI, Monitor timeout, or
agent teardown — these leave no transcript marker), or they may have been
running when the previous Claude Code process exited. They have been marked
stopped. Task ids: b1p86bj3a, b0fkfex68. Task ids in this notification beginning
 with "__orphan_summary" are internal scan markers, not tasks.
  ⎿  1 skill available

❯ 請幫我看一下實驗跑得怎麼樣 並且幫我畫該畫的圖還有做該做的分析                 

● Failed to authenticate: OAuth session expired and could not be refreshed

❯ 請幫我看一下實驗跑得怎麼樣 並且幫我畫該畫的圖還有做該做的分析                 

● Bash(cd /home/tekai324/MLMP; date "+%m-%d %H:%M"; echo "alive: $(pgrep -cf
      'main_continual.py --adapt --method deyo_mlmp_adagate')"; echo; for r in
      save/ACDCDataset…)  ⎿  08-12 23:58
     alive: 11
     … +22 lines (ctrl+o to expand)

● 全部 20 個都跑完 150 輪了。收集結果：

● Bash(cd /home/tekai324/MLMP; python3 -c "
      import csv,os,statistics as st…)
  ⎿  #### 軸一：trend_thr 曲線

     === ACDC   基準 mean=31.06  last=31.57  peak=31.57
     … +23 lines (ctrl+o to expand)

● Bash(cd /home/tekai324/MLMP; python3 -c "
      import csv,os,statistics as st…)
  ⎿  #### 軸二：maxlag_shallow（全部固定 mad 0.5 + ecdf）

     === ACDC   基準 mean=31.06 last=31.57
     … +21 lines (ctrl+o to expand)

● 兩個軸都有清楚結果，而且你的 maxlag 假設被證實了。先做完整分析：

● Bash(cd /home/tekai324/MLMP; python3 -c "
      import os…)
  ⎿  每個 arm 在三個資料集上的 (Δmean, Δlast) —— 找「處處不劣化」的組態
     arm                       ACDC Δmean/Δlast     VOC20 Δmean/Δlast
     CITY Δmean/Δlast   判定
     … +14 lines (ctrl+o to expand)

● ABmad05_ecdf_ml20 是唯一在三個資料集的 mean 和 last 都不劣化的組態，但
  ABmad07_ecdf 的 mean 全面最好。下一步很明確——把兩者結合。先確認 GPU 再丟：

● Bash(cd /home/tekai324/MLMP; nvidia-smi
      --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader; echo
      "my runs: $(pgrep -cf 'main_continual.py --adapt --…)
  ⎿  0, 97252 MiB, 0 %
     1, 32617 MiB, 100 %
     2, 97252 MiB, 0 %
     … +4 lines (ctrl+o to expand)

● Update(bash/sweep_adagate.sh)
  ⎿  Added 6 lines
      86      ABmad05_ecdf_ml10) echo "mad 0.5 ecdf 1.5 10" ;;
      87      ABmad05_ecdf_ml15) echo "mad 0.5 ecdf 1.5 15" ;;
      88      ABmad05_ecdf_ml20) echo "mad 0.5 ecdf 1.5 20" ;;
      89 +    # thr x maxlag interaction: thr 0.7 has the best mean on al
         +l three datasets but a                                         
      90 +    # -0.75 VOC20 tail; maxlag 20 is the only arm non-degrading
         + everywhere but has a                                          
      91 +    # smaller mean. These cross them to see if the two gains co
         +mpose.                                                         
      92 +    ABmad07_ecdf_ml15) echo "mad 0.7 ecdf 1.5 15" ;;           
      93 +    ABmad07_ecdf_ml20) echo "mad 0.7 ecdf 1.5 20" ;;           
      94 +    ABmad03_ecdf_ml20) echo "mad 0.3 ecdf 1.5 20" ;;           
      95      *) return 1 ;;
      96    esac
      97  }

● Bash(cd /home/tekai324/MLMP; bash -n bash/sweep_adagate.sh && echo "syntax OK"
      nohup env GPU=3 ROUNDS=150 STAGGER=30 bash bash/sweep_adagate.sh acdc
      "ABmad07_ecdf_ml…)
  ⎿  syntax OK
     launched 9 interaction runs (thr x maxlag) on GPU 3/4/5

● Now the figures. Building a dedicated analysis page for these two axes:

● Bash(cd /home/tekai324/MLMP; python3 -c "
      import csv,json,os,statistics as st…)
  ⎿  bytes 35255

● Bash(cd /home/tekai324/MLMP; S=/tmp/claude-1014/-home-tekai324-MLMP/38eb7291-b
      267-40f5-924c-e4239395f620/scratchpad
      python3 - "$S" <<'PY'…)
  ⎿  Error: Exit code 1
     Traceback (most recent call last):
       File "<stdin>", line 3, in <module>
     FileNotFoundError: [Errno 2] No such file or directory: 
     '/tmp/claude-1014/-home-tekai324-MLMP/38eb7291-b267-40f5-924c-e4239395f620/
     scratchpad/adagate_report.html'

● 上個 session 的 scratchpad 已清空，我重建分析頁：

● Write(/tmp/claude-1014/-home-tekai324-MLMP/38eb7291-b267-40f5-924c-e4239395f62
       0/scratchpad/adagate_sweep.html)
  ⎿  Wrote 282 lines to ../../../tmp/claude-1014/-home-tekai324-MLMP/38eb72
     91-b267-40f5-924c-e4239395f620/scratchpad/adagate_sweep.html
       2 <style>AdaGate 參數掃描：trend_thr × maxlag_shallow</title>
       3 :root{
       4   color-scheme: light;
       5   --plane:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#525
         14e; --muted:#898781;
       7   --accent:#2a78d6; --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a;
       8   --good:#0ca30c; --warn:#fab219; --crit:#d03b3b;
       9   --wash:rgba(42,120,214,.07);
      10   --sans:system-ui,-apple-system,"Segoe UI","Noto Sans TC","Ping
         Fang TC","Microsoft JhengHei",sans-serif;
     … +376 lines (ctrl+o to expand)

● Bash(cd /home/tekai324/MLMP; S=/tmp/claude-1014/-home-tekai324-MLMP/38eb7291-b
      267-40f5-924c-e4239395f620/scratchpad
      { echo '<script>'; printf 'const D='; cat $S/D2.j…)
  ⎿  OK 60624 bytes; thr points per ds: {'acdc': 7, 'v20': 6, 'city': 6}

● 20 個 run 全部跑完 150 輪。分析頁：https://claude.ai/code/artifact/b97d37cf-2a
  11-4bb3-b6a0-e5c8f3ca415b（5 張圖 + 3 張表，軌跡圖可 hover）

  軸一：trend_thr — 三個資料集形狀完全不同

  ┌─────┬────────────┬─────────────┬──────────────────┐
  │ thr │ ACDC Δmean │ VOC20 Δmean │ Cityscapes Δmean │
  ├─────┼────────────┼─────────────┼──────────────────┤
  │ 0.3 │ +0.38      │ +0.61       │ +0.10            │
  ├─────┼────────────┼─────────────┼──────────────────┤
  │ 0.5 │ +0.25      │ +0.69       │ +0.12            │
  ├─────┼────────────┼─────────────┼──────────────────┤
  │ 0.7 │ +0.56      │ +0.77       │ +0.13            │
  ├─────┼────────────┼─────────────┼──────────────────┤
  │ 1.0 │ +0.62      │ +0.79       │ −0.10            │
  ├─────┼────────────┼─────────────┼──────────────────┤
  │ 1.3 │ +0.81      │ +0.68       │ +0.07            │
  ├─────┼────────────┼─────────────┼──────────────────┤
  │ 1.5 │ +0.74      │ +0.64       │ −3.45 ⚠️         │
  ├─────┼────────────┼─────────────┼──────────────────┤
  │ 2.0 │ +1.06      │ —           │ —                │
  └─────┴────────────┴─────────────┴──────────────────┘

  thr=0.7 在三個資料集的 mean 都優於先前推薦的 0.5，而且它就是 Cityscapes
  的頂點。觸發率 0.237/0.239/0.275（極差 1.16×），跨資料集一致性跟 0.5 一樣好。

  最重要的發現是 Cityscapes 在 thr=1.5 崩了：mean 20.39，peak 24.07@R20
  之後一路掉到 18.45。這證明觸發率有下限（約 4%），低於它 gate 就守不住。ACDC
  那條「越嚴越好」的單調線看不到轉折，只是因為它的崩潰點在更右邊——所以先前「一直
  調低觸發率」的想法確實有天花板，現在有直接證據了。

  軸二：maxlag_shallow — 你的假設是對的

  VOC20（唯一有效的資料集，因為那裡 100% 的 restore 都走 lag 路徑）：

  ┌─────────────┬──────────┬───────┬───────┬───────────┐
  │   maxlag    │ 平均 lag │ Δmean │ Δlast │ peak 輪次 │
  ├─────────────┼──────────┼───────┼───────┼───────────┤
  │ 6（原設定） │ 3.47     │ +0.69 │ −0.39 │ R102      │
  ├─────────────┼──────────┼───────┼───────┼───────────┤
  │ 10          │ 5.11     │ +0.56 │ −0.19 │ R121      │
  ├─────────────┼──────────┼───────┼───────┼───────────┤
  │ 15          │ 7.39     │ +0.43 │ −0.12 │ R125      │
  ├─────────────┼──────────┼───────┼───────┼───────────┤
  │ 20          │ 8.80     │ +0.31 │ −0.06 │ R145      │
  └─────────────┴──────────┴───────┴───────┴───────────┘

  Δlast 單調改善到幾乎打平基準（78.93 vs
  78.99）。機制也講得通：倒得更遠會把到頂輪次從 R102 推遲到
  R145，壓抑早期爬升但防住後期漂移——所以 Δmean 同時單調下降（+0.69 → +0.31）。

  這是一個**「爬得快」對「守得住」的權衡，不是免費的改進**。ACDC
  對這軸不敏感（純噪音），Cityscapes 方向相同但幅度極小。

  綜合：只有一個組態處處不劣化

  mad 0.5 + maxlag 20 是唯一在三個資料集的 mean 和 last 都不劣於基準的組態（最差
   Δmean +0.06、最差 Δlast −0.06）。但它的增益也最小。

  而 mad 0.7 + maxlag 6 的 mean 全面最佳（+0.56/+0.77/+0.13），代價是 VOC20 尾段
   −0.75。

  已經丟下去的後續（9 runs，GPU 3/4/5）

  兩個軸的最佳值來自不同組態，所以我直接把它們交叉了：mad 0.7 + ml15、mad 0.7 + 
  ml20、mad 0.3 + ml20，三個資料集各跑一組。目標是看 0.7 的 mean 增益在加深 lag
  之後還留下多少。

  一個必須先講的限制：Cityscapes 的增益只有 +0.06~+0.13，而它在
  thr=1.0（−0.10）和
  1.3（+0.07）之間出現非單調反轉——在這個尺度上分不出真訊號和噪音。Cityscapes
  的任何結論寫進論文前都需要多 seed。這件事和 trend_hist=50
  從未掃過，是目前兩個最大的空缺。