# 「Brain-Inspired BitNet Architecture」設計藍圖與實作路線 v1.0

> 目標：設計並實作一個參考 BitNet、具腦啟發特性的實驗型語言模型架構，
> 以大量微節點局部推理、短期關聯記憶、ACC 式衝突監測與持續學習為核心，
> 並內建防「過激反應」與「崩壞重複」的修補機制。


---

## 1. 系統總體目標與設計原則

### 1.1 系統目標

1. **BitNet 精神**  
   - 採用低位元權重（例如 ternary `{-1, 0, +1}`）。  
   - 降低記憶體與運算成本，便於實驗與擴展。  

2. **腦啟發特性**  
   - 神經元具膜電位、閾值與類脈衝行為。  
   - 由大量 **MicroCircuit（微迴路/小節點）** 先做局部推理，再由上層整合。  
   - 具短期 fast weights 記憶與長期慢權重結構，支援持續學習。  
   - 內建 ACC 類「衝突偵測」與 Meta-controller，實現自我懷疑、自我反思與多步推理。  

3. **穩定性與安全性修補**  
   - 降低模型「過激」反應機率。  
   - 避免生成過程「崩壞重複 / 無限 loop」。  
   - 避免 fast weights 與自反模組放大錯誤模式。  

4. **實驗友善**  
   - 初始版本設計為 **10M–100M 參數級**小模型，在單機或一般伺服器即可訓練與推理。  
   - 所有組件皆可在任意作業系統與硬體上實作，只要支援現代深度學習框架。  


### 1.2 設計原則

- **模組化**：基礎算子層、微結構層、整合層、記憶層、控制層、解碼層清楚分離。  
- **自底向上**：先保證單元（BitLinear、SpikingUnit、MicroCircuit）行為正確，再向上堆疊。  
- **可退回 baseline**：每一層皆可切換為「標準 Transformer/MLP」實作，以便對照與除錯。  
- **漸進式複雜化**：先實作最小可用版本，再逐步加入 fast weights、ACC、持續學習。  


---

## 2. 系統分層架構概觀

整體架構可分為六層：

1. **基礎算子層**  
   - `BitLinear`：BitNet 風格低位元線性層。  
   - `SpikingUnit`：具膜電位與閾值的類脈衝神經元單元。  

2. **微結構層（MicroCircuit）**  
   - N 個 `SpikingUnit` + 內部 recurrent 連線 + 輸入/輸出投影。  
   - 負責對局部資訊做小範圍推理，輸出局部抽象表示 `h_i`。  

3. **整合層（CorticalHub）**  
   - 收集所有 microcircuit 輸出。  
   - 使用「群組注意力 + fast weights + 抑制」整合成全局狀態 `g_t`。  
   - 適度回饋各 microcircuit。  

4. **記憶與學習層**  
   - `M_t` fast weights：短期關聯記憶，session 內有效。  
   - 中期記憶：adapter/LoRA/外部 episodic memory，支援持續學習。  
   - 長期記憶：主幹 BitNet 權重，訓練期逐步固化。  

5. **控制層（ACC + Meta-controller）**  
   - ACC 群簇：監測衝突與不確定度。  
   - Meta-controller：決定推理深度、自我反思輪數、解碼策略、語氣強度。  

6. **解碼層（Sampling + Anti-collapse）**  
   - 溫度 / top-p / top-k。  
   - 重複懲罰 / no-repeat-ngrams。  
   - 崩壞偵測與安全中止策略。  


---

## 3. 基礎算子層設計

### 3.1 BitLinear：低位元線性層

**介面**

- 輸入：`x ∈ R^{batch × d_in}`  
- 輸出：`y ∈ R^{batch × d_out}`  

**內部狀態**

- FP32 主權重：`W_float ∈ R^{d_out × d_in}`  
- 量化權重：`W_q ∈ {-1, 0, +1}^{d_out × d_in}`  
- 偏置：`b ∈ R^{d_out}`（可選）  

**量化函式 `Q`**（範例）

- 若 `|W_float| < τ` → 0  
- 否則 → `sign(W_float)`（即 -1 或 +1）  
- `τ` 可為固定常數或 learnable 閾值。  

**Forward**

```text
W_q = Q(W_float)
y = x @ W_q^T + b
```

**Backward**

- 對 `W_float` 計算梯度（使用 Straight-Through Estimator 或其他平滑近似）。  
- 更新 `W_float` 後重新量化產生新的 `W_q`。  

**目標**

- 對齊 BitNet 的低位元、低能耗精神。  
- 提供簡易、可攜的實作基礎（不限語言與硬體）。  


### 3.2 SpikingUnit：類脈衝神經元

**狀態變數**（對每個樣本與時間步）

- 膜電位：`v_t`  
- 前一時間步輸出：`s_{t-1}`  

**輸入**

- 凈輸入電流：`u_t`（可由 `BitLinear` 產生）。  

**更新方程（概念）**

```text
# 膜電位更新（漏電 + 發放後復位）
v_t = α * v_{t-1} + u_t - β * s_{t-1}

# 脈衝輸出（真 spike 或連續近似）
s_t = spike(v_t - θ)
```

其中：

- `α ∈ (0,1)`：漏電係數。  
- `β ≥ 0`：發放後復位係數。  
- `θ`：閾值（可為固定或 learnable）。  
- `spike()`：  
  - 理想情況：`1_{v_t > θ}`；  
  - 實作時採用 sigmoid/HardSigmoid + STE 做近似，利於反向傳播。  

**用途**

- 作為 MicroCircuit 內部與輸入/輸出層的基本運算單元。  
- 提供時間相依、非線性、類生物動力學行為。  


---

## 4. 微結構層設計：MicroCircuit

### 4.1 MicroCircuit 結構

一個 `MicroCircuit` 為一個小型 recurrent 子網路，包含：

- `N` 個 SpikingUnit（例如 32–256）。  
- 內部 recurrent 連線：`W_rec`（BitLinear）。  
- 外部輸入投影：`W_in`（BitLinear）。  
- 外部輸出 readout：`W_out`（BitLinear）。  

**設計意圖**

- 模擬皮質微柱/局部神經群，對局部資訊做「小範圍推理」。  
- 大量並列的 MicroCircuit 構成整體模型的「小節點群」。  


### 4.2 單步更新流程

對時間步 t：

1. **接收局部輸入 `x_local_t`**  
   - 可以是：  
     - 一個 token 或少數幾個 token 的 embedding。  
     - 上一層（或上一時間步）全局狀態 `g_{t-1}` 的一部分。  

2. **計算輸入電流**  

```text
u_ext = W_in(x_local_t)      # 外部輸入
u_rec = W_rec(s_{t-1})       # 內部 recurrent
u     = u_ext + u_rec
```

3. **更新 N 個 SpikingUnit 狀態**  
   - 對每個單元使用 SpikingUnit 更新方程計算 `v_t` 與 `s_t`。  

4. **聚合輸出局部摘要 `h_i_t`**  

```text
h_i_t = W_out(s_t)
```

其中 `h_i_t` 為該 MicroCircuit 在時間 t 的局部表示向量。  


### 4.3 多 MicroCircuit 並行

- 系統中存在 M 個 `MicroCircuit`（i = 1..M）。  
- 每個 `MicroCircuit` 可透過不同的輸入路由獲得不同區域/功能的資訊。  
- 在實作上可以視為 batch/模組維度上的並行運算。  


---

## 5. 整合層設計：CorticalHub（腦風格注意力）

### 5.1 整體功能

`CorticalHub` 負責：

1. 收集所有 MicroCircuit 的輸出 `h_i_t`。  
2. 管理 fast weights `M_t` 以表徵短期關聯記憶。  
3. 對 `h_i_t` 執行群組/局部注意力與抑制。  
4. 產生新的全局狀態 `g_t`，並可對 `h_i_t` 做調制回饋。  


### 5.2 Fast Weights `M_t` 設計

- `M_t ∈ R^{d_h × d_h}`（`d_h` 為 `h_i` 維度）。  
- 作為整體短期關聯記憶矩陣。  

**更新規則（含修補）**

```text
M_{t+1} = λ * M_t + η * Σ_i (k_i ⊗ v_i)
M_{t+1} = clamp(M_{t+1}, -c, c)
```

說明：

- `0 < λ < 1`：衰減，避免舊記憶永久累積。  
- `η`：學習率/更新強度（小）。  
- `k_i, v_i`：由 `h_i` 經 BitLinear 產生的 key/value。  
- `clamp`：限制元素值於 `[-c, c]`，防止爆炸。  

> 此設計避免 fast weights 在單次錯誤寫入後將錯誤模式長期放大。  


### 5.3 群組注意力 + 抑制

對每個 microcircuit 的局部表示 `h_i`：  

1. **投影為 Q/K/V**  

```text
q_i = W_q(h_i)
k_i = W_k(h_i)
v_i = W_v(h_i)
```

2. **計算相似度**  

```text
S_base(i, j) = q_i · k_j
S_fast(i, j) = q_i · (M_t k_j)
S_ij         = S_base(i, j) + S_fast(i, j)
```

3. **局部化 / 分組**  
   - 對每個 i，只在附近/同組的 j 上計算 softmax（群組或局部注意力），而非全域。  

4. **抑制機制**  
   - 對於同一群中 activation 過高的單元，對其他單元施加減權或抑制，模擬局部競爭。  

5. **輸出更新後的 `h'_i` 與全局狀態 `g_t`**  

```text
h'_i = Σ_j softmax(S_i·)_j * v_j
g_t  = Readout({h'_i})      # 例如平均、加權平均或另一層 BitLinear
```


---

## 6. 記憶與學習層設計

### 6.1 三層記憶結構

1. **快記憶（短期，秒~分鐘）**  
   - 由 fast weights `M_t` 代表，僅在單一推理 session 中有效。  
   - 推理結束後清空。  

2. **中期記憶（中期，天~週）**  
   - 使用 adapter / LoRA 層或外部 key-value episodic memory。  
   - 每次任務完成時，將重要片段寫入此記憶。  
   - 週期性使用這些經驗做小批量再訓練或對 adapter 進行更新。  

3. **長期記憶（長期）**  
   - 主幹 BitNet 權重（`BitLinear`、`SpikingUnit` 等）。  
   - 在預訓練與定期再訓練過程中緩慢更新，逐步固化穩定知識。  


### 6.2 持續學習流程（概念）

1. **線上推理階段**  
   - 使用當前主幹權重 + 中期記憶（adapter / episodic memory）做推理。  
   - fast weights `M_t` 在 session 中持續更新。  

2. **任務/對話結束**  
   - 將 `(輸入, 內部狀態摘要, 輸出, 回饋)` 寫入 episodic buffer。  

3. **離線/間歇再訓練**  
   - 週期性選取 buffer 中高價值樣本，  
   - 更新 adapter 或選定子網權重，避免 catastrophic forgetting。  


---

## 7. 控制層設計：ACC + Meta-controller

### 7.1 ACC 群簇：衝突與不確定偵測

**輸入特徵（示意）**

- 各 `h_i` 之間的 variance / cosine similarity（反映微節點間分歧）。  
- 模型輸出 logits 的 entropy（反映不確定度）。  
- 若有多候選答案，候選間差異度（例如 KL divergence）。  

**輸出**

- `conflict_level ∈ [0, 1]`  
- `uncertainty ∈ [0, 1]`  

**校正（Calibration）**

1. 收集一批推理案例，標記為「明顯錯誤」「基本正確」等。  
2. 訓練一個小 MLP 或 MicroCircuit 當 ACC classifier。  
3. 對 ACC 原始輸出做 z-score 正規化：

```text
conflict_norm = (conflict_raw - μ_conflict) / σ_conflict
```

4. 設定門檻，例如 `conflict_norm > 2` 才視為高衝突。  


### 7.2 Meta-controller：推理策略管理

Meta-controller 根據 ACC 的輸出，控制：

1. **反思 / 多步推理輪數**  

- 定義 `max_reflection_steps`（建議 1–3）。  
- 若 `conflict_norm` 高且目前反思次數 < 上限：  
  - 觸發額外一輪內部 reasoning / 自我檢查。  
- 若已達上限：  
  - 直接選用目前最佳答案，並建議模型以保守語氣表達不確定性。  

2. **解碼策略（溫度與採樣）**  

- 若 `conflict_norm` 高：  
  - 可略微降低溫度或多 sample 幾次，再由裁決器選擇。  
  - 或切換「保守模式」prompt：鼓勵列出多種可能。  

3. **語氣控制**  

- 若 `uncertainty` 高：  
  - 在 prompt 或控制信號中加入「應避免絕對斷言，允許表達不確定」。  

> 關鍵修補：ACC **不直接否決答案**，僅調整策略與語氣，並由反思輪數上限避免無限 loop。  


---

## 8. 解碼層設計：Anti-collapse 與行為修補

### 8.1 基本採樣參數建議

- `temperature ∈ [0.8, 1.0]`  
- `top_p ≈ 0.9`  
- `top_k ≈ 40–100`  

可依模型大小與實測行為調整。  


### 8.2 重複控制

1. **repetition penalty**  

- 記錄已生成 token 次數 `freq[token]`。  
- 在每步解碼前：

```text
for each token:
    if freq[token] > N_threshold:
        logits[token] /= penalty_factor   # penalty_factor > 1
```

- 範例：`N_threshold = 3, penalty_factor = 1.2`。  

2. **no-repeat-ngrams**  

- 維護一個已生成 n-grams 的集合（建議 3-gram 或 4-gram）。  
- 對於任何會產生已存在 n-gram 的候選 token，將其 logit 設為負無窮（禁止選擇）。  


### 8.3 崩壞偵測與中止

在解碼過程中持續監控：  

- 若最近 `L` 個 token 中，某單一 token 的比例 > `p_high`（例如 70%）。  
- 或偵測到長度 ≥ `N` 的子串重複出現 2 次以上。  

則：

1. 視為 **崩壞生成**。  
2. 立即停止本輪生成。  
3. 將該情況記錄給上層（可作為訓練負樣本）。  
4. Meta-controller 可選擇：  
   - 嘗試一次重新生成（不同 seed/策略）。  
   - 或直接進入保守模式回答不確定。  


---

## 9. 實作路線（分階段）

以下實作路線假定使用任意深度學習框架（如 PyTorch、JAX、TensorFlow 等），
但不依賴特定作業系統或特定 GPU/CPU。


### 第 0 階段：環境與 baseline 準備

**目標：** 建立一個可運行的小型 baseline 語言模型做對照。

1. 準備通用環境：  
   - 安裝 Python 3.10+ 或任意支援的語言環境。  
   - 安裝支援 GPU/CPU 的深度學習框架。  

2. 實作或採用現成的小型 GPT-like 模型（20M–50M 參數級）：  
   - 僅需支援基本 language modeling。  
   - 在少量語料上訓練至可用 perplexity。  

3. 確認：  
   - baseline 能穩定訓練與推理。  
   - 解碼時行為合理。  


### 第 1 階段：基礎算子層實作與測試

**目標：** 實作 `BitLinear` 與 `SpikingUnit`，並驗證其正確性與可訓練性。

1. 實作 `BitLinear`：  
   - 替代框架中的標準線性層。  
   - 寫單元測試：比較 FP32 vs 量化輸出誤差；檢查梯度是否正常。  

2. 實作 `SpikingUnit`：  
   - 利用簡易函式（sigmoid + STE）實作 spike 近似。  
   - 在簡單任務（例如一維序列分類、簡單時間序列）上驗證能學到模式。  


### 第 2 階段：MicroCircuit 與 CorticalHub 原型

**目標：** 建立並驗證「大量微節點 + 整合中心」的核心結構。

1. 實作 `MicroCircuit`：  
   - 實作 N 個 SpikingUnit + `W_in / W_rec / W_out`。  
   - 使用小規模序列任務測試（例如 parity、copy、pattern detection）。  

2. 實作 `CorticalHub`：  
   - 先使用少量 MicroCircuit（8–16 個）作原型。  
   - 實作 fast weights `M_t` 更新（含衰減與 clamp）。  
   - 實作群組注意力與抑制。  
   - 用簡單任務測試整合能力（例如從多個輸入中選取關鍵資訊）。  


### 第 3 階段：整合為「BrainBlock」與玩具語言模型

**目標：** 將 MicroCircuit + CorticalHub 整合為可重複堆疊的 block，建立初版語言模型。

1. 定義 `BrainBlock`：  

   - 輸入：token embedding 序列。  
   - 內部：  
     - 多個 MicroCircuit 並行處理局部資訊。  
     - CorticalHub 整合並更新全局狀態。  
   - 輸出：更新後的序列表示。  

2. 用若干 `BrainBlock` 堆疊成完整模型：  
   - Embedding → 多層 BrainBlock → Output head（BitLinear）。  

3. 在小語料上訓練為 language model：  
   - 記錄 loss / perplexity，與 baseline 比較。  
   - 檢查生成行為（即使用粗糙模型，也應避免立即崩壞）。  


### 第 4 階段：引入 fast weights 完整流程與 Anti-collapse 解碼

**目標：** 將 fast weights `M_t` 與解碼修補整合進主模型。

1. 在完整 LM 推理流程中啟用 fast weights：  
   - 每步更新 `M_t`。  
   - 在整合層注意力中加入 `q_i · (M_t k_j)` 項。  

2. 加入 Anti-collapse 解碼層：  
   - 設定合適的溫度 / top-p / top-k。  
   - 實作 repetition penalty 與 no-repeat-ngrams。  
   - 實作崩壞偵測與中止機制。  

3. 以固定測試 prompt 比較：  
   - 有/無 Anti-collapse 的差異。  
   - 有/無 fast weights 的差異。  


### 第 5 階段：ACC 與 Meta-controller

**目標：** 實現自我懷疑、自我反思與推理深度控制。

1. **ACC 初版（規則型）**：  
   - 以 logits entropy、`h_i` 分歧度等作為粗略 `conflict_level`。  
   - 將此值提供給 Meta-controller。  

2. **Meta-controller 初版**：  
   - 加入固定上限 `max_reflection_steps`（建議 1–3）。  
   - 若衝突高且未達上限 → 再啟動一輪內部推理；否則輸出。  
   - 若不確定度高 → 將提示/控制信號調整為「保守語氣」。  

3. **ACC 升級為學習型**：  
   - 收集模型推理記錄，標註錯誤/正確。  
   - 訓練小 MLP 或 MicroCircuit 作為 ACC。  
   - 對輸出做校正（z-score + 門檻重設）。  


### 第 6 階段：持續學習與經驗式成長

**目標：** 讓模型隨互動逐步成長，而非訓練完即凍結。

1. 建立 episodic buffer：  
   - 儲存 `(輸入, 內部狀態摘要, 輸出, 回饋/標籤)`。  

2. 設計週期性再訓練程序：  
   - 定期抽取 buffer 中的樣本，更新 adapter 或局部權重。  
   - 控制學習率與更新頻率，避免破壞主幹能力。  

3. 設計可量測任務：  
   - 例如特定領域問答的正確率隨時間上升。  
   - 以此驗證經驗式成長機制確實生效。  


---

## 10. 後續擴展方向與注意事項

1. **可擴展性**  
   - 若小模型實驗成功，可逐步放大 MicroCircuit 數量、BrainBlock 層數與 embedding 維度。  

2. **架構替代與對照實驗**  
   - 可將 SpikingUnit 替換為標準激活函數，觀察差異。  
   - 可禁用 fast weights 或 ACC，逐一 ablation，評估其貢獻。  

3. **安全與風格調控**  
   - 若仍觀察到過激回答，可在訓練階段追加「冷靜/中性/承認不確定」風格樣本。  
   - 可加入專門的語氣分類器，將其輸出作為 ACC/Meta-controller 的額外訊號。  

4. **實作語言與框架選擇**  
   - 架構不依賴特定語言/硬體，可用：  
     - Python + PyTorch/JAX/TensorFlow。  
     - Rust + burn/dfdx 等。  
     - C++ + 自研 CUDA/Metal/ROCm 核心。  

5. **實驗哲學**  
   - 初期以「行為差異與穩定性」為主：是否更少崩壞、是否能表現出合理的自我懷疑。  
   - 再逐步追求「效能」與「可擴展性」。  
