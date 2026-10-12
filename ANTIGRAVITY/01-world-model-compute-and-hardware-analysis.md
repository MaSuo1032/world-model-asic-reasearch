# World Model 運算本質、推論機制與硬體架構深度剖析

日期：2026-10-12｜作者：Antigravity。本報告針對 World Model（世界模型）的運算組成、參數分佈、Prefill/Decode 機制、權重搬移特性、Batch 維度，以及演算法流程與 LLM 之本質差異進行深度分析，並聚焦其對硬體資料流與 ASIC 架構設計之意涵。

---

## 1. World Model 除了 Attention Block 以外還有哪些計算？

在以 Transformer（如 DiT 或 Autoregressive）或混合架構為主的現代世界模型中，注意力機制（Self/Cross Attention）僅承擔全域關聯建模，其餘非 Attention 運算佔據了極大比例的晶片面積與執行延遲：

| 模組分類 | 具體運算內容 | 執行時機與資料流特點 |
| :--- | :--- | :--- |
| **感官 Tokenizer / Encoder** | 2D/3D CNN（卷積、Stride 降採樣）、ViT Patchify、空間群組歸一化（GroupNorm）、向量量化（Vector Quantization / Codebook 距離計算） | 取得新觀測畫面時執行；空間局部性強，高度依賴圖像快取與卷積/GEMM 資料流。 |
| **FFN / MLP 子層** | 線性投影（$W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$）、SwiGLU/GeLU 逐元素非線性激活 | 在 Transformer 每層中佔據約 $\frac{2}{3}$ 參數量與運算量，為計算密集（Compute-bound）之典型矩陣乘法。 |
| **動作條件化（Action Conditioning）** | 小型 MLP 投影、AdaLN（Adaptive LayerNorm）/ FiLM 調製（計算 $\gamma, \beta, \alpha$ 縮放、位移與門控係數） | 每次輸入物理動作（方向盤、關節角、按鍵）時執行；涉及動態權重調製與頻繁之 Element-wise 計算。 |
| **狀態轉移（State Dynamics）** | RSSM（Recurrent State Space Model）、GRU/LSTM 門控遞迴更新、重參數化採樣（Reparameterization Trick, Gumbel-Softmax）、高斯隨機變量生成 | 於潛在空間中推進時間步；若為非 Attention 決策模型（如 TD-MPC2、Dreamer），此為主要推演核心。 |
| **擴散去噪求解器（若為 Diffusion 路線）** | 多步 ODE/SDE 數值積分（如 DDIM, Euler, DPM-Solver）、時間步 Embedding 投影 | 每次推演單幀或短片需重複 10~50 步全序列前向傳播，大幅增加反覆計算負擔。 |
| **解碼與幾何渲染（Renderer）** | 轉置卷積（ConvTranspose）、PixelShuffle、3D 高斯點雲（3DGS）投影、Tile-based 點雲光柵化與 Alpha 混合（如 World Labs Spark） | 需將潛在狀態還原為人眼或下游視覺系統可見信號時執行；3DGS 高度依賴基數排序（Radix Sort）與特殊光柵化硬體單元。 |

**ASIC 設計意涵**：晶片不能僅依賴傳統脈動陣列（Systolic Array）處理稠密 GEMM；必須同時整合專屬之影像壓縮前端管線、低延遲非線性 Element-wise/Norm 處理單元，以及若涉及 3D 渲染時之排序/光柵化加速器。

---

## 2. 參數量大約多少？

世界模型並非單一尺度，根據任務即時性、物理精度與是否需要解碼像素，參數量橫跨三個主要量級梯隊：

```
[邊緣端 / 輕量控制] (4M ~ 200M)  ───> TD-MPC2 (1M~317M), DreamerV3 (12M~400M)
[具身智能 / 自動駕駛] (1B ~ 14B)   ───> Meta V-JEPA 2 (1.3B), Google Genie (11B), NVIDIA Cosmos (4B~14B)
[大型多模態 / 空間世界] (10B ~ 32B+) ──> Meta CWM (32B), World Labs RTFM, LWM (7B~32B)
```

1. **輕量級 / 機器人強化學習專用（4M ～ 200M）**：
   * **代表系統**：Ha & Schmidhuber (2018) 約 4M；TD-MPC2（1M–317M）；DreamerV3（12M–400M）。
   * **特點**：潛在狀態空間極小（例如向量維度 512），不輸出像素，專注於極速動作規劃。全權重可完整常駐於片上 SRAM（On-chip SRAM）。
2. **中型具身基座 / 視訊世界模型（1B ～ 14B）**：
   * **代表系統**：Meta V-JEPA 2（ViT-g Encoder ~1B + Action Predictor ~300M，合計約 1.3B）；Google Genie（~11B）；Wayve GAIA-1。
   * **特點**：具備高維時空物理推演能力。權重無法常駐片上，需仰賴外部 LPDDR5 或 HBM。
3. **大型生成式多模態世界模型（10B ～ 32B+）**：
   * **代表系統**：Meta Code World Model (CWM, 32B)；UC Berkeley LWM（7B~32B，百萬視訊 Context）。
   * **特點**：支援多相機全景、長時序歷史與複雜文字/程式碼交叉理解，推論需多晶片張量並行（Tensor Parallelism）。

---

## 3. 每一次 Inference 一樣會有 Prefill 和 Decode 嗎？分別計算量多少？

是否具備傳統 LLM 之 Prefill 與 Decode 階段，完全取決於世界模型之**演算法架構**：

### A. 自回歸世界模型（Autoregressive World Model, 如 Genie, Oasis, GAIA-1）
* **階段劃分**：**完全對應 Prefill 與 Decode**。
* **Prefill 階段**：
  * **任務**：處理歷史 $T$ 幀觀測與動作。若每幀畫面壓縮為 $16 \times 16 = 256$ 個 Token，4 幀即為 $1,024$ 個 Token。模型一次性計算並存儲初始 KV Cache。
  * **計算特性**：**Compute-bound（計算密集型）**。
  * **FLOPs 估算**：約為 $2 \times N \times L_{\text{history}} + \mathcal{O}(L_{\text{history}}^2)$。由於視覺 Token 數量遠大於日常文字 Prompt，Prefill 算力消耗劇烈。
* **Decode 階段**：
  * **任務**：根據外部輸入動作，逐個 Token（或逐幀 Block）生成未來的視覺 Token。
  * **計算特性**：**Memory Bandwidth-bound（訪存頻寬密集型）**。
  * **FLOPs 估算**：每步生成 1 個 Token 的矩陣運算約為 $2 \times N$ FLOPs。生成一完整影格（256 Tokens）需循環 256 次。

### B. 擴散世界模型（Diffusion World Model / DiT）
* **機制**：**無自回歸單 Token Decode**。
* **流程**：包含「條件編碼」（類似 Prefill，只算一次）與「多步去噪迴圈（Denoising Loop）」。
* **計算特性**：每一步去噪均為對全空間 Token 的**全序列前向運算（Full Forward Pass）**。若採用 20 步採樣，計算量等同於 20 次完整的深度網路推論，全流程維持 Compute-bound。

### C. 潛在空間 MPC 模型（Latent MPC, 如 TD-MPC2）
* **機制**：**無 LLM 式 KV Cache**。
* **流程**：一次性觀測編碼（$o_t \to z_t$），接著由輕量 Dynamics MLP 在潛在空間對數百條候選軌跡進行多步平行推演（Rollout），計算量由候選數 $K$、預測步數 $H$ 與搜尋輪次 $I$ 決定。

---

## 4. Prefill 與 Decode 各需要 Load 多少權重？

從晶片記憶體階層（DRAM/HBM $\to$ 片上 SRAM $\to$ 運算單元暫存器）觀察：

```
[外部記憶體 DRAM/HBM] ──(權重傳輸)──> [片上快取 SRAM] ──(資料饋入)──> [暫存器/運算陣列]
```

假設模型參數量為 $N$，採用 BF16/FP16 精度（靜態權重容量為 $2N$ Bytes）：

* **Prefill 階段**：
  * **只需將全部靜態權重從外部記憶體載入 1 次（$2N$ Bytes）**。
  * 由於處理的 Context 長度 $L_{\text{history}}$ 大，權重矩陣被所有 Token 同時共用，**算術強度（Arithmetic Intensity = $\frac{\text{FLOPs}}{\text{Bytes}}$）極高**，能充分隱匿權重搬移延遲。
* **Decode 階段（自回歸單 Token 推演）**：
  * **每產生 1 個 Token，就必須把整個模型的靜態權重（$2N$ Bytes）從外部記憶體重新載入一次**。
  * **額外負擔**：必須同時加載持續增長的 **KV Cache**。
  * **瓶頸總結**：KV Cache 並不能減少模型靜態權重的讀取量，它僅避免了歷史特徵的重複計算。在 Batch = 1 的自回歸推論中，生成 256 個 Token 需重新載入權重 256 次，推論速度完全受制於外部記憶體頻寬（Memory Wall）。

---

## 5. 是否可以把 World Model 視為 Batch = 1 的模型？

**結論：在外部環境互動層為 Batch = 1；但在晶片內部計算層通常不為 1。**

1. **外部輸入的物理約束（External Batch = 1）**：
   * 在機器人實體控制、自駕車車端決策或即時可玩模擬器中，實體代理人每毫秒產生的觀測與動作反饋是單一閉環。無法跨時間或跨機器組 Batch，否則產生致命延遲。
2. **內部推演的平行維度（Internal Batch Expansion）**：
   * **候選軌跡維度（$K$）**：線上 MPC（如 MPPI/CEM）在當前瞬間會平行推演 $K=512$ 條候選動作序列，使 Dynamics 運算之有效 Batch 擴展為 512。
   * **空間 Token 維度（$M$）**：在 Transformer 中，即使只處理單一畫面（$B=1$），空間 Patch 切割後進入線性層的矩陣維度仍為 $M=256$ 或 $1024$，在 GEMM 視角下等同於批次大小為 $M$。
   * **Ensemble 維度（$E$）**：若採用 Q-ensemble 或多個 Dynamics 模型評估不確定性，計算規模亦呈倍數放大。

**ASIC 設計意涵**：加速器應特別針對「小型 GEMM、多路平行候選軌跡（Batch = $K$）」進行資料流最佳化，而非盲目採用針對巨大單一矩陣設計的超大脈動陣列（避免陣列利用率低落至 10% 以下）。

---

## 6. 模型實際運作的演算法流程與 LLM 之本質差異

### A. 世界模型之演算法運作流程（以閉環控制為例）

```mermaid
flowchart TD
    O[1. 感測觀測 o_t] --> T[2. 特徵編碼 / Tokenizer]
    T --> S[3. 潛在狀態 z_t]
    S --> A[4. 動作注入 / 候選採樣 a_t^(1..K)]
    A --> D[5. 動力學推演 Dynamics Rollout]
    D --> E[6. 價值評估 / 距離計分 J]
    E --> C{搜尋收斂?}
    C -- 否 --> U[更新動作分佈] --> A
    C -- 是 --> P[7. 輸出最佳首步動作 a*_t]
    P --> ACT[8. 實體執行於環境]
    ACT --> R[環境狀態轉移] --> O
```

1. **感官觀測與編碼**：相機/光達取得即時物理訊號 $o_t$，透過 Encoder 壓縮至潛在幾何與語義空間 $z_t$。
2. **動作注入與條件調製**：由規劃器採樣 $K$ 條候選動作序列，透過 AdaLN 注入動力學主幹。
3. **動力學潛在推演（Mental Simulation）**：模型在神經網路內部模擬未來的狀態演進 $z_{t+1} \sim P(z_{t+1} \mid z_t, a_t)$，並同時計算代價/獎勵。
4. **決策最佳化（MPC Search）**：評估候選軌跡的期望回報，迭代更新動作分佈，選定最優的第一步動作 $a_t^*$。
5. **實體執行與閉環修正**：動作輸出至機械致動器；隨後接收真實物理世界的新觀測 $o_{t+1}$，重置誤差，展開下一輪推演。

---

### B. World Model 與 LLM 的三大本質差異

| 核心維度 | 主流 LLM（大語言模型） | 決策/生成型 World Model（世界模型） |
| :--- | :--- | :--- |
| **信號本體與空間幾何** | **1D 離散符號序列**。<br>Token 具備高抽象語義，無真實空間座標、距離或幾何遮擋概念。 | **3D/4D 連續時空訊號**。<br>必須嚴格隱式/顯式建模**物體恆存性（Object Permanence）、幾何透視、遮擋與三維旋轉一致性**。 |
| **生成機制與因果關係** | **單向語義接龍**：$P(w_t \mid w_{<t})$。<br>文字生成不涉及外部物理干預，為自發性符號推演。 | **動作條件化模擬器（Simulator）**：$P(s_{t+1} \mid s_t, a_t)$。<br>未來狀態由**外部施加的動作干預（Intervention）**決定，具備主動探索與實體因果反饋。 |
| **幻覺的物理後果** | **事實與邏輯偏差**。<br>文字表面通常依然語法流暢、合乎語言習慣。 | **物理定律崩壞**。<br>表現為穿牆、碰撞失效、物體懸空或幾何撕裂；受客觀物理規律的剛性檢驗。 |

---

## 7. 總結與 ASIC 架構建議

1. **記憶體頻寬是第一瓶頸**：在線自回歸生成受限於 Batch=1 的權重反覆載入，必須仰賴高頻寬記憶體（HBM3/LPDDR5X）或足夠容納關鍵模組的片上 SRAM（16MB~64MB）。
2. **靈活的陣列維度**：適配 $K=512$ 候選並行或 $M=256$ 空間 Token 之運算尺度，避免單一超大脈動陣列造成算力浪費。
3. **異質運算單元整合**：將 GEMM 加速器與影像編碼器、AdaLN 調製單元及（可選的）3DGS 光柵化硬體深度耦合，方能實現端到端低延遲的世界模擬。
