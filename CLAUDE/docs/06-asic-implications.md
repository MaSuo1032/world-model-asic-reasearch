# 06 · 對 World Model / VLA 推論加速器的設計含義

> **修訂（2026-10-12，回應 GPT/04 的保留）**：§6.2 與 §6.9 的「128–256 MB 片上 SRAM」是需要以面積（N5 級約 70–90 mm²）、漏電與良率權衡的**選項**，不是預設值；若系統採用 best-of-K 候選平行，權重讀取會被攤提，常駐需求隨之降低。詳見 [08 §8.4](08-cross-review-with-gpt.md)。加速技巧的逐項可用性見 [07](07-llm-accelerator-techniques.md)。

本 repo 所屬專案的目標是「分析打造 world model 推論加速器的關鍵」。以下把前五章的數據轉成具體的設計結論。每一條都附上它依據的數字。

---

## 6.1 先決定要服務哪一種工作負載

前面的分析顯示，「world model」這個詞底下其實有三種計算形態，對晶片的要求相互矛盾：

| 形態 | 代表 | 瓶頸 | 每單位計算 | 晶片該往哪偏 |
|---|---|---|---|---|
| **A. Flow-matching VLA** | π0、GR00T、SmolVLA | 權重頻寬（迴圈 AI ≈ 50） | 0.7–5 TFLOP / chunk | 大 SRAM、低 M 效率 |
| **B. 串流影片世界模型 / WAM** | Matrix-Game、DreamZero | 算力 + 注意力 + KV 容量 | 37–550 TFLOP / 段 | 大算力、注意力引擎、KV ring buffer、卷積 |
| **C. 潛在空間 MPC** | V-JEPA 2-AC | 純算力（AI ≈ 6,000） | ~2,000 TFLOP / 動作 | 大算力、可大 batch |

一顆**邊緣晶片**（30–100 W）在 2026 年能合理服務的是 **A 全部 + B 的小型串流模型（Matrix-Game 2.0 級）**。DreamZero 級的 WAM 和 C 類 MPC 在 2026 年仍屬資料中心工作負載（見 6.6 的估算）。

## 6.2 結論一：為「迴圈階段」設計片上記憶體

**依據**：π0 的 action expert 佔 9% FLOPs 卻佔 Thor 上 56% 延遲；權重佔 expert 流量 97%；若權重常駐片上，10 步延遲從 41.9 ms 降到 21.3 ms（[04 §4.7](04-compute-loading.md)）。

**迴圈階段的權重大小**（[results/sensitivity.md](../results/sensitivity.md) §4）：

| 模型 | FP8 | FP4 |
|---|---|---|
| SmolVLA expert | 97 MB | 48 MB |
| π0 / π0.5 expert | 311 MB | 156 MB |
| GR00T N1.7 DiT | 415 MB | 208 MB |

**建議**：
- 片上 SRAM（或 3D 堆疊 SRAM / 近記憶體）容量目標 **128–256 MB**，足以在 FP4 下讓 π0 / GR00T 級的 expert 常駐，FP8 下讓 SmolVLA 級常駐。
- 權重常駐後，**迴圈步數幾乎免費**——硬體不再需要依賴模型端把步數蒸餾到 1 步，演算法可以保留 10 步的精度。
- 若 SRAM 不足以放下整個 expert，至少做 **layer-wise weight stationary + 多步融合排程**：把同一層在 K 步之間的計算重排是不可能的（步與步有資料依賴），但可以讓**多個請求（多手臂、best-of-N 取樣）的同一步**共用一次權重讀取。

## 6.3 結論二：矩陣引擎要在 M ≈ 40–64 時仍有效率

**依據**：所有 flow VLA 的迴圈 GEMM，M = chunk 長度 + 1 = 41–51；LLM decode 的 M = 1；影片 DiT 的 M = 數千。

**建議**：
- 選擇陣列形狀時，把 **M = 64** 當作一級設計點，而非只看大 M 的峰值。例如多個 64 寬的小陣列，優於一個 256 × 256 大陣列。
- 支援把**整個 K 步迴圈下放到加速器上的序列器**執行（openpi 用 `jax.lax.while_loop` 編譯成一張圖正是為此），避免每步回主機：π0 一次推論約 1,260 次 kernel 呼叫。
- Euler 更新 `x ← x + dt · v`、時間步 embedding、adaRMSNorm 的 scale/shift/gate 生成，要能在向量單元**與 GEMM 融合**，不落地 DRAM。

## 6.4 結論三：不需要 LLM 式的 KV cache 管理，但需要 ring buffer

**依據**：VLA 常駐狀態 0.7–18 MB、固定大小、每次重規劃丟棄；Matrix-Game 2.0 滑動視窗 KV 973 MB；DreamZero 估計 7.7 GB（[03](03-memory-io.md)）。

**建議**：
- VLA：prefix KV 直接放片上 scratchpad（< 20 MB），**不需要分頁、不需要 offload**。這是相對 LLM 加速器可以砍掉的複雜度。
- 串流世界模型：硬體管理的 **KV ring buffer**（以 latent 幀為單位進出），配合 local/sliding-window attention 的位址產生。容量目標 1–2 GB（在 DRAM），頻寬目標見 6.6。
- 記憶檢索式（Matrix-Game 3.0）：需要依相機位姿挑選歷史 latent → 小型 gather/DMA 引擎即可。

## 6.5 結論四：注意力與非 GEMM 運算的支援

**依據**：影片 DiT 注意力佔 FLOPs 24–83%；3D RoPE、qk-norm、adaLN 普遍存在；VLA 注意力遮罩是 prefix 雙向 + suffix 因果。

**建議**：
- **FlashAttention 式的串流注意力引擎**，支援：任意遮罩（prefix-LM、block-causal、frame-causal、滑動視窗）、cross-attention（Q 與 KV 來源不同寬度，如 GR00T 1536 vs 2048）、MQA/GQA/MHA。
- **3D RoPE** 硬體：時間 / 高 / 寬三組頻率的旋轉。
- softmax、LayerNorm/RMSNorm、GELU/SiLU、adaLN 調變的**向量單元吞吐**要按影片 DiT 的 token 數設計（每層每步數千～數萬 token）。

## 6.6 結論五：影片世界模型的算力預算，與最有效的槓桿

以每秒需要的持續算力衡量（BF16，理想值）：

| Workload | 每單位 TFLOP | 單位週期 | 需要的持續 TFLOP/s |
|---|---|---|---|
| π0（10 Hz 重規劃） | 4.4 | 100 ms | 44 |
| Matrix-Game 2.0（25 FPS） | 37.2 | 480 ms | 77 |
| Matrix-Game 3.0（40 FPS、720p） | 242 | 400 ms | 605 |
| DreamZero（7 Hz、4 步） | 548 | 143 ms | 3,840 |
| DreamZero（7 Hz、1 步 + KV 寫入） | ~219 | 143 ms | ~1,530 |

DreamZero 級 WAM 需要 **PFLOP/s 等級**的持續算力，對邊緣晶片仍遙遠。最有效的槓桿依序為：

1. **Latent 壓縮率**：token 數下降 → GEMM 線性下降、注意力平方下降。Wan2.1 VAE（8×8）換成 Wan2.2 VAE（16×16）讓每幀 token 少 4 倍。這是模型端的選擇，但晶片設計者應該把**目標 token 數**（而非解析度）當作規格。
2. **步數蒸餾**：35×CFG2 → 3–4 步 → 1 步，可省 20–70 倍。
3. **低精度**：FP8 / FP4 各 2 倍。
4. **注意力稀疏**：滑動視窗（Matrix-Game 2.0 的 6 幀）讓注意力成本與總長度無關。

## 6.7 結論六：VAE 與感測器 I/O 不能忽略

- 世界模型的 **3D 卷積 VAE** 是獨立的大型工作負載：Matrix-Game 3.0 需要另開行程並剪枝 50% 才達 40 FPS。純 Transformer 加速器需要卷積支援（im2col 轉 GEMM 也可，但 activation footprint 大），或與 ISP/NPU 分工。
- 機器人端：**相機 → ISP → ViT** 直通路徑可省去 CPU 前處理與 PCIe 搬運；多路相機的 ViT 可以批次化（π0 的 3 張圖共用一次 411 MB 權重讀取）。

## 6.8 結論七：精度分級

| 階段 | 建議精度 | 理由 |
|---|---|---|
| ViT、VLM backbone | FP8 / FP4 | 特徵抽取，對量化容忍度高；LLM 社群經驗可直接套用 |
| 動作頭（expert / DiT） | FP8，輸出層 BF16 | 輸出直接驅動馬達，ODE 多步積分累積誤差 |
| 影片 DiT | FP8 / INT8（Matrix-Game 3.0 已用 INT8） | 算力瓶頸，低精度直接換算力 |
| VAE decode | FP16 / INT8 | 影響畫質 |

## 6.9 一顆「邊緣 VLA + 串流世界模型」晶片的粗略規格

把上面的數字加總，以「π0 級 VLA 10 Hz + Matrix-Game 2.0 級世界模型 25 FPS 同時運行」為目標：

| 規格 | 目標 | 推導 |
|---|---|---|
| 持續算力 | ~120 TFLOP/s（BF16 等效） | 44 + 77；以 50% 利用率計，峰值 ~250 TFLOP/s，或 FP8 下 ~125 峰值 |
| 片上 SRAM | 128–256 MB | FP4 下放 π0/GR00T expert（156–208 MB）或 FP8 下放 SmolVLA（97 MB） |
| DRAM 頻寬 | ≥ 200 GB/s | VLA 前段每次 ~2.4 GB（FP8）× 10 Hz = 24 GB/s；世界模型 14.6 GB / 0.48 s ≈ 30 GB/s；加上餘裕與延遲需求（單次推論延遲 = 位元組 / 頻寬） |
| DRAM 容量 | 16 GB | 權重 ~4 GB（FP8）+ KV ring buffer ~1 GB + 影像緩衝 |
| 注意力引擎 | 任意遮罩、3D RoPE、滑動視窗 | §6.5 |
| 卷積 | VAE decode 支援或專用單元 | §6.7 |
| 控制 | 片上序列器執行完整去噪迴圈 | §6.3 |

這只是由分析模型推出的一階規格，正式設計前需要以實際 kernel 在 RTL/周期級模擬器上驗證利用率。

## 6.10 尚待研究的問題

1. **長時記憶 VLA**（π MEM、GR00T N2 的世界模型整合）會不會讓常駐狀態重新變大？
2. **WAM 是否會取代 VLA**？若是，邊緣晶片的設計點會從「頻寬 + SRAM」整體右移到「算力 + 注意力」。
3. **Best-of-N / 多候選動作取樣**是否成為主流——它能把 VLA 迴圈的 M 推高，改變頻寬瓶頸。
4. **VAE 是否會被 Transformer tokenizer 取代**，讓世界模型變成純 Transformer 工作負載。
