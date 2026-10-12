# 09 · 從 roofline 到週期級模擬，以及可執行的實驗套件

> **修訂（2026-10-12，回應 GPT 審查 R1、R5–R8）**：(1) 多核心模型改為同層 attention heads × batch 平行排程，數字已更新；(2) 新增 12 個未參與擬合的 SCALE-Sim 驗證案例，最大誤差 0.057%；(3) Q-head 測試原本使用 Q 輸出層為零的初始化模型，沒有鑑別力，已改用非零且各 head 不同的 fixture 加負向測試重做；(4) openpi 套件改名為「prefix 只算一次 + KV 複製」，並在檢查失敗時中止。詳見 [reviews/2026-10-12-reply-to-gpt-review.md](../reviews/2026-10-12-reply-to-gpt-review.md)。

日期：2026-10-12（第三輪）。

前兩輪的數字都是 roofline 解析估算。GPT 與 CLAUDE 的共識是「下一步要量測」。本輪做兩件事：

1. **週期級模擬**：用 SCALE-Sim v3（學界常用的 systolic array 週期級模擬器）校準一個快速解析模型，再套到所有模型的真實 GEMM 形狀上，取得 roofline 看不到的**陣列利用率**。
2. **實驗套件 + Colab notebook**：把 [review](../reviews/pre2026-llm-acceleration-review.md) 的 E1–E3 寫成可執行程式，並做成一鍵執行的 Colab notebook。

證據等級：單一陣列的 GEMM 週期是 **SCALE-Sim 校準**（擬合 + 獨立驗證）；多核心組態是**解析外推**（fold 分配到各核心、同層 heads 平行、層間依序、未計核心間通訊）；§9.4 的邊緣晶片表是**模擬 + 解析**；§9.5 的 Q-head 等價性是**本環境實測**（CPU）；GPU 量測**尚未執行**，待 Colab。

---

## 9.1 校準：解析模型與 SCALE-Sim 逐週期吻合

SCALE-Sim 很慢（每個 GEMM 1–5 分鐘），無法直接掃描所有形狀。做法是先跑 30 個案例（4 種真實 GEMM 形狀 × 32×32、128×128、16×64、64×16 陣列 × OS/WS/IS 三種 dataflow），再用它擬合解析模型的每 fold 週期公式：

| Dataflow | 陣列映射 | 每 fold 週期 | 最大誤差 |
|---|---|---|---|
| OS（output stationary） | M×N 映射到陣列，K 串流 | `K + R + C − 2` | 0.003% |
| WS（weight stationary，TPU 式） | K×N 映射，M 串流 | `M + 2R + C − 2` | 0.002% |
| IS（input stationary） | K×M 映射，N 串流 | `N + 2R + C − 2` | 0.003% |

非正方形陣列（16×64、64×16）是分開 R 與 C 係數的關鍵；只用正方形陣列時擬合會不唯一（本輪一開始就踩到這點，已修正）。原始資料在 [`results/scalesim_validation.csv`](../results/scalesim_validation.csv)。

上表誤差是在同一份資料上擬合後報告的。為了獨立檢驗，另外跑了 12 個**未參與擬合**的案例（不規則形狀 33×200×72、K 只有 48 的 17×1000×48、48×24 陣列、Flex-π 的 32×3072×1024），**最大誤差 0.057%**，每個預測都只差 1 個週期（[`results/scalesim_holdout.csv`](../results/scalesim_holdout.csv)）。

> 工具備註：SCALE-Sim v3.0.0 在 numpy 2.x 下有 3 處 `int(max(array))` 型別錯誤，本研究改成 `int(np.max(...))` 後執行；修改不影響週期計算邏輯。

## 9.2 結果：同樣 16,384 個 MAC，不同陣列形狀的利用率

![systolic](../figures/systolic_utilization.png)

完整表格：[`results/systolic.md`](../results/systolic.md)。重點：

| 工作負載 | 1 × 128×128 WS（TPU 式） | 16 × 32×32 OS | 16 × 32×32 每 GEMM 選最佳 |
|---|---|---|---|
| π0 動作專家每步（M=51） | **12%** | 75% | 77% |
| GR00T DiT 每步（M=41） | **9%** | 61% | 62% |
| SmolVLA 動作專家每步（M=50） | **10%** | 67% | 69% |
| π0 best-of-16（M=816） | 37% | 88% | 91% |
| π0 VLM prefill（M=816） | 68% | 96% | 97% |
| TD-MPC2-317M rollout（M=512） | 57% | 98% | 98% |
| V-JEPA 2-AC rollout（M=103,200） | 79% | 93% | 98% |
| Matrix-Game 2.0 去噪（M=2,640） | 86% | 91% | 98% |
| Llama-8B decode b=1（M=1） | 0% | 3% | 3% |

### 發現 1：大型 WS 陣列對 VLA 動作迴圈是最差的選擇

WS 在每個 fold 先把一塊 R×C 的權重載入陣列，再讓 M 個輸入流過。M = 41–51 時，載入與排空（2R + C）遠大於有效工作（M）：128×128 時是 382 個週期的開銷對 51 個週期的工作。**這就是 TPU 式大陣列在 VLA 迴圈只有 9–12% 利用率的原因。**

### 發現 2：小陣列多核心 + 可切換 dataflow 對所有類別都好

16 × 32×32、每個 GEMM 選最佳 dataflow，對 VLA 迴圈是 62–77%，對 prefill / 影片 DiT / latent MPC 是 88–98%。沒有任何一類掉到 60% 以下（LLM decode 除外）。

### 發現 3：best-of-K 同時解決了頻寬與利用率

K=16 時 M 從 51 變成 816，128×128 WS 從 12% 升到 37%，小陣列升到 91%。這和 [08 §8.3](08-cross-review-with-gpt.md) 的頻寬分析互相印證：**GPT 提議的「多候選 + 評分」架構，在硬體上同時把動作專家從頻寬瓶頸與陣列利用率瓶頸中解放出來。**

### 發現 4：LLM decode 在 systolic array 上幾乎不能用

b=1 時 0–3%；即使 b=64，每個序列對自己 8K KV 的 attention 仍是 M=1 的 GEMV，利用率只有 1–12%（同層 heads 平行排程後由 8% 升到 12%，GPT 審查 R5 的反例）。這從週期層面解釋了為什麼 LLM decode 加速器（DFX、FlightLLM）用向量 / GEMV 引擎，以及為什麼 NeuPIMs / AttAcc 要把 attention 搬到 PIM——而這些設計對機器人推論不必要（[07](07-llm-accelerator-techniques.md)）。

## 9.3 對 [06](06-asic-implications.md) 的修正

06 §6.3 建議「把 M = 64 當作一級設計點」。週期模擬給出具體形狀：

- **不要用單一大型 WS 陣列**（128×128 以上）作為唯一矩陣引擎。
- **建議 32×32（或 64×64）多核心、支援 OS 與 WS/IS 切換**；32×32 時 OS 對 M=51 就有 74%。
- 長條陣列（256×64、64×256）在固定 dataflow 下都有明顯弱點（例如 256×64 OS 對 M=51），需要可切換 dataflow 才能彌補。

## 9.4 把利用率接回頻寬：Thor 等級晶片上的 π0 一個 chunk

8 塊 × 16,384 MAC、1 GHz（BF16 峰值 262 TFLOP/s，與 Thor 推導值相近）、273 GB/s DRAM。每個階段取 max（陣列週期 ÷ 頻率，DRAM 位元組 ÷ 頻寬）：

| 每塊組態 | Dataflow | 權重每步串流 | 動作專家權重常駐 |
|---|---|---|---|
| 1 × 128×128 | WS | 46.4 ms | **36.4 ms** |
| 1 × 128×128 | OS | 42.4 ms | 25.7 ms |
| 4 × 64×64 | 每 GEMM 最佳 | 41.9 ms | 22.7 ms |
| 16 × 32×32 | 每 GEMM 最佳 | 41.9 ms | **22.5 ms** |

讀法：
- **權重串流時，陣列形狀幾乎不重要**（42–46 ms），因為瓶頸是 DRAM——這和 roofline（41.9 ms）一致。
- **權重常駐後，陣列形狀變成主角**：128×128 WS 36.4 ms vs 小陣列 22.5 ms，差 1.62×。因為常駐移除了頻寬瓶頸，剩下的是陣列能多有效地處理 M=51。
- 因此 SRAM 常駐與陣列形狀必須**一起**決定；只做其中一個，收益有限。這也呼應 GPT 的保留：常駐不會消除算力與迴圈延遲（[review Q3(b)](../reviews/pre2026-llm-acceleration-review.md)）。

## 9.5 實驗套件與 Colab notebook

| 檔案 | 測什麼 | 狀態 |
|---|---|---|
| [`experiments/tdmpc2_plan_profile.py`](../experiments/tdmpc2_plan_profile.py) | E1-latent：TD-MPC2 官方 `_plan` 的延遲、運算子數、`torch.compile`（CUDA graphs）效果；E3(ii)：Q-head 先抽樣的加速與等價性 | **修正後已在 CPU 重測**（[`results/tdmpc2_qsubsample_cpu.csv`](../results/tdmpc2_qsubsample_cpu.csv)）。原測試無效：上游初始化把 Q 輸出層設為零，所有 head 相同（GPT R1）。新 fixture 讓各 head 非零且不同（head 間差異 3.0–4.6），固定 head 編號比較 logits 差 = 0、抽錯 head 的 Q 差 = 3.2–15.9（負向測試通過），規劃器輸出相同。**但 CPU 上沒有加速（0.80–1.01×）**：這個規模下 Q 只佔小部分，取參數子集的開銷抵消收益 |
| [`experiments/openpi_bestofk_profile.py`](../experiments/openpi_bestofk_profile.py) | E1-VLA：π0 / π0.5 best-of-K（prefix 只算一次，KV 以 `jnp.repeat` 複製給每個候選，**不是實體共享 KV**，GPT R6）；E3(i)：真實權重下相鄰 flow 步的 expert 輸出相似度與可用 4 bit 表示的差值比例 | 內建兩項與官方 `sample_actions` 的一致性檢查，超過容忍值或出現 NaN 時**在計時前中止**（GPT R7）；記錄峰值記憶體；**尚未執行**，待 Colab |
| [`experiments/colab_experiments.ipynb`](../experiments/colab_experiments.ipynb) | 把上面全部串起來，GPU 上一鍵執行，結果打包下載或推到 `colab-results/*` 分支 | 待執行 |

每個實驗的預測與失敗條件見 [review Q4](../reviews/pre2026-llm-acceleration-review.md)。最關鍵的一個：**π0 在 GPU 上 K=16 的延遲若超過 K=1 的 2 倍，CLAUDE 對動作專家「受頻寬限制」的結論就要撤回。**

## 9.6 限制

- SCALE-Sim 設為理想記憶體頻寬，只量測陣列計算週期；記憶體影響由 §9.4 的解析方式加入。
- 只模擬 GEMM（含 attention 的 QKᵀ 與 PV），未含 softmax、LayerNorm、adaLN 等向量運算。
- 多核心假設 fold 能完美分配到各核心，未計核心間通訊與同步。
- 陣列頻率、晶片規模為假設值，用於比較陣列形狀，不代表特定產品。
