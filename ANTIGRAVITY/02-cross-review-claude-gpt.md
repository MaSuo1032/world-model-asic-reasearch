# 02 · 對 CLAUDE 與 GPT 世界模型研究之交叉審查

日期：2026-10-12｜審查者：Antigravity。
審查對象：
- [`CLAUDE/`](../CLAUDE/README.md) 系列文件（01–09、SCALE-Sim 週期級模擬與開源張量分析）。
- [`GPT/`](../GPT/README.md) 系列文件（00–08、決策成本公式、LLM 加速遷移調查與運算六問）。

---

## 1. 總體評價與共識確認

本專案兩位協作者的研究深度令人驚艷，彼此形成了高度互補：
* **CLAUDE 的優勢**在於**工程落地與真實張量形狀**：從 TD-MPC2、$\pi_0$、V-JEPA 2、SmolVLA 原始碼中萃取確切維度，並以 SCALE-Sim 進行週期級陣列模擬，揭示了 TPU 式大陣列在動作迴圈利用率低（9–12%）的實測現象。
* **GPT 的優勢**在於**概念抽象與數學邊界**：建立了決策成本公理化公式 $C_{\text{decision}} \approx I \cdot K \cdot H \cdot U \cdot C_{\text{transition}}$，並嚴格區隔模型容量、記憶體搬移與算術強度。

### 三方一致認同的關鍵基石：
1. **World Model 不是單一模型架構**：不可把純影片生成、潛在空間 MPC 與顯式 3D 物理引擎混為一談。
2. **VLA $\neq$ 顯式世界模型**：VLA 輸出動作，世界模型模擬狀態轉移；兩者不可混為一談。
3. **外部輸入 Batch=1 $\neq$ 晶片內部 Batch=1**：多候選（$K$）平行性是解放硬體利用率的關鍵。
4. **LLM Decode 的硬體加速方案難以直接平移**：LLM Decode 是極端的 $M=1$ GEMV 訪存瓶頸，而具備多候選或空間 Token 的世界模型本質上可轉化為中大型 GEMM。

---

## 2. 對 CLAUDE 研究的關鍵質疑與盲點剖析

CLAUDE 的週期級模擬和熱圖分析極具說服力，但存在三個關鍵盲點：

### 盲點 2.1：忽視非 GEMM 延遲造成的 Amdahl's Law 陷阱
* **問題**：CLAUDE 的利用率分析完全建立在 SCALE-Sim 的 GEMM 迴圈上。然而在真實的具身決策迴圈中，**非 GEMM 運算**（影像前端去畸變、ViT Patchify、AdaLN 條件調製、Softmax/LayerNorm、MPPI 候選篩選與 Top-K 排序）佔據相當大的時鐘週期。
* **風險**：若 ASIC 僅把 GEMM 加速 $10\times$，根據阿姆達爾定律（Amdahl's Law），若非 GEMM 部分佔比 20%，整機端到端加速上限僅有 $3.57\times$。CLAUDE 未在延遲預算中為非 GEMM 資料流預留安全餘量。

### 盲點 2.2：Best-of-K 平行度「免費午餐」的暫存器與 SRAM 溢出風險
* **問題**：CLAUDE 在 08 §8.3 指出：「在 Thor 上產生 16 個候選 chunk 只比 1 個多 25% 的時間，多出來的候選搭在原本就要做的權重讀取上」。
* **質疑**：權重讀取固然被攤銷，但 **Activation 快取與 KV Cache 流量隨 $K$ 成線性增長**。當 $K=16 \to 64$ 時，中間層特徵圖與注意力快取極易突破邊緣晶片的片上 SRAM（通常僅 16~32MB），迫使 Activation 頻繁溢出（Spill）至外部 LPDDR5/HBM，瞬間抵消權重攤銷帶來的延遲紅利。

### 盲點 2.3：V-JEPA 2-AC Token 級世界模型在邊緣端的可行性未予直接否定
* **問題**：CLAUDE 計算出 V-JEPA 2-AC 每次轉移需 245 GFLOPs，一次決策（8,000 次轉移）高達 **1,962 TFLOPs**。
* **修正建議**：在 10 Hz 的機器人控制迴圈中，這要求 **19.6 PFLOPS 的持續算力**。這相當於數台 NVidia H100 伺服器集群，在車載或機器人邊緣端 ASIC（算力上限約 1~2 PFLOPS INT8）**在物理與功耗上是絕對不可行的**。CLAUDE 應明確指出：**以目前邊緣矽智財能力，Token 級的稠密世界模型不能在線上執行百級數候選之 MPC**，必須依賴超緊湊潛在向量（如 TD-MPC2 式）或重度剪枝/稀疏化。

---

## 3. 對 GPT 研究的關鍵質疑與盲點剖析

GPT 的架構分類與公式推導十分嚴謹，但存在以下硬體與演算法斷層：

### 盲點 3.1：忽略「層優先候選管線化（Layer-by-Layer Candidate Streaming）」
* **問題**：GPT 在 04 與 08 中討論權重載入時，將「權重是否能常駐片上」視為二元對立（要麼全放 SRAM，要麼每步全部重讀）。
* **修正建議**：在硬體實作中，當多候選（$K=512$）推演時，最佳資料流不是逐候選走完所有層，而是**層優先串流（Layer-wise Streaming）**：
  $$\text{外部記憶體載入 Layer } l \text{ 權重至 SRAM} \implies \text{將全部 } K \text{ 個候選之 Activation 流過該層} \implies \text{換 Layer } l+1$$
  透過這種排程，即使總模型高達 300MB（遠超 32MB SRAM 容量），**每個時間步也只需由外部記憶體讀取模型權重剛好 1 次**！GPT 應將此排程納入記憶體流量公式中。

### 盲點 3.2：空間智能世界模型（3DGS / Splatting）的結構性缺位
* **問題**：GPT 的文獻視野主要集中在 TD-MPC2 與 V-JEPA 2，忽視了 2025/2026 年李飛飛 World Labs（Spark 3DGS）、Meta WorldGen 等空間世界模型。
* **實質差異**：空間世界模型並非純 Transformer，其核心負擔包含 **3D 高斯投影、基數排序（Radix Sort）與 Tile 光柵化**。這些運算的記憶體存取模式隨相機視角極度不規則，無法被傳統 GEMM 脈動陣列加速。若 ASIC 研究只針對 GEMM，將完全無法支援未來的空間智能管線。

### 盲點 3.3：忽視複合誤差（Compounding Error）對長 Horizon 規劃的懲罰
* **問題**：GPT 假設固定 $H=15$ 或 $H=3$。但在無真實物理回饋的神經模擬中，自回歸推演會迅速累積特徵漂移與物理幻覺。
* **影響**：實務上超過 $H > 5$ 的潛在預測常已失真，盲目增加 $H$ 只是浪費算力與記憶體。自適應 Horizon 或搭配保守性價值評估（Conservative Q-learning）是演算法必經之路，硬體控制器應支援動態提前終止（Early Termination）。

---

## 4. 交叉綜合：三方研究成果整合對照矩陣

| 評估維度 | CLAUDE 視角 | GPT 視角 | ANTIGRAVITY 裁決與補充 |
| :--- | :--- | :--- | :--- |
| **陣列架構選擇** | 推薦 $16 \times 32\times32$ 多核心 OS 陣列，否定 $128\times128$ WS 大陣列。 | 著重於總 FLOPs 負擔，未深入陣列映射。 | **完全贊同 CLAUDE**。但補充：需搭配異質向量單元以防 Amdahl's Law 瓶頸。 |
| **權重與頻寬瓶頸** | 主張 128~256MB SRAM 常駐，但遭 GPT 質疑面積過大。 | 指出邊緣 SRAM 難以容納大模型，強調 Memory Bandwidth 限制。 | **折衷方案**：SRAM 配置 16~32MB，採用**層優先候選串流（Layer-wise Streaming）**，既保證面積控制在 15~25 mm²，又能將外部權重讀取降至最低。 |
| **模型選擇落地** | 具體量化了 V-JEPA 2 與 $\pi_0$。 | 提出 $I \cdot K \cdot H$ 決策成本框架。 | **明確界定**：邊緣端在線 MPC 僅能採用 **向量型潛在模型（$\le 20\text{M}$）**；Token 級視訊/空間模型只能作非同步離線背景評估或 Cloud-Edge 分離部署。 |

---

## 5. 給 ASIC 設計團隊之核心建議綱要

1. **拒絕單一巨大脈動陣列**：採用多核小陣列（如 $8 \sim 16$ 組 $32\times32$ 彈性資料流處理單元），針對 $K=128 \sim 512$ 之候選批次優化利用率。
2. **SRAM 配置甜蜜點為 16MB ～ 32MB**：在 N5/N4 製程下約佔晶片面積 $15 \sim 25\text{ mm}^2$，配合層優先排程（Layer-wise candidate streaming），足以鎖定各層權重與候選 Activation。
3. **專屬非 GEMM 硬體管線**：配置專用 AdaLN 係數生成器、LayerNorm/Softmax 融合加速器，以及支援 Top-K 候選剪枝之硬體排序單元，消弭 Amdahl's Law 的延遲拖累。
