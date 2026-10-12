# ANTIGRAVITY 研究資料分區

本資料夾保存 Antigravity 針對世界模型（World Model）、多模態 3D/4D 空間智能、推論運算特徵與 ASIC 硬體加速架構的獨立研究與分析。

---

## 📑 建議閱讀順序

1. **[給人類看的最新精華（HUMAN.md）](updates/2026-10-12-cross-review-claude-gpt/HUMAN.md)**：本輪對 CLAUDE 與 GPT 交叉審查的 300 字白話精華。
2. **[02 · 對 CLAUDE 與 GPT 世界模型研究之交叉審查](02-cross-review-claude-gpt.md)**：
   * 指出 CLAUDE 忽視非 GEMM 運算之 Amdahl's Law 延遲瓶頸、Best-of-K 的 SRAM/Activation 溢出風險，以及 V-JEPA 2-AC Token 級模型在邊緣端不可行之事實。
   * 指出 GPT 忽視「層優先候選串流（Layer-wise candidate streaming）」、3DGS 空間世界模型結構性缺位，以及複合誤差對長 Horizon 之懲罰。
   * 提供三方視角對照矩陣與具體 ASIC 架構建議。
3. **[01 · World Model 運算本質、推論機制與硬體架構深度剖析](01-world-model-compute-and-hardware-analysis.md)**：
   * Q1: Attention 以外的核心運算（Tokenizer, FFN, AdaLN 條件調製, 狀態轉移, 3DGS 渲染）。
   * Q2: 參數規模層級（4M~200M 輕量決策、1B~14B 具身基座、10B~32B+ 大型多模態）。
   * Q3: Prefill 與 Decode 在自回歸、擴散模型與潛在 MPC 中的真實計算負擔。
   * Q4: 晶片記憶體階層（DRAM/HBM $\to$ SRAM $\to$ 暫存器）的權重載入特性與 KV Cache 限制。
   * Q5: Batch = 1 的迷思：外部環境 Batch = 1 與內部多候選 ($K=512$) / 空間 Token ($M=256$) 平行維度剖析。
   * Q6: 實體閉環推演演算法流程與 World Model vs LLM 之三大本質差異。
3. **[最新更新記錄](updates/2026-10-12-world-model-compute-six-questions/HUMAN.md)**。

---

## 🔬 研究焦點與定位

* **空間智能與多模態前沿**：涵蓋李飛飛 World Labs（Marble / RTFM / Spark 3DGS）、Meta JEPA 體系（V-JEPA 2）、Meta WorldGen 3D 空間生成與 Meta CWM 代碼世界模型。
* **ASIC 與資料流瓶頸**：結合計算密集度（Compute-bound）與記憶體頻寬牆（Memory Bandwidth Wall）分析，為專用硬體加速器提供架構設計指導。
