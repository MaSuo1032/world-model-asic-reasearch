# GPT：World Model 與決策運算研究

整理日期：2026-10-12（Asia/Taipei）。內容整理自本次對話的公開資料研究與工程推導。

研究焦點由一般 world model 進展，收斂為使用者指定的「機器生成足以做決策的未來」。

## 核心結論

主流自回歸 LLM 的典型核心迴圈是依上下文生成下一個 token。線上規劃用的 world model 則依狀態與候選動作預測後續狀態，評估候選未來，再選擇近期動作。兩者可以共用 Transformer 與矩陣乘法；主要差異在預測目標、資料流、候選平行、狀態管理及決策期限。

World model 是功能分類，不是一種專屬神經網路架構。LLM 可以參與世界建模；世界模型也可能只在訓練階段用來產生想像經驗，部署時由 policy 直接控制。不能把所有 world model 都當成線上 MPC。

對 ASIC 而言，應優化固定延遲與能耗預算下的決策效益，而非僅比較峰值 TOPS、模型參數或 tokens/s。以 policy 提出候選、world model 評估候選作為研究起點，是本報告的工程建議，不是已驗證的最優架構。

## 文件

| 文件 | 內容 |
|---|---|
| [00-methodology.md](00-methodology.md) | 證據分級、研究範圍、重要限制 |
| [01-progress-2026.md](01-progress-2026.md) | Genie、Waymo、V-JEPA、Cosmos、World Labs、Dreamer 等路線 |
| [02-world-model-vs-llm.md](02-world-model-vs-llm.md) | 訓練、推論、平行性、記憶體與硬體瓶頸 |
| [03-decision-world-models.md](03-decision-world-models.md) | 任務相關狀態、MPC、價值函數、不確定性與部署方式 |
| [04-asic-implications.md](04-asic-implications.md) | 加速器模組、資料重用、精度及可比較的評估方案 |
| [05-worked-estimates.md](05-worked-estimates.md) | 可逐步核算的運算量及狀態容量例子 |
| [06-pre2026-llm-acceleration-survey.md](06-pre2026-llm-acceleration-survey.md) | 2026 年以前 28 項 LLM 加速研究：硬體、kernel、量化、共享與移植限制 |
| [07-transfer-and-experiments.md](07-transfer-and-experiments.md) | 適用矩陣、前三個實驗、PIM／稀疏門檻與給 Claude 的審查意見 |
| [08-world-model-compute-six-questions.md](08-world-model-compute-six-questions.md) | 運算六問：參數、prefill/decode、權重、batch、演算法流程與兩組解析算例 |
| [references-pre2026.md](references-pre2026.md) | 26 篇論文及 2 項技術文章／實作的一手來源與證據類型 |
| [results/transfer-estimates.md](results/transfer-estimates.md) | 候選批次算術強度、共享 KV 容量、Amdahl 上限；可用附錄程式重算 |
| [collaboration/claude-review-request.md](collaboration/claude-review-request.md) | GitHub 協作請求與回覆狀態 |
| [references.md](references.md) | 一手資料及各來源支持的主張 |

## 已完成與未完成

已完成公開資料整理、概念架構分析、公式推導與示意數值核算。

未完成模型實際推論 profiling、機器人閉環實驗、模型量化實測、RTL、綜合、PPA 或實體晶片 benchmark。因此本資料夾的 GFLOPs、容量與效能推論不應被當成特定模型的實測結果。

本資料夾沒有沿用 CLAUDE 的數值結果作為自己的實驗證據。
