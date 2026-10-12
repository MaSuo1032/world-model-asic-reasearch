# 給人類看的最新精華

日期：2026-10-12｜作者：Antigravity｜本輪：對 CLAUDE 與 GPT 世界模型研究之交叉審查

CLAUDE 的「多核小陣列優於單一大陣列」結論獲得強烈支持，但其忽視了非 GEMM 運算的阿姆達爾定律（Amdahl's Law）延遲瓶頸，以及多候選擴張時的 Activation/SRAM 溢出風險；GPT 的決策成本公式精闢，但將權重讀取簡化為全放或每步重讀，忽略了「層優先候選串流」能在有限 SRAM 下完全攤銷權重。在邊緣端，以 V-JEPA 2 為代表的 Token 級世界模型做在線 MPC 在物理算力上不可行（需近 20 PFLOPS），邊緣晶片必須錨定於向量型潛在模型（$\le 20$M）。

避免硬體架構師掉入兩大陷阱：一是誤以為加速 GEMM 就解決了控制延遲，結果被歸一化、調製與排序拖慢；二是誤把資料中心級別的視訊 Token 模型當作邊緣 ASIC 的在線規格，導致開出不切實際的算力與面積預算。建議在模擬器中加入非 GEMM 運算延遲，並驗證 16~32MB SRAM 下採用層優先排程時之記憶體流量。本輪分析屬於文獻歸納與資料流理論推導，非實體晶片量測。

本次新增：Antigravity 對 CLAUDE 與 GPT 之交叉審查、三方結論矩陣與 ASIC 核心建議。

- [本輪摘要：交叉審查（Antigravity）](ANTIGRAVITY/updates/2026-10-12-cross-review-claude-gpt/HUMAN.md)
- [完整交叉審查報告（Antigravity）](ANTIGRAVITY/02-cross-review-claude-gpt.md)
- [上一輪：推論六問與硬體分析（Antigravity）](ANTIGRAVITY/01-world-model-compute-and-hardware-analysis.md)
- [GPT previous summary: 運算六問](GPT/updates/2026-10-12-six-compute-questions/HUMAN.md)
- [Claude previous summary: 週期級模擬與實驗](CLAUDE/updates/2026-10-12-systolic-and-colab/HUMAN.md)
- [所有協作者必須遵守的寫作規則](AGENTS.md)
