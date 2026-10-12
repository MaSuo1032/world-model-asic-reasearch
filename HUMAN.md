# 給人類看的最新精華

日期：2026-10-12｜作者：Claude｜本輪：Flex-π 深入研究（第四輪）

Flex-π 是一個 60 億參數的世界動作模型。它每次推論分兩段：先「看一次」當下的畫面、點雲與語意（prefill，約 3.6 TFLOP、讀 9.8 GB 權重），再對未來動作去噪 4 次（decode）。只出動作時，每步只需讀 2 GB 的動作頭權重；連未來畫面一起生成時，每步要讀 12 GB。整體計算有 94% 是一般矩陣乘法，attention 分數只佔 2–5%。

這些數字是用官方程式碼直接數出來的，和官方在 RTX 5090 上量到的延遲相比吻合 85–93%。只出動作模式只發揮 46% 的硬體效率，因為動作頭每次只處理 32 個 token、受記憶體頻寬限制。本研究沒有實際執行 Flex-π。

GPT 同一天提交了對 Claude 第三輪與 Antigravity 的審查（R1–R8），Claude 正在逐項回應與修正。

- [本輪摘要](CLAUDE/updates/2026-10-12-flexpi-deep-dive/HUMAN.md)
- [Flex-π 逐題回答（6 個問題）](flex-pi/CLAUDE/README.md)
- [GPT 審查 Claude 與 Antigravity](GPT/updates/2026-10-12-review-claude-antigravity/HUMAN.md)
- [Antigravity 交叉審查](ANTIGRAVITY/updates/2026-10-12-cross-review-claude-gpt/HUMAN.md)
- [寫作規則](AGENTS.md)
