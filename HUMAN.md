# 給人類看的最新精華

日期：2026-10-12｜作者：Antigravity｜本輪：World Model 運算本質、推論機制與硬體架構深度剖析

世界模型不只是 Attention，更包含前端影像 Tokenizer、高佔比 FFN、動作條件化（AdaLN）與可選的 3D 高斯渲染。在線即時互動的外部輸入確實是 Batch=1，但內部推演通常透過候選軌跡（如 512 條）或空間 Token 形成平行計算。自回歸推論在單 Token 解碼時面臨嚴苛的顯存頻寬牆（每步皆需重讀全權重），而擴散模型則是全序列多步去噪的純算力密集負擔。

文字模型是符號接龍，世界模型則是受動作干預（Intervention）驅動的物理模擬器，對空間一致性與幾何守恆有剛性要求。ASIC 設計不能只堆砌超大矩陣乘法陣列，否則在低 Batch 或非 Attention 運算時硬體利用率極低。建議評估 16~32MB 片上 SRAM 對候選 Rollout 的快取效果，並整合異質加速單元。本輪分析屬於文獻歸納與資料流理論推導，非實體晶片量測。

本次新增：Antigravity 獨立研究分區、推論六題深度技術剖析與硬體資料流架構建議。

- [本輪摘要（Antigravity）](ANTIGRAVITY/updates/2026-10-12-world-model-compute-six-questions/HUMAN.md)
- [完整六題深度分析（Antigravity）](ANTIGRAVITY/01-world-model-compute-and-hardware-analysis.md)
- [GPT previous summary: 運算六問](GPT/updates/2026-10-12-six-compute-questions/HUMAN.md)
- [Claude previous summary: 週期級模擬與實驗](CLAUDE/updates/2026-10-12-systolic-and-colab/HUMAN.md)
- [所有協作者必須遵守的寫作規則](AGENTS.md)
