# 給人類看的最新精華

日期：2026-10-12｜作者：GPT｜本輪：World Model 運算六問

世界模型不一定有 attention，也不一定有 LLM 式的 prefill 和 decode。決策系統通常先編碼觀測，再以同一套模型模擬多條候選未來，評分後執行近期動作。除了 attention，還有大量 MLP、狀態更新、價值評估與搜尋運算。

一台機器人只代表外部輸入可能是 batch 1，內部卻可同時評估512條候選。參數量不等於一次決策成本：本次假設的5.09M模型，在512候選、3步、6輪設定下，主要矩陣計算約46.27 GFLOPs。

權重應區分啟動載入、外部記憶體搬移與片上重用；KV cache不會讓模型權重消失。下一步必須固定模型與規劃設定，才能選擇ASIC規模。上述數字是解析算例，不是實測，不能外推所有世界模型。

本次新增：六題詳細解答、演算法流程及可重算程式。

- [本輪摘要](GPT/updates/2026-10-12-six-compute-questions/HUMAN.md)
- [完整六題解答](GPT/08-world-model-compute-six-questions.md)
- [Claude previous summary](CLAUDE/updates/2026-10-12-systolic-and-colab/HUMAN.md)
- [GPT previous summary](GPT/updates/2026-10-12-human-summary-rule/HUMAN.md)
- [所有協作者必須遵守的寫作規則](AGENTS.md)
