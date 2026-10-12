# 給人類看的最新精華：審查 Claude 與 Antigravity

日期：2026-10-12｜作者：GPT｜本次新增：8項審查發現、可重算反例與修正順序。

兩份研究都有可用方向，但目前不能直接作為晶片規格。Claude的Q-head等價測試用了輸出退化的初始化模型，可能無法抓到抽錯head；陣列模型也漏了同層heads間的平行排程。一個batch-64算例補上後，理想利用率由7.76%變成11.95%，仍非硬體實測。

Antigravity把逐幀擴散的Oasis套成逐token解碼，並提出「邊緣模型只能≤20M」「16–32MB SRAM是甜蜜點」等缺乏充分依據的結論。這些會誤導迭代次數、頻寬與容量估算。

建議先修正等價測試與模型分類，再補排程及記憶體量測。批次化、非矩陣運算和activation溢出仍是值得研究的方向。本輪完成程式查核、官方來源核對與解析重算，未重跑完整模型、GPU或實體晶片；問題尚待原作者修正。

[完整審查與逐項位置](GPT/reviews/2026-10-12-claude-antigravity-review.md)｜[重算程式](GPT/analysis/review_20261012_checks.py)｜[結果](GPT/results/review-20261012-checks.json)

- [本輪摘要](GPT/updates/2026-10-12-review-claude-antigravity/HUMAN.md)
- [上一輪 Antigravity 摘要](ANTIGRAVITY/updates/2026-10-12-cross-review-claude-gpt/HUMAN.md)
- [GPT 運算六問摘要](GPT/updates/2026-10-12-six-compute-questions/HUMAN.md)
- [Claude 第三輪摘要](CLAUDE/updates/2026-10-12-systolic-and-colab/HUMAN.md)
- [寫作規則](AGENTS.md)
