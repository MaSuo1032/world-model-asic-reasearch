# 給人類看的最新精華

日期：2026-10-12｜作者：Claude｜本輪：陣列形狀模擬與 Colab 實驗套件（第三輪）

如果加速器只放一個 TPU 式的大型矩陣陣列，機器人動作模型反覆計算的那一段只能用到 9–12% 的算力。改成多個小陣列、並讓每個運算自選資料流動方式後，利用率升到 59–77%；同一設計跑影片世界模型、規劃模型與 VLM 前段時仍有 88–98%。

這讓「把動作頭權重放在晶片上」的建議變得更具體：只有搭配小陣列才有效。在 Thor 等級的晶片上，π0 一段動作從 38 ms 降到 23 ms。讓 policy 一次產生 16 個候選，也能同時改善頻寬和陣列利用率，和 GPT 提出的「多候選 + 評分」架構相互支持。

陣列利用率是週期級模擬（以 SCALE-Sim 校準，誤差小於 0.01%）。TD-MPC2 Q 值頭先抽樣再計算、結果與原版逐位元相同，是本機實測。GPU 量測尚未執行。下一步是在 Colab 執行實驗 notebook，檢查 π0 產生 16 個候選時，延遲是否真的不超過 1 個候選的 2 倍。

- [本輪摘要](CLAUDE/updates/2026-10-12-systolic-and-colab/HUMAN.md)
- [Colab 一鍵開啟實驗](https://colab.research.google.com/github/MaSuo1032/world-model-asic-reasearch/blob/main/CLAUDE/experiments/colab_experiments.ipynb)
- 前一輪（GPT）摘要：[研究摘要與寫作規則](GPT/updates/2026-10-12-human-summary-rule/HUMAN.md)
- [寫作規則](AGENTS.md)
