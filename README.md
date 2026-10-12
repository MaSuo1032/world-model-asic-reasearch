# World Model ASIC Research

研究主題：world model、機器決策與主流 LLM 的運算差異，以及資料流、記憶體與 ASIC 設計。

**先讀 [HUMAN.md：給人類看的精華](HUMAN.md)。** 每次研究寫作或實質修訂，都必須新增本輪 `HUMAN.md` 並更新根目錄摘要；完整規則見 [AGENTS.md](AGENTS.md)，適用於 GPT、Claude 與其他協作者。

## 研究資料分區

| 資料夾 | 內容 |
|---|---|
| [ANTIGRAVITY](ANTIGRAVITY/README.md) | 本次 Antigravity 研究：世界模型運算本質、推論機制與硬體架構深度剖析（六問解答） |
| [CLAUDE](CLAUDE/README.md) | 本次整理前的全部既有檔案，原樣移入；包含分析程式、文件、圖表與結果 |
| [GPT](GPT/README.md) | 本次 GPT 研究：2026 進展、LLM 比較、決策型潛在世界模型與 ASIC 分析 |

整理日期：2026-10-12（Asia/Taipei）。

ANTIGRAVITY、CLAUDE 與 GPT 是依使用者指定的資料夾名稱區分研究來源，不表示彼此已交叉驗證或達成相同結論。既有內容的模型規格、估算與結論未在本次搬移中重新驗證。

## 既有分析的執行位置

CLAUDE 內原有相對路徑保持不變。從 repository 根目錄執行：

    cd CLAUDE
    python analysis/run_all.py
    python analysis/sensitivity.py

原專案的依賴及限制請見 [CLAUDE README](CLAUDE/README.md)。上述命令會重新產生結果，本次整理沒有重新執行或覆寫這些既有結果。

## CLAUDE（2026-10-13）：給教授的口頭報告稿與投影片

- [口頭報告稿](CLAUDE/reports/2026-10-13-oral-report.md)：整個 repo 的定性濃縮，約 8 分鐘，報告日期 2026-10-13。
- [投影片（.pptx）](CLAUDE/reports/World_Model_ASIC_report_2026-10-13.pptx)：使用實驗室簡報模板，共 12 頁，講稿在備註。
- [本輪人類精華摘要](CLAUDE/updates/2026-10-13-oral-report/HUMAN.md)

## ANTIGRAVITY 第二輪（2026-10-12）：對 CLAUDE 與 GPT 之交叉審查

- [02 · 對 CLAUDE 與 GPT 世界模型研究之交叉審查](ANTIGRAVITY/02-cross-review-claude-gpt.md)：指出 CLAUDE 忽視非 GEMM 延遲與 SRAM/Activation 溢出、指出 GPT 忽視層優先候選串流與 3DGS 空間智能，提供三方成果整合矩陣與 ASIC 核心建議。
- [Antigravity 第二輪人類精華摘要（最新）](ANTIGRAVITY/updates/2026-10-12-cross-review-claude-gpt/HUMAN.md)

## ANTIGRAVITY 第一輪（2026-10-12）：世界模型運算本質與推論六問

- [01 · World Model 運算本質、推論機制與硬體架構深度剖析](ANTIGRAVITY/01-world-model-compute-and-hardware-analysis.md)：剖析 Attention 以外運算、三層參數梯隊、自回歸 vs 擴散 vs 潛在 MPC 的 Prefill/Decode 負擔、權重搬移與外部頻寬牆、Batch=1 迷思與物理因果閉環演算法流程。
- [Antigravity 第一輪人類精華摘要](ANTIGRAVITY/updates/2026-10-12-world-model-compute-six-questions/HUMAN.md)

## CLAUDE 第二輪（2026-10-12）

- [07 · 2026 年以前 LLM 加速器論文與技巧的可用性](CLAUDE/docs/07-llm-accelerator-techniques.md)：約 60 項硬體與軟體技巧，逐項量化對 VLA / 世界模型 / WAM 的效果。
- [08 · 與 GPT 研究的交叉審查](CLAUDE/docs/08-cross-review-with-gpt.md)：以真實模型代入 GPT 的決策成本公式、回應 GPT 的保留意見並修正 CLAUDE 的說法，以及給 GPT 下一輪的建議。
- [Claude 對 GPT 審查請求的回覆](CLAUDE/reviews/pre2026-llm-acceleration-review.md)：回覆 [GPT/collaboration/claude-review-request.md](GPT/collaboration/claude-review-request.md) 的四個問題（一手來源驗證層級、可移植性、三項差異、三個可驗證實驗）。

## Flex-π 研究區（2026-10-12）

- [flex-pi/](flex-pi/README.md)：針對 6B World Action Model [Flex-π](https://github.com/MaSuo1032/flex-pi) 的深入研究。
- [Claude：逐題回答 6 個問題](flex-pi/CLAUDE/README.md)：以官方程式碼精確計數參數、FLOPs、prefill / decode、權重載入，並與官方 RTX 5090 實測延遲交叉驗證。

## CLAUDE 第三輪（2026-10-12）：週期級模擬與 Colab 實驗

- [09 · 週期級陣列模擬與實驗套件](CLAUDE/docs/09-systolic-simulation-and-experiments.md)：SCALE-Sim 校準的陣列利用率；大型 WS 陣列跑 VLA 動作迴圈只有 9–12%。
- [實驗套件](CLAUDE/experiments/README.md)｜[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MaSuo1032/world-model-asic-reasearch/blob/main/CLAUDE/experiments/colab_experiments.ipynb)

## GPT 建議閱讀順序

1. [摘要與研究範圍](GPT/README.md)
2. [World model 與 LLM 的本質運算差異](GPT/02-world-model-vs-llm.md)
3. [供機器決策的潛在世界模型](GPT/03-decision-world-models.md)
4. [ASIC 資料流與評估方案](GPT/04-asic-implications.md)
5. [完整來源清單](GPT/references.md)
6. [2026 年以前 LLM 加速研究及可移植性](GPT/06-pre2026-llm-acceleration-survey.md)
7. [移植實驗與 Claude 審查焦點](GPT/07-transfer-and-experiments.md)
8. [World Model 運算六問：參數、prefill/decode、權重與完整流程](GPT/08-world-model-compute-six-questions.md)
9. [GPT 審查 Claude 與 Antigravity：8項發現與重算證據](GPT/reviews/2026-10-12-claude-antigravity-review.md)

## GitHub 協作

依使用者指定，透過 CLAUDE/GPT 資料夾交換研究與審查意見。GPT 已整理 [28 項加速研究來源](GPT/references-pre2026.md)，並提供 [Claude 審查請求](GPT/collaboration/claude-review-request.md)。2026-10-12 同步時已確認 [Claude 回覆](CLAUDE/reviews/pre2026-llm-acceleration-review.md)存在；GPT 尚未完成逐項核驗，不代表雙方結論已全部確認。
