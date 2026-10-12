# World Model ASIC Research

研究主題：world model、機器決策與主流 LLM 的運算差異，以及資料流、記憶體與 ASIC 設計。

## 研究資料分區

| 資料夾 | 內容 |
|---|---|
| [CLAUDE](CLAUDE/README.md) | 本次整理前的全部既有檔案，原樣移入；包含分析程式、文件、圖表與結果 |
| [GPT](GPT/README.md) | 本次 GPT 研究：2026 進展、LLM 比較、決策型潛在世界模型與 ASIC 分析 |

整理日期：2026-10-12（Asia/Taipei）。

CLAUDE 與 GPT 是依使用者指定的資料夾名稱區分研究來源，不表示彼此已交叉驗證或達成相同結論。既有內容的模型規格、估算與結論未在本次搬移中重新驗證。

## 既有分析的執行位置

CLAUDE 內原有相對路徑保持不變。從 repository 根目錄執行：

    cd CLAUDE
    python analysis/run_all.py
    python analysis/sensitivity.py

原專案的依賴及限制請見 [CLAUDE README](CLAUDE/README.md)。上述命令會重新產生結果，本次整理沒有重新執行或覆寫這些既有結果。

## CLAUDE 第二輪（2026-10-12）

- [07 · 2026 年以前 LLM 加速器論文與技巧的可用性](CLAUDE/docs/07-llm-accelerator-techniques.md)：約 60 項硬體與軟體技巧，逐項量化對 VLA / 世界模型 / WAM 的效果。
- [08 · 與 GPT 研究的交叉審查](CLAUDE/docs/08-cross-review-with-gpt.md)：以真實模型代入 GPT 的決策成本公式、回應 GPT 的保留意見並修正 CLAUDE 的說法，以及給 GPT 下一輪的建議。

## GPT 建議閱讀順序

1. [摘要與研究範圍](GPT/README.md)
2. [World model 與 LLM 的本質運算差異](GPT/02-world-model-vs-llm.md)
3. [供機器決策的潛在世界模型](GPT/03-decision-world-models.md)
4. [ASIC 資料流與評估方案](GPT/04-asic-implications.md)
5. [完整來源清單](GPT/references.md)
6. [2026 年以前 LLM 加速研究及可移植性](GPT/06-pre2026-llm-acceleration-survey.md)
7. [移植實驗與 Claude 審查焦點](GPT/07-transfer-and-experiments.md)

## GitHub 協作

依使用者指定，透過 CLAUDE/GPT 資料夾交換研究與審查意見。GPT 已整理 [28 項加速研究來源](GPT/references-pre2026.md)，並提供 [Claude 審查請求](GPT/collaboration/claude-review-request.md)。此請求不代表 Claude 已收到執行通知或完成審查；目前尚未收到回覆。
