# World Model 與主流 LLM 的本質運算差異

## 核心差異

典型自回歸 LLM：p(x[t+1] | x[<=t])，依上下文生成下一個 token。

決策用 world model：p(z[t+1] | z[t], a[t])，依目前狀態與動作預測後續狀態。規劃器將預測轉為候選評分，再選動作。

這是功能與執行流程的差異，不是「一種用 Transformer，另一種不用」。World model 可以用 MLP、循環模型、Transformer 或混合架構。LLM 若具備合適的狀態／動作表示與訓練，也能參與世界建模。

## 推論比較

| 面向 | 典型自回歸文字 LLM | 線上規劃型 latent world model |
|---|---|---|
| 輸出 | Token 序列 | 供評估使用的未來狀態，最後選出動作 |
| 迴圈 | Decode、更新 KV、下一 token | Rollout、評分、候選更新、執行、重新觀測 |
| 平行來源 | 多請求／多生成序列 | 同一次決策中的候選／隨機結果 |
| 串行依賴 | 後續 token 依賴前文 | 後續狀態依賴前一步 |
| 主要暫存 | KV、權重及 activation | 狀態、必要歷史／KV、分數及搜尋資料 |
| 成本乘數 | 輸入與輸出長度、模型、batch | 候選數、rollout 步數、搜尋輪數、結果樣本 |
| 期限 | 首 token、token 間延遲 | 感測至動作的端到端期限、P99 與超時率 |
| 品質 | 回答及任務正確性 | 候選排序、閉環成功、風險與穩定性 |

LLM 的外部搜尋、多次採樣、工具呼叫也可形成分支評估，因此「有搜尋」不是 world model 獨有。重點是動作條件式環境預測及其在決策中的角色。

## 訓練比較

LLM 常用 next-token cross entropy；teacher forcing 使多個已知序列位置可在訓練時一起計算。不能將推論時逐 token 的串行方式原樣套到訓練成本。

World model 的訓練目標依路線不同：

- JEPA：預測 target encoder 的特徵，避免要求完整像素重建。
- 動態模型：以動作與歷史預測未來狀態，可加入多步一致性。
- 任務導向模型：共同學 reward／value 或其他控制相關訊號。
- 生成式模型：學離散 token、去噪或 flow 向量場，也可能混合上述目標。

常見 diffusion／flow 訓練會抽樣噪聲時間點；推論採樣 S 步不代表每個訓練樣本必須完整執行 S 步。不能把採樣倍率直接當作訓練倍率。

V-JEPA 的表徵預測與 action-conditioned 後訓練例子見 [官方程式庫](https://github.com/facebookresearch/vjepa2)；任務導向潛在模型與 terminal value 的例子見 [TD-MPC](https://github.com/nicklashansen/tdmpc)。

## 三類不同的 world model 成本

1. **影片生成**：一批時空 tokens 反覆去噪／flow 更新，外加 encoder／decoder。
2. **潛在規劃**：在 latent 狀態展開候選，可能完全不生成畫面。
3. **顯式 3D**：先生成資產，再多次渲染；生成與瀏覽的硬體成本不同。

Autoregressive 與 diffusion 可以同時存在：時間上逐段輸出，每段內去噪。[Atlas](https://www.worldlabs.ai/blog/atlas) 是公開的混合架構案例。World model 不等於 diffusion。

## 記憶體與算力

典型低 batch LLM decode 常受權重與 KV 讀取限制；prefill 常有較高算術強度。實際瓶頸仍取決於模型、長度、batch 及硬體。[NVIDIA 分析](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)

World model 可以同時評估多個候選以重用權重，但也可能因密集 visual tokens、attention 或多步 activation 搬移而受頻寬限制。不能直接把兩者概括為「LLM memory-bound，world model compute-bound」。

固定前綴的 KV 可以重用；diffusion 中正在改變的 tokens 不能直接當固定前綴。Latent world model 是否使用 KV、是否需要保存完整歷史，取決於轉移模型。Markov 狀態與歷史 Transformer 的容量需求不同。

對完整 attention，每層投影與 MLP 大致隨 N*d^2 增長，attention 隨 N^2*d 增長。FlashAttention 減少 HBM I/O 與中間矩陣儲存，沒有把完整 attention 的算術量自動變成線性。[原始論文](https://arxiv.org/abs/2205.14135)

## ASIC 評價單位

LLM 常用 tokens/s、TTFT、每 token 能耗。決策系統還應量測成功率、決策延遲、每次成功任務能耗、超時率，以及受限預算下增加模型評估是否值得。

Tokens/s 或 rollouts/s 可作輔助指標，但不能單獨代表決策價值。
