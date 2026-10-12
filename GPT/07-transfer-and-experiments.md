# 從 LLM 加速移植到決策型 World Model：實驗與取捨

日期：2026-10-12。這份文件是可執行的研究設計，不是已完成的模型 benchmark。來源與機制詳見[綜述](06-pre2026-llm-acceleration-survey.md)和[28 項文獻](references-pre2026.md)。

## 1. 優先矩陣

✓：適用機制明確但仍須測量；△：有結構／工作負載條件；—：該工作負載通常沒有對應運算。

| 方法 | 緊湊 latent MLP/recurrent | 密集 latent Transformer | 自回歸 action-token 模型 | 優先度 |
|---|---|---|---|---|
| 候選 batching、權重重用、算子融合 | ✓ | ✓ | ✓ | P0 |
| 固定形狀排程／graph replay | ✓ | ✓ | △，長度變化需管理 | P0 |
| W8A8／FP8 或 W4 配套 kernel | △ | △ | △ | P1 |
| FlashAttention／streaming attention | — | ✓，支援 mask/shape 才成立 | ✓ | P1 |
| 共用 encoder 輸出 | △，若有 encoder | ✓，相同觀測且不受候選影響 | △ | P1 |
| Hydragen 類共享 prefix 計算 | — | △，需不變的 prefix KV | △，確有共同 prefix | P1 |
| PagedAttention／RadixAttention | — | △，可變長 KV 且重用顯著 | △ | P2 |
| GQA、稀疏化、Softermax | △，只有對應算子 | △，常需校準／再訓練 | △ | P2 |
| NeuPIMs／AttAcc 類分工 | △，權重 GEMV 若主導 | △，共享後仍低 AI 才考慮 | △，attention/GEMV 比例足夠 | P2 |
| 精確 speculative decoding | — | —，一般連續 dynamics 不滿足條件 | △，機率與驗證規則成立 | P2 |
| NAND in-storage 權重運算 | △，通常不值得 | △，先證明容量瓶頸 | △，超大模型容量需求 | P3 |

矩陣不代表所有模型都可直接換 kernel。決定移植性的最小資訊是算子形狀、mask、state/KV 語義、精度、候選數、時間步數與期限。

## 2. 先收集一份足以決策的 trace

至少選兩類代表工作負載：A 為緊湊狀態轉移 MLP/recurrent；B 為多空間 token 的 latent Transformer。V-JEPA 類 predictor 可屬 B，但不得用 A 的低成本推估 B。另有實際目標模型後，所有優先度應由其 trace 更新。

每份 trace 記錄：

- 模型／權重 commit、框架與 kernel 版本、裝置、實際 dtype、dense/sparse 路徑。
- 每算子的 M/N/K、attention query/KV 長度、heads、mask、batch、資料排列與時間。
- encoder、每步 predictor、value/reward head、候選採樣／更新及同步各自佔比。
- 外部記憶體 bytes、片上峰值工作集、有效頻寬、有效算力；量不到者標為估算。
- 候選數 K、horizon H、規劃迭代 I、隨機樣本／ensemble 數 E、可重用的觀測與 prefix。

MPC 的粗略控制成本可寫成：

`T_control = T_observe_encode + Σ_i Σ_t T_predict_and_score(K_i, E_i, shape_t) + T_sampling_update + T_sync`。

其中 predictor 的 batch 已包含 K/E，不能再把同一 batch latency 乘一次 K。也不能將所有階段峰值 FLOPs 相加後直接除以峰值 TFLOPS，當成可實現延遲。Roofline 只能給單階段理想下界：`max(F/P_eff, Bytes/BW_eff)`，還需處理依賴、啟動、搬移重疊與同步。

## 3. 實驗一：候選 batching、融合與片上重用（最先做）

**假設：**目前不少時間來自重複載入權重、中間 activation 搬移或小 kernel 啟動；在相同候選與規劃設定下，合法 batching／融合能降低每次決策成本。

比較四個版本：原始逐候選／既有基線；相同時間步的候選 batching；加入 kernel fusion／固定圖重放；再掃描 SRAM 工作集與 subbatch。若基線本來已 batching，保留其最佳合理設定，不能刻意製造弱基準。

掃描 K={1,8,32,128,256}、H={1,4,16}，以及實际模型尺寸；硬體容量不足的點明確標記。這些是實驗設計值，不是某論文模型的預設。每個 sweep 固定其他變因。

保留候選獨立性：線性層可合併 batch 維；attention 不能把不同候選當成同一無遮罩序列。跨時間步仍遵守 z(t+1)=f(z(t),a(t)) 的依賴。若存在跨候選統計運算，先證明重排等價。

量測 P50/P95/P99 完整觀測到動作延遲、deadline miss rate、J/decision、bytes/decision、峰值記憶體與各階段時間；同一組候選下比較輸出數值誤差、評分及動作选择差異。功耗包含主機與裝置的範圍必須說明，不能只把 kernel 功率當系統能耗。

**失敗條件：**batching 增加等待或 spill，P99 變差；fusion 引入超大暫存；算力已飽和而搬移不再主導；或錯誤跨候選混合改變結果。這些結果應導向縮小 subbatch／重做 tiling，不是繼續加 SRAM。

對 ASIC 的輸出：矩陣陣列利用率、SRAM 容量／頻寬、DMA、歸約／非線性單元吞吐及 sequencer 成本需求。相關來源：DFX、FlightLLM、Orca、FlashInfer。

## 4. 實驗二：共享觀測與共享 prefix 要分開測

**先決条件：**逐層確認同一觀測在不同 action 候選下的 encoder 特徵或 prefix K/V 確實一致。若 action 影響 prefix 或 attention 是全雙向，縮小到可合法重用的 encoder 層，不強行套用 KV cache。

以相同有效候選、mask、position 做三組比較：完整複製；僅容量共享；共享前綴 attention 分解及批次化。第三組應如 Hydragen 類方法，對 prefix/suffix 的局部最大值與 softmax normalization 正確合併，不可直接平均兩個 attention 輸出。

觀察四種不同收益：encoder 重算減少、KV 容量減少、外部讀取減少、attention kernel 形狀改善。固定候選數比較一次，再在固定記憶體／期限下比較能容納的候選數；後者是新增決策预算，不是原 kernel 加速率。

**失敗條件：**prefix 不具不變性；prefix 太短使拆分／歸約成本大於收益；只有容量變小而 latency 沒改善；跨控制周期誤用過時觀測。測試需刻意改變動作、位置與觀測，確認 cache invalidation 正確。

示例：64 條候選、512 個共享 prefix tokens、每候選 32 個私有 tokens、12 層、4 KV heads、head dimension 64、每元素 2 bytes：複製 KV 為 408 MiB，共享為 30 MiB。這是 **13.6 倍容量差**，不是 13.6 倍推論加速；32 tokens 也不代表 32 個控制步。公式與程式見[解析估算](results/transfer-estimates.md)。

## 5. 實驗三：量化先看 kernel，再看長期決策

從原生 BF16/FP16 基線開始，比較裝置實際支援的 W8A8／FP8，再比較 W4A16 或 W4A8。名稱清楚拆分權重、activation、KV 與 accumulator 的精度；如某段實際回退 FP16，應列入時間。

校準資料包含真實觀測、動作邊界、接觸／遮擋／快速運動，以及模型自身 rollout 產生的狀態，訓練／校準與測試情境分開。不要只校準 encoder，忽略 predictor 反覆產生的分布偏移。

分三層評估：

1. **算子：**dequant／reorder／scale 成本、實際 kernel throughput、bytes 與能耗。
2. **預測／規劃：**不同 H 的狀態誤差、reward/value 偏差、候選排序、top-k 重合及選中軌跡相對高精度參考的分數差。這個差值只是參考規劃器下的 regret proxy，不等於真實世界 regret。
3. **閉環：**同一情境及配對隨機 seed 下的成功率、任務回報、碰撞／約束違反、deadline misses，報告樣本數與不確定性。

同時做「固定 K/H/I/E」與「固定控制期限」兩種比較。後者可回答：省下時間後增加候選，能否補償量化誤差？先預定任務可接受的品質退化與 deadline 門檻，不能看到結果後才調門檻。

**失敗條件：**W4 解壓抵銷搬移節省；單步誤差小但長期發散；關鍵候選排序翻轉；平均速度提高而 P99 或閉環成功率變差。必要時只量化 encoder／部分線性層，保留敏感狀態、歸約、value head 的精度。

相關來源：GPTQ、SmoothQuant、AWQ、QuaRot、QServe、Marlin。文獻的語言任務品質並不是本專案閉環品質證據。

## 6. PIM、稀疏與 speculative 的後續門檻

**PIM：**先使用最佳可行 batching/shared-prefix 基線，再確認 memory-bound 的矩陣向量／attention 佔比。評估正常記憶體存取與 PIM 並行限制、資料轉置、跨引擎輸送、PIM 精度、thermal/power、bank 利用率及尾延遲。權重或狀態已片上常駐時，不把消失的外部流量算成 PIM 的收益。

**稀疏：**從硬體可利用的 structured sparsity 做起；把排序、索引、壓縮／解壓及負载不均加入成本。以小物件、少見接觸與長 horizon 作針對性測試，不能只量 FLOPs 或平均 token 保留率。

**粗模型篩選：**量測保留高精度最佳候選的 recall、被錯刪候選的代價、驗證成本及實際閉環結果。即使速度變快，也不稱為連續動態上的精確 speculative decoding。若用離散 action-token AR 模型，才另行驗證原始方法的機率校正與接受率。

## 7. 解析例子能說明什麼

令一個線性層 W∈R^(D×D)，M 個獨立列向量共同運算，FLOPs=2MD²。假設權重只讀一次、input/output 各搬一次，則：

`AI = 2MD² / (b_w D² + 2 b_a M D)` FLOP/byte。

這是指定記憶體邊界下的理想流量，不含 scale、packing、cache conflict、中間 tensor、通信或額外重讀。D=1024、BF16 時，M 從 1 增至 256，AI 從約 1 增至 170.7 FLOP/byte。W4A16 壓縮可增加理想 AI，但實際執行還有 dequant 成本；較高 AI 也不直接等於速度倍數。

即使某 kernel 快 4 倍，若它只佔原本控制時間 40%，理想 Amdahl 上限也只有 `1/(0.6+0.4/4)=1.43` 倍，尚未加入額外開銷。[附錄程式](analysis/transfer_estimates.py)只重算這些解析例子，不載入模型，也不是 benchmark。

## 8. 給 Claude 的交叉審查焦點

既有 CLAUDE 估算與本文件可以互補，但需避免下列外推：

| 待檢視項目 | GPT 的判斷／待確認事項 |
|---|---|
| CLAUDE 以 V-JEPA2-AC、候選與多輪 CEM 得到大量算力需求 | 那是指定 predictor／K/H/I 的結果；不能推成所有 latent world model 都一樣 compute-bound |
| SRAM 常駐與多次迭代 | 可以減少外部搬移；算術、時間依賴、SRAM 帶寬及控制開銷仍存在 |
| 由 FP4 sparse 峰值折算 BF16 的硬體 roofline | 僅是建模假設，須與實際支援、頻率及實測吞吐區分，不能當官方 dense BF16 規格 |
| VLM／LLM 量化經驗直接用於 world model | 可當實驗起點，但不能當閉環精度或固定倍數加速的證據 |
| VLA 短序列不需 paging | 對固定短序列可能成立；仍取決於併行數、KV 壽命與共享需求 |

請 Claude 以[協作請求](collaboration/claude-review-request.md)指定方式回覆，附審查 commit。以上是 GPT 的獨立審查意見，尚未得到 Claude 確認；沒有修改 CLAUDE 的原有結論或結果。
