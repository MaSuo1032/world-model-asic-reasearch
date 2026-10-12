# 2026 年以前 LLM 加速器：哪些能用在決策型 World Model？

整理日期：2026-10-12。收錄首次公開早於 2026-01-01 的 28 項代表工作：26 篇論文、2 項技術文章／實作。這是針對可移植性的選讀，不是完整系統性文獻回顧。年份以首次公開為主，與會議年份可能不同；持續更新的程式庫只採用截至 2025 年已公開的機制。

## 結論

**最值得移植的是資料重用、融合、候選批次化，以及有對應高效 kernel 的量化。Transformer 型世界模型再加入 FlashAttention 與條件成立時的共同前綴共享。PIM、動態 KV 分頁與 speculative decoding 都不是通用答案。**

這是由文獻機制推導的研究優先順序，尚未完成 world model 推論、閉環控制或晶片測量。所有來源與證據類型見[文獻清單](references-pre2026.md)；實驗設計見[移植與驗證](07-transfer-and-experiments.md)。

## 1. 先對齊運算，而不是對齊模型名稱

| 工作負載 | 典型核心工作 | 優先觀察的瓶頸 | 與 LLM 加速的關係 |
|---|---|---|---|
| 單請求自回歸 LLM decode | 逐 token 線性層與歷史 attention | 權重／KV 搬移、小矩陣、序列依賴 | 常見 LLM decode 加速器的出發點 |
| 大 batch LLM／prefill | 大 GEMM 與大量 attention queries | 算力、attention IO、排程 | 更接近大量候選的密集 predictor |
| 緊湊 latent MLP／recurrent dynamics | 狀態、動作到下一狀態 | 權重重用、啟動延遲、時間步依賴 | 多數 KV 專用技巧不適用 |
| 多空間 token 的 latent Transformer | 多層 attention／MLP，跨候選 rollout | GEMM、activation、attention、候選數 | FA、融合、混合精度可能有效 |
| 擴散式 predictor | 多步去噪，各步包含密集計算 | 去噪步數、activation、算力 | 通用 kernel 可重用；LLM token 驗證不直接適用 |

世界模型是功能分類，不是固定微架構。線上 MPC 常有候選平行與時間步依賴；也有世界模型只在訓練時使用，部署時根本不 rollout。不能把「latent」當成小模型，也不能把整個類別當成大型 V-JEPA predictor。

候選數增加會改變硬體結論。對同一權重矩陣，把多個候選的線性運算合成 GEMM，權重載入成本可攤提；原本針對 GEMV 設計的 PIM 優势便可能下降。相反地，候選各自產生的歷史 KV 未必能共享。

## 2. 硬體架構：可移植的是機制，效能數字不能照搬

| 工作 | 公開年份／證據 | 值得借用的機制 | 移植限制與優先度 |
|---|---|---|---|
| [SpAtten](https://arxiv.org/abs/2012.09852) | 2020；HPCA 2021；RTL、週期／記憶體模擬及綜合估算 | token/head pruning、漸進精度、top-k datapath | 近似計算；低注意力不等於低決策風險。P2 |
| [Softermax](https://arxiv.org/abs/2103.09301) | 2021；HLS、RTL 模擬、綜合／功耗分析 | 低精度、硬體友善 softmax | 改變數值／運算近似；要驗證長 rollout。P2 |
| [DFX](https://arxiv.org/abs/2209.10797) | 2022；多 U280 FPGA 實作 | 可程式序列控制、算子映射、記憶體通道利用 | 借用控制與資料流，不複製 GPT-2 工作負載假設。P1 |
| [SpAtten-Chip](https://hanlab.mit.edu/projects/spatten-chip) | 2023；實體晶片量測 | 能耗可調、模型配置與 word elimination | 小型語言理解模型的晶片，並非大 LLM 或閉環 WM 驗證。P2 |
| [FlightLLM](https://arxiv.org/abs/2401.03868) | 2024；U280 FPGA 實作 | sparse DSP、混合精度、算子融合、長度感知編譯 | always-on-chip decode 指中間 activation，不代表全模型權重常駐。P1 |
| [NeuPIMs](https://arxiv.org/abs/2403.00579) | 2024；ONNXim／DRAMsim3 模擬 | GEMM 與 attention GEMV 分工、subbatch 交錯 | 要先確認低重用 GEMV 比例；多候選共享可改變瓶頸。P2 |
| [AttAcc](https://github.com/scale-snu/attacc_simulator) | 2024；xPU 模型與 Ramulator 模擬 | attention 移至 PIM、其餘運算留 xPU | 需要大而低重用的 KV 工作負載。P2 |
| [Cambricon-LLM](https://arxiv.org/abs/2409.15654) | 2024；SSDsim／NPU 週期模擬 | chiplet、NAND in-storage computing、容量分工 | 解決大型權重容量，對小型控制迴圈通常不是首要問題。P3 |
| [HPIM](https://arxiv.org/abs/2509.12993) | 2025；週期模擬及硬體單元評估 | SRAM-PIM／HBM-PIM 與 tensor 運算分工 | 文中 batch 1、FP16 的評估不可外推大量 MPC 候選。P2/P3 |

P0 是建立基線及優先執行；P1 是第一輪有條件實作；P2 是瓶頸與品質證據成立後才投入；P3 是目前低優先。此排序只對本研究的決策型 world model 成立。

特別區分：SpAtten 的模擬／綜合結果與 SpAtten-Chip 的量測是不同工作。PIM 模擬也不能寫成已量產晶片實測。FPGA 數字受資源、頻率、壓縮及基準版本影響，不可直接換算 ASIC 能耗優勢。

## 3. Kernel 與資料流：最容易先取得可測量成果

### Attention tiling 與融合

[FlashAttention 1](https://arxiv.org/abs/2205.14135) 避免將完整 attention score 矩陣寫回外部記憶體；[FA2](https://arxiv.org/abs/2307.08691) 改善工作分配；[FA3](https://arxiv.org/abs/2407.08608) 利用非同步及運算重疊。對密集 attention 的數學運算，這類重排可保持等價，但不保證逐 bit 一致，也不把 dense attention 的算術複雜度變成線性；FP8 路徑另外引入精度議題。

可移植到 ASIC 的是 SRAM tiling、online softmax、矩陣引擎與歸約單元重疊，不是原封不動搬入 Hopper 指令。沒有 attention 的 latent MLP 不會因 FlashAttention 受益。

[Flash-Decoding](https://pytorch.org/blog/flash-decoding/) 對少量 query、長 KV 分片運算再歸約；短序列或大量候選 query 可能已足夠平行，新增歸約反而有成本。[FlashInfer](https://arxiv.org/abs/2501.01005) 的可組合 attention 配置與排程，適合測試不同 predictor 形狀。固定形狀亦可評估 graph replay 與融合，減少每一步 dispatch，但必須量完整控制迴圈。

### 權重、觀測及共同前綴共享是三件不同的事

1. 同一步的多候選線性層共用權重，通常可以批次化。共享只減少搬移，沒有免除每個候選的乘加。
2. 相同觀測的 encoder 輸出可共用，但 action-conditioned predictor 的中間值可能不同。
3. 只有計算結果確實相同的前綴 KV 才可直接重用。位置、mask、模型權重、條件及精度均需一致；雙向 attention 或動作改變前綴表徵時，不能假設前綴不變。

[PagedAttention](https://arxiv.org/abs/2309.06180) 著重 KV 分頁、配置及共享容量；[SGLang](https://arxiv.org/abs/2312.07104) 的 RadixAttention 重用已算好的共同前綴。[Hydragen](https://arxiv.org/abs/2402.05099) 進一步把不同序列對同一前綴的 query 批次運算，增加 KV 重用，並正確合併前綴／私有後綴 softmax 結果。**省下複製容量，不必然省下每次 attention 的讀取；要分開測量。**

對固定、短小狀態的控制任務，預配置連續 scratchpad 可能比動態分頁更合適。對大量候選共用較長觀測前綴的 Transformer，Hydragen 類機制值得優先驗證。

### 排程與硬體分工

[Orca](https://www.usenix.org/conference/osdi22/presentation/yu) 的逐迭代排程可啟發候選 subbatch 排程；機器控制仍應以 deadline 為界，不能為湊滿 batch 無限等待。[DistServe](https://arxiv.org/abs/2401.09670) 的 prefill/decode 分工提示不同階段可用不同資源，但 encoder/rollout 只是類比；跨裝置搬移與同步可能抵銷收益。

在同一 rollout 內，z(t+1) 通常依賴 z(t) 的完整 predictor 輸出。不能為了讓某一層權重常駐，就先執行所有未來時間步的同一層。可合法利用的是同時間步候選批次、層內 tiling、或容量允許時全 predictor 常駐。

## 4. 精度與稀疏：壓縮率、加速率、決策品質各自驗證

| 方法 | 可用之處 | 不能直接假設的事 |
|---|---|---|
| [GPTQ](https://arxiv.org/abs/2210.17323)、[AWQ](https://arxiv.org/abs/2306.00978) | 權重量化；以實際觀測、動作、rollout 分布校準 | 文字校準集足夠；W4 代表所有乘法皆為 INT4 |
| [SmoothQuant](https://arxiv.org/abs/2211.10438) | 以尺度轉換緩解 activation outlier，支援 W8A8 | 等價縮放不表示後續量化沒有誤差 |
| [QuaRot](https://arxiv.org/abs/2404.00456) | 合法的特徵旋轉可改善低位元量化 | 任意架構／物理狀態座標可不改後續運算直接旋轉 |
| [QServe](https://arxiv.org/abs/2405.04532) | W4A8KV4 與配套 kernel；量化與資料排列共同設計 | 低位元儲存自然就帶來高吞吐 |
| [Marlin](https://github.com/IST-DASLab/marlin) | W4A16 的 unpack/dequant/GEMM 融合 | 所有 batch、尺寸、裝置都有固定倍數加速 |
| [GQA](https://arxiv.org/abs/2305.13245) | 減少 KV heads／容量，適合模型設計或 uptraining | 對任意預訓練 MHA 切一個 runtime 開關就等價 |
| [SparseGPT](https://arxiv.org/abs/2301.00774)、SpAtten | 結構化權重／token／head 減量 | 零值自動變快，或被刪除的小目標與接觸事件不影響決策 |

量化先做有成熟 kernel 的 W8A8 或 FP8 基線，再視硬體支援評估 W4A16、W4A8；並非預言 8-bit 一定最佳。大量候選使權重已充分重用時，weight-only 壓縮的效益可能下降。誤差應看多步狀態、候選排序、價值估計及閉環成功率，不能只看單步 MSE。

## 5. Speculative decoding 的邊界

[原始 speculative decoding](https://arxiv.org/abs/2211.17192) 用便宜模型草擬 token，再由目標模型平行驗證；精確保持目標抽樣分布依賴其接受與校正規則。這不是「只要先猜幾步，任何模型都可以精確加速」的定理。

若模型是符合條件的自回歸離散 action-token 模型，可評估原方法。連續 latent dynamics、確定性轉移、擴散去噪沒有直接相同的 token 機率接受規則。用小模型先篩候選、大模型再評分可以研究，但那是近似規劃，必須測量錯刪最佳候選與閉環損失；不能標成精確 speculative decoding。

## 6. 對 ASIC 方向的含義

第一版建議研究「可批次矩陣引擎＋可配置 SRAM／DMA＋融合歸約與非線性＋低延遲序列控制」，保留足夠精度及形狀彈性。Transformer 工作負載再加入 attention streaming 與共享前綴支援。這是工程推論，尚不是已綜合的微架構。

只有 trace 顯示外部記憶體搬移主導、且共享／批次化後依然如此，才值得投入 PIM。只有近似誤差與閉環品質可接受，才值得把 pruning、極低精度或粗模型篩選固定進硬體。最終優化目標是給定能耗與期限內的決策品質，而不是單看 tokens/s 或峰值 TOPS。
