# 示意成本：可逐步核算，非模型 benchmark

本文件的所有數值均為設定參數後的解析計算；未量測真實模型或 GPU。FLOPs 與記憶體慣例見 [方法](00-methodology.md)。

## 1. 線上規劃運算量

假設每秒決策 f=20 次，K=256 條候選，每條 H=15 步，I=4 輪搜尋，U=1 個結果樣本。每次狀態轉移加評分成本固定為 40,000,000 FLOPs。

    transitions_per_decision = I*K*H*U
                             = 4*256*15*1
                             = 15,360

    FLOPs_per_decision = 15,360*40,000,000
                       = 614,400,000,000
                       = 614.4 GFLOPs

    sustained_compute = 20*614.4
                      = 12,288 GFLOPs/s
                      = 12.288 TFLOPs/s

每週期為 1/20 秒，即 50 ms。這是整個系統週期，Encoder、感測與其他工作也需要時間。12.288 TFLOPs/s 是僅此 rollout 假設下的持續有效運算需求，不是晶片峰值規格。

未計入 Encoder、policy proposal、搜尋更新、傳輸、同步及控制。40 MFLOPs 是示意輸入，不對應某個已查核的參數量。

## 2. 潛在狀態容量

假設 K=256、D=1024、每元素 b=2 bytes。一個時間切片需要 K*S*D*b bytes。

| 每候選 state tokens S | 一個切片 bytes | 一個切片 MiB | 兩個 buffer MiB |
|---|---:|---:|---:|
| 1 | 524,288 | 0.5 | 1 |
| 128 | 67,108,864 | 64 | 128 |

這只計候選狀態，不含權重、內部 activation、KV、動作序列、分數與 metadata。兩個 buffer 僅適用於可逐步更新且不需保留完整歷史的情境。

若每個轉移時間片都把全部狀態讀寫一次至外部記憶體，未做融合／壓縮時，粗略流量為：

    bytes_per_decision ≈ 2*I*H*K*S*D*b

此式是刻意簡化的讀一次、寫一次模型；最初共享狀態、最後步是否需寫回、cache 重用與分塊都會改變結果，不能當成完整 DRAM profiling。

## 3. 影片 token 數：僅作另一種工作負載對照

假設 5 秒、24 FPS、1280x720，時間壓縮 4 倍，空間各壓縮 8 倍，再使用 2x2 latent patch：

    T = 5*24 = 120
    N = (120/4)*(720/(8*2))*(1280/(8*2))
      = 30*45*80
      = 108,000 tokens

寬與高各加倍時，N 增為四倍；若使用完整 attention，其配對量約增為十六倍。這不是聲稱任何列出的模型使用這組壓縮配置。

影片生成多步採樣的概念成本：

    C_video ≈ sampling_steps*C_backbone(N) + C_encode_decode

決策型 latent world model 可以不解碼像素，因此不能把這個影片例子用作 latent MPC 的需求。

## 4. 如何替換成實際規格

先從特定 checkpoint 與推論路徑取得真實張量形狀、算子 FLOPs、步數、候選數及狀態歷史長度，再用 profiler 取得記憶體流量及延遲。

用相同精度與品質條件比較。若採 early stopping、pruning、候選共享或分層規劃，應用實際執行數替代最大設定值，並同時報告最壞情況。
