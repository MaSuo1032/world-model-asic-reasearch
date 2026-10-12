# 2026 年以前的加速研究來源

整理：2026-10-12。28 項選讀＝26 篇論文＋2 項技術文章／實作。年份為首次公開年份，必要時另註會議年份。所有連結為原始論文、作者／機構網站或作者程式庫。沒有將 2026 年新增功能計入。

## 硬體／架構（9 篇論文）

| ID | 年份與工作 | 一手來源 | 本次採用的證據範圍 |
|---|---|---|---|
| H01 | 2020 / HPCA 2021 — SpAtten: Efficient Sparse Attention Architecture with Cascade Token and Head Pruning | [論文](https://arxiv.org/abs/2012.09852) | §V-A：SpinalHDL/RTL、Verilator、Ramulator、綜合與能耗模型；不是 silicon measured speedup |
| H02 | 2021 — Softermax: Hardware/Software Co-Design of an Efficient Softmax for Transformers | [論文](https://arxiv.org/abs/2103.09301) | §V–VI：HLS、VCS、邏輯綜合與功耗分析；數值／模型評估 |
| H03 | 2022 — DFX: A Low-latency Multi-FPGA Appliance for Accelerating Transformer-based Text Generation | [論文](https://arxiv.org/abs/2209.10797) | 多 Alveo U280 FPGA、GPT-2 推論評估 |
| H04 | 2023 — A Fully-Integrated Energy-Scalable Transformer Accelerator Supporting Adaptive Model Configuration and Word Elimination for Language Understanding on Edge Devices | [作者 SpAtten-Chip 專頁](https://hanlab.mit.edu/projects/spatten-chip)；[DOI](https://doi.org/10.1109/ISLPED58423.2023.10244459) | ISLPED；實體晶片能耗與效能量測；小型語言理解網路 |
| H05 | 2024 — FlightLLM: Efficient Large Language Model Inference with a Complete Mapping Flow on FPGAs | [論文](https://arxiv.org/abs/2401.03868) | FPGA 實作；§4 on-chip 中間 activation 管理，不等於所有權重 on-chip |
| H06 | 2024 — NeuPIMs: NPU-PIM Heterogeneous Acceleration for Batched LLM Inferencing | [論文](https://arxiv.org/abs/2403.00579) | ONNXim 與修改 DRAMsim3 的模擬；GEMM/GEMV 分工及 subbatch interleaving |
| H07 | 2024 — AttAcc! Unleashing the Power of PIM for Batched Transformer-based Generative Model Inference | [作者論文](https://scale.snu.ac.kr/papers/2024-04-Conference-ASPLOS-AttAcc.pdf)；[作者模擬器](https://github.com/scale-snu/attacc_simulator) | ASPLOS；本次 PDF 全文擷取失敗，模擬方法由作者 artifact 確認；不引述未核對的速度倍數 |
| H08 | 2024 — Cambricon-LLM: A Chiplet-Based Hybrid Architecture for On-Device Inference of 70B LLM | [論文](https://arxiv.org/abs/2409.15654) | NPU/NAND 架構；SSDsim 與 NPU 週期模擬 |
| H09 | 2025 — HPIM: Heterogeneous Processing-In-Memory-based Accelerator for Large Language Models Inference | [論文](https://arxiv.org/abs/2509.12993) | 模擬與硬體單元評估；OPT、batch 1、FP16 的設定，非 WM 量測 |

## Kernel（4 篇論文＋2 項非論文）

| ID | 年份與工作 | 一手來源 | 本次採用的證據範圍 |
|---|---|---|---|
| K01 | 2022 — FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | [論文](https://arxiv.org/abs/2205.14135) | IO-aware tiling；避免完整 score 矩陣 materialization |
| K02 | 2023 — FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | [論文](https://arxiv.org/abs/2307.08691) | 工作分配與 GPU 利用率 |
| K03 | 2024 — FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision | [論文](https://arxiv.org/abs/2407.08608) | Hopper 非同步／重疊，FP8 路徑另有數值考量 |
| K04 | 2023 — Flash-Decoding for long-context inference | [作者技術文章](https://pytorch.org/blog/flash-decoding/) | 非獨立學術論文；2023-10-13；沿 KV 分片及歸約 |
| K05 | 2025 — FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving | [論文](https://arxiv.org/abs/2501.01005) | 2025-01-02 首次公開；以當時論文界定功能，不採後續 repository 新功能 |
| K06 | 2024 — Marlin | [作者實作](https://github.com/IST-DASLab/marlin) | 非獨立學術論文；FP16 activation／INT4 weight kernel 與作者基準；不視為所有尺寸的保證 |

## 推論服務與共享（5 篇論文）

| ID | 年份與工作 | 一手來源 | 本次採用的證據範圍 |
|---|---|---|---|
| S01 | 2023 — Efficient Memory Management for Large Language Model Serving with PagedAttention | [論文](https://arxiv.org/abs/2309.06180) | vLLM、KV 分頁與共享配置 |
| S02 | 2023 / 2024 — SGLang: Efficient Execution of Structured Language Model Programs | [論文](https://arxiv.org/abs/2312.07104) | RadixAttention、共享 prefix 與結構化執行 |
| S03 | 2024 — Hydragen: High-Throughput LLM Inference with Shared Prefixes | [論文](https://arxiv.org/abs/2402.05099) | 共享前綴 attention 分解、跨 query 批次及正確合併 |
| S04 | 2022 — Orca: A Distributed Serving System for Transformer-Based Generative Models | [USENIX 論文頁](https://www.usenix.org/conference/osdi22/presentation/yu) | 逐迭代排程與 selective batching；非機器人 deadline 保證 |
| S05 | 2024 — DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving | [論文](https://arxiv.org/abs/2401.09670) | prefill/decode 分離與 SLO goodput；WM encoder/rollout 是工程類比 |

## 量化（5 篇論文）

| ID | 年份與工作 | 一手來源 | 本次採用的證據範圍 |
|---|---|---|---|
| Q01 | 2022 / ICLR 2023 — GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers | [論文](https://arxiv.org/abs/2210.17323) | 近似二階權重 PTQ |
| Q02 | 2022 / ICML 2023 — SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models | [論文](https://arxiv.org/abs/2211.10438) | outlier 尺度轉移、W8A8 |
| Q03 | 2023 / MLSys 2024 — AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration | [論文](https://arxiv.org/abs/2306.00978) | activation-aware channel scaling、weight-only 量化 |
| Q04 | 2024 — QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs | [論文](https://arxiv.org/abs/2404.00456) | 可保持全精度運算等價的旋轉與低位元量化；等價性有架構前提 |
| Q05 | 2024 — QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM Serving | [論文](https://arxiv.org/abs/2405.04532) | 量化、資料排列及配套 kernel；不把儲存壓縮當端到端加速 |

## 架構與近似方法（3 篇論文）

| ID | 年份與工作 | 一手來源 | 本次採用的證據範圍 |
|---|---|---|---|
| A01 | 2023 — GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints | [論文](https://arxiv.org/abs/2305.13245) | KV heads 分組與 uptraining；非任意模型的等價 runtime 替換 |
| A02 | 2023 — SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot | [論文](https://arxiv.org/abs/2301.00774) | 權重重建式剪枝；結構化與非結構化稀疏的硬體差異 |
| A03 | 2022 / ICML 2023 — Fast Inference from Transformers via Speculative Decoding | [論文](https://arxiv.org/abs/2211.17192) | 自回歸 token 草擬／驗證與抽樣分布保持條件 |

## 解讀規則

上述工作證明其各自實驗設定下的結果，未證明本專案 world model 的加速比、能耗或決策品質。本研究沒有把不同硬體、不同精度、不同 batch 的 speedup 相乘。技術可移植性與優先度為 GPT 的工程判斷，等待獨立實驗及 Claude 審查。
