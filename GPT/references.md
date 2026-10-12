# 來源清單

查閱與整理日期：2026-10-12。以下來源於本次對話透過網路工具查閱。官方能力為作者／廠商自述；論文是否完成同儕審查需另查會議紀錄，新近 arXiv 結果尤其不應當成普遍定論。

| ID | 一手來源 | 本報告使用範圍 |
|---|---|---|
| R01 | [Genie 3 官方模型頁](https://deepmind.google/models/genie/) | 互動環境及查閱時公布的輸出規格 |
| R02 | [Genie 3 發布說明，2025-08-05](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/) | 發布時的能力與限制 |
| R03 | [Waymo World Model，2026-02-06](https://community.waymo.com/blog/2026/02/the-waymo-world-model-a-new-frontier-for-autonomous-driving-simulation/) | Genie 基礎、多感測器與可控模擬 |
| R04 | [V-JEPA 2，arXiv:2506.09985](https://arxiv.org/abs/2506.09985) | 自監督視覺表示、action-conditioned 後訓練與規劃 |
| R05 | [V-JEPA 2.1，arXiv:2603.14482](https://arxiv.org/abs/2603.14482) | 密集特徵、空間與時間一致性 |
| R06 | [Meta 官方 V-JEPA 程式庫](https://github.com/facebookresearch/vjepa2) | 模型家族與訓練流程說明；本次未執行其程式 |
| R07 | [Cosmos 3，arXiv:2606.02800](https://arxiv.org/abs/2606.02800) | 多模態理解與生成 |
| R08 | [Cosmos 3 技術報告 HTML v1](https://arxiv.org/html/2606.02800v1) | 推論快取、context parallelism、權重及 VAE 分塊 |
| R09 | [World API，2026-01-21](https://www.worldlabs.ai/blog/announcing-the-world-api) | Marble 世界生成 API |
| R10 | [Atlas，2026-09-01](https://www.worldlabs.ai/blog/atlas) | 空間上下文、autoregressive diffusion 架構、early access |
| R11 | [World Labs：Bigger and Better Worlds，2025-09-16](https://www.worldlabs.ai/blog/bigger-better-worlds) | 可持續瀏覽的 3D 資產與 Gaussian splats |
| R12 | [Training Agents Inside of Scalable World Models，arXiv:2509.24527](https://arxiv.org/abs/2509.24527) | Dreamer 4、在想像環境中訓練行為 |
| R13 | [Hierarchical Planning with Latent World Models，arXiv:2604.03208](https://arxiv.org/abs/2604.03208) | 多時間尺度規劃研究 |
| R14 | [V-JEPA Policy，arXiv:2609.37250](https://arxiv.org/abs/2609.37250) | 潛在 world-action model 研究方向 |
| R15 | [The Planning Limits of Latent World Models，arXiv:2609.39235](https://arxiv.org/abs/2609.39235) | 特定設定的規劃範圍及候選篩選結果 |
| R16 | [TD-MPC 官方程式庫](https://github.com/nicklashansen/tdmpc) | Task-oriented latent dynamics、MPC 與 terminal value |
| R17 | [NVIDIA：Mastering LLM Techniques—Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/) | Prefill/decode 及常見瓶頸區分 |
| R18 | [FlashAttention，arXiv:2205.14135](https://arxiv.org/abs/2205.14135) | Attention I/O、HBM/SRAM 重用與中間儲存 |

## 本報告新增的推導

候選數乘以 horizon 與搜尋輪數的簡化成本模型、示意 614.4 GFLOPs／decision、狀態 buffer 容量、ASIC 模組建議及對照實驗方案，均為本報告工程推導，不是上述來源報告的實測數字。

未引用 CLAUDE 的既有解析結果作為獨立實驗證據。若日後交叉比較，應固定 checkpoint、精度、候選／上下文設定、硬體規格與成本定義。
