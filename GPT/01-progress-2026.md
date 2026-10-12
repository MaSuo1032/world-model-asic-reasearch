# 2026 World Model 進展

截至 2026-10-12，公開成果顯示 world model 已能支援有界的互動模擬、3D 場景生成及部分機器人規劃。尚不能由這些成果認定通用、精確、長時間可靠的物理模擬已被解決。

## 主要路線

| 路線 | 已公開進展 | 對本研究的意義 |
|---|---|---|
| Google Genie 3 | 2025 年公布；查閱時官方模型頁列 720p、20–24 FPS 互動環境，並提供 Project Genie 實驗體驗入口 | 動作影響生成環境，但展示畫面不保證精確物理或任意動作支援 |
| Waymo World Model | 2026-02 公布；基於 Genie 3，生成相機與 LiDAR 資料，接受駕駛輸入、語言及場景配置控制 | 世界模型進入特定產業的多感測器模擬；仍需任務驗證 |
| Meta V-JEPA 2.1 | 2026-03 發布；強化密集特徵的空間結構及時間一致性 | 以表徵預測為核心，避免把像素重建當成唯一目標 |
| NVIDIA Cosmos 3 | 2026-06 論文：整合理解與生成，涵蓋文字、影像、影片、聲音及動作 | 顯示語言／視覺理解與世界生成的架構融合 |
| World Labs Marble / Atlas | World API 於 2026-01 公布；Atlas 於 2026-09 公布，採 multimodal autoregressive diffusion Transformer，以空間上下文條件化輸出；當時為 early access | 空間重建、生成及模擬整合，但不同能力的精確性須分別評估 |
| Dreamer 4 | 2025-09 論文提出在學到的世界模型內訓練行為，展示離線資料下的 Minecraft 長程任務 | 世界模型運算可集中在訓練端，部署成本不能只由模型訓練流程推論 |

資料來源：[Genie](https://deepmind.google/models/genie/)、[Waymo](https://community.waymo.com/blog/2026/02/the-waymo-world-model-a-new-frontier-for-autonomous-driving-simulation/)、[V-JEPA 2.1](https://arxiv.org/abs/2603.14482)、[Cosmos 3](https://arxiv.org/abs/2606.02800)、[World API](https://www.worldlabs.ai/blog/announcing-the-world-api)、[Atlas](https://www.worldlabs.ai/blog/atlas)、[Dreamer 4](https://arxiv.org/abs/2509.24527)。

## 對決策最相關的研究

**V-JEPA 2-AC** 將動作條件式預測建立於視覺潛在表示上，用目標影像支援機器人規劃。這提供「先學表徵，再用互動資料適配動態」的研究範例；不意味只用無動作標記的網路影片就能可靠控制任意機器人。[論文](https://arxiv.org/abs/2506.09985)

**Hierarchical Planning with Latent World Models（2026-04）** 研究多時間尺度的潛在模型與分層規劃，在其任務中改善長程控制並降低部分規劃成本。對硬體的啟示是預測步距與分層策略也會改變工作量，不能只靠擴大固定 horizon。[論文](https://arxiv.org/abs/2604.03208)

**V-JEPA Policy（2026-09）** 在凍結的 V-JEPA 2.1 encoder 潛在空間上建立 world-action model，代表預測與動作學習的整合方向。這是新近研究，其 benchmark 結果不等於跨所有場景的部署驗證。[論文](https://arxiv.org/abs/2609.37250)

**The Planning Limits of Latent World Models（2026-09）** 在其模型、目標評分與任務設定中，觀察到短 rollout 對遠目標的動作排序受限，也測試用世界模型篩選 VLA 候選。這支持研究「候選評估」及「近程子目標」，但不是世界模型能力的普遍上限；評分方式、terminal value、資料與搜尋方法都可能改變結果。[論文](https://arxiv.org/abs/2609.39235)

## 仍需解決

- 部分可觀測：遮擋、速度、摩擦、負載等無法從單張影像完整取得。
- 多步誤差：微小狀態誤差可改變接觸與長程結果。
- 分布外動作：搜尋可能偏好模型不熟悉、預測過度樂觀的方案。
- 規劃指標：預測準確不必然等於候選排序或閉環成功。
- 模型與環境時間尺度：固定一步對不同速度及接觸事件未必合適。

Google 在 Genie 3 的原始發布說明也列出動作空間、多代理互動及持續時間等限制。該列表是發布時的自述，不應當成所有後續版本的固定規格。[官方說明](https://deepmind.google/blog/genie-3-a-new-frontier-for-world-models/)
