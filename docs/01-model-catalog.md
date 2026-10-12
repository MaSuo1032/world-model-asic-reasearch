# 01 · 開源模型目錄與推論流程（從原始碼讀出）

本章逐一拆解 12 個模型（11 個由原始碼、DreamZero 由論文）的**推論時**架構。所有層數、寬度、步數都直接取自程式碼或 config（檔案路徑附在每節最後）。凡是部署參數（相機數、prompt 長度、片段大小）屬於假設的，會標示 **[假設]**。

分類：

| 類別 | 代表模型 | 一次推論的產出 |
|---|---|---|
| **VLA-flow**（VLM + flow-matching 動作專家） | π0、π0.5、SmolVLA、GR00T N1.7 | 一段 action chunk（40–50 步連續動作） |
| **VLA-AR**（自回歸動作 token） | OpenVLA、π0-FAST | 7 個或約 60 個離散動作 token |
| **WM-JEPA**（潛在空間預測 + MPC） | V-JEPA 2-AC | 一個經規劃的動作 |
| **WM-gen**（影片擴散世界模型） | Cosmos-Predict2.5、Matrix-Game 2.0 / 3.0 | 一段影片（12–77 幀） |
| **WAM**（世界-動作聯合模型） | DreamZero | 未來影片潛變量 + 一段動作 |
| **LLM 對照組** | Llama-3.1-8B | 一個文字 token |

---

## 1. π0 / π0.5（Physical Intelligence, openpi）

**架構：兩個 Transformer 共用一個注意力空間（mixture-of-transformers）**

| 元件 | 層數 | d_model | heads × head_dim | KV heads | FFN | 參數 |
|---|---|---|---|---|---|---|
| SigLIP So400m/14（224², 256 tok/圖） | 27 | 1152 | 16 × 72 | 16 | 4304 (GELU) | 411 M |
| PaliGemma Gemma-2B（VLM） | 18 | 2048 | 8 × 256 | **1 (MQA)** | 16384 (GeGLU) | 1.98 B（+0.53 B embedding） |
| Action expert（Gemma-300M） | 18 | 1024 | 8 × 256 | **1 (MQA)** | 4096 (GeGLU) | 311 M |

**推論流程**（`pi0.py::sample_actions`）：

1. 3 張相機影像各自過 SigLIP → 3 × 256 = 768 個影像 token。
2. 加上語言 token（π0：48；π0.5：200，且機器人狀態被離散化進文字 prompt），組成 **prefix**。
3. **prefix 只過一次 VLM**，產生 KV cache（`_, kv_cache = self.PaliGemma.llm([prefix_tokens, None], ...)`）。
4. 從高斯雜訊 `x_1` 開始，**以 Euler 法積分 10 步**（`num_steps=10`, `dt=-0.1`）：每一步只把 suffix（1 個狀態 token + 50 個動作 token，π0.5 則為 50 個）送進 action expert，suffix 的 query 會注意 prefix 的 KV cache 與 suffix 本身。
5. 輸出 50 × 32 維的動作 chunk。

**兩代差異**：π0.5 把 flow-matching 的時間步以 **adaRMSNorm** 注入（`use_adarms=[False, True]`），而非把時間 embedding 拼到 token 上；state 改成離散 token 放在 prefix。

**對硬體重要的細節**：
- VLM 與 expert 的 KV 都是 **1 個 head × 256 維**（MQA），所以 816 個 prefix token 的 KV cache 只有 **15 MB**（18 層 × 2 × 256 × 816 × 2 B）。
- expert 每一步的 GEMM 的 M 維只有 **51**——這是整個模型的效能關鍵（見 [04](04-compute-loading.md)）。

> 來源：`openpi/src/openpi/models/{pi0.py, pi0_config.py, gemma.py, siglip.py}`；`model.py: IMAGE_RESOLUTION=(224,224)`；commit 2026-08-24。

## 2. π0-FAST（自回歸版本）

同樣的 SigLIP + Gemma-2B，但動作先做 DCT 再 BPE 壓縮成離散 token（FAST tokenizer），由 VLM **逐 token 自回歸解碼**。每個解碼步都要讀完整的 Gemma-2B 權重 **加上 257,152 × 2048 的詞表輸出層**（約 0.53 B 參數、1 GB BF16）。

- 每段 chunk 的 token 數 **[假設] ≈ 60**（FAST 壓縮率依任務而定，openpi 中 `max_token_len=180` 為 prompt+動作總長上限）。
- 這讓 π0-FAST 的資料流與 LLM decode 幾乎相同：**每個 token 讀 ~5 GB 權重，算術強度 ≈ 1 FLOP/B**。

> 來源：`openpi/src/openpi/models/pi0_fast.py`、`training/config.py`。

## 3. OpenVLA 7B

| 元件 | 規格 |
|---|---|
| 雙視覺編碼器 | DINOv2 ViT-L/14（24L, 1024）+ SigLIP So400m/14，224²，**特徵在 channel 維拼接** → 256 token |
| LLM | Llama-2-7B（32L, 4096, **MHA 32 heads**, FFN 11008） |
| 動作輸出 | 7 個 token（每個自由度一個），256 bins 均勻量化，`max_new_tokens = action_dim` |

一次推論 = 1 次 prefill（約 280 token）+ 7 次 decode。因為是 MHA，KV 每 token 512 KB，雖然序列短但比 π0 大 10 倍。

> 來源：`openvla/prismatic/extern/hf/modeling_prismatic.py::predict_action`、`configuration_prismatic.py (n_action_bins=256)`、`backbones/vision/dinosiglip_vit.py`。

## 4. SmolVLA（Hugging Face LeRobot，0.45B）

| 元件 | 規格 |
|---|---|
| 視覺 | SigLIP-B/16 @512²（12L, 768）→ 1024 patch → **pixel-shuffle 壓到 64 token/圖** |
| VLM | SmolLM2-360M，**只用前 16/32 層**（`num_vlm_layers=16`） |
| Action expert | 寬度 = 0.75 × 960 = 720，16 層，FFN 2048；**每隔一層交替 cross-attn（看 VLM KV）與 self-attn（動作 token 之間）**（`self_attn_every_n_layers=2`） |
| 推論 | `chunk_size=50`、`num_steps=10`、`use_cache=True` |

設計重點是「**砍層 + 砍 token**」：VLM 只跑一半，影像 token 壓 16 倍，expert 也縮小。結果是整個模型能在消費級 GPU/CPU 上跑。

> 來源：`lerobot/src/lerobot/policies/smolvla/{configuration_smolvla.py, smolvlm_with_expert.py}`。

## 5. GR00T N1.7（NVIDIA，2026-10 GA）

| 元件 | 規格 |
|---|---|
| VLM | **Cosmos-Reason2-2B**（Qwen3-VL-2B 架構）：ViT 24L/1024，patch 16，2×2 merge → 256² 影像 = 64 token；LLM 28L/2048、16 heads、8 KV heads、FFN 6144 |
| 截斷 | `select_layer=12`：**LLM 只保留前 12 層**（`while len(layers) > select_layer: pop`） |
| Action head | **AlternateVL-DiT**：16 層、32 heads × 48 = 1536 寬、GELU FFN 6144、adaLN；**奇數層 self-attn，偶數層 cross-attn**，且 cross-attn 每兩次在「影像 token」與「文字 token」之間交替（`attend_text_every_n_blocks=2`） |
| 推論 | `num_inference_timesteps=4`（4 步 flow-matching）、`action_horizon=40`、`max_action_dim=132`、32 種 embodiment 各自的狀態/動作 MLP |

**與 π0 的關鍵差異**：GR00T 的 backbone **不保留 KV cache**，只把第 12 層的 hidden state（約 168 × 2048 × 2 B ≈ 0.7 MB）當作 DiT 的 cross-attn 記憶。DiT 只跑 4 步。

> 來源：`Isaac-GR00T/gr00t/configs/model/gr00t_n1d7.py`、`model/modules/{dit.py, qwen3_backbone.py}`、`model/gr00t_n1d7/gr00t_n1d7.py`；commit 2026-10-07。

## 6. V-JEPA 2-AC（Meta，潛在空間世界模型 + MPC）

| 元件 | 規格 |
|---|---|
| Encoder | ViT-g/16：40L、1408、22 heads、MLP 6144；256²、tubelet 2 → 每個時間步 256 token |
| Action-conditioned predictor | 24L、1024、16 heads；**frame-causal**；每幀 256 個影像 token + 動作/狀態 token |
| 規劃器 | **CEM（Cross-Entropy Method）**：`samples=400, cem_steps=10, rollout=2, topk=10` |

**這是唯一一個「不產生像素也不直接輸出動作」的模型**：它在潛在空間中想像 400 組候選動作序列的結果，和目標影像的表徵比 L1 距離，迭代 10 輪選最佳。一次決策要跑 **400 × 10 × 2 = 8,000 次 predictor rollout**。

計算特性因此和其他模型完全相反：batch=400，GEMM 的 M 維高達 20 萬，**純算力瓶頸**（AI ≈ 6,000 FLOP/B）。

> 來源：`vjepa2/src/models/vision_transformer.py (vit_giant_xformers)`、`configs/train/vitg16/droid-256px-8f.yaml`、`notebooks/utils/{mpc_utils.py, world_model_wrapper.py}`。

## 7. Cosmos-Predict2.5（NVIDIA，影片擴散世界基礎模型）

| 元件 | 2B 版 | 14B 版 |
|---|---|---|
| DiT | 28 blocks、2048、16 heads、MLP ×4 | 36 blocks、5120、40 heads |
| Patch | 1×2×2（時間×高×寬） | 同 |
| Tokenizer | Wan2.1 VAE：時間 4×、空間 8×8、16 channel | 同 |
| 取樣 | `num_steps=35`、CFG guidance 7（**每步跑 2 次**：有/無條件） | 同 |

Token 數算法：720p（704×1280）→ latent 88×160 → patch 2 → **3,520 token/latent 幀**；77 幀 → 20 latent 幀 → **70,400 token**。全雙向注意力，所以注意力是 70,400² 等級。

機器人動作條件版（`robot/action-cond`）以 `chunk_size=12` 逐段生成，[假設] 解析度 480×640 → 1,200 token/latent 幀、每段 4 latent 幀。

> 來源：`cosmos-predict2.5/cosmos_predict2/_src/predict2/configs/text2world/defaults/net.py`、`cosmos_predict2/config.py`、`docs/inference_robot_action_cond.md`。

## 8. Matrix-Game 2.0 / 3.0（Skywork，即時互動世界模型）

**2.0（1.3B，串流）**

| 項目 | 值 |
|---|---|
| 架構 | CausalWan-1.3B：30L、1536、12 heads、FFN 8960；鍵盤/滑鼠動作模組注入前 15 層 |
| 影像 | 352×640 → latent 44×80 → patch 2 → **880 token/latent 幀** |
| 生成單位 | 每個 block 3 個 latent 幀（=12 個影片幀） |
| 注意力 | **causal + local window 6 latent 幀**（`local_attn_size=6`），以 KV cache 實作 |
| 取樣 | 蒸餾後 **3 步**（`denoising_step_list: [1000, 666, 333]`），之後再用乾淨 latent 跑一次寫入 KV cache |

**3.0（5B，720p）**：README 宣稱 5B 模型以 INT8 + 蒸餾 VAE 達到 720p 40 FPS；採用 Wan2.2 VAE（`vae_stride=(4,16,16)`、48 channel）、3 步 DMD 取樣。長時記憶的做法是**依相機視野挑選過去的 latent 幀，重新注入為 token**（`select_memory_idx_fov`），而不是無限延長 KV cache。

> ⚠️ 程式碼不一致：`Matrix-Game-3/wan/configs/config.py` 寫的是 dim 5120 / 40 層（≈14B 規格），但權重由 checkpoint 自帶的 config 以 `WanModel.from_pretrained` 載入。本研究依 README 的 5B 敘述，採 Wan2.2-TI2V-5B 規格（3072 寬、30 層）。片段大小（4 新 + 4 記憶 latent 幀）為 **[假設]**。

> 來源：`Matrix-Game-2/configs/distilled_model/universal/config.json`、`configs/inference_yaml/inference_universal.yaml`、`Matrix-Game-3/README.md`、`pipeline/inference_interactive_pipeline.py`。

## 9. DreamZero（NVIDIA，World Action Model）

| 項目 | 值 |
|---|---|
| Backbone | Wan2.1-I2V-14B-480P（40L、5120、40 heads、FFN 13824） |
| 輸出 | **同時**去噪未來影片 latent 與機器人動作 |
| 閉環 | 每段動作執行後，把預測的未來幀換成真實觀測寫回 KV cache（消除累積誤差） |
| 速度 | 論文：GB200 上經系統與模型層優化共 **38×** 加速後達 **7 Hz** 閉環控制；DreamZero-Flash 可降到單步取樣 |

[假設]：每段 2 個 latent 幀（480×832 → 1,560 token/幀）+ 32 個動作 token、KV cache 保留前 6 個 latent 幀、4 步去噪。這些數字論文頁面沒有直接給，分析時僅用來看量級。

> 來源：arXiv 2602.15922（DreamZero: World Action Models are Zero-shot Policies）。

## 10. 對照組：Llama-3.1-8B

32L、4096、32 heads、**8 KV heads（GQA）**、FFN 14336、詞表 128,256。分析四種情境：decode（batch 1，context 8K / 128K）、decode（batch 64，8K）、prefill 2K。

---

## 參數量驗證

分析模型算出的參數量（不含 embedding 表）與公開數字比對，誤差都在 embedding 或位置編碼的量級內：

| Stack | 本模型計算 | 公開數字 |
|---|---|---|
| SigLIP So400m | 411 M | ~400 M |
| Gemma-2B（PaliGemma） | 1.98 B + 0.53 B emb | 2.5 B |
| π0 action expert | 311 M | 311 M（程式碼註解） |
| Llama-2-7B | 6.48 B + 0.26 B emb | 6.74 B |
| Llama-3.1-8B | 6.98 B + 2 × 0.53 B emb/head | 8.03 B |
| V-JEPA 2 ViT-g | 1.01 B | ~1 B |
| Wan 1.3B / 5B / 14B | 1.39 / 4.91 / 14.05 B | 1.3 / 5 / 14 B |
| Cosmos-Predict2.5-2B DiT | 1.88 B | 2 B |
