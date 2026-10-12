# 00 · 方法論與假設

## 資料來源

1. **架構參數**：直接讀取開源 repo 的程式碼與 config（clone 於 2026-10-12），不依賴二手報導。各 repo 版本見 [references.md](references.md)。
2. **Hugging Face config**：Qwen3-VL-2B 的層數/寬度以官方 `config.json` 確認。
3. **閉源或未完整公開的模型**（DreamZero 的片段大小、Matrix-Game 3.0 的片段組成）：以論文/README 已知數字為錨點，其餘標示 **[假設]**，只用來看量級。

## 分析模型（`analysis/perf_model.py`）

每個 workload 被拆成多個 **Pass**（一次 Transformer stack 的前向計算），每個 Pass 描述：

| 欄位 | 意義 |
|---|---|
| `n_q` | 本次計算的 query token 數（= GEMM 的 M） |
| `n_kv_self` | self-attention 看到的 token 數（含 cache） |
| `n_ctx` | cross-attention 記憶的 token 數 |
| `kv_cached` | 從 DRAM 讀的 cache token 數 |
| `repeat` | 重複次數（去噪步數、CFG、decode 長度） |
| `batch` | 共用權重的獨立序列數（ViT 多相機、CEM 候選軌跡） |
| `attn_frac` | 實際計算的注意力矩陣比例（causal ≈ 0.5） |

### FLOPs

- GEMM：`2 × B × n_q × (作用在 query token 上的權重數)`
- Self-attention：`4 × B × n_q × n_kv × (heads × head_dim) × layers × attn_frac`
- Cross-attention：`4 × B × n_q × n_ctx × (heads × head_dim) × cross_layers`；記憶端的 K/V 投影假設在迴圈外算一次（hoisted）
- 忽略 norm、softmax、RoPE、activation 函數（< 3% FLOPs）

### DRAM 流量

- **權重**：每個 Pass 讀一次全部權重（假設片上 SRAM 遠小於模型——GPU 與現有邊緣 SoC 的實際情況）。
- **KV / cross-attn 記憶**：每層每 Pass 讀一次；新產生的 K/V 寫一次。
- **Activation**：每層邊界寫一次、讀一次（假設層內已融合，例如 FlashAttention + fused MLP）。

### 延遲

`latency = Σ_pass max(FLOPs / peak, bytes / bandwidth)` —— **理想 roofline 下限**。實際系統通常是這個值的 1.5–4 倍（kernel 啟動、小 M 利用率低、記憶體效率 < 100%）。

## 硬體參數

| 硬體 | BF16 dense 峰值 | 記憶體頻寬 | Ridge (FLOP/B) | 來源 / 推導 |
|---|---|---|---|---|
| H100 SXM | 989 TFLOP/s | 3.35 TB/s | 295 | NVIDIA datasheet |
| GB200（每 GPU） | 2.5 PFLOP/s | 8.0 TB/s | 312 | NVIDIA datasheet |
| Jetson AGX Thor | ~259 TFLOP/s | 273 GB/s | ~949 | **[假設]** 官方 2070 TFLOPS 為 FP4 稀疏 → ÷2（稀疏）÷4（FP4→BF16） |
| Jetson AGX Orin 64GB | ~42.5 TFLOP/s | 204.8 GB/s | ~208 | **[假設]** GPU 170 INT8 稀疏 TOPS ÷ 4 |

精度敏感度分析中，FP8 / FP4 假設峰值各 ×2 / ×4、權重 bytes 各 ÷2 / ÷4。

## 已知限制

- **不模擬 kernel 層級效率**：小 M GEMM 在真實硬體上的利用率會更差，所以 VLA 迴圈的實際延遲比本模型更差，結論只會更強。
- **不含 VAE、文字編碼器（T5/CLIP）、前處理**：世界模型的 VAE decode 是可觀的卷積負載，此處只做定性討論。
- **不含 CPU-GPU 傳輸與主機開銷**。
- **單一請求**：除 LLM b=64 情境外，都是 batch = 1 的機器人推論。
- 部分 workload 形狀為假設，已在 `analysis/models.py` 中以 `# ASSUMPTION` 標註，修改後重跑 `python analysis/run_all.py` 即可。
