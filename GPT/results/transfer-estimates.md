# LLM 加速移植：解析估算（非 benchmark）

由 `GPT/analysis/transfer_estimates.py` 產生；所有數值為假設工作負載。

## 線性層的理想算術強度

D=1024；FLOPs=2MD²；bytes=b_w D²+2b_a MD。權重讀一次，input/output 各一次。
BF16：b_w=b_a=2；W4A16：b_w=0.5、b_a=2。後者忽略 scale、padding 與 dequant 成本。
此表只有 FLOP/byte，沒有預測實際 speedup。

| M | BF16 FLOP/byte | W4A16 FLOP/byte |
|---:|---:|---:|
| 1 | 0.998 | 3.969 |
| 8 | 7.877 | 30.118 |
| 32 | 30.118 | 102.400 |
| 128 | 102.400 | 256.000 |
| 256 | 170.667 | 341.333 |

## 共同 prefix 的 KV 容量

K=64、prefix=512 tokens、suffix=32 tokens/candidate、L=12、KV heads=4、head dim=64、2 bytes/element。
每 token 的所有層 K+V = 2×L×heads×dim×bytes = 12,288 bytes。
複製容量=K(P+S)×12,288；共享容量=(P+KS)×12,288。

- 複製：427,819,008 bytes = 408.000 MiB。
- 共享：31,457,280 bytes = 30.000 MiB。
- 容量比：13.600×。

不含頁表、對齊、activation、weights 或 workspace；前提為 prefix K/V 完全可重用。
容量比不是速度比；suffix token 數不是控制步數。

## Amdahl 理想上限

原本佔時間 40% 的部分加速 4 倍：1/(0.6+0.4/4) = 1.429×。
額外調度／搬移／驗證開銷會再降低端到端收益。
