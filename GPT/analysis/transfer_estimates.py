"""Reproducible analytical examples only; no model or hardware benchmark.

Run from repository root: python GPT/analysis/transfer_estimates.py
Writes GPT/results/transfer-estimates.md, independent of current directory.
"""

from pathlib import Path


def linear_ai(m: int, d: int, weight_bytes: float, activation_bytes: float) -> float:
    if min(m, d, weight_bytes, activation_bytes) <= 0:
        raise ValueError("All dimensions and element sizes must be positive")
    flops = 2 * m * d * d
    traffic = weight_bytes * d * d + 2 * activation_bytes * m * d
    return flops / traffic


def kv_capacity(candidates: int, prefix: int, suffix: int, layers: int,
                kv_heads: int, head_dim: int, element_bytes: int) -> tuple[int, int]:
    if min(candidates, layers, kv_heads, head_dim, element_bytes) <= 0:
        raise ValueError("Counts and element size must be positive")
    if min(prefix, suffix) < 0:
        raise ValueError("Token lengths must be nonnegative")
    per_token = 2 * layers * kv_heads * head_dim * element_bytes
    replicated = candidates * (prefix + suffix) * per_token
    shared = (prefix + candidates * suffix) * per_token
    return replicated, shared


def render() -> str:
    lines = [
        "# LLM 加速移植：解析估算（非 benchmark）", "",
        "由 `GPT/analysis/transfer_estimates.py` 產生；所有數值為假設工作負載。", "",
        "## 線性層的理想算術強度", "",
        "D=1024；FLOPs=2MD²；bytes=b_w D²+2b_a MD。權重讀一次，input/output 各一次。",
        "BF16：b_w=b_a=2；W4A16：b_w=0.5、b_a=2。後者忽略 scale、padding 與 dequant 成本。",
        "此表只有 FLOP/byte，沒有預測實際 speedup。", "",
        "| M | BF16 FLOP/byte | W4A16 FLOP/byte |",
        "|---:|---:|---:|",
    ]
    for m in (1, 8, 32, 128, 256):
        lines.append(f"| {m} | {linear_ai(m, 1024, 2, 2):.3f} | {linear_ai(m, 1024, .5, 2):.3f} |")
    replicated, shared = kv_capacity(64, 512, 32, 12, 4, 64, 2)
    mib = 2 ** 20
    lines.extend([
        "", "## 共同 prefix 的 KV 容量", "",
        "K=64、prefix=512 tokens、suffix=32 tokens/candidate、L=12、KV heads=4、head dim=64、2 bytes/element。",
        "每 token 的所有層 K+V = 2×L×heads×dim×bytes = 12,288 bytes。",
        "複製容量=K(P+S)×12,288；共享容量=(P+KS)×12,288。", "",
        f"- 複製：{replicated:,} bytes = {replicated / mib:.3f} MiB。",
        f"- 共享：{shared:,} bytes = {shared / mib:.3f} MiB。",
        f"- 容量比：{replicated / shared:.3f}×。",
        "", "不含頁表、對齊、activation、weights 或 workspace；前提為 prefix K/V 完全可重用。",
        "容量比不是速度比；suffix token 數不是控制步數。", "",
        "## Amdahl 理想上限", "",
        f"原本佔時間 40% 的部分加速 4 倍：1/(0.6+0.4/4) = {1 / (.6 + .4 / 4):.3f}×。",
        "額外調度／搬移／驗證開銷會再降低端到端收益。", "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[1] / "results" / "transfer-estimates.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(), encoding="utf-8", newline="\n")
    print(output)
