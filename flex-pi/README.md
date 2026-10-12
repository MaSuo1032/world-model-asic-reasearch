# Flex-π 研究區

研究對象：[Flex-π: A Multi-Stream World-Action Model with Compute Flexibility](https://arxiv.org/abs/2608.10860)（6B 參數 World Action Model，fork：[MaSuo1032/flex-pi](https://github.com/MaSuo1032/flex-pi)）。

依 repo 慣例，各協作者的研究放在自己的子資料夾，互相審查請標示 commit 與範圍。

| 子資料夾 | 內容 | 狀態 |
|---|---|---|
| [CLAUDE](CLAUDE/README.md) | 以官方程式碼精確計數 Flex-π 的參數、FLOPs、權重載入，逐題回答：attention 以外的計算、參數量、prefill / decode、權重載入、batch、演算法流程與 LLM 差異 | 2026-10-12 完成第一版 |
| GPT | （待 GPT 加入） | – |

重現 CLAUDE 的計數（不需要 GPU 或權重）：

```bash
git clone https://github.com/MaSuo1032/flex-pi /tmp/flex-pi
pip install torch timm einops rich termcolor omegaconf pyyaml matplotlib
python flex-pi/CLAUDE/analysis/flexpi_flops.py --flexpi /tmp/flex-pi
python flex-pi/CLAUDE/analysis/flexpi_regimes.py
```
