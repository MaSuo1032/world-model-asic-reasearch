# 實驗套件（E1–E3）

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MaSuo1032/world-model-asic-reasearch/blob/main/CLAUDE/experiments/colab_experiments.ipynb)

在 Colab 選 **GPU 執行環境**（π0 部分建議 L4 或 A100；T4 沒有 bf16 tensor core，π0 計時會失真），然後「全部執行」。結果會存到 `/content/results/<GPU>_<時間>/`，最後打包下載；若在 Colab secrets 設定 `GITHUB_TOKEN`，也可推到本 repo 的 `colab-results/*` 分支（不會動 `main`）。

| 腳本 | 實驗 | 需要 |
|---|---|---|
| `tdmpc2_plan_profile.py` | E1-latent（TD-MPC2 規劃延遲 vs 模型大小、候選數、CUDA graphs）、E3(ii)（Q-head 先抽樣） | 任何 GPU，或 CPU |
| `openpi_bestofk_profile.py` | E1-VLA（π0 / π0.5 共享 prefix 的 best-of-K）、E3(i)（相鄰 flow 步的相似度，需官方權重） | L4 / A100 |
| `colab_experiments.ipynb` | 以上全部 | Colab GPU |
| `make_colab_notebook.py` | 產生 notebook（改 notebook 請改這支） | – |

E1 用隨機初始化、真實尺寸的模型：延遲與權重數值無關。E3(i) 需要下載 `gs://openpi-assets/checkpoints/pi05_droid/params`。這些實驗**都不量測任務成功率**（E2 需要模擬器或真機）。

預測與失敗條件見 [`../reviews/pre2026-llm-acceleration-review.md`](../reviews/pre2026-llm-acceleration-review.md) Q4；背景見 [`../docs/09`](../docs/09-systolic-simulation-and-experiments.md)。

本機執行（無 GPU 也可跑 TD-MPC2）：

```bash
git clone https://github.com/nicklashansen/tdmpc2 /tmp/tdmpc2 && (cd /tmp/tdmpc2 && git checkout e9f5932)
pip install torch tensordict
python tdmpc2_plan_profile.py --repo /tmp/tdmpc2 --device cpu --sizes 1 5 --samples 64 512
```
