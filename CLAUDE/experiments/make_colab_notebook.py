"""Generates colab_experiments.ipynb (kept as code so the notebook stays reviewable in diffs).

    python make_colab_notebook.py
"""
import json
import os

REPO = "https://github.com/MaSuo1032/world-model-asic-reasearch"
TDMPC2_COMMIT = "e9f5932"
OPENPI_COMMIT = "215abfb"

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": src.strip("\n").splitlines(keepends=True)})


md(f"""
# World-model / VLA accelerator research — Colab experiments (CLAUDE round 3)

Runs the experiments proposed in `CLAUDE/reviews/pre2026-llm-acceleration-review.md` on a Colab GPU.

| Exp | What it measures | Needs |
|---|---|---|
| **E1-latent** | TD-MPC2 MPPI planner latency vs model size and candidate count K; op counts; `torch.compile` (CUDA graphs) effect | any GPU |
| **E3(ii)** | Q-ensemble head subsampling: speedup and bit-exact equality | any GPU |
| **E1-VLA** | openpi π0 / π0.5 best-of-K with the **prefix computed once** (KV replicated per candidate, not physically shared): does K=16 cost ≤ 30% more than K=1? | **L4 or A100 recommended** (T4 has no bf16 tensor cores) |
| **E3(i)** | cosine similarity of the action expert's outputs between consecutive flow steps (π0.5-DROID weights) | ≥ 24 GB GPU memory (A100/L4) |

**How to run:** `Runtime → Change runtime type → GPU`, then `Runtime → Run all`. Results land in `/content/results/` and are zipped for download at the end (optionally pushed to a branch of the repo).

Weights for E1 are random (latency does not depend on weight values); E3(i) downloads the official checkpoint. Nothing here measures task success.
""")

code("""
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
import subprocess, datetime, os, json
GPU = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().replace(" ", "_")
STAMP = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M")
OUT = f"/content/results/{GPU}_{STAMP}"
os.makedirs(OUT, exist_ok=True)
print("results ->", OUT)
""")

code(f"""
%cd /content
!git clone -q {REPO} research
!git clone -q https://github.com/nicklashansen/tdmpc2 tdmpc2 && cd tdmpc2 && git checkout -q {TDMPC2_COMMIT}
!git clone -q https://github.com/Physical-Intelligence/openpi openpi && cd openpi && git checkout -q {OPENPI_COMMIT}
!ls research/CLAUDE/experiments
""")

md("## E1-latent + E3(ii): TD-MPC2 planner (PyTorch)")

code("""
!pip -q install tensordict
!cd /content/research/CLAUDE/experiments && python tdmpc2_plan_profile.py --repo /content/tdmpc2 --device cuda \\
    --sizes 1 5 19 48 317 --samples 64 256 512 1024 --reps 10 --warmup 3 --out {OUT}/tdmpc2_eager.csv
""".replace("{OUT}", "$OUT"))

code("""
# Same, with torch.compile(mode="reduce-overhead") = CUDA graphs: removes per-kernel launch overhead
!cd /content/research/CLAUDE/experiments && python tdmpc2_plan_profile.py --repo /content/tdmpc2 --device cuda \\
    --sizes 1 5 48 317 --samples 512 --reps 10 --warmup 5 --compile --out $OUT/tdmpc2_compiled.csv
""")

md("""
## E1-VLA: openpi π0 / π0.5 best-of-K, prefix computed once + replicated KV (JAX)

Installs the minimal JAX stack openpi's model code needs (not the full training stack). First run takes a few minutes for XLA compilation per (steps, K) shape.
""")

code("""
!pip -q install "jax[cuda12]==0.5.3" "flax==0.10.2" einops "jaxtyping==0.2.36" "beartype==0.19.0" augmax dm-tree \\
    "ml_collections==1.0.0" sentencepiece safetensors filelock fsspec gcsfs "orbax-checkpoint==0.11.13" numpydantic \\
    tqdm-loggable "transformers==4.53.2"
!python -c "import jax; print(jax.__version__, jax.devices())"
""")

code("""
%env XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
# pi0 (base shapes: Gemma-2B VLM + Gemma-300M expert, 50-step chunk)
!cd /content/research/CLAUDE/experiments && python openpi_bestofk_profile.py --openpi /content/openpi \\
    --steps 1 5 10 --K 1 4 16 64 --reps 10 --warmup 2 --out $OUT/openpi_pi0_bestofk.csv
""")

code("""
# pi0.5 shapes (adaRMS time conditioning, 200-token prompt)
!cd /content/research/CLAUDE/experiments && python openpi_bestofk_profile.py --openpi /content/openpi --pi05 \\
    --steps 10 --K 1 4 16 64 --reps 10 --warmup 2 --out $OUT/openpi_pi05_bestofk.csv
""")

md("""
## E3(i): step-to-step similarity with real π0.5-DROID weights

Downloads ~ several GB from the public `gs://openpi-assets` bucket. Skip this cell on GPUs with < 24 GB memory.
""")

code("""
import gcsfs, os
fs = gcsfs.GCSFileSystem(token="anon")
dst = "/content/pi05_droid_params"
if not os.path.exists(dst):
    fs.get("openpi-assets/checkpoints/pi05_droid/params", dst, recursive=True)
!du -sh /content/pi05_droid_params
!cd /content/research/CLAUDE/experiments && python openpi_bestofk_profile.py --openpi /content/openpi --pi05 --horizon 15 \\
    --params /content/pi05_droid_params --steps 10 --K 1 4 16 --reps 5 --warmup 1 --similarity \\
    --out $OUT/openpi_pi05_droid_real.csv
""")

md("## Package results")

code("""
import shutil, glob
print("\\n".join(sorted(glob.glob(OUT + "/*"))))
zip_path = shutil.make_archive(OUT, "zip", OUT)
try:
    from google.colab import files
    files.download(zip_path)
except Exception as e:
    print("download manually:", zip_path, e)
""")

md("""
### (Optional) push results to a branch so Claude / GPT can analyse them

Add a GitHub token with `contents:write` on this repo as a Colab secret named `GITHUB_TOKEN` (key icon in the left sidebar), then run the cell. Results go to branch `colab-results/<gpu>-<time>`, never to `main`.
""")

code("""
from google.colab import userdata
token = userdata.get("GITHUB_TOKEN")
branch = f"colab-results/{GPU}-{STAMP}"
dest = f"/content/research/CLAUDE/experiments/results_colab/{os.path.basename(OUT)}"
shutil.copytree(OUT, dest, dirs_exist_ok=True)
!cd /content/research && git checkout -q -b $branch && git add CLAUDE/experiments/results_colab && \\
    git -c user.name="colab-runner" -c user.email="colab@users.noreply.github.com" commit -q -m "Colab results $branch" && \\
    git push -q https://x-access-token:{token}@github.com/MaSuo1032/world-model-asic-reasearch.git $branch && echo "pushed $branch"
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "L4"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colab_experiments.ipynb")
json.dump(nb, open(path, "w"), indent=1, ensure_ascii=False)
print("wrote", path)
