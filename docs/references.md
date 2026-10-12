# References

## 原始碼（2026-10-12 shallow clone）

| Repo | 用到的檔案 | Commit 日期 |
|---|---|---|
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | `src/openpi/models/{pi0.py, pi0_config.py, pi0_fast.py, gemma.py, siglip.py, model.py}`, `training/config.py` | 2026-08-24 |
| [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | `gr00t/configs/model/gr00t_n1d7.py`, `gr00t/model/modules/{dit.py, qwen3_backbone.py}`, `gr00t/model/gr00t_n1d7/gr00t_n1d7.py` | 2026-10-07 |
| [openvla/openvla](https://github.com/openvla/openvla) | `prismatic/extern/hf/{modeling_prismatic.py, configuration_prismatic.py}`, `prismatic/models/backbones/vision/dinosiglip_vit.py` | 2025-03-23 |
| [huggingface/lerobot](https://github.com/huggingface/lerobot) | `src/lerobot/policies/smolvla/{configuration_smolvla.py, smolvlm_with_expert.py}` | 2026-10-08 |
| [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) | `src/models/vision_transformer.py`, `configs/train/vitg16/droid-256px-8f.yaml`, `notebooks/utils/{mpc_utils.py, world_model_wrapper.py}` | 2026-03-23 |
| [nvidia-cosmos/cosmos-predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) | `cosmos_predict2/_src/predict2/configs/text2world/defaults/net.py`, `cosmos_predict2/config.py`, `docs/inference_robot_action_cond.md`, `.../tokenizers/wan2pt1.py` | 2026-06-08 |
| [SkyworkAI/Matrix-Game](https://github.com/SkyworkAI/Matrix-Game) | `Matrix-Game-2/configs/...`, `Matrix-Game-3/{README.md, wan/configs/config.py, pipeline/*.py}` | 2026-09-29 |

## Model cards / configs

- Qwen3-VL-2B-Instruct `config.json`（Hugging Face）：text 28L/2048/16 heads/8 KV/FFN 6144；vision depth 24、patch 16、spatial merge 2。
- GR00T N1.7 README：backbone 為 Cosmos-Reason2-2B（Qwen3-VL 架構）。

## 論文與報導

- DreamZero: *World Action Models are Zero-shot Policies*, arXiv:2602.15922（Wan2.1-I2V-14B backbone、7 Hz、GB200 38× 加速）。
- *From World Models to World Action Models: A Concise Tutorial for Robotics*, arXiv:2607.00836。
- V-JEPA 2: *Self-Supervised Video Models Enable Understanding, Prediction and Planning*, arXiv:2506.09985。
- Physical Intelligence 部落格：π0.7（2026-04-16）、MEM（2026-03-03）、RL Token（2026-03-19）。
- Gemini Robotics 2（2026-07-30，閉源，未納入定量分析）。
- NVIDIA Cosmos 3（2026-05-31，GTC Taipei；權重開放，但本研究使用程式碼已公開且可查證的 Cosmos-Predict2.5）。
