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

## 第二輪：LLM / 擴散模型加速器論文與技巧（docs/07）

會議與年份依原始發表；只有 arXiv 編號者表示截至查閱時未確認正式會議版本。

**硬體**：A3 (HPCA'20)、SpAtten (HPCA'21)、ELSA (ISCA'21)、Sanger (MICRO'21)、DOTA (ASPLOS'22)、FACT (ISCA'23)、DFX (MICRO'22)、ANT (MICRO'22)、Mokey (ISCA'22)、OliVe (ISCA'23)、FIGNA (HPCA'24)、Tender (ISCA'24)、BitMoD (HPCA'25)、[LUT Tensor Core](https://arxiv.org/abs/2408.06003) (ISCA'25)、Samsung HBM-PIM (ISSCC'21/ISCA'21)、AttAcc (ASPLOS'24)、[NeuPIMs](https://cs.paperswithcode.com/paper/neupims-npu-pim-heterogeneous-acceleration) (ASPLOS'24)、SpecPIM (ASPLOS'24)、Duplex (MICRO'24)、Cambricon-LLM (MICRO'24)、Groq TSP (ISCA'20; [GroqChip brief](https://groq.com/wp-content/uploads/2022/10/GroqChip%E2%84%A2-Processor-Product-Brief-v1.5.pdf))、TPU v4 (ISCA'23)、[SambaNova SN40L](https://arxiv.org/abs/2405.07518)、[d-Matrix Corsair](https://chipsandcheese.com/p/d-matrix-corsair-256gb-of-lpddr-for)、Cerebras WSE-3、Cambricon-D (ISCA'24)、[EXION](https://castlab.kaist.ac.kr/?p=5168) (HPCA'25)、[Ditto](https://arxiv.org/abs/2501.11211)、[SD-Acc](https://arxiv.org/abs/2507.01309)。

**軟體 / 系統**：FlashAttention 1/2/3、FlashDecoding、FlashInfer、SageAttention 1/2 (ICLR'25/ICML'25)、Sliding Tile Attention (2025)、Sparse VideoGen (ICML'25)、PagedAttention/vLLM (SOSP'23)、SGLang RadixAttention (2024)、Orca (OSDI'22)、Sarathi-Serve (OSDI'24)、DistServe (OSDI'24)、Splitwise (ISCA'24)、StreamingLLM (ICLR'24)、H2O (NeurIPS'23)、SnapKV (2024)、KIVI (ICML'24)、MQA/GQA/MLA、Speculative decoding (ICML'23)、Medusa / EAGLE (ICML'24)、PD-VLA (2025)、LLM.int8、SmoothQuant (ICML'23)、GPTQ (ICLR'23)、AWQ (MLSys'24)、QuaRot (NeurIPS'24)、SpinQuant (2024)、OCP MX formats、SparseGPT (ICML'23)、Wanda (ICLR'24)、DejaVu (ICML'23)、PowerInfer (SOSP'24)、DeepCache (CVPR'24)、FORA、Δ-DiT、PAB、TeaCache (CVPR'25)、LCM、DMD2 (NeurIPS'24)、Self-Forcing (2025)、CausVid (CVPR'25)、xDiT/PipeFusion、[Mirage MPK](https://arxiv.org/abs/2512.22219)、Hazy Research megakernel (2025)、[VLA-Cache](https://arxiv.org/abs/2502.02175) (NeurIPS'25)、Real-Time Chunking (Physical Intelligence 2025)、FAST tokenizer (2025)。

## 第三輪：模擬工具與實驗程式

- [SCALE-Sim v3](https://github.com/scalesim-project/SCALE-Sim)（PyPI `scalesim` 3.0.0；repo commit 9f98c43, 2025-12-17）。本研究修補 3 處 numpy-2 相容性（`int(max(arr))` → `int(np.max(arr))`）。
- [TD-MPC2](https://github.com/nicklashansen/tdmpc2) commit e9f5932（2026-07-13）：`tdmpc2/config.yaml`、`common/__init__.py`、`common/world_model.py`、`tdmpc2.py`。
- openpi commit 215abfb：`src/openpi/models/{pi0.py, gemma.py, model.py}`（實驗套件的 shared-prefix sampler 由 `Pi0.sample_actions` 改寫）。
