"""
Model + workload definitions.

Every architecture number below was read from the open-source code / config
(commit dates in docs/references.md). Workload shapes (camera count, prompt
length, segment sizes) are representative deployment settings; anything that
is an assumption rather than a value from code is marked  # ASSUMPTION.
"""
from perf_model import TransformerStack as TS, Pass, Workload, Hardware

# =============================================================================
# Vision encoders  (ViT: MHA, non-gated GELU MLP)
# =============================================================================
SIGLIP_SO400M = TS("SigLIP-So400m/14", layers=27, d_model=1152, n_heads=16, head_dim=72,
                   n_kv_heads=16, d_ff=4304, gated_ffn=False)        # openpi siglip.py
DINOV2_L = TS("DINOv2-L/14", 24, 1024, 16, 64, 16, 4096, gated_ffn=False)  # openvla dinosiglip
SIGLIP_B16 = TS("SigLIP-B/16-512", 12, 768, 12, 64, 12, 3072, gated_ffn=False)  # SmolVLM2-500M
QWEN3VL_VIT = TS("Qwen3-VL-2B ViT", 24, 1024, 16, 64, 16, 4096, gated_ffn=False)  # Cosmos-Reason2-2B
VJEPA2_G = TS("V-JEPA2 ViT-g/16", 40, 1408, 22, 64, 22, 6144, gated_ffn=False)  # vision_transformer.py
VJEPA2_AC = TS("V-JEPA2-AC predictor", 24, 1024, 16, 64, 16, 4096, gated_ffn=False)  # droid-256px-8f.yaml

# =============================================================================
# Language / reasoning backbones (decoder-only, gated FFN, GQA/MQA)
# =============================================================================
GEMMA_2B = TS("PaliGemma Gemma-2B", 18, 2048, 8, 256, 1, 16384)       # openpi gemma.py
GEMMA_2B_DEC = TS("Gemma-2B (+LM head)", 18, 2048, 8, 256, 1, 16384,
                  extra_params=257_152 * 2048, weight_group="PaliGemma Gemma-2B")                           # vocab head read per token
GEMMA_300M = TS("pi0 action expert (Gemma-300M)", 18, 1024, 8, 256, 1, 4096)
LLAMA2_7B = TS("Llama-2-7B", 32, 4096, 32, 128, 32, 11008)
LLAMA2_7B_DEC = TS("Llama-2-7B (+LM head)", 32, 4096, 32, 128, 32, 11008, extra_params=32000 * 4096, weight_group="Llama-2-7B")
SMOLLM2_16L = TS("SmolLM2-360M (first 16/32 L)", 16, 960, 15, 64, 5, 2560)
SMOLVLA_EXPERT = TS("SmolVLA action expert", 16, 720, 15, 64, 5, 2048,
                    cross_attn_layers=8, d_ctx=320, cross_replaces_self=True)
QWEN3_2B_12L = TS("Cosmos-Reason2-2B LLM (first 12/28 L)", 12, 2048, 16, 128, 8, 6144)
GR00T_DIT = TS("GR00T N1.7 AlternateVL-DiT", 16, 1536, 32, 48, 32, 6144, gated_ffn=False,
               cross_attn_layers=8, d_ctx=2048, cross_replaces_self=True)
LLAMA31_8B = TS("Llama-3.1-8B", 32, 4096, 32, 128, 8, 14336)
LLAMA31_8B_DEC = TS("Llama-3.1-8B (+LM head)", 32, 4096, 32, 128, 8, 14336, extra_params=128_256 * 4096, weight_group="Llama-3.1-8B")

# =============================================================================
# Video diffusion transformers (world models / WAM)
# =============================================================================
COSMOS25_2B = TS("Cosmos-Predict2.5-2B DiT", 28, 2048, 16, 128, 16, 8192, gated_ffn=False,
                 cross_attn_layers=28, d_ctx=2048)                       # net.py
WAN13_CAUSAL = TS("Matrix-Game-2 CausalWan-1.3B", 30, 1536, 12, 128, 12, 8960, gated_ffn=False,
                  cross_attn_layers=30, d_ctx=1536)                      # distilled config.json
WAN22_5B = TS("Matrix-Game-3 (Wan2.2-5B class)", 30, 3072, 24, 128, 24, 14336, gated_ffn=False,
              cross_attn_layers=30, d_ctx=3072)                          # ASSUMPTION: Wan2.2-TI2V-5B dims
WAN21_14B = TS("DreamZero (Wan2.1-I2V-14B)", 40, 5120, 40, 128, 40, 13824, gated_ffn=False,
               cross_attn_layers=40, d_ctx=5120)


# =============================================================================
# Workloads
# =============================================================================
def kv_resident(stack: TS, tokens: int, bpa: float = 2.0, layers=None) -> float:
    L = stack.layers if layers is None else layers
    return 2 * L * tokens * stack.kv_width * bpa


W = []

# ---- pi0 : 3 cams x 256 img tokens + 48 text = 816 prefix; 10 flow steps on 51 suffix tokens
W.append(Workload(
    "pi0 (3B)", "VLA-flow", "50-step action chunk",
    [Pass(SIGLIP_SO400M, "ViT x3 cams", 256, 256, batch=3),
     Pass(GEMMA_2B, "VLM prefill (816 tok)", 816, 816, write_kv=True),
     Pass(GEMMA_300M, "action expert x10 steps", 51, 816 + 51, kv_cached=816, repeat=10)],
    resident_state_bytes=kv_resident(GEMMA_2B, 816),
    unit_period_s=0.5,  # 50 actions @50 Hz, typical replan every 25 -> 0.5 s
    source="openpi pi0.py sample_actions(), pi0_config.py, gemma.py"))

# ---- pi0.5 : state goes into discrete prompt (max_token_len 200), suffix = 50 actions, adaRMS time
W.append(Workload(
    "pi0.5 (3B)", "VLA-flow", "50-step action chunk",
    [Pass(SIGLIP_SO400M, "ViT x3 cams", 256, 256, batch=3),
     Pass(GEMMA_2B, "VLM prefill (968 tok)", 968, 968, write_kv=True),
     Pass(GEMMA_300M, "action expert x10 steps", 50, 968 + 50, kv_cached=968, repeat=10)],
    resident_state_bytes=kv_resident(GEMMA_2B, 968),
    unit_period_s=0.5,
    source="openpi pi0_config.py (pi05=True, max_token_len=200)"))

# ---- pi0-FAST : autoregressive DCT/BPE action tokens through Gemma-2B
W.append(Workload(
    "pi0-FAST (3B)", "VLA-AR", "action chunk (~60 tokens)",
    [Pass(SIGLIP_SO400M, "ViT x3 cams", 256, 256, batch=3),
     Pass(GEMMA_2B, "VLM prefill (888 tok)", 888, 888, write_kv=True, attn_frac=0.6),
     Pass(GEMMA_2B_DEC, "AR decode x60 tok", 1, 918, kv_cached=918, write_kv=True, repeat=60)],  # ASSUMPTION ~60 tok
    resident_state_bytes=kv_resident(GEMMA_2B, 948),
    unit_period_s=0.5,
    source="openpi pi0_fast.py; token count is an ASSUMPTION (FAST compresses 10x7 chunk to ~30-60 tok)"))

# ---- OpenVLA 7B : DINOv2+SigLIP fused (256 tok), Llama-2-7B, 7 action tokens
W.append(Workload(
    "OpenVLA (7.5B)", "VLA-AR", "1 action (7 DoF)",
    [Pass(DINOV2_L, "DINOv2-L (1 img)", 256, 256),
     Pass(SIGLIP_SO400M, "SigLIP (1 img)", 256, 256),
     Pass(LLAMA2_7B, "LLM prefill (280 tok)", 280, 280, write_kv=True, attn_frac=0.5),
     Pass(LLAMA2_7B_DEC, "AR decode x7 tok", 1, 284, kv_cached=284, write_kv=True, repeat=7)],
    resident_state_bytes=kv_resident(LLAMA2_7B, 287),
    unit_period_s=1 / 5,  # single-step policy, typically 5-15 Hz on WidowX/Franka
    source="openvla prismatic/extern/hf/modeling_prismatic.py predict_action()"))

# ---- SmolVLA : 3 cams, pixel-shuffle 1024->64 tok/img, first 16 VLM layers, expert w/ interleaved CA/SA
W.append(Workload(
    "SmolVLA (0.45B)", "VLA-flow", "50-step action chunk",
    [Pass(SIGLIP_B16, "ViT x3 cams (1024 patch)", 1024, 1024, batch=3),
     Pass(SMOLLM2_16L, "VLM prefill 16L (241 tok)", 241, 241, write_kv=True),
     Pass(SMOLVLA_EXPERT, "expert x10 steps", 50, 50, n_ctx=241, repeat=10, self_attn_layers=8, attn_frac=0.5)],
    resident_state_bytes=kv_resident(SMOLLM2_16L, 241),
    unit_period_s=0.5,
    source="lerobot configuration_smolvla.py, smolvlm_with_expert.py"))

# ---- GR00T N1.7 : Cosmos-Reason2-2B truncated at layer 12, DiT 16L, 4 flow steps, horizon 40
W.append(Workload(
    "GR00T N1.7 (3B)", "VLA-flow", "40-step action chunk",
    [Pass(QWEN3VL_VIT, "ViT x2 cams (256 patch)", 256, 256, batch=2),
     Pass(QWEN3_2B_12L, "LLM 12L (168 tok)", 168, 168, attn_frac=0.5),
     Pass(GR00T_DIT, "DiT x4 steps", 41, 41, n_ctx=84, repeat=4, self_attn_layers=8)],
    resident_state_bytes=168 * 2048 * 2,     # only the layer-12 hidden states are kept
    unit_period_s=40 / 30 / 2,                # ASSUMPTION 30 Hz control, replan at half chunk
    source="Isaac-GR00T configs/model/gr00t_n1d7.py, modules/dit.py, qwen3_backbone.py"))

# ---- V-JEPA 2-AC : latent MPC with CEM (400 samples x 10 iters x horizon 2)
W.append(Workload(
    "V-JEPA 2-AC (1.3B)", "WM-JEPA (MPC)", "1 planned action",
    [Pass(VJEPA2_G, "encoder ctx+goal frames", 256, 256, batch=2),
     Pass(VJEPA2_AC, "CEM rollout h=1 (400x10)", 258, 258, batch=400, repeat=10, attn_frac=0.5),
     Pass(VJEPA2_AC, "CEM rollout h=2 (400x10)", 516, 516, batch=400, repeat=10, attn_frac=0.75)],
    resident_state_bytes=2 * 256 * 1408 * 2,
    unit_period_s=None,
    source="vjepa2 notebooks/utils/mpc_utils.py cem(), world_model_wrapper.py mpc_args"))

# ---- Cosmos-Predict2.5-2B robot action-conditioned: 480x640, 1+12 frames, 35 steps, CFG
W.append(Workload(
    "Cosmos-Predict2.5-2B action-cond", "WM-gen (video DiT)", "12-frame rollout",
    [Pass(COSMOS25_2B, "DiT 4 lat-frames x35 steps x CFG2", 4 * 1200, 4 * 1200, n_ctx=512, repeat=70)],
    unit_period_s=None,
    source="cosmos-predict2.5 config.py (num_steps=35, guidance 7), docs/inference_robot_action_cond.md (chunk_size 12)"))

# ---- Cosmos-Predict2.5-2B video2world 720p 77 frames (synthetic data generation)
W.append(Workload(
    "Cosmos-Predict2.5-2B 720p/77f", "WM-gen (video DiT)", "77-frame 720p clip",
    [Pass(COSMOS25_2B, "DiT 20 lat-frames x35 x CFG2", 20 * 3520, 20 * 3520, n_ctx=512, repeat=70)],
    unit_period_s=None,
    source="cosmos-predict2.5 config.py num_output_frames=77; Wan2.1 VAE 4x8x8, patch 1x2x2"))

# ---- Matrix-Game 2.0 : causal Wan-1.3B, 3 latent frames/block, local window 6, 3 steps + cache write
mg2_blk, mg2_win = 3 * 880, 6 * 880
W.append(Workload(
    "Matrix-Game 2.0 (1.3B, streaming)", "WM-gen (AR video)", "3 latent = 12 video frames",
    [Pass(WAN13_CAUSAL, "denoise x3 (KV window 6 lat)", mg2_blk, mg2_win, n_ctx=257, kv_cached=mg2_win - mg2_blk, repeat=3),
     Pass(WAN13_CAUSAL, "clean pass -> write KV", mg2_blk, mg2_win, n_ctx=257, kv_cached=mg2_win - mg2_blk, write_kv=True)],
    resident_state_bytes=kv_resident(WAN13_CAUSAL, mg2_win),
    unit_period_s=12 / 25,
    source="Matrix-Game-2 configs/distilled_model/universal/config.json, inference_universal.yaml"))

# ---- Matrix-Game 3.0 : 5B, 704x1280, 3 DMD steps, memory frames re-injected as tokens
W.append(Workload(
    "Matrix-Game 3.0 (5B, 720p)", "WM-gen (AR video)", "4 latent = 16 video frames",
    [Pass(WAN22_5B, "segment 4 new + 4 memory lat x3 steps", 8 * 880, 8 * 880, n_ctx=512, repeat=3)],  # ASSUMPTION segment
    unit_period_s=16 / 40,
    notes="README: 40 fps @720p with INT8 + distilled VAE",
    source="Matrix-Game-3 README, wan/configs/config.py (vae_stride 4x16x16, patch 1x2x2)"))

# ---- DreamZero (WAM): Wan2.1-I2V-14B 480p, joint video+action, KV cache with GT obs replacement
dz_new, dz_ctx = 2 * 1560 + 32, 6 * 1560
W.append(Workload(
    "DreamZero WAM (14B)", "WAM", "1 action chunk (2 latent frames)",
    [Pass(WAN21_14B, "joint video+action denoise x4", dz_new, dz_ctx + dz_new, n_ctx=769, kv_cached=dz_ctx, repeat=4),
     Pass(WAN21_14B, "GT obs -> KV cache write", dz_new, dz_ctx + dz_new, n_ctx=769, kv_cached=dz_ctx, write_kv=True)],
    resident_state_bytes=kv_resident(WAN21_14B, dz_ctx),
    unit_period_s=1 / 7,   # paper: 7 Hz closed loop on GB200 after 38x optimization
    notes="segment sizes / steps are ASSUMPTIONS; backbone + 7 Hz from paper 2602.15922",
    source="arXiv 2602.15922"))

# ---- LLM references
W.append(Workload(
    "Llama-3.1-8B decode b=1 ctx=8K", "LLM", "1 token",
    [Pass(LLAMA31_8B_DEC, "decode", 1, 8192, kv_cached=8192, write_kv=True)],
    resident_state_bytes=kv_resident(LLAMA31_8B, 8192), unit_period_s=1 / 30,
    source="Llama-3.1 config"))
W.append(Workload(
    "Llama-3.1-8B decode b=1 ctx=128K", "LLM", "1 token",
    [Pass(LLAMA31_8B_DEC, "decode", 1, 131072, kv_cached=131072, write_kv=True)],
    resident_state_bytes=kv_resident(LLAMA31_8B, 131072), unit_period_s=1 / 30))
W.append(Workload(
    "Llama-3.1-8B decode b=64 ctx=8K", "LLM", "64 tokens (1/seq)",
    [Pass(LLAMA31_8B_DEC, "decode", 1, 8192, kv_cached=8192, write_kv=True, batch=64)],
    resident_state_bytes=64 * kv_resident(LLAMA31_8B, 8192), unit_period_s=1 / 30))
W.append(Workload(
    "Llama-3.1-8B prefill 2K", "LLM", "2K-token prompt",
    [Pass(LLAMA31_8B, "prefill", 2048, 2048, write_kv=True, attn_frac=0.5)],
    resident_state_bytes=kv_resident(LLAMA31_8B, 2048)))

WORKLOADS = W

# =============================================================================
# Hardware (dense BF16 peak; see docs/00-methodology.md for derivations)
# =============================================================================
HW = [
    Hardware("H100 SXM", 989e12, 3.35e12, 50e6, "BF16 dense, HBM3"),
    Hardware("GB200 (per GPU)", 2.5e15, 8.0e12, 126e6, "BF16 dense, HBM3e"),
    Hardware("Jetson AGX Thor", 259e12, 273e9, 0, "ASSUMPTION: 2070 FP4-sparse TF / 8; LPDDR5X"),
    Hardware("Jetson AGX Orin 64G", 42.5e12, 204.8e9, 4e6, "ASSUMPTION: GPU 170 INT8-sparse TOPS / 4"),
]
