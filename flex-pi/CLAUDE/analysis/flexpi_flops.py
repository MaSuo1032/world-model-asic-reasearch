"""Exact parameter and FLOP counts for Flex-π inference, measured on the OFFICIAL
model code (github.com/geyan21/flex-pi, fork MaSuo1032/flex-pi @ 20c1b2b).

Modules are instantiated on PyTorch's `meta` device (shapes only, no weights,
no memory), then the real inference functions (`MoT._prefill_video_cache_inner`,
`MoT._forward_action_with_video_cache_inner`, `MoT.prefill_joint_anchor_kv`,
`MoT.decode_joint_noisy`, DINOv3, Wan2.2 VAE) are executed under
`torch.utils.flop_counter.FlopCounterMode`, which counts every matmul / conv /
SDPA FLOP that would run on a GPU (masked SDPA counts the full dense product,
as the kernels compute it). Element-wise ops (norms, RoPE, adaLN, GELU) are not
counted by FlopCounterMode; they are < 1% of FLOPs here.

    python flexpi_flops.py --flexpi /path/to/flex-pi   # writes ../results/flexpi_flops.{json,md}

Token layout = shipped RoboTwin/YAM 3-camera `tshape_384x320` (docs/OVERVIEW.md §2):
  RGB 120 tok/latent-frame, pointmap 120, DINO 147 (3 cams x 7x7 after 2x2 fold),
  3 latent frames (1 observed + 2 future), 32 action tokens, 128 text + 1 proprio context.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import torch
from torch.utils.flop_counter import FlopCounterMode

RGB_TPF, PT_TPF, DINO_TPF, N_LAT, N_ACT, N_CTX = 120, 120, 147, 3, 32, 129
ACTION_DIM = 14            # configs/data/robotwin.yaml action_output_dim
META = torch.device("meta")
BF = torch.bfloat16


def build(flexpi_root):
    sys.path.insert(0, os.path.join(flexpi_root, "src"))
    import yaml
    from flexpi.models.wan_video_dit import WanVideoDiT
    from flexpi.models.action_dit import ActionDiT
    from flexpi.models.mot import MoT
    cfg = yaml.safe_load(open(os.path.join(flexpi_root, "configs/model/flexpi.yaml")))
    vcfg = {k: v for k, v in cfg["video_dit_config"].items()
            if k in ("has_image_input", "patch_size", "in_dim", "hidden_dim", "ffn_dim", "freq_dim", "text_dim",
                     "out_dim", "num_heads", "attn_head_dim", "num_layers", "eps", "seperated_timestep",
                     "require_clip_embedding", "require_vae_embedding", "fuse_vae_embedding_in_latents")}
    acfg = {k: v for k, v in cfg["action_dit_config"].items() if k != "use_gradient_checkpointing"}
    acfg["action_dim"] = ACTION_DIM
    with META:
        video = WanVideoDiT(**vcfg).to(BF)
        action = ActionDiT(**acfg).to(BF)
        mot = MoT({"video": video, "action": action}, mot_checkpoint_mixed_attn=False, attn_mode="sdpa",
                  hbridge_enabled=True, hbridge_bottom_ratio=0.25, hbridge_top_ratio=0.25).eval()
    return cfg, video, action, mot


def t(*shape, dtype=BF):
    return torch.empty(*shape, device=META, dtype=dtype)


def freqs(n):          # RoPE table: [S, 1, head_dim/2] complex (MoT docstring)
    return torch.empty(n, 1, 64, device=META, dtype=torch.complex64)


def mask(q, k):
    return torch.ones(q, k, device=META, dtype=torch.bool)


CATS = [  # (category, substring of the module path inside an expert) -- first match wins
    ("cross_attn_kv_on_text", ".cross_attn.k"), ("cross_attn_kv_on_text", ".cross_attn.v"),
    ("cross_attn_q_o", ".cross_attn."), ("self_attn_qkvo", ".self_attn."), ("ffn", ".ffn."),
]
_LINEAR_LABELS = {}


def label_linears(*named_roots):
    """Remember each nn.Linear's full path so a forward hook can attribute its FLOPs."""
    for root_name, root in named_roots:
        for name, m in root.named_modules():
            if isinstance(m, torch.nn.Linear):
                _LINEAR_LABELS[m] = f"{root_name}.{name}"


def count(fn):
    by = defaultdict(int)

    def hook(m, inp, _out):
        path = _LINEAR_LABELS.get(m, "")
        cat = next((c for c, sub in CATS if sub in path + "."), "other_linear")
        x = inp[0]
        by[cat] += 2 * (x.numel() // x.shape[-1]) * m.in_features * m.out_features

    hs = [m.register_forward_hook(hook) for m in _LINEAR_LABELS]
    try:
        with FlopCounterMode(display=False) as fc:
            fn()
    finally:
        for h in hs:
            h.remove()
    g = fc.get_flop_counts().get("Global", {})
    by["attention_scores_sdpa"] = sum(n for op, n in g.items() if "bmm" in str(op) or "scaled_dot" in str(op))
    by["conv"] = sum(n for op, n in g.items() if "convolution" in str(op))
    total = fc.get_total_flops()
    by["unattributed"] = total - sum(by.values())
    return total, dict(by)


def params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flexpi", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    a = ap.parse_args()
    cfg, video, action, mot = build(a.flexpi)
    D_V, D_A = 3072, 1024
    label_linears(("video", video), ("action", action))
    R = {}

    # ---------------- parameters ----------------
    R["params"] = {
        "visual_trunk_total": params(video),
        "visual_trunk_blocks": sum(params(b) for b in video.blocks),
        "action_expert_total": params(action),
        "action_expert_blocks": sum(params(b) for b in action.blocks),
        "per_trunk_block": params(video.blocks[0]),
        "per_action_block": params(action.blocks[0]),
    }
    blk = video.blocks[0]
    R["params"]["trunk_block_breakdown"] = {
        "self_attn": params(blk.self_attn), "cross_attn": params(blk.cross_attn),
        "ffn": params(blk.ffn), "modulation+norms": params(blk) - params(blk.self_attn) - params(blk.cross_attn) - params(blk.ffn)}
    ablk = action.blocks[0]
    R["params"]["action_block_breakdown"] = {
        "self_attn": params(ablk.self_attn), "cross_attn": params(ablk.cross_attn),
        "ffn": params(ablk.ffn), "modulation+norms": params(ablk) - params(ablk.self_attn) - params(ablk.cross_attn) - params(ablk.ffn)}

    n_anchor = RGB_TPF + DINO_TPF + PT_TPF                     # 387 observed first-frame tokens
    sub_lens = [RGB_TPF, DINO_TPF, PT_TPF]
    sub_masks = [mask(n, n) for n in sub_lens]
    ctx_v = {"context": t(1, N_CTX, D_V), "mask": torch.ones(1, n_anchor, N_CTX, device=META, dtype=torch.bool)}

    # ---------------- PREFILL: anchors through the 5B trunk (once per call) ----------------
    def prefill():
        return mot._prefill_video_cache_inner(
            video_tokens=t(1, n_anchor, D_V), video_freqs=freqs(n_anchor), video_t_mod=t(1, n_anchor, 6, D_V),
            video_context_payload=ctx_v, video_attention_mask=mask(n_anchor, n_anchor),
            video_sub_stream_lens=sub_lens, video_sub_stream_self_masks=sub_masks)
    R["prefill_anchor"] = count(prefill)
    _, ck, cv = prefill()

    # ---------------- DECODE (action-only regime): 32 action tokens through the 1B expert ----------------
    ctx_a = {"context": t(1, N_CTX, D_A), "mask": torch.ones(1, N_ACT, N_CTX, device=META, dtype=torch.bool)}

    def decode_action():
        return mot._forward_action_with_video_cache_inner(
            action_tokens=t(1, N_ACT, D_A), action_freqs=freqs(N_ACT), action_t_mod=t(1, 6, D_A),
            action_context_payload=ctx_a, video_cache_k=ck, video_cache_v=cv,
            action_attention_mask=mask(N_ACT, n_anchor + N_ACT), action_only_attention_mask=mask(N_ACT, N_ACT))
    R["decode_action_only_step"] = count(decode_action)

    # ---------------- DECODE (joint regimes): noisy future visual tokens + actions ----------------
    k_all, v_all = torch.stack(ck), torch.stack(cv)
    regimes = {   # name: noisy future visual tokens generated per step
        "joint_rgb": (N_LAT - 1) * RGB_TPF,
        "joint_rgb_pointmap": (N_LAT - 1) * (RGB_TPF + PT_TPF),
        "joint_full": (N_LAT - 1) * (RGB_TPF + PT_TPF + DINO_TPF),
    }
    for name, n_noisy in regimes.items():
        seq = n_anchor + n_noisy

        def dec(n_noisy=n_noisy, seq=seq):
            return mot.decode_joint_noisy(
                noisy_video_tokens=t(1, n_noisy, D_V), action_tokens=t(1, N_ACT, D_A),
                noisy_video_freqs=freqs(n_noisy), action_freqs=freqs(N_ACT),
                noisy_video_t_mod=t(1, n_noisy, 6, D_V), action_t_mod=t(1, 6, D_A),
                noisy_video_context_payload={"context": t(1, N_CTX, D_V),
                                             "mask": torch.ones(1, n_noisy, N_CTX, device=META, dtype=torch.bool)},
                action_context_payload=ctx_a, k_all=k_all, v_all=v_all,
                anchor_pos_idx=torch.arange(n_anchor, device=META), noisy_pos_idx=torch.arange(n_noisy, device=META),
                video_seq_len=seq, decode_inner_mask=mask(n_noisy + N_ACT, seq + N_ACT),
                decode_outer_mask=mask(n_noisy + N_ACT, seq + N_ACT))
        R[f"decode_{name}_step"] = count(dec)
        R[f"decode_{name}_step"] = (*R[f"decode_{name}_step"], n_noisy)

    # ---------------- encoders ----------------
    try:
        import timm
        with META:
            dino = timm.create_model(cfg["dino_model_name"].split(".")[0], pretrained=False, num_classes=0).to(BF)
        label_linears(("dino", dino))
        R["params"]["dinov3_vitb16"] = params(dino)
        R["encode_dino_3cams"] = count(lambda: dino.forward_features(t(3, 3, 224, 224)))
    except Exception as e:  # noqa: BLE001
        R["encode_dino_3cams"] = f"skipped: {type(e).__name__}: {e}"
    try:
        from flexpi.models.wan_video_vae import VideoVAE38_
        with META:
            vae = VideoVAE38_(z_dim=48, dim=160).to(BF).eval()
        R["params"]["wan22_vae_encoder"] = params(vae.encoder) + params(vae.conv1)
        R["params"]["wan22_vae_decoder"] = params(vae.decoder) + params(vae.conv2)
        scale = [t(48), t(48)]
        R["encode_vae_rgb_first_frame"] = count(lambda: vae.encode(t(1, 3, 1, 320, 384), scale))
        R["decode_vae_rgb_9frames"] = count(lambda: vae.decode(t(1, 48, N_LAT, 20, 24), scale))
    except Exception as e:  # noqa: BLE001
        R["encode_vae_rgb_first_frame"] = f"skipped: {type(e).__name__}: {e}"

    os.makedirs(a.out, exist_ok=True)
    json.dump(R, open(os.path.join(a.out, "flexpi_flops.json"), "w"), indent=1, default=str)
    for k, v in R.items():
        if isinstance(v, tuple):
            print(f"{k:34s} {v[0]/1e9:10.1f} GFLOP   " + ", ".join(f"{op}={n/1e9:.1f}" for op, n in v[1].items()))
        else:
            print(k, v if not isinstance(v, dict) else json.dumps(v, default=str)[:400])


if __name__ == "__main__":
    main()
