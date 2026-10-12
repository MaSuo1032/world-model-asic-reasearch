"""Apply pre-2026 LLM / diffusion acceleration techniques to every workload and
measure the roofline speedup they would give.

    cd CLAUDE && python analysis/techniques.py

Each technique is a *first-order* transform on the analytical model, with the
magnitude taken from the paper that introduced it (see docs/07). The point is
not the exact number but WHICH techniques move WHICH workloads, and why.
"""
import copy
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from models import WORKLOADS, HW  # noqa: E402
from perf_model import Hardware  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
WL = {w.name: w for w in WORKLOADS}

LOOP_FAMILIES = {"VLA-flow", "WM-gen (video DiT)", "WM-gen (AR video)", "WAM"}
VIDEO_FAMILIES = {"WM-gen (video DiT)", "WM-gen (AR video)", "WAM"}


def is_loop(w, p):
    """The iterative-refinement stage (flow / diffusion denoiser)."""
    return w.family in LOOP_FAMILIES and p.repeat > 1


def is_decode(p):
    return p.n_q == 1


def is_prefill(w, p):
    return w.family in {"VLA-flow", "VLA-AR"} and p.repeat == 1 and p.n_q > 1 and p.batch == 1


# Each technique: fn(workload) -> (workload', bpw, bpa, peak_multiplier) or None if N/A
def t_w4_weight_only(w):          # GPTQ / AWQ: INT4 weights, BF16 compute
    for p in w.passes:
        p.wbytes_scale *= 0.25
    return w, 2, 2, 1


def t_fp8(w):                     # SmoothQuant / FP8 W8A8: half bytes, 2x peak
    return w, 1, 1, 2


def t_fp4(w):                     # MXFP4 / NVFP4 W4A4(A8)
    return w, 0.5, 1, 4


def t_sparse24(w):                # SparseGPT / Wanda + 2:4 sparse tensor cores
    for p in w.passes:
        p.gemm_scale *= 0.5
        p.wbytes_scale *= 0.5625  # 50% values + 2-bit index per kept element (BF16)
    return w, 2, 2, 1


def t_loop_resident(w):           # Groq / d-Matrix / SN40L-style: looped weights stay on-chip
    hit = False
    for p in w.passes:
        if is_loop(w, p):
            p.weight_resident = True
            hit = True
    return (w, 2, 2, 1) if hit else None


def t_step_cache(w):              # DeepCache / FORA / TeaCache / EXION FFN-reuse / Cambricon-D:
    hit = False                   # reuse ~50% of the denoiser's work from the previous step
    for p in w.passes:
        if is_loop(w, p) and p.repeat >= 3:
            p.gemm_scale *= 0.5
            p.attn_scale *= 0.5
            p.wbytes_scale *= 0.5  # skipped blocks' weights not fetched
            hit = True
    return (w, 2, 2, 1) if hit else None


def t_step_distill(w):            # DMD2 / self-forcing / CausVid: <=4 steps, CFG distilled away
    hit = False
    for p in w.passes:
        if is_loop(w, p):
            p.repeat = min(p.repeat, 4)
            hit = True
    return (w, 2, 2, 1) if hit else None


def t_one_step(w):                # consistency / shortcut / DreamZero-Flash: 1-step sampler
    hit = False
    for p in w.passes:
        if is_loop(w, p):
            p.repeat = 1
            hit = True
    return (w, 2, 2, 1) if hit else None


def t_sparse_video_attn(w):       # Sliding Tile Attention / Sparse VideoGen (~3x attention)
    if w.family not in VIDEO_FAMILIES:
        return None
    for p in w.passes:
        p.attn_scale *= 1 / 3
    return w, 2, 2, 1


def t_int8_attn(w):               # SageAttention 1/2: INT8 QK^T, FP8 PV (~2x attention)
    for p in w.passes:
        p.attn_scale *= 0.5
    return w, 2, 2, 1


def t_visual_token_cache(w):      # VLA-Cache: reuse KV of static visual tokens (-27% FLOPs)
    hit = False
    for p in w.passes:
        if is_prefill(w, p):
            p.gemm_scale *= 0.73
            p.attn_scale *= 0.73
            hit = True
    return (w, 2, 2, 1) if hit else None


def t_spec_decode(w):             # Medusa / EAGLE: ~2.5 tokens accepted per verify step
    hit = False
    for p in w.passes:
        if is_decode(p) and p.repeat > 1:
            p.repeat = math.ceil(p.repeat / 2.5)
            p.n_q = 4
            hit = True
    if w.family == "LLM":       # single-token unit: amortize one verify over 2.5 tokens
        for p in w.passes:
            p.n_q = 4
            p.gemm_scale /= 2.5 / 1  # per-token share of the 4-token verify GEMM
            p.attn_scale /= 2.5
            p.wbytes_scale /= 2.5
            p.kv_scale /= 2.5
        hit = True
    return (w, 2, 2, 1) if hit else None


def t_kv_quant(w):                # KIVI / KV-cache INT4
    for p in w.passes:
        p.kv_scale *= 0.25
    return w, 2, 2, 1


def t_pim_attn(w):                # AttAcc / NeuPIMs / Duplex: KV reads served inside memory
    for p in w.passes:
        p.kv_scale = 0.0
    return w, 2, 2, 1


def combo(*fns):
    def f(w):
        bpw, bpa, mult, hit = 2, 2, 1, False
        for fn in fns:
            r = fn(w)
            if r is None:
                continue
            w, b1, b2, m = r
            bpw, bpa, mult = min(bpw, b1), min(bpa, b2), max(mult, m)
            hit = True
        return (w, bpw, bpa, mult) if hit else None
    return f


TECHNIQUES = [
    ("W4 weight-only (GPTQ/AWQ)", t_w4_weight_only),
    ("FP8 W8A8 (SmoothQuant/FP8)", t_fp8),
    ("FP4 (MXFP4/NVFP4)", t_fp4),
    ("2:4 sparsity", t_sparse24),
    ("Loop weights on-chip (SRAM-first)", t_loop_resident),
    ("Cross-step reuse (DeepCache/TeaCache/EXION)", t_step_cache),
    ("Step distillation (<=4 steps, no CFG)", t_step_distill),
    ("One-step sampler (consistency/Flash)", t_one_step),
    ("Sparse video attention (STA/SVG)", t_sparse_video_attn),
    ("INT8 attention (SageAttention)", t_int8_attn),
    ("Visual token cache (VLA-Cache)", t_visual_token_cache),
    ("Speculative decoding (EAGLE/Medusa)", t_spec_decode),
    ("KV-cache INT4 (KIVI)", t_kv_quant),
    ("PIM attention offload (AttAcc/NeuPIMs)", t_pim_attn),
    ("RECIPE edge-VLA: FP8 + loop on-chip", combo(t_fp8, t_loop_resident)),
    ("RECIPE world-model: FP8 + distill + sparse/INT8 attn",
     combo(t_fp8, t_step_distill, t_sparse_video_attn, t_int8_attn)),
    ("RECIPE WAM-on-edge: FP4 + 1-step + sparse/INT8 attn",
     combo(t_fp4, t_one_step, t_sparse_video_attn, t_int8_attn)),
]

TARGETS = ["pi0 (3B)", "GR00T N1.7 (3B)", "SmolVLA (0.45B)", "pi0-FAST (3B)", "OpenVLA (7.5B)",
           "V-JEPA 2-AC (1.3B)", "Cosmos-Predict2.5-2B action-cond", "Matrix-Game 2.0 (1.3B, streaming)",
           "Matrix-Game 3.0 (5B, 720p)", "DreamZero WAM (14B)",
           "Llama-3.1-8B decode b=1 ctx=8K", "Llama-3.1-8B decode b=1 ctx=128K"]
SHORT = {"pi0 (3B)": "π0", "GR00T N1.7 (3B)": "GR00T", "SmolVLA (0.45B)": "SmolVLA",
         "pi0-FAST (3B)": "π0-FAST", "OpenVLA (7.5B)": "OpenVLA", "V-JEPA 2-AC (1.3B)": "V-JEPA2-AC",
         "Cosmos-Predict2.5-2B action-cond": "Cosmos-AC", "Matrix-Game 2.0 (1.3B, streaming)": "MG-2.0",
         "Matrix-Game 3.0 (5B, 720p)": "MG-3.0", "DreamZero WAM (14B)": "DreamZero",
         "Llama-3.1-8B decode b=1 ctx=8K": "LLM 8K", "Llama-3.1-8B decode b=1 ctx=128K": "LLM 128K"}


def speedup(name, fn, hw):
    base = WL[name]
    t0 = base.roofline_latency(hw)
    r = fn(copy.deepcopy(base))
    if r is None:
        return None
    w, bpw, bpa, mult = r
    hw2 = Hardware(hw.name, hw.peak_flops * mult, hw.bw)
    return t0 / w.roofline_latency(hw2, bpw, bpa)


def run(hw):
    table = []
    for tname, fn in TECHNIQUES:
        table.append((tname, [speedup(n, fn, hw) for n in TARGETS]))
    return table


def md(table, hw):
    h = "| Technique | " + " | ".join(SHORT[n] for n in TARGETS) + " |"
    lines = [f"### {hw.name} (ridge {hw.ridge:.0f} FLOP/B)\n", h, "|" + "---|" * (len(TARGETS) + 1)]
    for tname, vals in table:
        cells = ["–" if v is None else (f"**{v:.2f}×**" if v >= 1.5 else f"{v:.2f}×") for v in vals]
        lines.append(f"| {tname} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    thor, h100 = HW[2], HW[0]
    tt, th = run(thor), run(h100)
    out = ["# Technique applicability (auto-generated by analysis/techniques.py)\n",
           "Roofline speedup of one inference unit vs. the BF16 baseline. `–` = technique does not apply "
           "to that workload's structure. **Bold** = ≥1.5×. Magnitudes per technique are taken from the "
           "originating papers (docs/07-llm-accelerator-techniques.md) and are first-order.\n",
           md(tt, thor), md(th, h100)]
    with open(os.path.join(RES, "techniques.md"), "w") as fh:
        fh.write("\n".join(out))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
    cmap = LinearSegmentedColormap.from_list("blue", ["#f1f6fd", "#9cc3ef", "#2a78d6", "#103d73"])
    for hw, tab, fn in ((thor, tt, "techniques_heatmap_thor.png"), (h100, th, "techniques_heatmap_h100.png")):
        import numpy as np
        M = np.array([[np.nan if v is None else v for v in vals] for _, vals in tab])
        fig, ax = plt.subplots(figsize=(12.5, 7.2), facecolor=SURF)
        ax.set_facecolor("#ecebe7")
        im = ax.imshow(M, cmap=cmap, norm=LogNorm(vmin=1, vmax=64), aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if np.isnan(v):
                    ax.text(j, i, "–", ha="center", va="center", color=INK2, fontsize=8)
                else:
                    ax.text(j, i, f"{v:.1f}×", ha="center", va="center", fontsize=7.5,
                            color="white" if v >= 6 else INK)
        ax.set_xticks(range(len(TARGETS)))
        ax.set_xticklabels([SHORT[n] for n in TARGETS], rotation=30, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(tab)))
        ax.set_yticklabels([t for t, _ in tab], fontsize=8.5)
        ax.axhline(13.5, color=SURF, lw=3)
        for x in (4.5, 9.5):
            ax.axvline(x, color=SURF, lw=3)
        ax.set_title(f"Roofline speedup per technique on {hw.name} (BF16 baseline = 1×; – = not applicable)",
                     loc="left", fontsize=10, color=INK, pad=26)
        for x, lab in ((2, "VLA"), (7, "World model / WAM"), (10.5, "LLM")):
            ax.text(x, -0.72, lab, ha="center", va="bottom", fontsize=8.5, color=INK2, fontweight="bold")
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cb.set_ticks([1, 2, 4, 8, 16, 32, 64]); cb.set_ticklabels(["1×", "2×", "4×", "8×", "16×", "32×", "64×"])
        cb.outline.set_visible(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, fn), dpi=160)
        plt.close(fig)
    print("\n".join(out))
