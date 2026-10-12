"""Fast analytical systolic-array model, calibrated against SCALE-Sim v3
(results/scalesim_validation.csv), applied to the real GEMM shapes of every
workload stage.

Mapping follows SCALE-Sim:
    OS : array rows <- M, cols <- N, stream K
    WS : array rows <- K, cols <- N, stream M
    IS : array rows <- K, cols <- M, stream N
cycles = folds * (T + a*R + b*C + c)   with (a, b, c) fitted per dataflow.

    cd CLAUDE && python analysis/systolic.py
"""
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")

# per-fold overhead coefficients (T + a*R + b*C + c); fitted in fit_coeffs()
COEF = {"os": (1, 1, -2), "ws": (2, 1, -2), "is": (2, 1, -2)}


def map_dims(M, N, K, df):
    if df == "os":
        return M, N, K
    if df == "ws":
        return K, N, M
    if df == "is":
        return K, M, N
    raise ValueError(df)


def gemm_cycles(M, N, K, R, C, df, cores=1):
    sr, sc, t = map_dims(M, N, K, df)
    folds = math.ceil(sr / R) * math.ceil(sc / C)
    a, b, c = COEF[df]
    per_fold = t + a * R + b * C + c
    return math.ceil(folds / cores) * per_fold


def ideal_cycles(M, N, K, R, C, cores=1):
    return M * N * K / (R * C * cores)


def fit_coeffs(path=os.path.join(RES, "scalesim_validation.csv")):
    """Integer grid search of (a, b, c) per dataflow against SCALE-Sim runs; returns max rel. error.
    Non-square arrays in the data are what separate a (rows) from b (cols)."""
    rows = list(csv.DictReader(open(path)))
    errs = {}
    for df in ("os", "ws", "is"):
        rs = [r for r in rows if r["dataflow"] == df]
        if not rs:
            continue
        best = None
        for a in range(0, 4):
            for b in range(0, 4):
                for c in range(-4, 3):
                    COEF[df] = (a, b, c)
                    e = max(abs(gemm_cycles(int(r["M"]), int(r["N"]), int(r["K"]),
                                            *map(int, r["array"].split("x")), df) - int(r["cycles"]))
                            / int(r["cycles"]) for r in rs)
                    if best is None or e < best[0] - 1e-12:
                        best = (e, (a, b, c))
        errs[df], COEF[df] = best
        errs[df] = (best[0], len(rs), len({r["array"] for r in rs if r["array"].split("x")[0] != r["array"].split("x")[1]}))
    return errs


# ---------------------------------------------------------------------------
# GEMM lists per stage (one execution of the stage), from the specs in models.py
# ---------------------------------------------------------------------------
def transformer_gemms(n_q, n_kv, d, heads, hd, kv_heads, ff, layers, gated=True, batch=1, d_ctx=None,
                      n_ctx=0, cross_layers=0, cross_replaces=False):
    M = n_q * batch
    aw, kvw = heads * hd, kv_heads * hd
    g = [("q", M, aw, d, layers), ("k", M, kvw, d, layers - (cross_layers if cross_replaces else 0)),
         ("v", M, kvw, d, layers - (cross_layers if cross_replaces else 0)), ("o", M, d, aw, layers)]
    if gated:
        g += [("gate", M, ff, d, layers), ("up", M, ff, d, layers), ("down", M, d, ff, layers)]
    else:
        g += [("fc1", M, ff, d, layers), ("fc2", M, d, ff, layers)]
    self_layers = layers - (cross_layers if cross_replaces else 0)
    if n_kv:
        g += [("qk", n_q, n_kv, hd, heads * self_layers * batch), ("pv", n_q, hd, n_kv, heads * self_layers * batch)]
    if cross_layers and n_ctx:
        if not cross_replaces:
            g += [("xq", M, aw, d, cross_layers), ("xo", M, d, aw, cross_layers)]
        g += [("xqk", n_q, n_ctx, hd, heads * cross_layers * batch), ("xpv", n_q, hd, n_ctx, heads * cross_layers * batch)]
    return g


def mlp3(M, i, h, o, count=1):
    return [("l1", M, h, i, count), ("l2", M, h, h, count), ("l3", M, o, h, count)]


STAGES = {
    # name: (gemm list, repeats per inference unit, family)
    "π0 expert step (M=51)": (transformer_gemms(51, 867, 1024, 8, 256, 1, 4096, 18), 10, "VLA loop"),
    "π0 best-of-16 expert step (M=816)": (transformer_gemms(51, 867, 1024, 8, 256, 1, 4096, 18, batch=16), 10, "VLA loop"),
    "SmolVLA expert step (M=50)": (transformer_gemms(50, 50, 720, 15, 64, 5, 2048, 16, cross_layers=8, n_ctx=241,
                                                     cross_replaces=True), 10, "VLA loop"),
    "GR00T DiT step (M=41)": (transformer_gemms(41, 41, 1536, 32, 48, 32, 6144, 16, gated=False, cross_layers=8,
                                                n_ctx=84, cross_replaces=True), 4, "VLA loop"),
    "π0 VLM prefill (M=816)": (transformer_gemms(816, 816, 2048, 8, 256, 1, 16384, 18), 1, "VLA front"),
    "SigLIP ViT x3 (M=768)": (transformer_gemms(256, 256, 1152, 16, 72, 16, 4304, 27, gated=False, batch=3), 1, "VLA front"),
    "TD-MPC2-317M rollout step (M=512)": (mlp3(512, 1382, 4096, 1376) + mlp3(512, 1382, 4096, 101), 18, "latent MPC"),
    "TD-MPC2-5M rollout step (M=512)": (mlp3(512, 518, 512, 512) + mlp3(512, 518, 512, 101), 18, "latent MPC"),
    "V-JEPA2-AC rollout h=1 (M=103,200)": (transformer_gemms(258, 258, 1024, 16, 64, 16, 4096, 24, gated=False,
                                                              batch=400), 10, "latent MPC"),
    "Matrix-Game-2 denoise (M=2,640)": (transformer_gemms(2640, 5280, 1536, 12, 128, 12, 8960, 30, gated=False,
                                                          cross_layers=30, n_ctx=257), 3, "video WM"),
    "Llama-8B decode b=1 (M=1)": (transformer_gemms(1, 8192, 4096, 32, 128, 8, 14336, 32), 1, "LLM"),
    "Llama-8B decode b=64 (M=64)": (transformer_gemms(1, 8192, 4096, 32, 128, 8, 14336, 32, batch=64), 1, "LLM"),
}

# iso-PE configurations: 16,384 MACs total
CONFIGS = [
    ("1x 128x128", 128, 128, 1),
    ("4x 64x64", 64, 64, 4),
    ("16x 32x32", 32, 32, 16),
    ("1x 256x64", 256, 64, 1),
    ("1x 64x256", 64, 256, 1),
]


def stage_util(gemms, R, C, df, cores):
    cyc = sum(gemm_cycles(M, N, K, R, C, df, cores) * cnt for _, M, N, K, cnt in gemms)
    ideal = sum(ideal_cycles(M, N, K, R, C, cores) * cnt for _, M, N, K, cnt in gemms)
    return cyc, ideal / cyc


def best_df(gemms, R, C, cores):
    """Per-GEMM choice of dataflow (a reconfigurable array) vs fixed dataflow."""
    cyc = sum(min(gemm_cycles(M, N, K, R, C, df, cores) for df in ("os", "ws", "is")) * cnt
              for _, M, N, K, cnt in gemms)
    ideal = sum(ideal_cycles(M, N, K, R, C, cores) * cnt for _, M, N, K, cnt in gemms)
    return cyc, ideal / cyc


if __name__ == "__main__":
    out = ["# Systolic-array utilization on real GEMM shapes (auto-generated by analysis/systolic.py)\n"]
    if os.path.exists(os.path.join(RES, "scalesim_validation.csv")):
        errs = fit_coeffs()
        out.append("## Calibration against SCALE-Sim v3\n")
        out.append("Per-fold cycle model `T + a·R + b·C + c` fitted to cycle-accurate SCALE-Sim runs "
                   "(ideal memory bandwidth): " + ", ".join(
                       f"{df.upper()}: a={COEF[df][0]}, b={COEF[df][1]}, c={COEF[df][2]} "
                       f"(max error {e[0]*100:.3f}% over {e[1]} runs, {e[2]} non-square)"
                       for df, e in errs.items()) + "\n")
    out.append("## Compute-array utilization by stage (16,384 MACs in every configuration)\n")
    hdr = "| Stage | Family | " + " | ".join(f"{n} {df.upper()}" for n, *_ in CONFIGS[:3] for df in ("ws", "os")) + \
          " | best config + per-GEMM dataflow |"
    out.append(hdr)
    out.append("|" + "---|" * (2 + 6 + 1))
    rows = []
    for sname, (gemms, rep, fam) in STAGES.items():
        cells = []
        for cname, R, C, cores in CONFIGS[:3]:
            for df in ("ws", "os"):
                _, u = stage_util(gemms, R, C, df, cores)
                cells.append(f"{u*100:.0f}%")
        best = max(((cname,) + best_df(gemms, R, C, cores) for cname, R, C, cores in CONFIGS),
                   key=lambda x: x[2])
        rows.append(dict(stage=sname, family=fam, best_cfg=best[0], best_util=best[2]))
        out.append(f"| {sname} | {fam} | " + " | ".join(cells) + f" | {best[0]}: {best[2]*100:.0f}% |")
    # ---- couple array utilization with DRAM traffic: a pi0 chunk on an edge chip ----
    from models import WORKLOADS
    pi0 = next(w for w in WORKLOADS if w.name == "pi0 (3B)")
    F_HZ, BW, SCALE = 1.0e9, 273e9, 8     # 8 tiles x 16,384 MACs @ 1 GHz = 262 TFLOP/s BF16 (Thor-class), LPDDR5X
    stage_bytes = {"SigLIP ViT x3 (M=768)": pi0.passes[0].total_bytes(2, 2),
                   "π0 VLM prefill (M=816)": pi0.passes[1].total_bytes(2, 2),
                   "π0 expert step (M=51)": pi0.passes[2].total_bytes(2, 2) / 10}
    expert_w = pi0.passes[2].weight_bytes(2) / 10
    out.append("\n## π0 chunk on an edge chip: array shape × dataflow × weight residency\n")
    out.append("8 tiles × 16,384 MACs at 1 GHz (262 TFLOP/s BF16 peak, Thor-class) with 273 GB/s DRAM. Config names "
               "below describe one tile. Each stage takes "
               "max(array cycles / f, DRAM bytes / BW). ‘Resident’ = action-expert weights stay on-chip across "
               "the 10 flow steps (needs ≈ 622 MB BF16 / 311 MB FP8 of SRAM).\n")
    out.append("| Config | Dataflow | expert step: compute ms | expert step: DRAM ms | chunk ms (streamed) | "
               "chunk ms (resident) |")
    out.append("|---|---|---|---|---|---|")
    edge_rows = []
    for cname, R, C, cores in CONFIGS:
        for df in ("ws", "os", "best"):
            def cyc(name):
                g = STAGES[name][0]
                c8 = cores * SCALE
                return (best_df(g, R, C, c8) if df == "best" else stage_util(g, R, C, df, c8))[0]
            t_c = {n: cyc(n) / F_HZ for n in stage_bytes}
            t_m = {n: b / BW for n, b in stage_bytes.items()}
            front = sum(max(t_c[n], t_m[n]) for n in ("SigLIP ViT x3 (M=768)", "π0 VLM prefill (M=816)"))
            e = "π0 expert step (M=51)"
            streamed = front + 10 * max(t_c[e], t_m[e])
            t_m_res = (stage_bytes[e] - expert_w) / BW
            resident = front + max(t_c[e], expert_w / BW) + 9 * max(t_c[e], t_m_res)
            edge_rows.append((cname, df, streamed, resident))
            out.append(f"| {cname} | {df.upper() if df != 'best' else 'per-GEMM best'} | {t_c[e]*1e3:.2f} | "
                       f"{t_m[e]*1e3:.2f} | {streamed*1e3:.1f} | {resident*1e3:.1f} |")

    # ---- figure: utilization heatmap ----
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
    cols = [(c, df) for c in CONFIGS[:3] for df in ("ws", "os", "best")]
    names = list(STAGES)
    U = np.array([[(best_df(STAGES[n][0], c[1], c[2], c[3]) if df == "best"
                    else stage_util(STAGES[n][0], c[1], c[2], df, c[3]))[1] * 100 for c, df in cols] for n in names])
    cmap = LinearSegmentedColormap.from_list("blue", ["#f1f6fd", "#9cc3ef", "#2a78d6", "#103d73"])
    fig, ax = plt.subplots(figsize=(11.5, 6.4), facecolor=SURF)
    im = ax.imshow(U, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            ax.text(j, i, f"{U[i, j]:.0f}%", ha="center", va="center", fontsize=8,
                    color="white" if U[i, j] >= 60 else INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([("per-GEMM\nbest" if df == "best" else df.upper()) for _, df in cols], fontsize=8)
    for k, c in enumerate(CONFIGS[:3]):
        ax.text(3 * k + 1, -0.9, c[0], ha="center", fontsize=9, color=INK2, fontweight="bold")
        if k:
            ax.axvline(3 * k - 0.5, color=SURF, lw=3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"{n}  [{STAGES[n][2]}]" for n in names], fontsize=8.5)
    ax.set_title("Systolic-array compute utilization on real GEMM shapes (16,384 MACs per configuration; "
                 "SCALE-Sim-calibrated)", loc="left", fontsize=10, color=INK, pad=24)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("utilization", color=INK2)
    cb.outline.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "systolic_utilization.png"), dpi=160)
    plt.close(fig)

    with open(os.path.join(RES, "systolic.md"), "w") as fh:
        fh.write("\n".join(out) + "\n")
    with open(os.path.join(RES, "systolic.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["stage", "family", "config", "dataflow", "util"])
        for sname, (gemms, rep, fam) in STAGES.items():
            for cname, R, C, cores in CONFIGS:
                for df in ("ws", "os", "is"):
                    wr.writerow([sname, fam, cname, df, round(stage_util(gemms, R, C, df, cores)[1], 4)])
                wr.writerow([sname, fam, cname, "best-per-gemm", round(best_df(gemms, R, C, cores)[1], 4)])
    print("\n".join(out))
