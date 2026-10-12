"""
Analytical performance model for transformer-based embodied models
(VLA / World Model / World Action Model) vs. LLMs.

Everything here is first-order arithmetic:
  * FLOPs      : 2 * MACs, counting GEMMs + attention score/value matmuls.
                 Norms, softmax, RoPE, activations are ignored (<3% of FLOPs).
  * Weight I/O : every weight of a module is read once per forward pass
                 (i.e. no cross-pass on-chip reuse). This is the realistic
                 case for GPUs/edge SoCs whose SRAM << model size.
  * KV / ctx I/O: cached K/V (or cross-attn memory) read once per layer per pass,
                 newly produced K/V written once.
  * Act I/O    : layer-boundary activations written+read once per layer
                 (assumes intra-layer fusion, e.g. FlashAttention + fused MLP).

Latency = sum over passes of max(FLOPs/peak, Bytes/BW)  -> ideal roofline lower bound.
Real systems land 1.5-4x above this (kernel launch, low util at small M, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
@dataclass
class TransformerStack:
    """A homogeneous stack of transformer layers."""
    name: str
    layers: int
    d_model: int
    n_heads: int
    head_dim: int
    n_kv_heads: int
    d_ff: int
    gated_ffn: bool = True          # SwiGLU/GeGLU = 3 matrices, GELU MLP = 2
    q_dim: Optional[int] = None     # if q projection width != n_heads*head_dim
    # cross-attention (DiT style): extra Q/O proj + K/V proj from a memory of width d_ctx
    cross_attn_layers: int = 0      # how many of the layers have a cross-attn block
    d_ctx: Optional[int] = None
    # True  -> the cross-attn block REPLACES self-attn in those layers (GR00T AlternateVLDiT,
    #          SmolVLA expert). Q/O are already counted; K/V project from d_ctx.
    # False -> an extra cross-attn block is added (Wan / Cosmos DiT text cross-attn).
    cross_replaces_self: bool = False
    extra_params: int = 0           # adaLN / modulation / embeddings counted in weights
    weight_group: Optional[str] = None  # stacks sharing the same weights (e.g. decode view with LM head)

    @property
    def attn_width(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.n_kv_heads * self.head_dim

    def params_per_layer_self(self) -> int:
        q = self.d_model * self.attn_width
        kv = 2 * self.d_model * self.kv_width
        o = self.attn_width * self.d_model
        ffn = (3 if self.gated_ffn else 2) * self.d_model * self.d_ff
        return q + kv + o + ffn

    def params_cross_block(self) -> int:
        d_ctx = self.d_ctx or self.d_model
        kv = 2 * d_ctx * self.kv_width
        if self.cross_replaces_self:
            return kv - 2 * self.d_model * self.kv_width   # delta vs a self layer
        return self.d_model * self.attn_width * 2 + kv

    def gemm_params_on_query_tokens(self) -> int:
        """Weights that multiply the query-side tokens every pass
        (cross-attn K/V projections act on the memory, not on the queries)."""
        tot = self.layers * self.params_per_layer_self()
        if self.cross_attn_layers:
            if self.cross_replaces_self:
                tot -= self.cross_attn_layers * 2 * self.d_model * self.kv_width
            else:
                tot += self.cross_attn_layers * 2 * self.d_model * self.attn_width
        return tot

    @property
    def params(self) -> int:
        return (self.layers * self.params_per_layer_self()
                + self.cross_attn_layers * self.params_cross_block()
                + self.extra_params)


@dataclass
class Pass:
    """One forward pass of a stack over a set of query tokens.

    n_q        : tokens computed in this pass (query side)
    n_kv_self  : tokens the self-attention attends over (incl. cache), per layer
    n_ctx      : tokens of cross-attn memory (only for cross_attn_layers)
    kv_cached  : how many of n_kv_self come from a cache (read from memory)
    write_kv   : whether this pass writes its own K/V to a cache
    repeat     : how many times this exact pass is executed (denoise steps, CFG...)
    batch      : independent sequences processed together (weights shared)
    attn_frac  : fraction of the full n_q x n_kv attention matrix actually computed
                 (0.5 for causal prefill, 1.0 for bidirectional / decode)
    ctx_kv_precomputed : cross-attn K/V projections hoisted out of the step loop
    """
    stack: TransformerStack
    label: str
    n_q: int
    n_kv_self: int
    n_ctx: int = 0
    kv_cached: int = 0
    write_kv: bool = False
    repeat: int = 1
    batch: int = 1
    attn_frac: float = 1.0
    ctx_kv_precomputed: bool = True
    self_attn_layers: Optional[int] = None   # if only some layers do self-attn

    # ---- FLOPs ----
    def flops(self) -> float:
        s = self.stack
        B, Nq = self.batch, self.n_q
        L_self = s.layers if self.self_attn_layers is None else self.self_attn_layers
        # GEMMs acting on the query tokens (QKV/O projections + FFN [+ cross Q/O])
        proj = 2 * B * Nq * s.gemm_params_on_query_tokens()
        # self-attention score + value: 2 matmuls * 2 flops/mac
        attn = 4 * B * Nq * self.n_kv_self * s.attn_width * L_self * self.attn_frac
        # cross attention scores/values (+ K/V projection of memory if not hoisted)
        cross = 0.0
        if s.cross_attn_layers and self.n_ctx:
            d_ctx = s.d_ctx or s.d_model
            cross += 4 * B * Nq * self.n_ctx * s.attn_width * s.cross_attn_layers
            if not self.ctx_kv_precomputed:
                cross += 2 * B * self.n_ctx * (2 * d_ctx * s.kv_width) * s.cross_attn_layers
        return (proj + attn + cross) * self.repeat

    # ---- Bytes ----
    def weight_bytes(self, bpw: float) -> float:
        s = self.stack
        w = s.gemm_params_on_query_tokens() + s.extra_params
        if s.cross_attn_layers and not self.ctx_kv_precomputed:
            w = s.params
        return w * bpw * self.repeat

    def kv_bytes(self, bpa: float) -> float:
        s = self.stack
        L_self = s.layers if self.self_attn_layers is None else self.self_attn_layers
        rd = 2 * self.kv_cached * s.kv_width * L_self * bpa * self.batch
        wr = (2 * self.n_q * s.kv_width * L_self * bpa * self.batch) if self.write_kv else 0
        ctx = 0.0
        if s.cross_attn_layers and self.n_ctx:
            ctx = 2 * self.n_ctx * s.kv_width * s.cross_attn_layers * bpa * self.batch
        return (rd + wr + ctx) * self.repeat

    def act_bytes(self, bpa: float) -> float:
        s = self.stack
        return 2 * self.batch * self.n_q * s.d_model * s.layers * bpa * self.repeat

    def total_bytes(self, bpw: float, bpa: float) -> float:
        return self.weight_bytes(bpw) + self.kv_bytes(bpa) + self.act_bytes(bpa)


@dataclass
class Workload:
    """One 'inference unit': an action chunk, a video block, or one LLM token."""
    name: str
    family: str                  # VLA / VLA-AR / WAM / WM-gen / WM-JEPA / LLM
    unit: str                    # what one execution produces
    passes: List[Pass]
    resident_state_bytes: float = 0.0   # persistent cache that must live between units
    unit_period_s: Optional[float] = None  # required wall-clock budget per unit
    notes: str = ""
    source: str = ""

    def params(self) -> int:
        groups = {}
        for p in self.passes:
            g = p.stack.weight_group or p.stack.name
            groups[g] = max(groups.get(g, 0), p.stack.params)
        return sum(groups.values())

    def flops(self) -> float:
        return sum(p.flops() for p in self.passes)

    def bytes(self, bpw=2.0, bpa=2.0) -> float:
        return sum(p.total_bytes(bpw, bpa) for p in self.passes)

    def weight_bytes(self, bpw=2.0) -> float:
        return sum(p.weight_bytes(bpw) for p in self.passes)

    def kv_bytes(self, bpa=2.0) -> float:
        return sum(p.kv_bytes(bpa) for p in self.passes)

    def act_bytes(self, bpa=2.0) -> float:
        return sum(p.act_bytes(bpa) for p in self.passes)

    def intensity(self, bpw=2.0, bpa=2.0) -> float:
        return self.flops() / self.bytes(bpw, bpa)

    def roofline_latency(self, hw: "Hardware", bpw=2.0, bpa=2.0) -> float:
        t = 0.0
        for p in self.passes:
            f = p.flops()
            b = p.total_bytes(bpw, bpa)
            t += max(f / hw.peak_flops, b / hw.bw)
        return t

    def bound_breakdown(self, hw: "Hardware", bpw=2.0, bpa=2.0):
        out = []
        for p in self.passes:
            f = p.flops()
            b = p.total_bytes(bpw, bpa)
            tc, tm = f / hw.peak_flops, b / hw.bw
            out.append(dict(stage=p.label, flops=f, bytes=b, ai=f / b if b else 0,
                            t_compute=tc, t_mem=tm,
                            bound="compute" if tc >= tm else "memory",
                            t=max(tc, tm)))
        return out


@dataclass
class Hardware:
    name: str
    peak_flops: float   # dense, at the chosen precision
    bw: float           # bytes/s DRAM/HBM
    sram_bytes: float = 0.0
    notes: str = ""

    @property
    def ridge(self) -> float:
        return self.peak_flops / self.bw


def fmt_si(x: float, unit: str = "") -> str:
    for s, k in (("P", 1e15), ("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(x) >= k:
            return f"{x / k:.2f} {s}{unit}"
    return f"{x:.2f} {unit}"
