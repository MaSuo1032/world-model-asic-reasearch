"""E1 + E3(ii) kit: profile TD-MPC2's MPPI planner (the official `_plan` code)
and test Q-head subsampling.

Runs the *unmodified* planner from github.com/nicklashansen/tdmpc2 on CPU or GPU.
Weights are randomly initialised, then the Q-ensemble and reward heads are overwritten with
distinct non-zero values (upstream init zeroes their last layers, which would make every head
identical -- GPT review R1). The Q-subsampling check is a head-level unit test with fixed head
IDs, logits/Q comparison and a negative control (wrong heads must differ), followed by a
planner-level comparison. Latency and op counts are valid; task quality is NOT measured.

    git clone https://github.com/nicklashansen/tdmpc2 /path/tdmpc2
    pip install torch tensordict
    python tdmpc2_plan_profile.py --repo /path/tdmpc2 --device cpu  --sizes 1 5 19
    python tdmpc2_plan_profile.py --repo /path/tdmpc2 --device cuda --sizes 1 5 19 48 317 --compile

Outputs a markdown table (stdout) and a CSV (--out).
"""
import argparse
import csv
import os
import statistics
import sys
import time
import types


def build_cfg(size, obs_dim, action_dim, num_samples, iterations):
    from common import MODEL_SIZE  # noqa: E402  (tdmpc2/common/__init__.py)
    c = dict(  # defaults from tdmpc2/config.yaml (commit e9f5932)
        task="profile", obs="state", episodic=False, multitask=False, task_dim=0,
        mpc=True, iterations=iterations, num_samples=num_samples, num_elites=64, num_pi_trajs=24,
        horizon=3, min_std=0.05, max_std=2, temperature=0.5, log_std_min=-10, log_std_max=2,
        entropy_coef=1e-4, num_bins=101, vmin=-10, vmax=10, num_enc_layers=2, enc_dim=256,
        num_channels=32, mlp_dim=512, latent_dim=512, num_q=5, dropout=0.01, simnorm_dim=8,
        obs_shape={"state": (obs_dim,)}, action_dim=action_dim, episode_length=1000,
        discount_denom=5, discount_min=0.95, discount_max=0.995, compile=False,
    )
    c.update(MODEL_SIZE[size])
    c["bin_size"] = (c["vmax"] - c["vmin"]) / (c["num_bins"] - 1)
    if action_dim >= 20:          # same heuristic as TDMPC2.__init__
        c["iterations"] += 2
    return types.SimpleNamespace(**c)


def make_planner(cfg, device, compile_plan):
    import torch
    from common.world_model import WorldModel
    from tdmpc2 import TDMPC2

    p = types.SimpleNamespace()
    p.cfg, p.device = cfg, torch.device(device)
    with p.device:                       # build in place: Ensemble keeps a meta-device template module
        p.model = WorldModel(cfg).eval()
    p.discount = TDMPC2._get_discount(p, cfg.episode_length)
    p._prev_mean = torch.zeros(cfg.horizon, cfg.action_dim, device=p.device)
    p._estimate_value = types.MethodType(TDMPC2._estimate_value, p)
    p._plan = types.MethodType(TDMPC2._plan, p)
    if compile_plan:
        p._plan = torch.compile(p._plan, mode="reduce-overhead")
    return p


def patch_q_subsample(model):
    """Draw the 2 Q heads first, then evaluate only those (output-identical for a fixed seed)."""
    import torch
    from common import math as tmath

    def Q(self, z, a, task, return_type="min", target=False, detach=False):
        assert return_type in {"min", "avg", "all"}
        z = torch.cat([z, a], dim=-1)
        qnet = self._target_Qs if target else (self._detach_Qs if detach else self._Qs)
        if return_type == "all":
            return qnet(z)
        qidx = torch.randperm(self.cfg.num_q, device=z.device)[:2]
        out = torch.vmap(qnet._call, (0, None), randomness="different")(qnet.params[qidx], z)
        Qv = tmath.two_hot_inv(out, self.cfg)
        return Qv.min(0).values if return_type == "min" else Qv.sum(0) / 2

    model.Q = types.MethodType(Q, model)


def randomize_value_heads(model, seed=0, scale=0.2):
    """Test fixture (GPT review R1): upstream init zeroes the last layer of every Q head and of the
    reward head, so all heads emit identical logits and a wrong-head bug would go unnoticed.
    Overwrite every Q-ensemble / reward parameter with distinct non-zero values."""
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for _k, p in model._Qs.params.items(include_nested=True, leaves_only=True):
            p.copy_((torch.randn(p.shape, generator=g) * scale).to(p.device, p.dtype))
        model._target_Qs_params.data.copy_(model._Qs.params.data)
        for p in model._reward.parameters():
            p.copy_((torch.randn(p.shape, generator=g) * scale).to(p.device, p.dtype))


def q_subsample_unit_test(model, cfg, device, n=257, atol=0.0):
    """Head-level equivalence with a negative control. Returns a dict of diagnostics; raises on failure."""
    import torch
    from common import math as tmath
    if cfg.num_q <= 2:   # subsampling 2 of 2 heads is a permutation: no saving, no meaningful negative control
        return dict(head_spread=float("nan"), logit_maxdiff=float("nan"), q_maxdiff=float("nan"),
                    wrong_head_q_maxdiff=float("nan"), atol=atol, passed="n/a (num_q<=2)")
    qnet = model._Qs
    torch.manual_seed(7)
    z = torch.randn(n, cfg.latent_dim + cfg.action_dim, device=device)
    full = qnet(z)                                               # [num_q, n, bins] -- upstream computes all heads
    head_spread = float((full - full.mean(0, keepdim=True)).abs().max())
    if head_spread == 0.0:
        raise AssertionError("fixture invalid: all Q heads identical (zero-initialised heads)")
    qidx = torch.tensor([cfg.num_q - 1, 1], device=device)       # fixed, non-trivial selection
    sub = torch.vmap(qnet._call, (0, None), randomness="different")(qnet.params[qidx], z)
    logit_diff = float((full[qidx] - sub).abs().max())
    q_diff = float((tmath.two_hot_inv(full[qidx], cfg) - tmath.two_hot_inv(sub, cfg)).abs().max())
    wrong = torch.tensor([0, 2], device=device)                  # negative control: wrong heads must differ
    wrong_diff = float((tmath.two_hot_inv(full[qidx], cfg) - tmath.two_hot_inv(full[wrong], cfg)).abs().max())
    ok = logit_diff <= atol and q_diff <= atol and wrong_diff > 1e-4
    res = dict(head_spread=head_spread, logit_maxdiff=logit_diff, q_maxdiff=q_diff,
               wrong_head_q_maxdiff=wrong_diff, atol=atol, passed=ok)
    if not ok:
        raise AssertionError(f"Q-subsample unit test failed: {res}")
    return res


def time_plan(p, obs, reps, warmup, device):
    import torch
    sync = (lambda: torch.cuda.synchronize()) if device.startswith("cuda") else (lambda: None)
    for _ in range(warmup):
        p._plan(obs, t0=True, eval_mode=True)
    sync()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        p._plan(obs, t0=True, eval_mode=True)
        sync()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts), min(ts)


def count_ops(p, obs):
    import torch
    from torch.profiler import profile, ProfilerActivity
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if p.device.type == "cuda" else [])
    with profile(activities=acts) as prof:
        p._plan(obs, t0=True, eval_mode=True)
    ev = prof.key_averages()
    aten = sum(e.count for e in ev if e.key.startswith("aten::"))
    mm = sum(e.count for e in ev if e.key in ("aten::addmm", "aten::mm", "aten::bmm", "aten::linear", "aten::matmul"))
    kern = sum(e.count for e in ev if getattr(e, "device_type", None) is not None and "CUDA" in str(e.device_type))
    return aten, mm, kern


def analytic_flops(cfg):
    """Same arithmetic as CLAUDE/analysis/decision_system.py §3 (forward MACs x2)."""
    L, h, A, B = cfg.latent_dim, cfg.mlp_dim, cfg.action_dim, cfg.num_bins
    mlp = lambda i, o: i * h + h * h + h * o  # noqa: E731
    step = 2 * (mlp(L + A, L) + mlp(L + A, B))
    term = 2 * (mlp(L, 2 * A) + cfg.num_q * mlp(L + A, B))
    return cfg.iterations * cfg.num_samples * (cfg.horizon * step + term)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path to a clone of nicklashansen/tdmpc2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 5, 19])
    ap.add_argument("--samples", type=int, nargs="+", default=[512])
    ap.add_argument("--iterations", type=int, default=6)
    ap.add_argument("--obs-dim", type=int, default=24)
    ap.add_argument("--action-dim", type=int, default=6)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--compile", action="store_true", help="torch.compile(mode='reduce-overhead'), GPU")
    ap.add_argument("--out", default="tdmpc2_plan_profile.csv")
    a = ap.parse_args()

    sys.path.insert(0, os.path.join(os.path.abspath(a.repo), "tdmpc2"))
    import torch
    torch.manual_seed(0)
    rows = []
    for size in a.sizes:
        for ns in a.samples:
            cfg = build_cfg(size, a.obs_dim, a.action_dim, ns, a.iterations)
            p = make_planner(cfg, a.device, a.compile)
            randomize_value_heads(p.model)            # GPT review R1: make heads distinct and non-zero
            obs = torch.randn(1, a.obs_dim, device=a.device)
            flops = analytic_flops(cfg)
            med, best = time_plan(p, obs, a.reps, a.warmup, a.device)
            aten, mm, kern = count_ops(p, obs)

            # E3(ii): head-level equivalence with negative control, then planner-level check + speed
            unit = q_subsample_unit_test(p.model, cfg, a.device)
            torch.manual_seed(123); ref = p._plan(obs, t0=True, eval_mode=True).clone()
            patch_q_subsample(p.model)
            torch.manual_seed(123); sub = p._plan(obs, t0=True, eval_mode=True).clone()
            med_sub, _ = time_plan(p, obs, a.reps, a.warmup, a.device)
            rows.append(dict(
                model=f"TD-MPC2-{size}M", device=a.device, num_samples=ns, iterations=cfg.iterations,
                params_M=round(sum(t.numel() for t in p.model.parameters()) / 1e6, 2),
                analytic_GFLOP=round(flops / 1e9, 2), median_ms=round(med * 1e3, 2), best_ms=round(best * 1e3, 2),
                achieved_GFLOPs=round(flops / med / 1e9, 1), aten_ops=aten, matmul_ops=mm, gpu_kernels=kern,
                qsub_median_ms=round(med_sub * 1e3, 2), qsub_speedup=round(med / med_sub, 3),
                qsub_identical=bool(torch.equal(ref, sub)),
                num_q=cfg.num_q, unit_head_spread=unit["head_spread"], unit_logit_maxdiff=unit["logit_maxdiff"],
                unit_q_maxdiff=unit["q_maxdiff"], unit_wrong_head_q_maxdiff=unit["wrong_head_q_maxdiff"],
                unit_passed=unit["passed"],
            ))
            print(rows[-1], flush=True)

    with open(a.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print("\n| model | device | K | GFLOP | median ms | achieved GFLOP/s | aten ops | Q-subsample speedup | identical |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | {r['device']} | {r['num_samples']} | {r['analytic_GFLOP']} | {r['median_ms']} | "
              f"{r['achieved_GFLOPs']} | {r['aten_ops']} | {r['qsub_speedup']}× | {r['qsub_identical']} |")


if __name__ == "__main__":
    main()
