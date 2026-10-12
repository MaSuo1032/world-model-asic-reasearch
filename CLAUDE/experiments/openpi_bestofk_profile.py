"""E1 kit for flow VLAs: profile openpi pi0 / pi0.5 action sampling vs. number of
flow steps and best-of-K candidates that SHARE one prefix (ViT + VLM) pass.

Hypothesis under test (CLAUDE/docs/08 §8.3): at batch 1 the action expert is
weight-bandwidth-bound, so K=16 candidate chunks cost <= ~30% more than K=1.

Timing (real pi0 shapes, random weights -- latency does not depend on weight values):
    python openpi_bestofk_profile.py --openpi /path/openpi --steps 1 2 5 10 --K 1 4 16 64
    python openpi_bestofk_profile.py --openpi /path/openpi --pi05 --steps 10 --K 1 4 16 64

E3(i) step-to-step similarity (needs trained weights):
    gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_droid/params ./pi05_droid_params
    python openpi_bestofk_profile.py --openpi /path/openpi --pi05 --horizon 15 \
        --params ./pi05_droid_params --steps 10 --K 1 --similarity

See CLAUDE/experiments/colab_experiments.ipynb for a ready-to-run Colab version.

The shared-prefix sampler is a copy of Pi0.sample_actions (openpi commit 215abfb)
with the prefix KV cache tiled along the batch axis; built-in checks confirm it
reproduces sample_actions exactly.
"""
import argparse
import csv
import os
import statistics
import sys
import time


def load_model(a):
    """Real layer shapes; weights random (timing only) or restored from an official checkpoint."""
    import jax
    from openpi.models import pi0_config
    from openpi.models import model as _model
    kw = dict(paligemma_variant=a.vlm, action_expert_variant=a.expert, pi05=a.pi05)
    if a.horizon:
        kw["action_horizon"] = a.horizon
    cfg = pi0_config.Pi0Config(**kw)
    import jax.numpy as jnp
    if a.params:
        params = _model.restore_params(a.params, restore_type=jax.Array, dtype=jnp.bfloat16)
        return cfg, cfg.load(params)
    from flax import nnx
    model = cfg.create(jax.random.key(0))
    graphdef, state = nnx.split(model)            # cast random-init params to bf16 like real inference
    state = jax.tree.map(lambda x: x.astype(jnp.bfloat16) if jnp.issubdtype(x.dtype, jnp.floating) else x, state)
    return cfg, nnx.merge(graphdef, state)


def make_samplers(model):
    import einops
    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask

    def prefix(observation):
        observation = _model.preprocess_observation(None, observation, train=False)
        pt, pm, par = model.embed_prefix(observation)
        positions = jnp.cumsum(pm, axis=1) - 1
        _, kv = model.PaliGemma.llm([pt, None], mask=make_attn_mask(pm, par), positions=positions)
        return observation, pm, kv

    def loop(observation, prefix_mask, kv_cache, noise, num_steps):
        B = noise.shape[0]
        dt = -1.0 / num_steps

        def step(carry):
            x_t, time_ = carry
            st, sm, sar, adarms = model.embed_suffix(observation, x_t, jnp.broadcast_to(time_, B))
            s_mask = make_attn_mask(sm, sar)
            p_mask = einops.repeat(prefix_mask, "b p -> b s p", s=st.shape[1])
            full = jnp.concatenate([p_mask, s_mask], axis=-1)
            pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(sm, axis=-1) - 1
            (_, so), _ = model.PaliGemma.llm([None, st], mask=full, positions=pos, kv_cache=kv_cache,
                                            adarms_cond=[None, adarms])
            v_t = model.action_out_proj(so[:, -model.action_horizon:])
            return x_t + dt * v_t, time_ + dt

        x0, _ = jax.lax.while_loop(lambda c: c[1] >= -dt / 2, step, (noise, 1.0))
        return x0

    def best_of_k(observation, noise, num_steps):
        """One prefix pass, K = noise.shape[0] candidate chunks."""
        K = noise.shape[0]
        obs, pm, kv = prefix(observation)
        tile = lambda x: jnp.repeat(x, K, axis=0)  # noqa: E731
        obs_k = jax.tree.map(tile, obs)
        kv_k = jax.tree.map(lambda x: jnp.repeat(x, K, axis=1), kv)   # KVCache = [layers, batch, ...]
        return loop(obs_k, tile(pm), kv_k, noise, num_steps)

    def prefix_only(observation):
        return prefix(observation)[2]

    def step_trace(observation, noise, num_steps):
        """Python-loop version of the flow loop returning per-step expert outputs (for E3(i))."""
        obs, pm, kv = prefix(observation)
        B, dt = noise.shape[0], -1.0 / num_steps

        @jax.jit
        def one(x_t, time_):
            st, sm, sar, adarms = model.embed_suffix(obs, x_t, jnp.broadcast_to(time_, B))
            full = jnp.concatenate([einops.repeat(pm, "b p -> b s p", s=st.shape[1]), make_attn_mask(sm, sar)], -1)
            pos = jnp.sum(pm, axis=-1)[:, None] + jnp.cumsum(sm, axis=-1) - 1
            (_, so), _ = model.PaliGemma.llm([None, st], mask=full, positions=pos, kv_cache=kv,
                                            adarms_cond=[None, adarms])
            v_t = model.action_out_proj(so[:, -model.action_horizon:])
            return x_t + dt * v_t, so, v_t

        x, t, hs, vs = noise, 1.0, [], []
        for _ in range(num_steps):
            x, h, v = one(x, t)
            hs.append(h); vs.append(v); t += dt
        return hs, vs

    return best_of_k, prefix_only, step_trace


def bench(fn, args, reps, warmup):
    import jax
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openpi", required=True)
    ap.add_argument("--params", help="local dir of an openpi checkpoint's params/ (e.g. pi05_droid/params)")
    ap.add_argument("--horizon", type=int, help="action_horizon (pi05_droid=15, pi0 base=50)")
    ap.add_argument("--similarity", action="store_true",
                    help="E3(i): cosine similarity of expert outputs between consecutive flow steps")
    ap.add_argument("--vlm", default="gemma_2b")
    ap.add_argument("--expert", default="gemma_300m")
    ap.add_argument("--pi05", action="store_true")
    ap.add_argument("--steps", type=int, nargs="+", default=[10])
    ap.add_argument("--K", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default="openpi_bestofk_profile.csv")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.openpi, "src"))
    sys.path.insert(0, os.path.join(a.openpi, "packages", "openpi-client", "src"))

    import jax
    import jax.numpy as jnp
    import numpy as np
    cfg, model = load_model(a)
    model.eval() if hasattr(model, "eval") else None
    obs = cfg.fake_obs(batch_size=1)
    # replace the all-ones images with random pixels so attention patterns are not degenerate
    obs = obs.replace(images={k: jax.random.uniform(jax.random.key(i), v.shape, minval=-1, maxval=1)
                              for i, (k, v) in enumerate(obs.images.items())})
    H, D = model.action_horizon, model.action_dim
    best_of_k, prefix_only, step_trace = make_samplers(model)

    # ---- correctness checks (against the official sampler) ----
    rng = jax.random.key(1)
    noise1 = jax.random.normal(rng, (1, H, D))
    ref = model.sample_actions(rng, obs, num_steps=a.steps[0], noise=noise1)
    mine = best_of_k(obs, noise1, a.steps[0])
    err_k1 = float(jnp.max(jnp.abs(ref - mine)))
    noise4 = jax.random.normal(jax.random.key(2), (4, H, D))
    batch4 = best_of_k(obs, noise4, a.steps[0])
    err_rows = max(float(jnp.max(jnp.abs(batch4[i:i + 1] - model.sample_actions(rng, obs, num_steps=a.steps[0],
                                                                                    noise=noise4[i:i + 1]))))
                   for i in range(4))
    print(f"check K=1 vs sample_actions: max|diff|={err_k1:.3e};  check row-wise K=4: max|diff|={err_rows:.3e}")

    # ---- E3(i): step-to-step similarity of expert activations ----
    if a.similarity:
        hs, vs = step_trace(obs, noise1, a.steps[-1])
        cos = lambda x, y: float(jnp.sum(x * y) / (jnp.linalg.norm(x) * jnp.linalg.norm(y) + 1e-12))  # noqa: E731
        sim_rows = []
        for i in range(1, len(hs)):
            dh = (hs[i] - hs[i - 1]).astype(jnp.float32)
            rng_h = float(jnp.max(jnp.abs(hs[i - 1].astype(jnp.float32))))
            # Ditto-style: INT8-quantise with scale max|h|/127; delta fits signed 4-bit if |dq| <= 7
            frac4 = float(jnp.mean(jnp.abs(dh) <= 7 * rng_h / 127))
            sim_rows.append(dict(step=i, cos_hidden=round(cos(hs[i], hs[i - 1]), 4),
                                 cos_velocity=round(cos(vs[i], vs[i - 1]), 4), frac_delta_4bit=round(frac4, 4)))
            print(sim_rows[-1], flush=True)
        with open(os.path.splitext(a.out)[0] + "_similarity.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(sim_rows[0].keys()))
            wr.writeheader()
            wr.writerows(sim_rows)
        if a.K == [1] and a.steps == [a.steps[-1]] and a.reps == 0:
            return

    # ---- timing ----
    jit_prefix = jax.jit(prefix_only)
    t_prefix = bench(jit_prefix, (obs,), a.reps, a.warmup)
    rows = []
    for s in a.steps:
        f = jax.jit(best_of_k, static_argnums=2)
        base = None
        for K in a.K:
            noise = jax.random.normal(jax.random.key(K), (K, H, D))
            t = bench(f, (obs, noise, s), a.reps, a.warmup)
            base = base or t
            rows.append(dict(steps=s, K=K, total_ms=round(t * 1e3, 2), prefix_ms=round(t_prefix * 1e3, 2),
                             loop_ms=round((t - t_prefix) * 1e3, 2), rel_to_K1=round(t / base, 3),
                             device=str(jax.devices()[0]), check_k1=err_k1, check_rows=err_rows))
            print(rows[-1], flush=True)
    with open(a.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print("\n| steps | K | total ms | prefix ms | loop ms | vs K=1 |\n|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['steps']} | {r['K']} | {r['total_ms']} | {r['prefix_ms']} | {r['loop_ms']} | {r['rel_to_K1']}× |")


if __name__ == "__main__":
    main()
