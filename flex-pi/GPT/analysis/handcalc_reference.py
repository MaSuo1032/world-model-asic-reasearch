"""Scalar teaching model: 32 action rows, 2 channels, 1 head/layer, 4 Euler steps.

All inputs and weights are disclosed here and in the output JSON. This is NOT
the pretrained Flex-pi model. The forward calculation uses only the Python
standard library. --check adds an independently vectorized NumPy comparison.
"""
import argparse
import json
import math
from pathlib import Path

EPS = 1e-6
WEIGHTS = {
    "enc": ([[1., .2], [-.1, .8]], [.05, -.02]),
    "q": ([[1., .1], [.2, .9]], [.01, -.03]),
    "k": ([[.7, -.2], [.3, 1.]], [-.02, .04]),
    "v": ([[.5, .2], [-.3, .8]], [.03, -.01]),
    "o": ([[.8, -.1], [.2, .6]], [0., .02]),
    "cq": ([[.6, .1], [-.2, .7]], [.02, 0.]),
    "ck": ([[.9, -.1], [.1, .5]], [0., -.02]),
    "cv": ([[.4, .3], [-.1, .6]], [.01, .02]),
    "co": ([[.7, .2], [-.2, .5]], [-.01, 0.]),
    "ff1": ([[.5, -.2], [.1, .6], [-.4, .3]], [.01, -.02, .03]),
    "ff2": ([[.3, -.1, .2], [-.2, .4, .1]], [0., .01]),
    "head": ([[.9, -.2], [.1, .7]], [.02, -.03]),
}
RAW_ANCHORS = [[1., 0.], [0., 1.], [-1., 1.]]
ANCHOR_ANGLES = [0., .5, 1.]
TEXT = [.3, -.2]
STATE_RAW = [.5, -.4]
STATE_MEAN = [.1, -.2]
STATE_STD = [.2, .4]
ACTION_MEAN = [1., -.5]
ACTION_STD = [.2, .4]
NOISE = [[.2 + .01*i, -.1 + .005*i] for i in range(32)]


def linear(x, key):
    w, b = WEIGHTS[key]
    return [b[o] + sum(w[o][d]*x[d] for d in range(len(x))) for o in range(len(w))]


def ln(x, affine=False):
    mean = sum(x)/len(x)
    var = sum((a-mean)**2 for a in x)/len(x)
    out = [(a-mean)/math.sqrt(var+EPS) for a in x]
    if affine:  # explicit norm3 affine parameters
        out = [1.1*out[0]+.01, .9*out[1]-.02]
    return out


def rms(x):
    den = math.sqrt(sum(a*a for a in x)/len(x)+EPS)
    return [a/den for a in x]  # RMS weights [1,1]


def rope(x, angle):
    c, s = math.cos(angle), math.sin(angle)
    return [x[0]*c-x[1]*s, x[0]*s+x[1]*c]


def gelu(x):
    return .5*x*(1+math.tanh(math.sqrt(2/math.pi)*(x+.044715*x**3)))


def attend(q, keys, values):
    scores = [sum(a*b for a, b in zip(q, key))/math.sqrt(2) for key in keys]
    m = max(scores)
    exps = [math.exp(s-m) for s in scores]
    probs = [e/sum(exps) for e in exps]
    out = [sum(p*v[d] for p, v in zip(probs, values)) for d in range(2)]
    return out, scores, probs


def attention_io(rows, angles, sigma):
    norm = [ln(x) for x in rows]
    # The toy time map is fully specified; no unspecified timestep weights.
    mod = [[a*(1+.1*sigma) for a in x] for x in norm]
    qr = [linear(x, "q") for x in mod]
    kr = [linear(x, "k") for x in mod]
    q = [rope(rms(x), a) for x, a in zip(qr, angles)]
    k = [rope(rms(x), a) for x, a in zip(kr, angles)]
    v = [linear(x, "v") for x in mod]
    return q, k, v, norm, mod, qr, kr


def block(rows, angles, sigma, context, cached=None):
    q, k, v, norm, mod, qr, kr = attention_io(rows, angles, sigma)
    all_k = (cached[0] if cached else []) + k
    all_v = (cached[1] if cached else []) + v
    ck = [rms(linear(x, "ck")) for x in context]
    cv = [linear(x, "cv") for x in context]
    out, traces = [], []
    for i, x in enumerate(rows):
        sa, scores, probs = attend(q[i], all_k, all_v)
        sap = linear(sa, "o")
        y = [a+(.5+.1*sigma)*b for a, b in zip(x, sap)]
        n3 = ln(y, affine=True)
        cq = rms(linear(n3, "cq"))
        ca, cs, cp = attend(cq, ck, cv)  # no RoPE in cross-attention
        cap = linear(ca, "co")
        yc = [a+b for a, b in zip(y, cap)]
        u = [a*(1-.05*sigma)+.02*sigma for a in ln(yc)]
        hpre = linear(u, "ff1")
        h = [gelu(a) for a in hpre]
        ff = linear(h, "ff2")
        xn = [a+(.25+.1*sigma)*b for a, b in zip(yc, ff)]
        out.append(xn)
        traces.append(dict(input=x, ln1=norm[i], modulated=mod[i], q_raw=qr[i],
                           k_raw=kr[i], q=q[i], k=k[i], v=v[i], sa_scores=scores,
                           sa_probs=probs, sa=sa, sa_projected=sap, y=y,
                           norm3=n3, cq=cq, ck=ck, cv=cv, cross_scores=cs,
                           cross_probs=cp, cross=ca, cross_projected=cap, yc=yc,
                           ff_input=u, ff_pre=hpre, ff_hidden=h, ff=ff, output=xn))
    return out, (k, v), traces


def scalar_run():
    state = [max(-5., min(5., (s-m)/(v+1e-8)))
             for s, m, v in zip(STATE_RAW, STATE_MEAN, STATE_STD)]
    context = [TEXT, state]  # toy state encoder/context projectors are identity
    _, cache, pre_trace = block(RAW_ANCHORS, ANCHOR_ANGLES, 0., context)
    current = [row[:] for row in NOISE]
    steps = []
    for sigma in [1., .75, .5, .25]:
        before = [row[:] for row in current]
        hidden = [linear(x, "enc") for x in current]
        final_hidden, _, trace = block(hidden, list(range(32)), sigma, context, cache)
        velocities = [linear(x, "head") for x in final_hidden]
        current = [[a-.25*v for a, v in zip(x, vel)] for x, vel in zip(current, velocities)]
        steps.append(dict(sigma=sigma, t=1000*sigma, delta=-.25, before=before,
                          first_action_trace=trace[0], velocities=velocities, after=current))
    physical = [[a*(s+1e-8)+m for a, s, m in zip(x, ACTION_STD, ACTION_MEAN)] for x in current]
    return dict(kind="teaching model, not pretrained Flex-pi inference",
                dimensions=dict(actions=32, action_dim=2, hidden_dim=2, heads=1, layers=1),
                weights=WEIGHTS, raw_anchors=RAW_ANCHORS, anchor_angles=ANCHOR_ANGLES,
                state_raw=STATE_RAW, state_mean=STATE_MEAN, state_std=STATE_STD,
                context=context, initial_actions=NOISE, cache_k=cache[0], cache_v=cache[1],
                prefill_first_trace=pre_trace[0], steps=steps, normalized_actions=current,
                action_mean=ACTION_MEAN, action_std=ACTION_STD, physical_actions=physical)


def numpy_run():
    """Independent whole-matrix implementation, with separate primitive code."""
    import numpy as np

    def lin(x, key):
        w, b = WEIGHTS[key]
        return x @ np.asarray(w).T + np.asarray(b)

    def norm(x, affine=False):
        y = (x-x.mean(axis=-1, keepdims=True))/np.sqrt(x.var(axis=-1, keepdims=True)+EPS)
        return y*np.array([1.1, .9])+np.array([.01, -.02]) if affine else y

    def rn(x):
        return x/np.sqrt((x*x).mean(axis=-1, keepdims=True)+EPS)

    def rot(x, angles):
        angles = np.asarray(angles)
        c, s = np.cos(angles), np.sin(angles)
        return np.stack([x[:, 0]*c-x[:, 1]*s, x[:, 0]*s+x[:, 1]*c], axis=-1)

    def att(q, k, v):
        z = q @ k.T / np.sqrt(2)
        p = np.exp(z-z.max(axis=-1, keepdims=True))
        p /= p.sum(axis=-1, keepdims=True)
        return p @ v

    state = np.clip((np.array(STATE_RAW)-STATE_MEAN)/(np.array(STATE_STD)+1e-8), -5, 5)
    ctx = np.stack([TEXT, state])

    def blk(x, angles, sigma, cached=None):
        u = norm(x)*(1+.1*sigma)
        q, k = rot(rn(lin(u, "q")), angles), rot(rn(lin(u, "k")), angles)
        v = lin(u, "v")
        ak, av = (np.concatenate([cached[0], k]), np.concatenate([cached[1], v])) if cached else (k, v)
        y = x+(.5+.1*sigma)*lin(att(q, ak, av), "o")
        yc = y+lin(att(rn(lin(norm(y, True), "cq")), rn(lin(ctx, "ck")), lin(ctx, "cv")), "co")
        f = lin(norm(yc)*(1-.05*sigma)+.02*sigma, "ff1")
        g = .5*f*(1+np.tanh(np.sqrt(2/np.pi)*(f+.044715*f**3)))
        return yc+(.25+.1*sigma)*lin(g, "ff2"), (k, v)

    _, cache = blk(np.array(RAW_ANCHORS), ANCHOR_ANGLES, 0.)
    x = np.array(NOISE)
    all_after, all_vel = [], []
    for sigma in [1., .75, .5, .25]:
        h, _ = blk(lin(x, "enc"), np.arange(32), sigma, cache)
        vel = lin(h, "head")
        x = x-.25*vel
        all_after.append(x.copy())
        all_vel.append(vel)
    return all_after, all_vel, x*(np.array(ACTION_STD)+1e-8)+ACTION_MEAN


def verify(result):
    import numpy as np
    after, vel, physical = numpy_run()
    errors = []
    for i in range(4):
        errors.append(float(np.max(np.abs(np.asarray(result["steps"][i]["after"])-after[i]))))
        errors.append(float(np.max(np.abs(np.asarray(result["steps"][i]["velocities"])-vel[i]))))
    errors.append(float(np.max(np.abs(np.asarray(result["physical_actions"])-physical))))
    assert max(errors) < 1e-12, errors
    # Distinguish key permutation from mismatched K/V: matching pairs preserve
    # attention; shuffling only V changes it. This guards against index mistakes.
    q = [.4, -.2]
    k = [[1., 0.], [0., 1.], [-.5, .3]]
    v = [[2., 1.], [-1., 3.], [.5, -2.]]
    base = attend(q, k, v)[0]
    order = [2, 0, 1]
    shuffled = attend(q, [k[i] for i in order], [v[i] for i in order])[0]
    mismatched = attend(q, k, [v[i] for i in order])[0]
    assert max(abs(a-b) for a, b in zip(base, shuffled)) < 1e-14
    assert max(abs(a-b) for a, b in zip(base, mismatched)) > .01
    # A normalized action is not the physical output.
    assert not np.allclose(result["normalized_actions"], result["physical_actions"])
    return dict(independent_numpy_max_abs_error=max(errors), compared="all 32x2 velocities and actions, all 4 rounds, physical outputs",
                matched_kv_permutation="passed", mismatched_kv_negative_control="passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="requires NumPy; compare independent vectorized implementation")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1]/"results/handcalc_reference.json")
    args = ap.parse_args()
    result = scalar_run()
    if args.check:
        result["verification"] = verify(result)
        print(json.dumps(result["verification"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    for i, row in enumerate(result["physical_actions"]):
        print(f"{i:2d}  {row[0]:.9f}  {row[1]:.9f}")


if __name__ == "__main__":
    main()
