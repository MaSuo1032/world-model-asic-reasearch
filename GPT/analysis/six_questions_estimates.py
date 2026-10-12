"""Analytical toy examples, not checkpoint counts or hardware measurements."""
import json
from pathlib import Path


def calculate():
    layers, width = 24, 1024
    params = 12 * layers * width ** 2
    transformer = {
        "parameters_stack_only": params,
        "bf16_weight_bytes": 2 * params,
        "dense_256_flops": 2 * params * 256 + 4 * layers * 256 ** 2 * width,
        "dense_256_ffn_flops": 16 * layers * 256 * width ** 2,
        "dense_256_attention_projections_flops": 8 * layers * 256 * width ** 2,
        "dense_256_attention_pairs_flops": 4 * layers * 256 ** 2 * width,
        "cached_one_query_256_keys_flops": 2 * params + 4 * layers * 256 * width,
        "dense_512_flops": 2 * params * 512 + 4 * layers * 512 ** 2 * width,
    }
    enc = 64 * 512 + 512 * 512
    dyn = 518 * 512 + 512 * 512 + 512 * 512
    reward = 518 * 512 + 512 * 512 + 512 * 101
    policy = 512 * 512 + 512 * 512 + 512 * 12
    q = 5 * reward
    step, terminal = dyn + reward, policy + q
    mlp = {
        "parameters_encoder": enc,
        "parameters_dynamics": dyn,
        "parameters_reward": reward,
        "parameters_policy": policy,
        "parameters_q_ensemble": q,
        "parameters_total": enc + step + terminal,
        "encode_one_observation_flops": 2 * enc,
        "transition_reward_one_candidate_flops": 2 * step,
        "terminal_one_candidate_flops": 2 * terminal,
        "decision_K512_H3_I6_flops": 2 * enc + 6 * 512 * (3 * 2 * step + 2 * terminal),
        "bf16_cold_resident_weight_bytes": 2 * (enc + step + terminal),
        "bf16_nonresident_decision_weight_bytes": 2 * enc + 6 * (3 * 2 * step + 2 * terminal),
    }
    return {"status": "Analytical examples only; see GPT/08 for assumptions and exclusions.",
            "transformer": transformer, "mlp": mlp}


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[1] / "results/six-questions-estimates.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(calculate(), indent=2) + "\n", encoding="utf-8", newline="\n")
    print(output)
