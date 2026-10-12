"""Read-only numerical review probes; does not execute or alter CLAUDE benchmarks.

The grouped-head schedule uses the SAME ideal fold-cost assumptions as the
reviewed model. It is a sensitivity/counterexample, not a validated accelerator.
"""
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run():
    spec = importlib.util.spec_from_file_location("reviewed_systolic", ROOT / "CLAUDE/analysis/systolic.py")
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    fit = model.fit_coeffs()
    cases = []
    for batch in (1, 64):
        layers, heads, r, c, cores = 32, 32, 32, 32, 16
        gemms = model.transformer_gemms(1, 8192, 4096, heads, 128, 8, 14336, layers, batch=batch)
        original, original_util = model.best_df(gemms, r, c, cores)
        grouped = 0
        for name, m, n, k, count in gemms:
            # Only batch/head instances within ONE layer may share cores.
            # Layers remain sequential. Linear GEMMs are unchanged.
            parallel = heads * batch if name in ("qk", "pv") else 1
            assert count % parallel == 0
            sequential = count // parallel
            alternatives = []
            for dataflow in ("os", "ws", "is"):
                sr, sc, stream = model.map_dims(m, n, k, dataflow)
                folds = math.ceil(sr / r) * math.ceil(sc / c)
                a, b, offset = model.COEF[dataflow]
                per_fold = stream + a * r + b * c + offset
                alternatives.append(sequential * math.ceil(parallel * folds / cores) * per_fold)
            grouped += min(alternatives)
        ideal = sum(model.ideal_cycles(m, n, k, r, c, cores) * count for _, m, n, k, count in gemms)
        assert grouped <= original
        cases.append({"batch": batch, "array": "16 x 32x32", "original_cycles": original,
                      "grouped_same_layer_heads_cycles": grouped,
                      "original_utilization": original_util,
                      "grouped_utilization": ideal / grouped})
    assert cases[1]["original_cycles"] == 459483648
    assert cases[1]["grouped_same_layer_heads_cycles"] == 298461696
    # No framework or checkpoint is involved: a logical witness showing why
    # equality on all-zero head logits cannot detect an incorrect head index.
    zero_logits = [[0.0] * 101 for _ in range(5)]
    first_pair = [zero_logits[i] for i in (0, 1)]
    wrong_pair = [zero_logits[i] for i in (3, 4)]
    assert first_pair == wrong_pair
    return {
        "reviewed_commit": "231df78150c3570c1ccb472a6aa53c03d8bab49f",
        "evidence": "Analytical recomputation and logical witness, not GPU or silicon measurements",
        "fit_on_existing_calibration_rows": fit,
        "same_layer_parallel_schedule": cases,
        "zero_head_wrong_indices_still_equal": first_pair == wrong_pair,
        "bf16_weight_capacity_MB": {"20M": 40, "200M": 400, "317M": 634},
        "candidate_state_MiB_K512_S256_D1024_BF16": 512 * 256 * 1024 * 2 / 2**20,
        "activation_example": "Whole slice is 256 MiB; tiling may reduce live SRAM, so this is not a minimum SRAM requirement",
    }


if __name__ == "__main__":
    output = ROOT / "GPT/results/review-20261012-checks.json"
    output.write_text(json.dumps(run(), indent=2) + "\n", encoding="utf-8", newline="\n")
    print(output)
