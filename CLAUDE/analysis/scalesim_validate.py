"""Run SCALE-Sim v3 (cycle-accurate systolic array simulator) on a small set of
real GEMM shapes, to calibrate the fast analytical tiling model in systolic.py.

    pip install scalesim            # v3.0.0; see docs/09 for a numpy-2 patch
    cd CLAUDE && python analysis/scalesim_validate.py   # slow: minutes per case

Memory bandwidth is set to CALC (ideal), so reported cycles are compute-array
cycles only (fill/drain + folds). Memory stalls are modelled separately.
"""
import csv
import os
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results", "scalesim_validation.csv")

SHAPES = {                       # name: (M, N, K)  -- real layer shapes
    "pi0_expert_up": (51, 4096, 1024),       # Gemma-300M gate/up proj, 51 suffix tokens
    "pi0_expert_down": (51, 1024, 4096),
    "mid_512": (512, 1024, 1024),
    "llm_decode_gemv": (1, 4096, 4096),      # Llama-class decode, b=1
}
ARRAYS = [(32, 32), (128, 128)]
EXTRA = [("pi0_expert_up", 16, 64), ("pi0_expert_up", 64, 16)]   # non-square: separates R and C terms
# HOLDOUT (GPT review R8): never used for fitting; ragged shapes, small reductions, other array sizes
HOLDOUT = [("ragged_33x200x72", (33, 200, 72), 32, 32), ("small_k_17x1000x48", (17, 1000, 48), 16, 64),
           ("flexpi_act_32x3072x1024", (32, 3072, 1024), 64, 64), ("odd_130x77x300", (130, 77, 300), 48, 24)]
HOLDOUT_OUT = os.path.join(ROOT, "results", "scalesim_holdout.csv")
DATAFLOWS = ["os", "ws", "is"]

CFG = """[general]
run_name = {name}

[architecture_presets]
ArrayHeight:    {h}
ArrayWidth:     {w}
IfmapSramSzkB:   4096
FilterSramSzkB:  4096
OfmapSramSzkB:   4096
IfmapOffset:    0
FilterOffset:   10000000
OfmapOffset:    20000000
Bandwidth : 1024
Dataflow : {df}
MemoryBanks:   1
ReadRequestBuffer: 32
WriteRequestBuffer: 32

[layout]
IfmapCustomLayout: False
IfmapSRAMBankBandwidth: 10
IfmapSRAMBankNum: 10
IfmapSRAMBankPort: 2
FilterCustomLayout: False
FilterSRAMBankBandwidth: 10
FilterSRAMBankNum: 10
FilterSRAMBankPort: 2

[sparsity]
SparsitySupport : false
SparseRep : ellpack_block
OptimizedMapping : false
BlockSize : 8
RandomNumberGeneratorSeed : 40

[run_presets]
InterfaceBandwidth: CALC
UseRamulatorTrace: False
"""

RUNNER = """
import sys
from scalesim.scale_sim import scalesim
s = scalesim(save_disk_space=True, verbose=False, config=sys.argv[1], topology=sys.argv[2],
             layout=sys.argv[2], input_type_gemm=True)
s.run_scale(top_path=sys.argv[3])
"""


def run_one(name, M, N, K, h, w, df, timeout=1800):
    with tempfile.TemporaryDirectory() as d:
        cfg, top, runner = (os.path.join(d, f) for f in ("c.cfg", "t.csv", "run.py"))
        open(cfg, "w").write(CFG.format(name=name, h=h, w=w, df=df))
        open(top, "w").write(f"Layer,M,N,K,\n{name},{M},{N},{K},\n")
        open(runner, "w").write(RUNNER)
        subprocess.run([sys.executable, "-I", runner, cfg, top, os.path.join(d, "out")],
                       check=True, timeout=timeout, capture_output=True)
        rep = os.path.join(d, "out", name, "COMPUTE_REPORT.csv")
        row = list(csv.reader(open(rep)))[1]
        return dict(cycles=int(float(row[2])), util=float(row[4]), map_eff=float(row[5]))


def cases(only_extra=False):
    if not only_extra:
        for sname, (M, N, K) in SHAPES.items():
            for h, w in ARRAYS:
                yield sname, M, N, K, h, w
    for sname, h, w in EXTRA:
        yield (sname, *SHAPES[sname], h, w)


def run_holdout():
    done = set()
    if os.path.exists(HOLDOUT_OUT):
        done = {(r["shape"], r["array"], r["dataflow"]) for r in csv.DictReader(open(HOLDOUT_OUT))}
    new = not os.path.exists(HOLDOUT_OUT)
    with open(HOLDOUT_OUT, "a", newline="") as fh:
        wr = csv.DictWriter(fh, ["shape", "M", "N", "K", "array", "dataflow", "cycles", "util_pct", "map_eff_pct"])
        if new:
            wr.writeheader()
        for sname, (M, N, K), h, w in HOLDOUT:
            for df in DATAFLOWS:
                if (sname, f"{h}x{w}", df) in done:
                    continue
                r = run_one(sname, M, N, K, h, w, df)
                wr.writerow(dict(shape=sname, M=M, N=N, K=K, array=f"{h}x{w}", dataflow=df, cycles=r["cycles"],
                                 util_pct=round(r["util"], 3), map_eff_pct=round(r["map_eff"], 3)))
                fh.flush()
                print("HOLDOUT OK", sname, h, w, df, r, flush=True)


if __name__ == "__main__":
    if "--holdout" in sys.argv:
        run_holdout()
        sys.exit(0)
    only_extra = "--extra" in sys.argv
    done = set()
    if os.path.exists(OUT):
        done = {(r["shape"], r["array"], r["dataflow"]) for r in csv.DictReader(open(OUT))}
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as fh:
        wr = csv.DictWriter(fh, ["shape", "M", "N", "K", "array", "dataflow", "cycles", "util_pct", "map_eff_pct"])
        if new:
            wr.writeheader()
        for sname, M, N, K, h, w in cases(only_extra):
            if True:
                for df in DATAFLOWS:
                    key = (sname, f"{h}x{w}", df)
                    if key in done:
                        continue
                    try:
                        r = run_one(sname, M, N, K, h, w, df)
                    except Exception as e:  # noqa: BLE001
                        print("FAIL", key, type(e).__name__, flush=True)
                        continue
                    wr.writerow(dict(shape=sname, M=M, N=N, K=K, array=f"{h}x{w}", dataflow=df,
                                     cycles=r["cycles"], util_pct=round(r["util"], 3),
                                     map_eff_pct=round(r["map_eff"], 3)))
                    fh.flush()
                    print("OK", key, r, flush=True)
