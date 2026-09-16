#!/usr/bin/env python3
"""
RSA-260 bidi: endogenous fine-incidence predictor probe under fixed CADO-shaped
4x4 preparation and fixed projection seed.

Source lineage:
- Eric Lu, "Factoring RSA-260", Cognition (2026-09-09), primary execution source.
- Don Coppersmith, Block Wiedemann, DOI 10.1090/S0025-5718-1994-1192970-7.
- Emmanuel Thome, vector generating polynomials / Block Wiedemann,
  DOI 10.1006/jsco.2002.0533.
- CADO-NFS BWC README mirror snapshot:
  MichaelBell/cado-nfs @ 792169c33c26d05a4f7c90e91584285133ea5c55,
  README blob 34b029d2ab166fa2d400dd13351e2d31adc8952d.

The measured structural fibres are derived from the candidate carrier itself:
  adjacency(A) = mean_r |S_r intersect (S_r + 1)|
  translation(A) = mean_r max_s |(S_r+s) intersect S_{r+1}|

They are candidate predictors, not proofs of causal mechanism or production
RSA-260 structure.
"""
from pathlib import Path
import json
import time
import numpy as np

BASE = Path(__file__).with_name("rsa260_bidi_candidate_robustness.py")
source = BASE.read_text()
prefix = source.split("adapters = {", 1)[0]
env = {}
exec(compile(prefix, str(BASE) + ":prefix", "exec"), env)

ROWS, COLS = env["ROWS"], env["COLS"]

def build_cyclic(step, offset=441):
    A = np.zeros((ROWS, COLS), dtype=np.uint8)
    for r in range(ROWS):
        degree = 151 if r < 6 else 150
        base = (r * step + offset) % COLS
        A[r, (base + np.arange(degree)) % COLS] = 1
    return A

def build_random(seed):
    rng = np.random.default_rng(seed)
    A = np.zeros((ROWS, COLS), dtype=np.uint8)
    for r in range(ROWS):
        degree = 151 if r < 6 else 150
        A[r, rng.choice(COLS, size=degree, replace=False)] = 1
    return A

def cado_grid_permutation(nh, nv):
    nz = COLS // (nh * nv)
    p = np.empty(COLS, dtype=int)
    for x in range(COLS):
        q, k = divmod(x, nz)
        i, j = divmod(q, nv)
        p[x] = (j * nh + i) * nz + k
    return p

def cyclic_adjacency_mean(A):
    return float(np.mean(np.sum(A * np.roll(A, -1, axis=1), axis=1)))

def best_translation_overlap_mean(A, sample_rows=64):
    values = []
    for r in range(min(sample_rows, ROWS - 1)):
        a, b = A[r], A[r + 1]
        values.append(max(int(np.sum(np.roll(a, s) * b)) for s in range(COLS)))
    return float(np.mean(values))

def main():
    perm = cado_grid_permutation(4, 4)
    original_A = env["A"]
    cases = [
        ("cyclic_step_433", build_cyclic(433), 40),
        ("cyclic_step_431", build_cyclic(431), 40),
        ("cyclic_step_251", build_cyclic(251), 40),
        ("cyclic_step_127", build_cyclic(127), 40),
        ("random_260001", build_random(260001), 80),
        ("random_260003", build_random(260003), 80),
        ("random_260007", build_random(260007), 80),
        ("random_260011", build_random(260011), 80),
    ]
    out = []
    started = time.perf_counter()
    try:
        for label, A, max_degree in cases:
            env["A"] = A
            env["MAXD"] = max_degree
            apply = env["make_apply_B"](perm)
            r = env["evaluate"](
                apply,
                env["BASEX"] ^ (1 << 48),
                env["BASEY"] ^ (1 << 40),
            )
            out.append({
                "carrier": label,
                "cyclic_adjacency_mean": cyclic_adjacency_mean(A),
                "best_adjacent_row_translation_overlap_mean":
                    best_translation_overlap_mean(A),
                "generator_degree": r.get("degree"),
                "relation_space_dim": r.get("relation_space_dim"),
                "kernel_weight_min": r.get("kernel_weight_min"),
                "consumer_pass": r.get("passed"),
            })
    finally:
        env["A"] = original_A

    report = {
        "fixed_preparation": "CADO-shaped 4x4 block-grid transpose analogue",
        "fixed_projection_seed_index": 0,
        "cases": out,
        "interpretation": {
            "tested_family_level_separation": True,
            "predictor_is_complete_degree_model": False,
            "predictor_is_causal_proof": False,
            "predictor_is_production_measurement": False,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    main()
