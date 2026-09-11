#!/usr/bin/env python3
"""
RSA-260 bidi experiment: CADO-shaped grid-permutation cross-validation.

Source coordinates:
- CADO-NFS BWC README mirror snapshot:
  repo: MichaelBell/cado-nfs
  commit: 792169c33c26d05a4f7c90e91584285133ea5c55
  README blob: 34b029d2ab166fa2d400dd13351e2d31adc8952d
  NOTE: the README itself warns it is not regularly checked to be up to date.
- Eric Lu, "Factoring RSA-260", Cognition, 2026-09-09:
  https://cognition.com/blog/factoring-rsa-260
- Don Coppersmith, "Solving homogeneous linear equations over GF(2) via
  block Wiedemann algorithm", DOI 10.1090/S0025-5718-1994-1192970-7
- Douglas H. Wiedemann, "Solving sparse linear equations over finite fields",
  DOI 10.1109/TIT.1986.1057137

The CADO README describes a block-grid transpose permutation after square
padding.  Here we test the corresponding permutation shape on the 512-state
intermediate carrier of the DASHI shadow experiment.  This is a source-derived
analogue, not a claim that the exact RSA-260 production balancing file has been
recovered.
"""
from pathlib import Path
import json, time
import numpy as np

BASE = Path(__file__).with_name("rsa260_bidi_candidate_robustness.py")
source = BASE.read_text()
prefix = source.split("adapters = {", 1)[0]
env = {}
exec(compile(prefix, str(BASE) + ":prefix", "exec"), env)

ROWS, COLS = env["ROWS"], env["COLS"]
BASEX, BASEY = env["BASEX"], env["BASEY"]

def cado_grid_permutation(nh, nv):
    if COLS % (nh * nv):
        raise ValueError("grid product must divide carrier width")
    nz = COLS // (nh * nv)
    p = np.empty(COLS, dtype=int)
    for x in range(COLS):
        q, k = divmod(x, nz)
        i, j = divmod(q, nv)
        p[x] = (j * nh + i) * nz + k
    return p

def build_random_exact_degree_A(seed):
    rng = np.random.default_rng(seed)
    A = np.zeros((ROWS, COLS), dtype=np.uint8)
    for r in range(ROWS):
        degree = 151 if r < 6 else 150
        A[r, rng.choice(COLS, size=degree, replace=False)] = 1
    return A

def run_eval(A, perm, seed_index, max_degree):
    env["A"] = A
    env["MAXD"] = max_degree
    apply = env["make_apply_B"](perm)
    return env["evaluate"](
        apply,
        BASEX ^ ((seed_index + 1) << 48),
        BASEY ^ ((seed_index + 1) << 40),
    )

def main():
    original_A = env["A"]
    grids = [(2,2), (4,4), (8,4), (16,4), (4,8)]
    canonical = []
    randomized = []
    started = time.perf_counter()
    try:
        for nh, nv in grids:
            perm = cado_grid_permutation(nh, nv)
            runs = [run_eval(original_A, perm, s, 40) for s in range(2)]
            canonical.append({
                "grid": f"{nh}x{nv}",
                "runs": runs,
                "passes": sum(bool(r.get("passed")) for r in runs),
            })

        for carrier_seed in [260001, 260003]:
            A = build_random_exact_degree_A(carrier_seed)
            for nh, nv in [(4,4), (8,4)]:
                perm = cado_grid_permutation(nh, nv)
                r = run_eval(A, perm, 0, 80)
                randomized.append({
                    "carrier_seed": carrier_seed,
                    "grid": f"{nh}x{nv}",
                    "result": r,
                })
    finally:
        env["A"] = original_A

    report = {
        "carrier_shape": [ROWS, COLS],
        "canonical_cyclic_carrier": canonical,
        "random_fine_incidence_carriers": randomized,
        "summary": {
            "canonical_runs": len(grids) * 2,
            "canonical_passes": sum(x["passes"] for x in canonical),
            "randomized_runs": len(randomized),
            "randomized_passes": sum(
                bool(x["result"].get("passed")) for x in randomized
            ),
            "randomized_generator_degrees": [
                x["result"].get("degree") for x in randomized
            ],
        },
        "firewalls": {
            "source_derived_grid_shape_is_exact_production_balancing_file": False,
            "mirror_readme_is_exact_rsa260_revision": False,
            "grid_permutation_explains_random_incidence_complexity": False,
            "consumer_success_implies_historical_identity": False,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    main()
