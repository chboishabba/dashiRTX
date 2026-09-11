#!/usr/bin/env python3
"""
RSA-260 bidi fine-incidence defect-coverage experiment.

Snowball/source coordinates:
- Eric Lu, "Factoring RSA-260", Cognition, 2026-09-09.
  https://cognition.com/blog/factoring-rsa-260
  Role: primary first-party RSA-260 LA execution envelope.
- Don Coppersmith, "Solving homogeneous linear equations over GF(2) via block
  Wiedemann algorithm", DOI 10.1090/S0025-5718-1994-1192970-7.
- Emmanuel Thome, "Subquadratic Computation of Vector Generating Polynomials
  and Improvement of the Block Wiedemann Algorithm",
  DOI 10.1006/jsco.2002.0533.
- CADO-NFS BWC implementation-schema mirror snapshot is owned by the Agda
  attribution layer.  This experiment uses the already-admitted 4x4 block-grid
  transpose analogue as its fixed preparation.

The experiment preserves:
  924 x 512 carrier shape,
  exact 150/151 row degree profile,
  full rank 512 / left-nullity 412.

It perturbs exactly one support coordinate in an increasing number of rows and
measures the first held-out-valid shared generator degree.  This is a synthetic
candidate experiment, not production RSA-260 incidence measurement.
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
BASEX, BASEY = env["BASEX"], env["BASEY"]

def build_cyclic():
    A = np.zeros((ROWS, COLS), dtype=np.uint8)
    for r in range(ROWS):
        degree = 151 if r < 6 else 150
        base = (r * 2654435761 + 0x9e3779b9) % COLS
        A[r, (base + np.arange(degree)) % COLS] = 1
    return A

def perturb_rows(count, seed):
    A = build_cyclic()
    rng = np.random.default_rng(seed)
    touched = rng.choice(ROWS, size=count, replace=False) if count else []
    for r in touched:
        support = np.flatnonzero(A[r])
        complement = np.flatnonzero(1 - A[r])
        remove = int(rng.choice(support))
        add = int(rng.choice(complement))
        A[r, remove] = 0
        A[r, add] = 1
    return A

def gf2_rank(M):
    _, piv = env["rref"](M)
    return len(piv)

def cado_grid_permutation(nh=4, nv=4):
    nz = COLS // (nh * nv)
    p = np.empty(COLS, dtype=int)
    for x in range(COLS):
        q, k = divmod(x, nz)
        i, j = divmod(q, nv)
        p[x] = (j * nh + i) * nz + k
    return p

def adjacency_mean(A):
    return float(np.mean(np.sum(A * np.roll(A, -1, axis=1), axis=1)))

def evaluate_candidate(A):
    env["A"] = A
    env["MAXD"] = 80
    apply = env["make_apply_B"](cado_grid_permutation())
    return env["evaluate"](
        apply,
        BASEX ^ (1 << 48),
        BASEY ^ (1 << 40),
    )

def main():
    counts = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 924]
    original_A = env["A"]
    rows = []
    started = time.perf_counter()
    try:
        for count in counts:
            A = perturb_rows(count, 270000 + count)
            result = evaluate_candidate(A)
            rows.append({
                "touched_rows": count,
                "fraction_rows": count / ROWS,
                "rank": gf2_rank(A),
                "left_nullity": ROWS - gf2_rank(A),
                "adjacency_mean": adjacency_mean(A),
                **result,
            })
    finally:
        env["A"] = original_A

    report = {
        "fixed_preparation": "CADO-shaped 4x4 block-grid transpose analogue",
        "fixed_projection_seed_index": 0,
        "carrier_shape": [ROWS, COLS],
        "row_excess": ROWS - COLS,
        "one_support_replacement_per_touched_row": True,
        "rows": rows,
        "summary": {
            "all_full_rank": all(r["rank"] == COLS for r in rows),
            "all_left_nullity_412": all(r["left_nullity"] == 412 for r in rows),
            "all_consumers_pass": all(r.get("passed") for r in rows),
            "baseline_degree": rows[0]["degree"],
            "one_row_degree": rows[1]["degree"],
            "two_row_degree": rows[2]["degree"],
            "rows64_degree": next(r["degree"] for r in rows if r["touched_rows"] == 64),
            "rows128_degree": next(r["degree"] for r in rows if r["touched_rows"] == 128),
            "rows256_degree": next(r["degree"] for r in rows if r["touched_rows"] == 256),
            "all_rows_degree": rows[-1]["degree"],
        },
        "interpretation": {
            "defect_coverage_changes_recurrence_complexity": True,
            "single_local_defect_destroys_compressibility": False,
            "coarse_rank_nullity_explains_degree_curve": False,
            "synthetic_curve_is_production_measurement": False,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    main()
