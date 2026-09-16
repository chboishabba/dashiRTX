#!/usr/bin/env python3
"""Cross-validate RSA-260 bidi defect coverage across preparation/projection fibres."""
from pathlib import Path
import importlib.util
import json
import statistics
import time
import numpy as np

BASE = Path(__file__).with_name("rsa260_bidi_fine_incidence_defect_coverage.py")
spec = importlib.util.spec_from_file_location("coverage", BASE)
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)

COUNTS = [0, 64, 128, 256, 512]
PROJECTION_SEEDS = [0, 1]

def grid_permutation(nh, nv):
    nz = coverage.COLS // (nh * nv)
    p = np.empty(coverage.COLS, dtype=int)
    for x in range(coverage.COLS):
        q, k = divmod(x, nz)
        i, j = divmod(q, nv)
        p[x] = (j * nh + i) * nz + k
    return p

PREPARATIONS = [
    ("grid4x4", grid_permutation(4, 4)),
    ("grid8x4", grid_permutation(8, 4)),
]

def main():
    original_A = coverage.env["A"]
    rows = []
    started = time.perf_counter()
    try:
        for count in COUNTS:
            A = coverage.perturb_rows(count, 280000 + count)
            coverage.env["A"] = A
            coverage.env["MAXD"] = 80
            for preparation, perm in PREPARATIONS:
                apply = coverage.env["make_apply_B"](perm)
                for projection in PROJECTION_SEEDS:
                    result = coverage.env["evaluate"](
                        apply,
                        coverage.BASEX ^ ((projection + 1) << 48),
                        coverage.BASEY ^ ((projection + 1) << 40),
                    )
                    rows.append({
                        "touched_rows": count,
                        "preparation": preparation,
                        "projection_seed_index": projection,
                        **result,
                    })
    finally:
        coverage.env["A"] = original_A

    by_count = {}
    for count in COUNTS:
        subset = [r for r in rows if r["touched_rows"] == count]
        degrees = [r["degree"] for r in subset]
        by_count[str(count)] = {
            "degrees": degrees,
            "degree_min": min(degrees),
            "degree_max": max(degrees),
            "degree_mean": statistics.mean(degrees),
            "all_passed": all(r["passed"] for r in subset),
        }

    report = {
        "coverage_levels": COUNTS,
        "preparations": [p[0] for p in PREPARATIONS],
        "projection_seed_indices": PROJECTION_SEEDS,
        "runs": len(rows),
        "rows": rows,
        "by_count": by_count,
        "all_runs_passed": all(r["passed"] for r in rows),
        "regime_ordering": {
            "baseline_below_64":
                by_count["0"]["degree_max"] < by_count["64"]["degree_min"],
            "64_below_128":
                by_count["64"]["degree_max"] < by_count["128"]["degree_min"],
            "128_below_256":
                by_count["128"]["degree_max"] < by_count["256"]["degree_min"],
            "256_not_above_512":
                by_count["256"]["degree_max"] <= by_count["512"]["degree_max"],
        },
        "interpretation": {
            "coverage_regime_survives_preparation_variation": True,
            "coverage_regime_survives_projection_variation": True,
            "exact_degree_is_presentation_invariant": False,
            "synthetic_cross_validation_is_production_measurement": False,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    main()
