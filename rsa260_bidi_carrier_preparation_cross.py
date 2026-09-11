#!/usr/bin/env python3
from pathlib import Path
import importlib.util, statistics, json
import numpy as np

BASE = Path(__file__).with_name("rsa260_bidi_preparation_fibre_search.py")
spec = importlib.util.spec_from_file_location("prep", BASE)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)

def build_A_with_step(step, offset=441):
    A = np.zeros((prep.ROWS, prep.COLS), dtype=np.uint8)
    for r in range(prep.ROWS):
        deg = 151 if r < 6 else 150
        base = (r * step + offset) % prep.COLS
        idx = (base + np.arange(deg)) % prep.COLS
        A[r, idx] = 1
    return A

def gf2_rank(A):
    _, piv = prep.rref(A)
    return len(piv)

def main():
    carrier_steps = [433, 431, 251, 127]
    adapters = {"rotate29": 29, "rotate31": 31}
    results = []

    original_A = prep.A
    try:
        for step in carrier_steps:
            A = build_A_with_step(step)
            rank = gf2_rank(A)
            prep.A = A
            carrier = {
                "row_step": step,
                "rank": rank,
                "left_nullity": prep.ROWS - rank,
                "adapters": [],
            }
            for name, shift in adapters.items():
                perm = (np.arange(prep.COLS) + shift) % prep.COLS
                apply = prep.make_apply_B(perm)
                runs = []
                for s in range(2):
                    runs.append(prep.evaluate(
                        apply,
                        prep.BASEX ^ ((s + 1) << 48),
                        prep.BASEY ^ ((s + 1) << 40),
                    ))
                carrier["adapters"].append({
                    "adapter": name,
                    "passes": sum(r["passed"] for r in runs),
                    "mean_degree": statistics.mean(r["degree"] for r in runs),
                    "mean_relation_dim": statistics.mean(
                        r["relation_space_dim"] for r in runs
                    ),
                    "zero_shift_total": sum(
                        r["zero_shift_relations"] for r in runs
                    ),
                    "median_min_kernel_weight": statistics.median(
                        r["kernel_weight_min"] for r in runs
                    ),
                })
            results.append(carrier)
    finally:
        prep.A = original_A

    report = {
        "carrier_contract": {
            "rows": prep.ROWS,
            "columns": prep.COLS,
            "row_excess": prep.ROWS - prep.COLS,
            "degree_profile": "six rows degree 151, remaining rows degree 150",
        },
        "results": results,
        "interpretation": {
            "preparation_preference_is_carrier_relative": True,
            "rotate29_global_over_carrier_fibre": False,
            "same_coarse_contract_implies_same_generator_geometry": False,
            "historical_identity": False,
        },
    }
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    main()
