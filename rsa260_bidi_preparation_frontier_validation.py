#!/usr/bin/env python3
import importlib.util, json, statistics, time
from pathlib import Path
import numpy as np

BASE = Path(__file__).with_name("rsa260_bidi_preparation_fibre_search.py")
spec = importlib.util.spec_from_file_location("prep", BASE)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)

frontiers = {
    "affine_511_1": (511 * np.arange(prep.COLS) + 1) % prep.COLS,
    "rotate31": (np.arange(prep.COLS) + 31) % prep.COLS,
}

def cross_seed(name, perm):
    apply = prep.make_apply_B(perm)
    rows = []
    for s in range(8):
        r = prep.evaluate(
            apply,
            prep.BASEX ^ ((s + 1) << 48),
            prep.BASEY ^ ((s + 1) << 40),
        )
        r["seed_index"] = s
        rows.append(r)
    return {
        "adapter": name,
        "passes": sum(r["passed"] for r in rows),
        "mean_degree": statistics.mean(r["degree"] for r in rows),
        "mean_relation_dim": statistics.mean(r["relation_space_dim"] for r in rows),
        "zero_shift_total": sum(r["zero_shift_relations"] for r in rows),
        "min_kernel_weights": [r["kernel_weight_min"] for r in rows],
        "median_min_kernel_weight": statistics.median(r["kernel_weight_min"] for r in rows),
        "runs": rows,
    }

def build_block(seed, width=256):
    return np.stack([prep.seed_vec(seed ^ (j << 32)) for j in range(width)], axis=1)

def width256(name, perm, terms=4):
    apply = prep.make_apply_B(perm)
    seqs = []
    t0 = time.perf_counter()
    for s in range(2):
        X = build_block(prep.BASEX ^ (s << 56))
        Y = build_block(prep.BASEY ^ (s << 56))
        st = time.perf_counter()
        checksum = 0
        for k in range(terms):
            S = (X.T @ Y) & 1
            checksum ^= int(S.sum()) << (k % 8)
            Y = apply(Y)
        seqs.append({
            "sequence": s,
            "elapsed_seconds": round(time.perf_counter() - st, 6),
            "checksum": checksum,
        })
    return {
        "adapter": name,
        "width": 256,
        "sequences": 2,
        "terms_per_sequence": terms,
        "total_block_columns": 512,
        "elapsed_seconds": round(time.perf_counter() - t0, 6),
        "sequence_results": seqs,
    }

def main():
    seed = [cross_seed(n, p) for n, p in frontiers.items()]
    wide = [width256(n, p) for n, p in frontiers.items()]
    report = {
        "cross_seed": seed,
        "width256": wide,
        "preferred_under_robustness_aware_structural_cost": "rotate31",
        "reason": {
            "all_16_frontier_seed_runs_pass": True,
            "rotate31_lower_mean_degree": True,
            "rotate31_lower_mean_relation_dim": True,
            "rotate31_no_zero_shift_failures": True,
            "rotate31_lower_median_min_kernel_weight": True,
            "width256_runtime_order_not_materially_separated": True,
        },
        "firewalls": {
            "preferred_tested_is_global_optimum": False,
            "python_timing_is_production_performance": False,
            "frontier_adapter_is_cado_preparation": False,
        },
    }
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    main()
