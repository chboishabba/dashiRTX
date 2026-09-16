#!/usr/bin/env python3
from pathlib import Path
import importlib.util, json, statistics, time
import numpy as np

BASE = Path(__file__).with_name("rsa260_bidi_preparation_fibre_search.py")
spec = importlib.util.spec_from_file_location("prep", BASE)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)

ADAPTERS = {
    "rotate29": (np.arange(prep.COLS) + 29) % prep.COLS,
    "rotate31": (np.arange(prep.COLS) + 31) % prep.COLS,
}

def cross_seed(name, perm):
    apply = prep.make_apply_B(perm)
    runs = []
    for s in range(8):
        r = prep.evaluate(
            apply,
            prep.BASEX ^ ((s + 1) << 48),
            prep.BASEY ^ ((s + 1) << 40),
        )
        r["seed_index"] = s
        runs.append(r)
    return {
        "adapter": name,
        "passes": sum(r["passed"] for r in runs),
        "mean_degree": statistics.mean(r["degree"] for r in runs),
        "mean_relation_dim": statistics.mean(r["relation_space_dim"] for r in runs),
        "zero_shift_total": sum(r["zero_shift_relations"] for r in runs),
        "median_min_kernel_weight": statistics.median(
            r["kernel_weight_min"] for r in runs
        ),
        "min_kernel_weights": [r["kernel_weight_min"] for r in runs],
        "runs": runs,
    }

def build_block(seed, width=256):
    return np.stack(
        [prep.seed_vec(seed ^ (j << 32)) for j in range(width)],
        axis=1,
    )

def width256(name, perm, terms=4):
    apply = prep.make_apply_B(perm)
    t0 = time.perf_counter()
    seqs = []
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
            "seconds": round(time.perf_counter() - st, 6),
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
    cross = [cross_seed(n, p) for n, p in ADAPTERS.items()]
    wide = [width256(n, p) for n, p in ADAPTERS.items()]
    by = {r["adapter"]: r for r in cross}
    preference = (
        by["rotate29"]["passes"] == 8
        and by["rotate31"]["passes"] == 8
        and by["rotate29"]["mean_degree"] < by["rotate31"]["mean_degree"]
        and by["rotate29"]["mean_relation_dim"] < by["rotate31"]["mean_relation_dim"]
        and by["rotate29"]["zero_shift_total"] <= by["rotate31"]["zero_shift_total"]
        and by["rotate29"]["median_min_kernel_weight"]
            < by["rotate31"]["median_min_kernel_weight"]
    )
    report = {
        "cross_seed": cross,
        "width256": wide,
        "preferred_tested_adapter": "rotate29" if preference else "unresolved",
        "firewalls": {
            "preferred_tested_is_global_optimum": False,
            "preferred_tested_is_cado_preparation": False,
            "python_timing_is_production_performance": False,
        },
    }
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    main()
