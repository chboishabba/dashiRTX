#!/usr/bin/env python3
"""
Fit a held-out recurrence-complexity model from repo-native structural fibres.

Training labels come from the exact-byte defect-coverage seed-replication
experiment.  The held-out perturbation seed is executed here through the same
coverage consumer so its labels are not assumed.

Features are the existing two-hop/common-neighbour family:
row/column common-neighbour means, stds, entropies, adjacent/distance-2
overlaps, and profile-step L1 variation.

This is a diagnostic model, not a causal mechanism and not a production
RSA-260 predictor.
"""
from pathlib import Path
import importlib.util
import json
import numpy as np

COVERAGE = Path(__file__).with_name("rsa260_bidi_fine_incidence_defect_coverage.py")
spec = importlib.util.spec_from_file_location("coverage", COVERAGE)
coverage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage)

COUNTS = [32, 64, 128, 256, 512, 924]
TRAIN_SEEDS = [271001, 271003, 271007]
TRAIN_LABELS = {
    32: [24, 22, 23],
    64: [30, 29, 29],
    128: [44, 41, 44],
    256: [63, 62, 64],
    512: [66, 65, 66],
    924: [66, 66, 65],
}
HELDOUT_SEED = 271011
RIDGE_LAMBDA = 0.3

FEATURE_NAMES = [
    "row_common_mean",
    "row_common_std",
    "row_common_entropy",
    "row_adjacent",
    "row_distance2",
    "row_profile_step_l1",
    "col_common_mean",
    "col_common_std",
    "col_common_entropy",
    "col_adjacent",
    "col_distance2",
    "col_profile_step_l1",
]

def entropy(values):
    _, counts = np.unique(values, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())

def fibres(A):
    X = A.astype(np.int16)
    RR = X @ X.T
    CC = X.T @ X
    rr = RR[~np.eye(coverage.ROWS, dtype=bool)]
    cc = CC[~np.eye(coverage.COLS, dtype=bool)]
    rr_diff = np.abs(
        RR[1:].astype(np.int32) - RR[:-1].astype(np.int32)
    ).sum(axis=1)
    cc_diff = np.abs(
        CC[1:].astype(np.int32) - CC[:-1].astype(np.int32)
    ).sum(axis=1)
    return np.array([
        float(rr.mean()),
        float(rr.std()),
        entropy(rr),
        float(np.diag(RR, 1).mean()),
        float(np.diag(RR, 2).mean()),
        float(rr_diff.mean()),
        float(cc.mean()),
        float(cc.std()),
        entropy(cc),
        float(np.diag(CC, 1).mean()),
        float(np.diag(CC, 2).mean()),
        float(cc_diff.mean()),
    ])

def main():
    X_train, y_train = [], []
    for count in COUNTS:
        for seed, degree in zip(TRAIN_SEEDS, TRAIN_LABELS[count]):
            A = coverage.perturb_rows(count, seed + count)
            X_train.append(fibres(A))
            y_train.append(degree)

    X_train = np.array(X_train)
    y_train = np.array(y_train, dtype=float)
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X_train - mu) / sd
    coef = np.linalg.solve(
        Z.T @ Z + RIDGE_LAMBDA * np.eye(Z.shape[1]),
        Z.T @ (y_train - y_train.mean()),
    )

    heldout = []
    original_A = coverage.env["A"]
    try:
        for count in COUNTS:
            A = coverage.perturb_rows(count, HELDOUT_SEED + count)
            feature = fibres(A)
            coverage.env["A"] = A
            coverage.env["MAXD"] = 80
            apply = coverage.env["make_apply_B"](coverage.cado_grid_permutation())
            result = coverage.env["evaluate"](
                apply,
                coverage.BASEX ^ (1 << 48),
                coverage.BASEY ^ (1 << 40),
            )
            prediction = float(
                y_train.mean() + ((feature - mu) / sd) @ coef
            )
            heldout.append({
                "touched_rows": count,
                "actual_degree": result["degree"],
                "predicted_degree": prediction,
                "abs_error": abs(prediction - result["degree"]),
                "consumer_passed": result["passed"],
            })
    finally:
        coverage.env["A"] = original_A

    mae = float(np.mean([r["abs_error"] for r in heldout]))
    largest = max(heldout, key=lambda r: r["abs_error"])
    report = {
        "feature_names": FEATURE_NAMES,
        "training_points": len(y_train),
        "heldout_seed": HELDOUT_SEED,
        "ridge_lambda": RIDGE_LAMBDA,
        "heldout": heldout,
        "heldout_mae": mae,
        "largest_residual": largest,
        "all_heldout_consumers_passed": all(
            r["consumer_passed"] for r in heldout
        ),
        "interpretation": {
            "portfolio_has_predictive_signal": mae < 5.0,
            "portfolio_is_exact_degree_predictor": False,
            "largest_residual_is_repair_target": True,
            "fit_is_causal_mechanism": False,
            "fit_is_production_measurement": False,
        },
    }
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    main()
