#!/usr/bin/env python3
"""
RSA-260 bidi held-out dynamic-rank repair experiment.

Purpose:
  Compare the existing 12-static-fibre recurrence-complexity predictor with
  the same predictor plus one dynamic fibre:
      r_80 = rank([BY, B^2Y, ..., B^80Y])
  for width-8 Krylov blocks under the fixed CADO-shaped 4x4 preparation.

Training degree labels are the exact-byte seed-replication receipt:
  coverage 32:  [24,22,23]
  coverage 64:  [30,29,29]
  coverage 128: [44,41,44]
  coverage 256: [63,62,64]
  coverage 512: [66,65,66]
  coverage 924: [66,66,65]

Held-out seed 271011 degree labels are the already-executed held-out receipt:
  [24,30,43,64,66,66].

This file recomputes every static and dynamic feature from each carrier.
It does not claim causation, a universal generator formula, or production
RSA-260 measurement.
"""
import json
import numpy as np

ROWS, COLS = 924, 512
BLOCK = 8
HORIZON = 80
MASK = (1 << 64) - 1
BASEY = 0x3c6ef372fe94f82b
COUNTS = [32, 64, 128, 256, 512, 924]
TRAIN_SEEDS = [271001, 271003, 271007]
HELDOUT_SEED = 271011
RIDGE_LAMBDA = 0.3

TRAIN_LABELS = {
    32: [24, 22, 23],
    64: [30, 29, 29],
    128: [44, 41, 44],
    256: [63, 62, 64],
    512: [66, 65, 66],
    924: [66, 66, 65],
}
HELDOUT_LABELS = {32:24, 64:30, 128:43, 256:64, 512:66, 924:66}

STATIC_NAMES = [
    "row_common_mean","row_common_std","row_common_entropy",
    "row_adjacent","row_distance2","row_profile_step_l1",
    "col_common_mean","col_common_std","col_common_entropy",
    "col_adjacent","col_distance2","col_profile_step_l1",
]

def mix64(x):
    x &= MASK
    x ^= x >> 30
    x = (x * 0xbf58476d1ce4e5b9) & MASK
    x ^= x >> 27
    x = (x * 0x94d049bb133111eb) & MASK
    x ^= x >> 31
    return x & MASK

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

def entropy(values):
    _, counts = np.unique(values, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())

def static_fibres(A):
    X = A.astype(np.int16)
    RR = X @ X.T
    CC = X.T @ X
    rr = RR[~np.eye(ROWS, dtype=bool)]
    cc = CC[~np.eye(COLS, dtype=bool)]
    rr_step = np.abs(RR[1:].astype(np.int32) - RR[:-1].astype(np.int32)).sum(axis=1)
    cc_step = np.abs(CC[1:].astype(np.int32) - CC[:-1].astype(np.int32)).sum(axis=1)
    return np.array([
        float(rr.mean()), float(rr.std()), entropy(rr),
        float(np.diag(RR,1).mean()), float(np.diag(RR,2).mean()),
        float(rr_step.mean()),
        float(cc.mean()), float(cc.std()), entropy(cc),
        float(np.diag(CC,1).mean()), float(np.diag(CC,2).mean()),
        float(cc_step.mean()),
    ], dtype=float)

def seed_vec(seed):
    bits = np.zeros(ROWS, dtype=np.uint8)
    for w in range(15):
        val = mix64(seed ^ ((0x9e3779b97f4a7c15 * (w + 1)) & MASK))
        lo, hi = 64*w, min(64*(w+1), ROWS)
        for b in range(hi-lo):
            bits[lo+b] = (val >> b) & 1
    return bits

def build_block(seed):
    return np.stack([seed_vec(seed ^ (j << 32)) for j in range(BLOCK)], axis=1)

def grid4x4():
    p = np.empty(COLS, dtype=int)
    nz = COLS // 16
    for x in range(COLS):
        q, k = divmod(x, nz)
        i, j = divmod(q, 4)
        p[x] = (j * 4 + i) * nz + k
    return p

PERM = grid4x4()

def apply_B(A, Y):
    T = (A.T @ Y) & 1
    T = T[PERM]
    return (A @ T) & 1

def rref_rank(M):
    M = M.copy()
    nr, nc = M.shape
    r = 0
    for c in range(nc):
        nz = np.flatnonzero(M[r:, c])
        if nz.size == 0:
            continue
        p = r + nz[0]
        if p != r:
            M[[r,p]] = M[[p,r]]
        nz = np.flatnonzero(M[:,c])
        nz = nz[nz != r]
        if nz.size:
            M[nz] ^= M[r]
        r += 1
        if r == nr:
            break
    return r

def reachable_rank80(A):
    Y = build_block(BASEY ^ (1 << 40))
    blocks = []
    for _ in range(HORIZON):
        Y = apply_B(A, Y)
        blocks.append(Y)
    return rref_rank(np.concatenate(blocks, axis=1))

def fit_ridge(X, y, lam):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    coef = np.linalg.solve(
        Z.T @ Z + lam * np.eye(Z.shape[1]),
        Z.T @ (y - y.mean()),
    )
    return mu, sd, y.mean(), coef

def predict(model, X):
    mu, sd, ybar, coef = model
    return ybar + ((X - mu) / sd) @ coef

def main():
    X_static, X_dynamic, y = [], [], []
    train_receipts = []
    for count in COUNTS:
        for seed, degree in zip(TRAIN_SEEDS, TRAIN_LABELS[count]):
            A = perturb_rows(count, seed + count)
            sf = static_fibres(A)
            rk = reachable_rank80(A)
            X_static.append(sf)
            X_dynamic.append(np.concatenate([sf, [rk]]))
            y.append(degree)
            train_receipts.append({"touched_rows":count,"seed":seed,"degree":degree,"rank80":rk})

    X_static = np.array(X_static)
    X_dynamic = np.array(X_dynamic)
    y = np.array(y, dtype=float)
    static_model = fit_ridge(X_static, y, RIDGE_LAMBDA)
    dynamic_model = fit_ridge(X_dynamic, y, RIDGE_LAMBDA)

    heldout = []
    static_errors, dynamic_errors = [], []
    for count in COUNTS:
        A = perturb_rows(count, HELDOUT_SEED + count)
        sf = static_fibres(A)
        rk = reachable_rank80(A)
        actual = HELDOUT_LABELS[count]
        ps = float(predict(static_model, sf[None,:])[0])
        pd = float(predict(dynamic_model, np.concatenate([sf,[rk]])[None,:])[0])
        es, ed = abs(ps-actual), abs(pd-actual)
        static_errors.append(es)
        dynamic_errors.append(ed)
        heldout.append({
            "touched_rows": count,
            "actual_degree": actual,
            "rank80": rk,
            "static_prediction": ps,
            "dynamic_prediction": pd,
            "static_abs_error": es,
            "dynamic_abs_error": ed,
        })

    static_mae = float(np.mean(static_errors))
    dynamic_mae = float(np.mean(dynamic_errors))
    target = next(r for r in heldout if r["touched_rows"] == 256)
    report = {
        "static_feature_count": 12,
        "dynamic_feature_count": 13,
        "dynamic_repair_fibre": "rank80",
        "training_points": len(y),
        "heldout_seed": HELDOUT_SEED,
        "ridge_lambda": RIDGE_LAMBDA,
        "training_rank_receipts": train_receipts,
        "heldout": heldout,
        "static_mae": static_mae,
        "dynamic_mae": dynamic_mae,
        "mae_improvement": static_mae - dynamic_mae,
        "target_256": target,
        "repair_boundary": {
            "dynamic_rank_reduces_overall_heldout_mae": dynamic_mae < static_mae,
            "dynamic_rank_reduces_256_row_residual":
                target["dynamic_abs_error"] < target["static_abs_error"],
            "dynamic_rank_eliminates_256_row_residual":
                target["dynamic_abs_error"] < 0.5,
            "dynamic_rank_is_universal_formula": False,
            "synthetic_repair_is_production_measurement": False,
        },
    }
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    main()
