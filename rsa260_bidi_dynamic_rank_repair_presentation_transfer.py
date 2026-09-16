#!/usr/bin/env python3
"""
RSA-260 bidi dynamic-rank repair transfer across preparation/projection fibres.

Train the same 12-static and 13-fibre (+r80) ridge models on the exact
seed-replication training portfolio used by the held-out repair experiment.

Validate on the already-paid defect-coverage presentation-cross carriers:
  carrier seed rule: 280000 + coverage
  coverage: 0,64,128,256,512
  presentation A: grid4x4 / projection 0
  presentation B: grid4x4 / projection 1
  presentation C: grid8x4 / projection 0

The actual generator degrees below are frozen from the exact-byte
rsa260_bidi_defect_coverage_prep_projection_cross.py receipt, so this runtime
only recomputes static fibres and r80 under each presentation.
"""
from pathlib import Path
import importlib.util
import json
import numpy as np

BASE = Path(__file__).with_name("rsa260_bidi_dynamic_rank_repair_heldout.py")
spec = importlib.util.spec_from_file_location("base", BASE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

COUNTS = [0,64,128,256,512]
RIDGE_LAMBDA = base.RIDGE_LAMBDA

PRESENTATIONS = [
    ("grid4x4_proj0", 4, 4, 0, {0:16,64:30,128:42,256:63,512:66}),
    ("grid4x4_proj1", 4, 4, 1, {0:16,64:31,128:43,256:62,512:65}),
    ("grid8x4_proj0", 8, 4, 0, {0:16,64:30,128:43,256:62,512:66}),
]

def grid_permutation(nh, nv):
    nz = base.COLS // (nh * nv)
    p = np.empty(base.COLS, dtype=int)
    for x in range(base.COLS):
        q, k = divmod(x, nz)
        i, j = divmod(q, nv)
        p[x] = (j * nh + i) * nz + k
    return p

def apply_B(A, perm, Y):
    T = (A.T @ Y) & 1
    T = T[perm]
    return (A @ T) & 1

def reachable_rank80(A, perm, projection):
    Y = base.build_block(base.BASEY ^ ((projection + 1) << 40))
    blocks = []
    for _ in range(base.HORIZON):
        Y = apply_B(A, perm, Y)
        blocks.append(Y)
    return base.rref_rank(np.concatenate(blocks, axis=1))

def train_models():
    Xs, Xd, y = [], [], []
    for count in base.COUNTS:
        for seed, degree in zip(base.TRAIN_SEEDS, base.TRAIN_LABELS[count]):
            A = base.perturb_rows(count, seed + count)
            sf = base.static_fibres(A)
            rk = base.reachable_rank80(A)
            Xs.append(sf)
            Xd.append(np.concatenate([sf,[rk]]))
            y.append(degree)
    Xs = np.array(Xs)
    Xd = np.array(Xd)
    y = np.array(y,dtype=float)
    return (
        base.fit_ridge(Xs,y,RIDGE_LAMBDA),
        base.fit_ridge(Xd,y,RIDGE_LAMBDA),
    )

def main():
    static_model, dynamic_model = train_models()
    results = []
    for name, nh, nv, projection, labels in PRESENTATIONS:
        perm = grid_permutation(nh,nv)
        rows = []
        for count in COUNTS:
            A = base.perturb_rows(count, 280000 + count)
            sf = base.static_fibres(A)
            rk = reachable_rank80(A,perm,projection)
            actual = labels[count]
            ps = float(base.predict(static_model,sf[None,:])[0])
            pd = float(base.predict(dynamic_model,np.concatenate([sf,[rk]])[None,:])[0])
            rows.append({
                "touched_rows":count,
                "actual_degree":actual,
                "rank80":rk,
                "static_abs_error":abs(ps-actual),
                "dynamic_abs_error":abs(pd-actual),
                "static_prediction":ps,
                "dynamic_prediction":pd,
            })
        sm = float(np.mean([r["static_abs_error"] for r in rows]))
        dm = float(np.mean([r["dynamic_abs_error"] for r in rows]))
        target = next(r for r in rows if r["touched_rows"]==256)
        results.append({
            "presentation":name,
            "static_mae":sm,
            "dynamic_mae":dm,
            "mae_improvement":sm-dm,
            "target_256":target,
            "dynamic_reduces_mae":dm<sm,
            "dynamic_reduces_256_residual":
                target["dynamic_abs_error"]<target["static_abs_error"],
            "rows":rows,
        })
    report = {
        "training_presentation":"grid4x4_proj0",
        "validation_presentations":[r["presentation"] for r in results],
        "results":results,
        "all_presentations_reduce_mae":
            all(r["dynamic_reduces_mae"] for r in results),
        "all_presentations_reduce_256_residual":
            all(r["dynamic_reduces_256_residual"] for r in results),
        "interpretation":{
            "dynamic_repair_transfers_across_tested_preparation_projection_fibres":
                all(r["dynamic_reduces_mae"] for r in results),
            "transfer_is_exact_degree_formula":False,
            "frozen_degree_labels_are_new_execution":False,
            "synthetic_transfer_is_production_measurement":False,
        },
    }
    print(json.dumps(report,indent=2))
    return report

if __name__ == "__main__":
    main()
