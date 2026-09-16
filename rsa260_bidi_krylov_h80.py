#!/usr/bin/env python3
"""
RSA-260 bidi dynamic fibre: first horizon attaining r80.

Define
  h80 = min { h <= 80 : rank([BY,...,B^h Y]) = r80 }.

Test h80 against previously paid shared-generator degrees on:
  - 18 training carriers,
  - 6 held-out carriers,
  - 15 preparation/projection validation cases.

Observed combined envelope:
  d - h80 in {-1,0,1}.

This is finite-horizon experimental evidence only.  h80 is not claimed to be
an infinite-time Krylov saturation index or a universal Block-Wiedemann formula.
"""
from pathlib import Path
import importlib.util
import json
import numpy as np

BASE = Path(__file__).with_name("rsa260_bidi_dynamic_rank_repair_heldout.py")
spec = importlib.util.spec_from_file_location("base", BASE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

def grid_permutation(nh, nv):
    nz = base.COLS // (nh * nv)
    p = np.empty(base.COLS, dtype=int)
    for x in range(base.COLS):
        q, k = divmod(x, nz)
        i, j = divmod(q, nv)
        p[x] = (j * nh + i) * nz + k
    return p

def col_to_int(col):
    value = 0
    for i in np.flatnonzero(col):
        value |= 1 << int(i)
    return value

def add_basis(value, basis):
    x = value
    while x:
        pivot = x.bit_length() - 1
        if pivot in basis:
            x ^= basis[pivot]
        else:
            basis[pivot] = x
            return True
    return False

def h80(A, perm, projection):
    Y = base.build_block(base.BASEY ^ ((projection + 1) << 40))
    basis = {}
    ranks = []
    for _step in range(1, 81):
        T = (A.T @ Y) & 1
        T = T[perm]
        Y = (A @ T) & 1
        for j in range(base.BLOCK):
            add_basis(col_to_int(Y[:,j]), basis)
        ranks.append(len(basis))
    r80 = ranks[-1]
    first = next(i + 1 for i, r in enumerate(ranks) if r == r80)
    return first, r80

PRESENTATIONS = [
    ("grid4x4_proj0", grid_permutation(4,4), 0,
     {0:16,64:30,128:42,256:63,512:66}),
    ("grid4x4_proj1", grid_permutation(4,4), 1,
     {0:16,64:31,128:43,256:62,512:65}),
    ("grid8x4_proj0", grid_permutation(8,4), 0,
     {0:16,64:30,128:43,256:62,512:66}),
]

def main():
    canonical_perm = grid_permutation(4,4)

    training = []
    for count in base.COUNTS:
        for seed, degree in zip(base.TRAIN_SEEDS, base.TRAIN_LABELS[count]):
            A = base.perturb_rows(count, seed + count)
            horizon, r80 = h80(A, canonical_perm, 0)
            training.append({
                "touched_rows":count,
                "seed":seed,
                "degree":degree,
                "h80":horizon,
                "r80":r80,
                "delta":degree-horizon,
            })

    heldout = []
    for count in base.COUNTS:
        A = base.perturb_rows(count, base.HELDOUT_SEED + count)
        horizon, r80 = h80(A, canonical_perm, 0)
        degree = base.HELDOUT_LABELS[count]
        heldout.append({
            "touched_rows":count,
            "degree":degree,
            "h80":horizon,
            "r80":r80,
            "delta":degree-horizon,
        })

    presentation = []
    for name, perm, projection, labels in PRESENTATIONS:
        for count, degree in labels.items():
            A = base.perturb_rows(count, 280000 + count)
            horizon, r80 = h80(A, perm, projection)
            presentation.append({
                "presentation":name,
                "touched_rows":count,
                "degree":degree,
                "h80":horizon,
                "r80":r80,
                "delta":degree-horizon,
            })

    all_rows = training + heldout + presentation
    deltas = [r["delta"] for r in all_rows]
    report = {
        "training_cases":len(training),
        "heldout_cases":len(heldout),
        "presentation_cases":len(presentation),
        "total_cases":len(all_rows),
        "training":training,
        "heldout":heldout,
        "presentation":presentation,
        "combined_delta_min":min(deltas),
        "combined_delta_max":max(deltas),
        "all_combined_deltas_within_minus1_plus1":
            all(-1 <= d <= 1 for d in deltas),
        "heldout_all_delta_plus1":
            all(r["delta"] == 1 for r in heldout),
        "presentation_delta_min":min(r["delta"] for r in presentation),
        "presentation_delta_max":max(r["delta"] for r in presentation),
        "interpretation":{
            "h80_is_tighter_than_rank80_ceiling_on_tested_portfolio":True,
            "h80_is_infinite_time_saturation_theorem":False,
            "d_equals_h80_plus_one_universally":False,
            "same_object_A_sequence_can_constrain_h80":True,
            "synthetic_h80_is_production_measurement":False,
        },
    }
    print(json.dumps(report,indent=2))
    return report

if __name__=="__main__":
    main()
