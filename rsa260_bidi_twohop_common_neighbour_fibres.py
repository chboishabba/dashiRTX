#!/usr/bin/env python3
import json
import numpy as np

ROWS, COLS = 924, 512
RNG_SEEDS = [2602027,2602029,2602031,2602033,2602039,2602041,2602043,2602047]
KNOWN_DEGREES = [66,65,65,66,65,65,65,66]

def build_base():
    A=np.zeros((ROWS,COLS),dtype=np.uint8)
    for r in range(ROWS):
        degree=151 if r<6 else 150
        base=(r*2654435761 + 0x9e3779b9) % COLS
        A[r,(base+np.arange(degree))%COLS]=1
    return A

def one_swap(A0,seed):
    rng=np.random.default_rng(seed); A=A0.copy()
    for r in range(ROWS):
        support=np.flatnonzero(A0[r]); complement=np.flatnonzero(A0[r]==0)
        A[r,rng.permutation(support)[0]]=0
        A[r,rng.permutation(complement)[0]]=1
    return A

def entropy(values):
    _,counts=np.unique(values,return_counts=True)
    p=counts/counts.sum()
    return float(-(p*np.log2(p)).sum())

def fibres(A):
    X=A.astype(np.int16)
    RR=X@X.T
    CC=X.T@X
    rr=RR[~np.eye(ROWS,dtype=bool)]
    cc=CC[~np.eye(COLS,dtype=bool)]
    rr_diff=np.abs(RR[1:].astype(np.int32)-RR[:-1].astype(np.int32)).sum(axis=1)
    cc_diff=np.abs(CC[1:].astype(np.int32)-CC[:-1].astype(np.int32)).sum(axis=1)
    return {
        "row_common_mean":float(rr.mean()),
        "row_common_std":float(rr.std()),
        "row_common_entropy_bits":entropy(rr),
        "row_common_adjacent_mean":float(np.diag(RR,1).mean()),
        "row_common_distance2_mean":float(np.diag(RR,2).mean()),
        "row_twohop_profile_step_l1_mean":float(rr_diff.mean()),
        "col_common_mean":float(cc.mean()),
        "col_common_std":float(cc.std()),
        "col_common_entropy_bits":entropy(cc),
        "col_common_adjacent_mean":float(np.diag(CC,1).mean()),
        "col_common_distance2_mean":float(np.diag(CC,2).mean()),
        "col_twohop_profile_step_l1_mean":float(cc_diff.mean()),
        "row_common_max":int(rr.max()),
        "col_common_max":int(cc.max()),
    }

def main():
    base=build_base()
    rows=[{"carrier":"base","generator_degree":16,**fibres(base)}]
    for seed,d in zip(RNG_SEEDS,KNOWN_DEGREES):
        rows.append({"carrier":f"swap_{seed}","generator_degree":d,**fibres(one_swap(base,seed))})
    out={
        "carrier_shape":[ROWS,COLS],
        "base_vs_one_swap_ensemble":rows,
        "summary":{
            "base_row_common_entropy_bits":rows[0]["row_common_entropy_bits"],
            "swap_row_common_entropy_min":min(r["row_common_entropy_bits"] for r in rows[1:]),
            "swap_row_common_entropy_max":max(r["row_common_entropy_bits"] for r in rows[1:]),
            "base_row_distance2_overlap":rows[0]["row_common_distance2_mean"],
            "swap_row_distance2_min":min(r["row_common_distance2_mean"] for r in rows[1:]),
            "swap_row_distance2_max":max(r["row_common_distance2_mean"] for r in rows[1:]),
            "base_col_common_entropy_bits":rows[0]["col_common_entropy_bits"],
            "swap_col_common_entropy_min":min(r["col_common_entropy_bits"] for r in rows[1:]),
            "swap_col_common_entropy_max":max(r["col_common_entropy_bits"] for r in rows[1:]),
            "base_col_profile_step_l1":rows[0]["col_twohop_profile_step_l1_mean"],
            "swap_col_profile_step_l1_min":min(r["col_twohop_profile_step_l1_mean"] for r in rows[1:]),
            "swap_col_profile_step_l1_max":max(r["col_twohop_profile_step_l1_mean"] for r in rows[1:]),
        },
        "firewalls":{
            "ensemble_separation_is_exact_degree_predictor":False,
            "structural_association_is_causal_mechanism":False,
            "synthetic_fibres_are_production_measurement":False,
        }
    }
    print(json.dumps(out,indent=2))
    return out

if __name__=="__main__":
    main()
