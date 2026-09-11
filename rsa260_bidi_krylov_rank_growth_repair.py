#!/usr/bin/env python3
"""
RSA-260 bidi residual inspection: Krylov rank-growth repair.

Compare:
  M13 = 12 static fibres + r80
  M16 = 12 static fibres + r20 + r40 + r60 + r80

Same 18 training carriers, same held-out seed 271011, same degree labels and
ridge lambda as the paid dynamic-rank repair experiment.
"""
from pathlib import Path
import importlib.util
import json
import numpy as np

BASE = Path(__file__).with_name("rsa260_bidi_dynamic_rank_repair_heldout.py")
spec = importlib.util.spec_from_file_location("base", BASE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

HORIZONS = [20,40,60,80]

def rank_growth(A):
    Y = base.build_block(base.BASEY ^ (1 << 40))
    blocks = []
    out = []
    target = set(HORIZONS)
    for step in range(1, max(HORIZONS)+1):
        Y = base.apply_B(A,Y)
        blocks.append(Y)
        if step in target:
            out.append(base.rref_rank(np.concatenate(blocks,axis=1)))
    return np.array(out,dtype=float)

def main():
    X13, X16, y = [], [], []
    for count in base.COUNTS:
        for seed, degree in zip(base.TRAIN_SEEDS, base.TRAIN_LABELS[count]):
            A = base.perturb_rows(count, seed+count)
            sf = base.static_fibres(A)
            rg = rank_growth(A)
            X13.append(np.concatenate([sf,[rg[-1]]]))
            X16.append(np.concatenate([sf,rg]))
            y.append(degree)
    X13=np.array(X13); X16=np.array(X16); y=np.array(y,float)
    m13=base.fit_ridge(X13,y,base.RIDGE_LAMBDA)
    m16=base.fit_ridge(X16,y,base.RIDGE_LAMBDA)

    rows=[]
    e13=[]; e16=[]
    for count in base.COUNTS:
        A=base.perturb_rows(count,base.HELDOUT_SEED+count)
        sf=base.static_fibres(A)
        rg=rank_growth(A)
        actual=base.HELDOUT_LABELS[count]
        p13=float(base.predict(m13,np.concatenate([sf,[rg[-1]]])[None,:])[0])
        p16=float(base.predict(m16,np.concatenate([sf,rg])[None,:])[0])
        a13=abs(p13-actual); a16=abs(p16-actual)
        e13.append(a13); e16.append(a16)
        rows.append({
            "touched_rows":count,
            "actual_degree":actual,
            "rank_growth":{str(h):int(r) for h,r in zip(HORIZONS,rg)},
            "rank80_prediction":p13,
            "rank_growth_prediction":p16,
            "rank80_abs_error":a13,
            "rank_growth_abs_error":a16,
        })
    target=next(r for r in rows if r["touched_rows"]==256)
    report={
        "rank_horizons":HORIZONS,
        "rank80_model_feature_count":13,
        "rank_growth_model_feature_count":16,
        "rank80_mae":float(np.mean(e13)),
        "rank_growth_mae":float(np.mean(e16)),
        "mae_improvement":float(np.mean(e13)-np.mean(e16)),
        "target_256":target,
        "boundary":{
            "rank_growth_reduces_overall_mae":bool(np.mean(e16)<np.mean(e13)),
            "rank_growth_reduces_256_residual":bool(
                target["rank_growth_abs_error"]<target["rank80_abs_error"]),
            "rank_growth_eliminates_256_residual":bool(
                target["rank_growth_abs_error"]<0.5),
            "rank_growth_is_complete_formula":False,
            "synthetic_rank_growth_is_production_measurement":False,
        },
        "rows":rows,
    }
    print(json.dumps(report,indent=2))
    return report

if __name__=="__main__":
    main()
