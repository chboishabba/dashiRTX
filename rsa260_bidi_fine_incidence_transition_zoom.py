#!/usr/bin/env python3
from pathlib import Path
import importlib.util, json, time
import numpy as np

BASE = Path(__file__).with_name('rsa260_bidi_preparation_fibre_search.py')
spec = importlib.util.spec_from_file_location('prep', BASE)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)

FRACTIONS = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.125]
RNG_SEED = 2602027

def gf2_rank(A):
    _, piv = prep.rref(A)
    return len(piv)

def cado_grid_permutation(nh=4, nv=4):
    nz = prep.COLS // (nh * nv)
    p = np.empty(prep.COLS, dtype=int)
    for x in range(prep.COLS):
        q, k = divmod(x, nz)
        i, j = divmod(q, nv)
        p[x] = (j * nh + i) * nz + k
    return p

def build_nested_plans(base):
    rng = np.random.default_rng(RNG_SEED)
    plans = []
    for r in range(prep.ROWS):
        support = np.flatnonzero(base[r])
        complement = np.flatnonzero(base[r] == 0)
        plans.append((rng.permutation(support), rng.permutation(complement)))
    return plans

def carrier_at(base, plans, alpha):
    A = base.copy()
    for r, (remove_order, add_order) in enumerate(plans):
        degree = len(remove_order)
        k = int(round(alpha * degree))
        if k:
            A[r, remove_order[:k]] = 0
            A[r, add_order[:k]] = 1
    return A

def adjacency(A):
    return float(np.mean(np.sum(A * np.roll(A, -1, axis=1), axis=1)))

def translation(A, sample_rows=64):
    vals=[]
    for r in range(min(sample_rows, prep.ROWS-1)):
        a,b=A[r],A[r+1]
        vals.append(max(int(np.sum(np.roll(a,s)*b)) for s in range(prep.COLS)))
    return float(np.mean(vals))

def main():
    base = prep.build_A()
    plans = build_nested_plans(base)
    perm = cado_grid_permutation(4,4)
    oldA, oldD = prep.A, prep.MAXD
    rows=[]; t0=time.perf_counter()
    try:
        prep.MAXD=80
        for alpha in FRACTIONS:
            A=carrier_at(base,plans,alpha); prep.A=A
            r=prep.evaluate(prep.make_apply_B(perm), prep.BASEX^(1<<48), prep.BASEY^(1<<40))
            rows.append({'rewire_fraction':alpha,'rank':gf2_rank(A),'left_nullity':prep.ROWS-gf2_rank(A),
                         'cyclic_adjacency_mean':adjacency(A),'best_translation_overlap_mean':translation(A),**r})
    finally:
        prep.A, prep.MAXD = oldA, oldD
    out={'nested_rewire_path':rows,'elapsed_seconds':round(time.perf_counter()-t0,3),
         'firewalls':{'observed_transition_is_universal_threshold':False,'synthetic_transition_is_production_measurement':False}}
    print(json.dumps(out,indent=2)); return out

if __name__=='__main__': main()
