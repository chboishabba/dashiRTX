#!/usr/bin/env python3
from pathlib import Path
import importlib.util, json
import numpy as np

BASE = Path(__file__).with_name('rsa260_bidi_preparation_fibre_search.py')
spec = importlib.util.spec_from_file_location('prep', BASE)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)

SEEDS = [2602027,2602029,2602031,2602033,2602039,2602041,2602043,2602047]

def cado_grid_permutation(nh=4, nv=4):
    nz = prep.COLS // (nh * nv)
    p = np.empty(prep.COLS, dtype=int)
    for x in range(prep.COLS):
        q, k = divmod(x, nz)
        i, j = divmod(q, nv)
        p[x] = (j * nh + i) * nz + k
    return p

def one_swap_per_row(base, seed):
    rng = np.random.default_rng(seed)
    A = base.copy()
    for r in range(prep.ROWS):
        support = np.flatnonzero(base[r])
        complement = np.flatnonzero(base[r] == 0)
        A[r, rng.permutation(support)[0]] = 0
        A[r, rng.permutation(complement)[0]] = 1
    return A

def main():
    base = prep.build_A(); perm = cado_grid_permutation(4,4)
    oldA, oldD = prep.A, prep.MAXD; runs=[]
    try:
        prep.MAXD = 80
        for seed in SEEDS:
            prep.A = one_swap_per_row(base, seed)
            runs.append({'seed':seed, **prep.evaluate(prep.make_apply_B(perm), prep.BASEX^(1<<48), prep.BASEY^(1<<40))})
    finally:
        prep.A, prep.MAXD = oldA, oldD
    out = {
        'intervention':'one degree-preserving nonzero swap independently in every row',
        'runs':runs,
        'summary':{
            'runs':len(runs),
            'passed':sum(bool(r['passed']) for r in runs),
            'minimum_generator_degree':min(r['degree'] for r in runs),
            'maximum_generator_degree':max(r['degree'] for r in runs),
            'all_shifted_rank_512':all(r['shifted_rank']==512 for r in runs),
        },
        'firewalls':{
            'ensemble_proves_universal_fragility':False,
            'synthetic_ensemble_is_production_measurement':False,
            'consumer_failure':False,
        }
    }
    print(json.dumps(out,indent=2)); return out

if __name__=='__main__': main()
