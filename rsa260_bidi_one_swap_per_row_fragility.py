#!/usr/bin/env python3
from pathlib import Path
import importlib.util, json
import numpy as np

BASE = Path(__file__).with_name('rsa260_bidi_preparation_fibre_search.py')
spec = importlib.util.spec_from_file_location('prep', BASE)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)

RNG_SEED = 2602027

def cado_grid_permutation(nh=4, nv=4):
    nz = prep.COLS // (nh * nv)
    p = np.empty(prep.COLS, dtype=int)
    for x in range(prep.COLS):
        q, k = divmod(x, nz)
        i, j = divmod(q, nv)
        p[x] = (j * nh + i) * nz + k
    return p

def one_swap_per_row(base):
    rng = np.random.default_rng(RNG_SEED)
    A = base.copy()
    for r in range(prep.ROWS):
        support = np.flatnonzero(base[r])
        complement = np.flatnonzero(base[r] == 0)
        remove = rng.permutation(support)[0]
        add = rng.permutation(complement)[0]
        A[r, remove] = 0
        A[r, add] = 1
    return A

def gf2_rank(A):
    _, piv = prep.rref(A)
    return len(piv)

def adjacency(A):
    return float(np.mean(np.sum(A * np.roll(A, -1, axis=1), axis=1)))

def translation(A, sample_rows=64):
    vals=[]
    for r in range(min(sample_rows, prep.ROWS-1)):
        a,b=A[r],A[r+1]
        vals.append(max(int(np.sum(np.roll(a,s)*b)) for s in range(prep.COLS)))
    return float(np.mean(vals))

def evaluate_carrier(A, perm):
    prep.A = A
    return prep.evaluate(prep.make_apply_B(perm), prep.BASEX^(1<<48), prep.BASEY^(1<<40))

def main():
    base = prep.build_A()
    perturbed = one_swap_per_row(base)
    perm = cado_grid_permutation(4,4)
    oldA, oldD = prep.A, prep.MAXD
    try:
        prep.MAXD = 80
        base_result = evaluate_carrier(base, perm)
        perturbed_result = evaluate_carrier(perturbed, perm)
    finally:
        prep.A, prep.MAXD = oldA, oldD
    out = {
        'carrier_shape':[prep.ROWS,prep.COLS],
        'degree_profile':'six rows degree 151; 918 rows degree 150',
        'intervention':'remove one existing nonzero and add one absent nonzero independently in every row',
        'base_rank':gf2_rank(base),
        'perturbed_rank':gf2_rank(perturbed),
        'base_left_nullity':prep.ROWS-gf2_rank(base),
        'perturbed_left_nullity':prep.ROWS-gf2_rank(perturbed),
        'base_adjacency':adjacency(base),
        'perturbed_adjacency':adjacency(perturbed),
        'base_translation':translation(base),
        'perturbed_translation':translation(perturbed),
        'base_result':base_result,
        'perturbed_result':perturbed_result,
        'firewalls':{
            'one_path_proves_universal_fragility':False,
            'synthetic_fragility_is_production_measurement':False,
            'consumer_failure_after_perturbation':False,
        }
    }
    print(json.dumps(out,indent=2)); return out

if __name__=='__main__': main()
