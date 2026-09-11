#!/usr/bin/env python3
from pathlib import Path
import importlib.util, json, time
import numpy as np

BASE = Path(__file__).with_name('rsa260_bidi_preparation_fibre_search.py')
spec = importlib.util.spec_from_file_location('prep', BASE)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)

FRACTIONS = [0.0, 0.125, 0.25, 0.5, 0.75, 1.0]
RNG_SEED = 2602026

def gf2_rank(A):
    _, piv = prep.rref(A)
    return len(piv)

def cyclic_base():
    return prep.build_A()

def interpolate_rows(alpha):
    A = cyclic_base().copy()
    rng = np.random.default_rng(RNG_SEED + int(round(alpha * 1000)))
    for r in range(prep.ROWS):
        degree = 151 if r < 6 else 150
        k = int(round(alpha * degree))
        if k == 0:
            continue
        current = np.flatnonzero(A[r])
        remove = rng.choice(current, size=k, replace=False)
        A[r, remove] = 0
        available = np.flatnonzero(A[r] == 0)
        add = rng.choice(available, size=k, replace=False)
        A[r, add] = 1
    return A

def cado_grid_permutation(nh=4, nv=4):
    nz = prep.COLS // (nh * nv)
    p = np.empty(prep.COLS, dtype=int)
    for x in range(prep.COLS):
        q, k = divmod(x, nz)
        i, j = divmod(q, nv)
        p[x] = (j * nh + i) * nz + k
    return p

def cyclic_adjacency_mean(A):
    return float(np.mean(np.sum(A * np.roll(A, -1, axis=1), axis=1)))

def best_translation_overlap_mean(A, sample_rows=64):
    vals = []
    for r in range(min(sample_rows, prep.ROWS - 1)):
        a, b = A[r], A[r + 1]
        vals.append(max(int(np.sum(np.roll(a, s) * b)) for s in range(prep.COLS)))
    return float(np.mean(vals))

def main():
    original_A = prep.A
    original_MAXD = prep.MAXD
    perm = cado_grid_permutation(4, 4)
    rows = []
    t0 = time.perf_counter()
    try:
        prep.MAXD = 80
        for alpha in FRACTIONS:
            A = interpolate_rows(alpha)
            prep.A = A
            rank = gf2_rank(A)
            apply = prep.make_apply_B(perm)
            r = prep.evaluate(apply, prep.BASEX ^ (1 << 48), prep.BASEY ^ (1 << 40))
            rows.append({
                'rewire_fraction_per_row': alpha,
                'rank': rank,
                'left_nullity': prep.ROWS - rank,
                'degree151_rows': int((A.sum(axis=1) == 151).sum()),
                'degree150_rows': int((A.sum(axis=1) == 150).sum()),
                'cyclic_adjacency_mean': cyclic_adjacency_mean(A),
                'best_translation_overlap_mean': best_translation_overlap_mean(A),
                **r,
            })
    finally:
        prep.A = original_A
        prep.MAXD = original_MAXD

    out = {
        'fixed_preparation': 'CADO-shaped 4x4 block-grid transpose analogue',
        'fixed_projection_seed_index': 0,
        'interpolation': rows,
        'firewalls': {
            'graded_trend_implies_causal_proof': False,
            'synthetic_interpolation_equals_production_matrix': False,
            'same_degree_profile_fixes_generator_degree': False,
        },
        'elapsed_seconds': round(time.perf_counter() - t0, 3),
    }
    print(json.dumps(out, indent=2))
    return out

if __name__ == '__main__':
    main()
