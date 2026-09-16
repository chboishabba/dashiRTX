#!/usr/bin/env python3
"""
RSA-260 bidi dynamic-fibre experiment: reachable block-Krylov rank.

Snowball/source coordinates:
- Eric Lu, "Factoring RSA-260", Cognition, 2026-09-09.
  https://cognition.com/blog/factoring-rsa-260
  Role: primary first-party RSA-260 linear-algebra execution envelope.
- Don Coppersmith, "Solving homogeneous linear equations over GF(2) via block
  Wiedemann algorithm", DOI 10.1090/S0025-5718-1994-1192970-7.
- Emmanuel Thome, "Subquadratic Computation of Vector Generating Polynomials
  and Improvement of the Block Wiedemann Algorithm",
  DOI 10.1006/jsco.2002.0533.

For selected defect-coverage levels and two CADO-shaped preparation analogues,
measure:
  1. first held-out-valid shared generator degree d;
  2. rank r_80 of 80 shifted width-8 Krylov blocks;
  3. ceil(r_80 / 8).

Observed synthetic relation:
  d - ceil(r_80/8) is in {0,1,2} on the tested portfolio.

This is an experimental envelope only.  It does not establish an exact
production formula, recover the historical RSA-260 matrix, or identify CADO's
exact production balancing file.
"""
import json
import math
import time
import numpy as np

ROWS, COLS = 924, 512
BLOCK, TERMS, TRAIN_LAST, MAXD = 8, 256, 191, 80
HORIZON = 80
MASK = (1 << 64) - 1
BASEX = 0xbb67ae8584caa73b
BASEY = 0x3c6ef372fe94f82b

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

def seed_vec(seed):
    bits = np.zeros(ROWS, dtype=np.uint8)
    for w in range(15):
        val = mix64(seed ^ ((0x9e3779b97f4a7c15 * (w + 1)) & MASK))
        lo = 64 * w
        hi = min(lo + 64, ROWS)
        for b in range(hi - lo):
            bits[lo + b] = (val >> b) & 1
    return bits

def build_block(seed):
    return np.stack(
        [seed_vec(seed ^ (j << 32)) for j in range(BLOCK)],
        axis=1,
    )

def grid_permutation(nh, nv):
    if COLS % (nh * nv):
        raise ValueError("grid product must divide carrier width")
    nz = COLS // (nh * nv)
    p = np.empty(COLS, dtype=int)
    for x in range(COLS):
        q, k = divmod(x, nz)
        i, j = divmod(q, nv)
        p[x] = (j * nh + i) * nz + k
    return p

def make_apply(A, perm):
    perm = np.asarray(perm)
    def apply(Y):
        T = (A.T @ Y) & 1
        T = T[perm]
        return (A @ T) & 1
    return apply

def gen_seq(apply):
    X = build_block(BASEX ^ (1 << 48))
    Y = build_block(BASEY ^ (1 << 40))
    seq = np.empty((TERMS, BLOCK, BLOCK), dtype=np.uint8)
    for k in range(TERMS):
        seq[k] = (X.T @ Y) & 1
        Y = apply(Y)
    return seq

def solve_gf2(M, b):
    M, b = M.copy(), b.copy()
    nr, nc = M.shape
    pivots = []
    r = 0
    for c in range(nc):
        nz = np.flatnonzero(M[r:, c])
        if nz.size == 0:
            continue
        p = r + nz[0]
        if p != r:
            M[[r, p]] = M[[p, r]]
            b[[r, p]] = b[[p, r]]
        nz = np.flatnonzero(M[:, c])
        nz = nz[nz != r]
        if nz.size:
            M[nz] ^= M[r]
            b[nz] ^= b[r]
        pivots.append(c)
        r += 1
        if r == nr:
            break
    if r < nr:
        zero = np.where(M[r:].sum(axis=1) == 0)[0]
        if zero.size and np.any(b[r:][zero]):
            return None
    x = np.zeros(nc, dtype=np.uint8)
    for rr, c in enumerate(pivots):
        x[c] = b[rr]
    return x

def fit_degree(seq, degree):
    kvals = TRAIN_LAST - degree + 1
    lhs = np.empty((kvals * BLOCK, BLOCK * degree), dtype=np.uint8)
    for k in range(kvals):
        for i in range(BLOCK):
            lhs[k * BLOCK + i] = seq[k:k + degree, i, :].reshape(-1)
    F = np.zeros((degree, BLOCK, BLOCK), dtype=np.uint8)
    for j in range(BLOCK):
        rhs = np.concatenate([seq[k + degree, :, j] for k in range(kvals)])
        sol = solve_gf2(lhs, rhs)
        if sol is None:
            return None
        F[:, :, j] = sol.reshape(degree, BLOCK)
    return F

def recurrence_holds(seq, F, start, end):
    degree = len(F)
    for k in range(start, end - degree):
        acc = seq[k + degree].copy()
        for l in range(degree):
            acc ^= (seq[k + l] @ F[l]) & 1
        if acc.any():
            return False
    return True

def first_generator_degree(seq):
    for degree in range(1, MAXD + 1):
        F = fit_degree(seq, degree)
        if F is None:
            continue
        if not recurrence_holds(seq, F, 0, TRAIN_LAST + 1):
            continue
        holdout_start = TRAIN_LAST + 1 - degree
        if recurrence_holds(seq, F, holdout_start, TERMS):
            return degree
    return None

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
            M[[r, p]] = M[[p, r]]
        nz = np.flatnonzero(M[:, c])
        nz = nz[nz != r]
        if nz.size:
            M[nz] ^= M[r]
        r += 1
        if r == nr:
            break
    return r

def reachable_rank(apply):
    K = [build_block(BASEY ^ (1 << 40))]
    for _ in range(HORIZON):
        K.append(apply(K[-1]))
    shifted = np.concatenate(K[1:], axis=1)
    return rref_rank(shifted)

def main():
    counts = [0, 128, 256, 512]
    preparations = [
        ("grid4x4", grid_permutation(4, 4)),
        ("grid8x4", grid_permutation(8, 4)),
    ]
    rows = []
    started = time.perf_counter()

    for count in counts:
        A = perturb_rows(count, 280000 + count)
        for preparation, perm in preparations:
            apply = make_apply(A, perm)
            degree = first_generator_degree(gen_seq(apply))
            rank80 = reachable_rank(apply)
            ceiling = math.ceil(rank80 / BLOCK)
            rows.append({
                "touched_rows": count,
                "preparation": preparation,
                "generator_degree": degree,
                "rank80": rank80,
                "block_width": BLOCK,
                "ceil_rank80_over_block": ceiling,
                "degree_minus_ceiling": degree - ceiling,
            })

    offsets = [r["degree_minus_ceiling"] for r in rows]
    report = {
        "horizon": HORIZON,
        "rows": rows,
        "offset_min": min(offsets),
        "offset_max": max(offsets),
        "all_offsets_between_0_and_2": all(0 <= x <= 2 for x in offsets),
        "interpretation": {
            "reachable_rank_is_dynamic_recurrence_scale_fibre": True,
            "ceil_rank_over_block_is_exact_generator_formula": False,
            "same_object_A_sequence_can_measure_dynamic_span": True,
            "synthetic_rank_is_production_measurement": False,
            "historical_matrix_identity_paid": False,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    main()
