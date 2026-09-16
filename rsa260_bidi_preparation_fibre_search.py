#!/usr/bin/env python3
import json, time
import numpy as np

ROWS, COLS = 924, 512
BLOCK, TERMS, TRAIN_LAST, MAXD = 8, 256, 191, 40
MASK = (1 << 64) - 1
BASEX = 0xbb67ae8584caa73b
BASEY = 0x3c6ef372fe94f82b

def mix64(x):
    x &= MASK
    x ^= x >> 30; x = (x * 0xbf58476d1ce4e5b9) & MASK
    x ^= x >> 27; x = (x * 0x94d049bb133111eb) & MASK
    x ^= x >> 31
    return x & MASK

def build_A():
    A = np.zeros((ROWS, COLS), dtype=np.uint8)
    for r in range(ROWS):
        deg = 151 if r < 6 else 150
        base = (r * 2654435761 + 0x9e3779b9) % COLS
        idx = (base + np.arange(deg)) % COLS
        A[r, idx] = 1
    return A

A = build_A()

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
    return np.stack([seed_vec(seed ^ (j << 32)) for j in range(BLOCK)], axis=1)

def make_apply_B(perm):
    perm = np.asarray(perm)
    def apply(Y):
        T = (A.T @ Y) & 1
        T = T[perm]
        return (A @ T) & 1
    return apply

def gen_seq(apply, xseed, yseed):
    X, Y = build_block(xseed), build_block(yseed)
    seq = np.empty((TERMS, BLOCK, BLOCK), dtype=np.uint8)
    for k in range(TERMS):
        seq[k] = (X.T @ Y) & 1
        Y = apply(Y)
    return seq

def solve_gf2(M, b):
    M, b = M.copy(), b.copy()
    nr, nc = M.shape
    piv, r = [], 0
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
        piv.append(c)
        r += 1
        if r == nr:
            break
    if r < nr:
        zero = np.where(M[r:].sum(axis=1) == 0)[0]
        if zero.size and np.any(b[r:][zero]):
            return None
    x = np.zeros(nc, dtype=np.uint8)
    for rr, c in enumerate(piv):
        x[c] = b[rr]
    return x

def fit_degree(seq, d):
    kvals = TRAIN_LAST - d + 1
    nu = BLOCK * d
    lhs = np.empty((kvals * BLOCK, nu), dtype=np.uint8)
    for k in range(kvals):
        for i in range(BLOCK):
            lhs[k * BLOCK + i] = seq[k:k+d, i, :].reshape(-1)
    F = np.zeros((d, BLOCK, BLOCK), dtype=np.uint8)
    for outj in range(BLOCK):
        rhs = np.empty(kvals * BLOCK, dtype=np.uint8)
        for k in range(kvals):
            rhs[k*BLOCK:(k+1)*BLOCK] = seq[k+d, :, outj]
        sol = solve_gf2(lhs, rhs)
        if sol is None:
            return None
        F[:, :, outj] = sol.reshape(d, BLOCK)
    return F

def recurrence_holds(seq, F, start, end):
    d = F.shape[0]
    for k in range(start, end - d):
        acc = seq[k+d].copy()
        for l in range(d):
            acc ^= (seq[k+l] @ F[l]) & 1
        if acc.any():
            return False
    return True

def first_generator(seq):
    for d in range(1, MAXD + 1):
        F = fit_degree(seq, d)
        if F is None:
            continue
        if recurrence_holds(seq, F, 0, TRAIN_LAST + 1):
            holdout_start = TRAIN_LAST + 1 - d
            if recurrence_holds(seq, F, holdout_start, TERMS):
                return d
    return None

def rref(M):
    M = M.copy()
    nr, nc = M.shape
    piv, r = [], 0
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
        piv.append(c)
        r += 1
        if r == nr:
            break
    return M, piv

def nullspace_basis(M):
    R, piv = rref(M)
    free = [c for c in range(M.shape[1]) if c not in set(piv)]
    out = []
    for f in free:
        v = np.zeros(M.shape[1], dtype=np.uint8)
        v[f] = 1
        for rr, p in enumerate(piv):
            if R[rr, f]:
                v[p] = 1
        out.append(v)
    return out, len(piv)

def krylov_basis(apply, yseed, degree):
    K = [build_block(yseed)]
    for _ in range(degree):
        K.append(apply(K[-1]))
    return K

def shifted_vector(K, rel, degree):
    v = np.zeros(ROWS, dtype=np.uint8)
    c = 0
    for l in range(1, degree + 1):
        for j in range(BLOCK):
            if rel[c]:
                v ^= K[l-1][:, j]
            c += 1
    return v

def evaluate(apply, xseed, yseed):
    seq = gen_seq(apply, xseed, yseed)
    degree = first_generator(seq)
    if degree is None:
        return {"degree": None, "passed": False}
    K = krylov_basis(apply, yseed, degree)
    shifted = np.concatenate(K[1:], axis=1)
    basis, rank = nullspace_basis(shifted)
    valid_nonzero, zero_shift, weights = 0, 0, []
    for rel in basis:
        v = shifted_vector(K, rel, degree)
        if not v.any():
            zero_shift += 1
            continue
        if not apply(v[:, None]).any() and not ((A.T @ v) & 1).any():
            valid_nonzero += 1
        weights.append(int(v.sum()))
    return {
        "degree": degree,
        "relation_space_dim": len(basis),
        "shifted_rank": rank,
        "valid_nonzero_kernels": valid_nonzero,
        "zero_shift_relations": zero_shift,
        "kernel_weight_min": min(weights) if weights else None,
        "kernel_weight_max": max(weights) if weights else None,
        "passed": valid_nonzero > 0,
    }

def bitrev9(x):
    return int(f"{x:09b}"[::-1], 2)

def adapter_family():
    out = {"identity": np.arange(COLS)}
    for k in (1, 3, 7, 31, 63):
        out[f"rotate{k}"] = (np.arange(COLS) + k) % COLS
    for a, b in ((3,1),(5,17),(7,29),(9,13),(17,5),(33,7),
                 (65,11),(127,19),(255,23),(511,1)):
        out[f"affine_{a}_{b}"] = (a * np.arange(COLS) + b) % COLS
    out["bitrev9"] = np.array([bitrev9(i) for i in range(COLS)])
    out["xor1"] = np.arange(COLS) ^ 1
    out["xor85"] = np.arange(COLS) ^ 85
    out["xor255"] = np.arange(COLS) ^ 255
    return out

def dominates(a, b):
    """Strict Pareto dominance over declared presentation costs."""
    ca = (a["degree"], a["relation_space_dim"],
          a["zero_shift_relations"], a["kernel_weight_min"], a["elapsed_seconds"])
    cb = (b["degree"], b["relation_space_dim"],
          b["zero_shift_relations"], b["kernel_weight_min"], b["elapsed_seconds"])
    return all(x <= y for x, y in zip(ca, cb)) and any(x < y for x, y in zip(ca, cb))

def main():
    results = []
    for name, perm in adapter_family().items():
        t0 = time.perf_counter()
        row = evaluate(make_apply_B(perm), BASEX, BASEY)
        row["adapter"] = name
        row["elapsed_seconds"] = round(time.perf_counter() - t0, 6)
        results.append(row)

    eligible = [r for r in results if r["passed"]]
    frontier = []
    for r in eligible:
        if not any(dominates(q, r) for q in eligible if q is not r):
            frontier.append(r)

    ranked = sorted(
        eligible,
        key=lambda r: (
            r["degree"],
            r["relation_space_dim"],
            r["zero_shift_relations"],
            r["kernel_weight_min"],
            r["kernel_weight_max"],
            r["adapter"],
        ),
    )
    structural_best = ranked[0]

    report = {
        "matrix_shape": [ROWS, COLS],
        "row_excess": ROWS - COLS,
        "consumer": "nonzero y with B_P y = 0 and A^T y = 0",
        "tested_adapters": len(results),
        "passed_adapters": len(eligible),
        "all_passed": len(eligible) == len(results),
        "structural_best": structural_best,
        "pareto_frontier": sorted(frontier, key=lambda r: r["adapter"]),
        "results": sorted(results, key=lambda r: r["adapter"]),
        "firewalls": {
            "best_tested_is_global_optimum": False,
            "preparation_equivalent_to_cado": False,
            "consumer_success_implies_historical_identity": False,
        },
    }
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    main()
