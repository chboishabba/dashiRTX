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

adapters = {
    "identity": np.arange(COLS),
    "rotate1": (np.arange(COLS) + 1) % COLS,
    "affine5": (5 * np.arange(COLS) + 17) % COLS,
    "bitrev9": np.array([bitrev9(i) for i in range(COLS)]),
}

t0 = time.time()

identity = make_apply_B(adapters["identity"])
seed_runs = []
for s in range(8):
    seed_runs.append({
        "seed_index": s,
        **evaluate(
            identity,
            BASEX ^ ((s + 1) << 48),
            BASEY ^ ((s + 1) << 40),
        ),
    })

adapter_runs = []
for name, perm in adapters.items():
    adapter_runs.append({
        "adapter": name,
        **evaluate(make_apply_B(perm), BASEX, BASEY),
    })

out = {
    "matrix_shape": [ROWS, COLS],
    "row_excess": ROWS - COLS,
    "seed_runs": seed_runs,
    "adapter_runs": adapter_runs,
    "all_seed_runs_passed": all(x["passed"] for x in seed_runs),
    "all_adapter_runs_passed": all(x["passed"] for x in adapter_runs),
    "elapsed_seconds": round(time.time() - t0, 3),
}
print(json.dumps(out, indent=2))
