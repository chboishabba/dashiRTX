#!/usr/bin/env python3
"""Hash-bound synthetic Block-Wiedemann/mksol action binding receipt.

This is a synthetic diagnostic, not CADO-NFS production semantics.  It retains
one exact executable context linking:

  apply_B, V, recovered F_i, K_0 = V, K_{i+1} = apply_B(K_i),
  action = XOR_i K_i F_i.

The purpose is to provide a concrete same-runtime object for the Lean generic
map F |-> sum_i (M^i V) F_i.  Cross-prover identity remains a separate proof.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np

ROWS, COLS = 924, 512
BLOCK, TERMS, TRAIN_LAST, MAXD = 8, 256, 191, 40
MASK = (1 << 64) - 1
BASEX = 0xBB67AE8584CAA73B
BASEY = 0x3C6EF372FE94F82B
SCHEMA = "rsa260-bidi-mksol-action-binding-v1"


def mix64(x: int) -> int:
    x &= MASK
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & MASK
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & MASK
    x ^= x >> 31
    return x & MASK


def build_A() -> np.ndarray:
    A = np.zeros((ROWS, COLS), dtype=np.uint8)
    for r in range(ROWS):
        deg = 151 if r < 6 else 150
        base = (r * 2654435761 + 0x9E3779B9) % COLS
        idx = (base + np.arange(deg)) % COLS
        A[r, idx] = 1
    return A


A = build_A()


def seed_vec(seed: int) -> np.ndarray:
    bits = np.zeros(ROWS, dtype=np.uint8)
    for w in range(15):
        val = mix64(seed ^ ((0x9E3779B97F4A7C15 * (w + 1)) & MASK))
        lo = 64 * w
        hi = min(lo + 64, ROWS)
        for b in range(hi - lo):
            bits[lo + b] = (val >> b) & 1
    return bits


def build_block(seed: int) -> np.ndarray:
    return np.stack([seed_vec(seed ^ (j << 32)) for j in range(BLOCK)], axis=1)


def make_apply_B(perm: np.ndarray):
    perm = np.asarray(perm)

    def apply(Y: np.ndarray) -> np.ndarray:
        T = (A.T @ Y) & 1
        T = T[perm]
        return (A @ T) & 1

    return apply


def gen_seq(apply, xseed: int, yseed: int) -> np.ndarray:
    X, Y = build_block(xseed), build_block(yseed)
    seq = np.empty((TERMS, BLOCK, BLOCK), dtype=np.uint8)
    for k in range(TERMS):
        seq[k] = (X.T @ Y) & 1
        Y = apply(Y)
    return seq


def solve_gf2(M: np.ndarray, b: np.ndarray):
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


def fit_degree(seq: np.ndarray, d: int):
    kvals = TRAIN_LAST - d + 1
    nu = BLOCK * d
    lhs = np.empty((kvals * BLOCK, nu), dtype=np.uint8)
    for k in range(kvals):
        for i in range(BLOCK):
            lhs[k * BLOCK + i] = seq[k : k + d, i, :].reshape(-1)
    F = np.zeros((d, BLOCK, BLOCK), dtype=np.uint8)
    for outj in range(BLOCK):
        rhs = np.empty(kvals * BLOCK, dtype=np.uint8)
        for k in range(kvals):
            rhs[k * BLOCK : (k + 1) * BLOCK] = seq[k + d, :, outj]
        sol = solve_gf2(lhs, rhs)
        if sol is None:
            return None
        F[:, :, outj] = sol.reshape(d, BLOCK)
    return F


def recurrence_holds(seq: np.ndarray, F: np.ndarray, start: int, end: int) -> bool:
    d = F.shape[0]
    for k in range(start, end - d):
        acc = seq[k + d].copy()
        for l in range(d):
            acc ^= (seq[k + l] @ F[l]) & 1
        if acc.any():
            return False
    return True


def first_generator_with_coefficients(seq: np.ndarray):
    for d in range(1, MAXD + 1):
        F = fit_degree(seq, d)
        if F is None:
            continue
        if recurrence_holds(seq, F, 0, TRAIN_LAST + 1):
            holdout_start = TRAIN_LAST + 1 - d
            if recurrence_holds(seq, F, holdout_start, TERMS):
                return d, F
    raise RuntimeError("no generator found")


def array_sha256(a: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(tuple(int(x) for x in a.shape)).encode("ascii"))
    h.update(b"|")
    h.update(str(a.dtype).encode("ascii"))
    h.update(b"|")
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def compute_binding_receipt() -> dict:
    perm = np.arange(COLS, dtype=np.int64)
    apply_B = make_apply_B(perm)
    seq = gen_seq(apply_B, BASEX, BASEY)
    degree, F = first_generator_with_coefficients(seq)

    V = build_block(BASEY)
    K = [V]
    for _ in range(1, degree):
        K.append(apply_B(K[-1]))
    K_stack = np.stack(K, axis=0)

    recurrence_ok = all(
        np.array_equal(K[i + 1], apply_B(K[i])) for i in range(degree - 1)
    )

    action_from_stored = np.zeros((ROWS, BLOCK), dtype=np.uint8)
    for i in range(degree):
        action_from_stored ^= (K[i] @ F[i]) & 1

    action_streaming = np.zeros((ROWS, BLOCK), dtype=np.uint8)
    Y = V.copy()
    for i in range(degree):
        action_streaming ^= (Y @ F[i]) & 1
        Y = apply_B(Y)

    # Finite executable sanity checks only; they are not formal linearity proofs.
    Y1 = build_block(BASEY ^ (1 << 40))
    Y2 = build_block(BASEY ^ (2 << 40))
    linearity_spotcheck = (
        np.array_equal(apply_B(Y1 ^ Y2), apply_B(Y1) ^ apply_B(Y2))
        and not apply_B(np.zeros_like(Y1)).any()
    )

    return {
        "schema": SCHEMA,
        "synthetic_only": True,
        "exact_cado_mksol_semantics": False,
        "rows": ROWS,
        "cols": COLS,
        "block": BLOCK,
        "degree": int(degree),
        "operator_formula": "Y -> A @ ((A.T @ Y)[perm]) mod 2",
        "permutation": "identity",
        "A_sha256": array_sha256(A),
        "permutation_sha256": array_sha256(perm),
        "V_sha256": array_sha256(V),
        "coefficient_family_sha256": array_sha256(F),
        "krylov_family_sha256": array_sha256(K_stack),
        "action_sha256": array_sha256(action_from_stored),
        "coefficient_family_shape": list(F.shape),
        "seed_block_shape": list(V.shape),
        "krylov_block_shape": list(K[0].shape),
        "action_shape": list(action_from_stored.shape),
        "krylov_recurrence_all_equal": bool(recurrence_ok),
        "action_stored_equals_streaming": bool(
            np.array_equal(action_from_stored, action_streaming)
        ),
        "operator_linearity_spotcheck": bool(linearity_spotcheck),
        "boundary": {
            "runtime_apply_B_is_formal_Lean_M": False,
            "runtime_V_is_formal_Lean_V": False,
            "runtime_F_is_formal_Lean_coefficient_family": False,
            "runtime_action_is_formal_Lean_krylovCoefficientFamilyAction": False,
            "production_RSA260": False,
        },
    }


def main() -> None:
    print(json.dumps(compute_binding_receipt(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
