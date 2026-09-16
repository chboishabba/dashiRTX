#!/usr/bin/env python3
"""Hash-bound synthetic Block-Wiedemann/mksol action binding receipt.

This producer specializes the existing `rsa260_bidi_candidate_robustness.py`
carrier to one deterministic baseline context and retains enough identity
information to bind the runtime objects to the generic Lean action

    F |-> sum_i (M^i V) F_i.

It is intentionally synthetic.  It does not claim exact CADO mksol semantics,
production RSA-260 custody, or a Lean/Agda same-object theorem.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

ROWS, COLS = 924, 512
BLOCK, TERMS, TRAIN_LAST, MAXD = 8, 256, 191, 40
MASK = (1 << 64) - 1
BASEX = 0xBB67AE8584CAA73B
BASEY = 0x3C6EF372FE94F82B


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
    perm = np.asarray(perm, dtype=np.int64)

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


def solve_gf2(M: np.ndarray, b: np.ndarray) -> np.ndarray | None:
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


def fit_degree(seq: np.ndarray, d: int) -> np.ndarray | None:
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


def first_generator(seq: np.ndarray) -> tuple[int, np.ndarray]:
    for d in range(1, MAXD + 1):
        F = fit_degree(seq, d)
        if F is None:
            continue
        if recurrence_holds(seq, F, 0, TRAIN_LAST + 1):
            holdout_start = TRAIN_LAST + 1 - d
            if recurrence_holds(seq, F, holdout_start, TERMS):
                return d, F
    raise RuntimeError("no generator")


def array_digest(a: np.ndarray) -> str:
    """Digest shape, dtype and exact bytes so carrier identity is explicit."""
    h = hashlib.sha256()
    h.update(str(tuple(int(x) for x in a.shape)).encode("ascii"))
    h.update(b"|")
    h.update(str(a.dtype).encode("ascii"))
    h.update(b"|")
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def coefficient_action_from_stored_krylov(K: list[np.ndarray], F: np.ndarray) -> np.ndarray:
    out = np.zeros_like(K[0])
    for i in range(F.shape[0]):
        out ^= (K[i] @ F[i]) & 1
    return out


def coefficient_action_streaming(apply, V: np.ndarray, F: np.ndarray) -> np.ndarray:
    out = np.zeros_like(V)
    Y = V.copy()
    for i in range(F.shape[0]):
        out ^= (Y @ F[i]) & 1
        Y = apply(Y)
    return out


def compute_binding_receipt() -> dict[str, Any]:
    perm = np.arange(COLS, dtype=np.int64)
    apply = make_apply_B(perm)
    seq = gen_seq(apply, BASEX, BASEY)
    degree, F = first_generator(seq)

    V = build_block(BASEY)
    K = [V.copy()]
    for _ in range(1, degree):
        K.append(apply(K[-1]))

    recurrence_flags = [
        np.array_equal(K[i + 1], apply(K[i])) for i in range(degree - 1)
    ]
    stored_action = coefficient_action_from_stored_krylov(K, F)
    streaming_action = coefficient_action_streaming(apply, V, F)

    # Deterministic linearity spot check on the actual runtime operator.
    X = build_block(BASEX)
    Y = build_block(BASEY)
    operator_linearity_spotcheck = np.array_equal(
        apply(X ^ Y), apply(X) ^ apply(Y)
    )

    Kstack = np.stack(K, axis=0)
    return {
        "schema": "rsa260-bidi-mksol-action-binding-v1",
        "runtime_source_donor": "rsa260_bidi_candidate_robustness.py",
        "context": "identity adapter / BASEX / BASEY",
        "rows": ROWS,
        "cols": COLS,
        "block": BLOCK,
        "degree": degree,
        "coefficient_family_shape": list(F.shape),
        "seed_block_shape": list(V.shape),
        "krylov_block_shape": list(K[0].shape),
        "krylov_family_shape": list(Kstack.shape),
        "action_shape": list(stored_action.shape),
        "krylov_recurrence_all_equal": all(recurrence_flags),
        "action_stored_equals_streaming": bool(
            np.array_equal(stored_action, streaming_action)
        ),
        "operator_linearity_spotcheck": bool(operator_linearity_spotcheck),
        "digests": {
            "A": array_digest(A),
            "identity_permutation": array_digest(perm),
            "V": array_digest(V),
            "F_family": array_digest(F),
            "K_family": array_digest(Kstack),
            "action": array_digest(stored_action),
        },
        "boundary": {
            "synthetic_runtime_binding_executed": True,
            "exact_cado_mksol_semantics": False,
            "runtime_apply_B_is_formal_lean_M": False,
            "runtime_V_is_formal_lean_V": False,
            "runtime_F_is_formal_lean_coefficient_family": False,
            "runtime_action_is_formal_lean_krylov_action": False,
            "production_rsa260": False,
        },
    }


def main() -> None:
    print(json.dumps(compute_binding_receipt(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
