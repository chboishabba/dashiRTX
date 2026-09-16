#!/usr/bin/env python3
"""Finite certificate for the synthetic two-V mksol action family.

For degree 17 and block width 8, one output column of the coefficient family has
17*8 = 136 GF(2) coordinates.  For a fixed seed block V, the mksol-style action
uses the 924x136 Krylov-column matrix

    K(V) = [V | M V | ... | M^16 V].

The baseline V leaves an 8-dimensional kernel in that 136-coordinate column
map (rank 128).  Stacking the baseline with the independent synthetic block

    V' = build_block(BASEY xor (1 << 40))

has full column rank 136.  Instead of retaining only that rank numeral, this
producer emits a finite proof certificate:

* 136 concrete rows of the stacked 1848x136 matrix;
* the selected 136x136 square minor;
* an explicit 136x136 GF(2) inverse, packed row-major;
* checks of both B*B^-1 = I and B^-1*B = I.

Because the 8 output columns of the full coefficient action are independent
copies of the same column map, this certificate witnesses zero intersection
kernel for the checked two-V synthetic family on the 17x8x8 coefficient carrier.

This is synthetic/runtime evidence only.  It is not exact CADO mksol semantics,
does not identify historical F.sols/V artifacts, and is not a Lean kernel
receipt.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np

from rsa260_bidi_mksol_action_binding import (
    BASEY,
    BLOCK,
    COLS,
    ROWS,
    array_sha256,
    build_block,
    make_apply_B,
)

SCHEMA = "rsa260-bidi-two-v-joint-kernel-certificate-v1"
DEGREE = 17
SECOND_SEED_XOR = 1 << 40


def gf2_rank(M: np.ndarray) -> int:
    R = M.copy().astype(np.uint8)
    nr, nc = R.shape
    row = 0
    for col in range(nc):
        nz = np.flatnonzero(R[row:, col])
        if nz.size == 0:
            continue
        pivot = row + int(nz[0])
        if pivot != row:
            R[[row, pivot]] = R[[pivot, row]]
        nz = np.flatnonzero(R[:, col])
        nz = nz[nz != row]
        if nz.size:
            R[nz] ^= R[row]
        row += 1
        if row == nr:
            break
    return row


def krylov_columns(yseed: int) -> np.ndarray:
    apply_B = make_apply_B(np.arange(COLS, dtype=np.int64))
    blocks = [build_block(yseed)]
    for _ in range(1, DEGREE):
        blocks.append(apply_B(blocks[-1]))
    out = np.concatenate(blocks, axis=1)
    if out.shape != (ROWS, DEGREE * BLOCK):
        raise AssertionError(f"unexpected Krylov-column shape {out.shape}")
    return out


def independent_row_indices(M: np.ndarray) -> list[int]:
    """Return pivot columns of M^T, i.e. independent row indices of M."""
    R = M.T.copy().astype(np.uint8)
    nr, nc = R.shape
    row = 0
    pivots: list[int] = []
    for col in range(nc):
        nz = np.flatnonzero(R[row:, col])
        if nz.size == 0:
            continue
        pivot = row + int(nz[0])
        if pivot != row:
            R[[row, pivot]] = R[[pivot, row]]
        nz = np.flatnonzero(R[:, col])
        nz = nz[nz != row]
        if nz.size:
            R[nz] ^= R[row]
        pivots.append(col)
        row += 1
        if row == nr:
            break
    return pivots


def gf2_inverse(M: np.ndarray) -> np.ndarray:
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"expected square matrix, got {M.shape}")
    n = M.shape[0]
    aug = np.concatenate([M.copy().astype(np.uint8), np.eye(n, dtype=np.uint8)], axis=1)
    row = 0
    for col in range(n):
        nz = np.flatnonzero(aug[row:, col])
        if nz.size == 0:
            raise ValueError("matrix is singular")
        pivot = row + int(nz[0])
        if pivot != row:
            aug[[row, pivot]] = aug[[pivot, row]]
        nz = np.flatnonzero(aug[:, col])
        nz = nz[nz != row]
        if nz.size:
            aug[nz] ^= aug[row]
        row += 1
    return aug[:, n:]


def packed_row_bytes(M: np.ndarray) -> bytes:
    """Pack row bits little-endian within each byte."""
    return np.packbits(M.astype(np.uint8), axis=1, bitorder="little").tobytes()


def compute_certificate() -> dict:
    baseline = krylov_columns(BASEY)
    second = krylov_columns(BASEY ^ SECOND_SEED_XOR)
    stacked = np.vstack([baseline, second])

    baseline_rank = gf2_rank(baseline)
    stacked_rank = gf2_rank(stacked)
    selected_rows = independent_row_indices(stacked)
    if len(selected_rows) != DEGREE * BLOCK:
        raise AssertionError(f"expected 136 independent rows, got {len(selected_rows)}")

    square = stacked[selected_rows, :]
    inverse = gf2_inverse(square)
    identity = np.eye(DEGREE * BLOCK, dtype=np.uint8)
    left_ok = np.array_equal((square @ inverse) & 1, identity)
    right_ok = np.array_equal((inverse @ square) & 1, identity)
    inverse_bytes = packed_row_bytes(inverse)

    return {
        "schema": SCHEMA,
        "synthetic_only": True,
        "degree": DEGREE,
        "block": BLOCK,
        "coefficient_column_bits": DEGREE * BLOCK,
        "baseline_seed": "BASEY",
        "second_seed_formula": "BASEY xor (1 << 40)",
        "baseline_krylov_column_shape": list(baseline.shape),
        "second_krylov_column_shape": list(second.shape),
        "stacked_krylov_column_shape": list(stacked.shape),
        "baseline_krylov_column_rank": int(baseline_rank),
        "stacked_krylov_column_rank": int(stacked_rank),
        "selected_row_indices": [int(x) for x in selected_rows],
        "selected_square_shape": list(square.shape),
        "selected_square_rank": int(gf2_rank(square)),
        "selected_square_sha256": array_sha256(square),
        "inverse_row_bytes_count": len(inverse_bytes),
        "inverse_row_bytes_hex": inverse_bytes.hex(),
        "inverse_row_bytes_sha256": hashlib.sha256(inverse_bytes).hexdigest(),
        "left_inverse_identity": bool(left_ok),
        "right_inverse_identity": bool(right_ok),
        "two_v_intersection_kernel_zero": bool(stacked_rank == DEGREE * BLOCK and left_ok and right_ok),
        "exact_cado_mksol_context_family": False,
        "production_rsa260": False,
        "boundary": {
            "certificate_is_lean_kernel_receipt": False,
            "certificate_is_production_cado_receipt": False,
            "same_object_historical_V_artifacts": False,
        },
    }


def main() -> None:
    print(json.dumps(compute_certificate(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
